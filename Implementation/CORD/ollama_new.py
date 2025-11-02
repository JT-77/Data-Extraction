import os
import json
import base64
from io import BytesIO
from PIL import Image
import requests
import numpy as np
from datasets import load_dataset
from thefuzz import fuzz
from thefuzz import process
import warnings
import time
import re
import sys
import logging
import Levenshtein

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="PIL")

# --- Configuration ---
OLLAMA_HOST = "http://localhost:11434"
# OLLAMA_MODEL_NAME = "mistral-small3.2:latest"
# OLLAMA_MODEL_NAME = "qwen2.5vl:latest"
OLLAMA_MODEL_NAME = "llama3.2-vision:latest"
MAX_SAMPLES_TO_PROCESS = 100
LOG_FILE = "new_results/llama_cord_evaluation_log.jsonl"

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define labels ---
TARGET_LABELS = frozenset({
    "menu.nm",
    "menu.cnt",
    "menu.price",
    "menu.num",          
    "menu.itemsubtotal", 
    "sub_total.discount_price",
    "total.total_price",
})

EXACT_MATCH_LABELS = frozenset({
    "menu.cnt",
    "menu.price",
    "menu.num", 
    "sub_total.discount_price", 
    "total.total_price", 
    "menu.itemsubtotal" 
})

GLOBAL_ALL_IOB2_LABELS = []
GLOBAL_LABEL2ID = {}
GLOBAL_ID2LABEL = {}

# --- Text Normalization Helper ---
def normalize_text_for_matching(text, is_numeric_field=False):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    if is_numeric_field:
        text = text.replace('-', '').replace(',', '').replace('.', '').replace('x', '')
        text = re.sub(r'[a-z]', '', text)
        text = text.strip()
        if re.match(r'^0+$', text):
            text = "0"
        elif text and re.match(r'^0[0-9]', text):
            text = text.lstrip('0')
            if not text:
                text = "0"
        return text.strip()

    text = text.replace('$', '').replace('€', '').replace('£', '')

    text = text.replace('%', '').replace('+', '').replace('*', '')
    text = re.sub(r'\s+[a-z0-9]$', '', text).strip()
    text = re.sub(r'^\d+\s*(?:x|pcs|gr|ml|oz)?\s*', '', text).strip()
    text = re.sub(r'^\-\s*', '', text, flags=re.IGNORECASE).strip()
    prefixes_to_remove_specific = [r'no:\s*', r'item\s*', r'product\s*', r'desc\s*', r'name\s*', r's\s*']
    for prefix in prefixes_to_remove_specific:
        text = re.sub(f'^{prefix}', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- Helper to convert PIL Image to base64 ---
def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Extract words, bboxes, and labels from ground truth ---
def extract_words_bboxes_and_labels_from_gt(ground_truth_json_str, image_width, image_height):
    data = json.loads(ground_truth_json_str)
    words_on_page = []
    bboxes_on_page = []
    semantic_entities = []
    labels_present_in_gt = set()

    def _add_gt_entity(label, text_value):
        if label in TARGET_LABELS:
            is_numeric_category = label in [
                "menu.cnt", "menu.price", "menu.itemsubtotal", "menu.num",
                "sub_total.subtotal_price", "sub_total.discount_price",
                "sub_total.service_price", "sub_total.tax_price",
                "total.total_price", "total.cashprice", "total.change_price",
                "total.creditcardprice", "total.menuqty_cnt"
            ]
            normalized_text = normalize_text_for_matching(text_value, is_numeric_field=is_numeric_category)
            if normalized_text:
                semantic_entities.append({"label": label, "text": normalized_text})
                labels_present_in_gt.add(label)

    gt_parse = data.get("gt_parse", {})
    sub_total_gt = gt_parse.get("sub_total", {})
    if isinstance(sub_total_gt, dict):
        if "subtotal_price" in sub_total_gt:
            _add_gt_entity("sub_total.subtotal_price", sub_total_gt["subtotal_price"])
        if "discount_price" in sub_total_gt:
            _add_gt_entity("sub_total.discount_price", sub_total_gt["discount_price"])

    total_gt = gt_parse.get("total", {})
    if isinstance(total_gt, dict):
        if "total_price" in total_gt:
            _add_gt_entity("total.total_price", total_gt["total_price"])

    menu_gt_items = gt_parse.get("menu")
    if isinstance(menu_gt_items, dict):
        menu_gt_items = [menu_gt_items]
    elif not isinstance(menu_gt_items, list):
        menu_gt_items = []

    for item in menu_gt_items:
        if isinstance(item, dict):
            if "nm" in item: _add_gt_entity("menu.nm", item["nm"])
            if "cnt" in item: _add_gt_entity("menu.cnt", item["cnt"])
            if "price" in item: _add_gt_entity("menu.price", item["price"])
            if "num" in item: _add_gt_entity("menu.num", item["num"])
            if "itemsubtotal" in item: _add_gt_entity("menu.itemsubtotal", item["itemsubtotal"])
            sub_item = item.get("sub")
            if isinstance(sub_item, dict):
                if "nm" in sub_item: _add_gt_entity("menu.nm", sub_item["nm"])
                if "cnt" in sub_item: _add_gt_entity("menu.cnt", sub_item["cnt"])
                if "price" in sub_item: _add_gt_entity("menu.price", sub_item["price"])
                if "num" in sub_item: _add_gt_entity("menu.num", sub_item["num"])
                if "itemsubtotal" in sub_item: _add_gt_entity("menu.itemsubtotal", sub_item["itemsubtotal"])
            elif isinstance(sub_item, list):
                for sub_sub_item in sub_item:
                    if isinstance(sub_sub_item, dict):
                        if "nm" in sub_sub_item: _add_gt_entity("menu.nm", sub_sub_item["nm"])
                        if "cnt" in sub_sub_item: _add_gt_entity("menu.cnt", sub_sub_item["cnt"])
                        if "price" in sub_sub_item: _add_gt_entity("menu.price", sub_sub_item["price"])
                        if "num" in sub_sub_item: _add_gt_entity("menu.num", sub_sub_item["num"])
                        if "itemsubtotal" in sub_sub_item: _add_gt_entity("menu.itemsubtotal", sub_sub_item["itemsubtotal"])

    for line_data in data["valid_line"]:
        for word_info in line_data["words"]:
            words_on_page.append(word_info["text"])
            quad = word_info["quad"]
            x_coords = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
            y_coords = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            normalized_bbox = [
                int(1000 * (x_min / image_width)),
                int(1000 * (y_min / image_height)),
                int(1000 * (x_max / image_width)),
                int(1000 * (y_max / image_height))
            ]
            normalized_bbox = [max(0, min(1000, coord)) for coord in normalized_bbox]
            if normalized_bbox[0] > normalized_bbox[2]: normalized_bbox[0], normalized_bbox[2] = normalized_bbox[2], normalized_bbox[0]
            if normalized_bbox[1] > normalized_bbox[3]: normalized_bbox[1], normalized_bbox[3] = normalized_bbox[3], normalized_bbox[1]
            bboxes_on_page.append(normalized_bbox)

    return words_on_page, bboxes_on_page, semantic_entities, frozenset(labels_present_in_gt)

# --- Dynamic Label Initialization ---
def initialize_global_labels(dataset):
    ALL_BASE_LABELS_LOCAL = set()
    for split in ["train", "validation", "test"]:
        for i in range(min(len(dataset[split]), 200)):
            ground_truth_str = dataset[split][i]["ground_truth"]
            image_width = dataset[split][i]["image"].size[0]
            image_height = dataset[split][i]["image"].size[1]
            _, _, entities, _ = extract_words_bboxes_and_labels_from_gt(ground_truth_str, image_width, image_height)
            for entity in entities:
                ALL_BASE_LABELS_LOCAL.add(entity['label'])
    
    sorted_base_labels = sorted(list(ALL_BASE_LABELS_LOCAL))
    all_iob2_labels = ["O"]
    for base_label in sorted_base_labels:
        all_iob2_labels.append(f"B-{base_label.upper()}")
        all_iob2_labels.append(f"I-{base_label.upper()}")
    
    global GLOBAL_ID2LABEL, GLOBAL_LABEL2ID, GLOBAL_ALL_IOB2_LABELS
    GLOBAL_ID2LABEL = {i: label for i, label in enumerate(all_iob2_labels)}
    GLOBAL_LABEL2ID = {label: i for i, label in enumerate(all_iob2_labels)}
    GLOBAL_ALL_IOB2_LABELS = all_iob2_labels

# --- Prompt for Ollama ---
def create_ollama_prompt_for_cord(labels_to_request: frozenset):
    prompt_fields_schema = {}
    prompt_fields_example = {}

    menu_item_schema = {}
    menu_item_example = {}
    if "menu.nm" in labels_to_request: 
        menu_item_schema["name"] = "string (Name of the item) except PB1 and % texts"
        menu_item_example["name"] = "Apples"
    if "menu.cnt" in labels_to_request: 
        menu_item_schema["count"] = "string (Quantity, e.g., '1x', '2')"
        menu_item_example["count"] = "2"
    if "menu.price" in labels_to_request:
        menu_item_schema["price"] = "string (Total price for this specific quantity of the item, e.g., '10.50' for 2 items @ 5.25 each)"
        menu_item_example["price"] = "10.50"
    if "menu.num" in labels_to_request: 
        menu_item_schema["num"] = "string (Item number or ID, if present)"
        menu_item_example["num"] = "A1"
    if "menu.itemsubtotal" in labels_to_request: 
        menu_item_schema["item_subtotal"] = "string (Subtotal for this specific item, if present)"
        menu_item_example["item_subtotal"] = "6.00"
    
    if menu_item_schema:
        prompt_fields_schema["menu_items"] = f"""
    - menu_items: array of objects exept "PB1" and text with "%" symbols (e.g., [{json.dumps(menu_item_example)}])
        - Each object should have:
            {chr(10).join([f"- {k}: {v}" for k,v in menu_item_schema.items()])}
            """
        prompt_fields_example["menu_items"] = [menu_item_example]

    sub_total_schema_parts = []
    sub_total_example_parts = []
    if "sub_total.subtotal_price" in labels_to_request: 
        sub_total_schema_parts.append('"subtotal_price": "string (Subtotal amount)"')
        sub_total_example_parts.append('"subtotal_price": "95.50"')
    if "sub_total.discount_price" in labels_to_request: 
        sub_total_schema_parts.append('"discount_price": "string (Discount amount, if any, e.g., \'-5.00\')"')
        sub_total_example_parts.append('"discount_price": "-5.00"')
    
    if sub_total_schema_parts:
        prompt_fields_schema["sub_total"] = f"""
    - sub_total: object (e.g., {{{", ".join(sub_total_example_parts)}}})
        - {chr(10).join([s.replace('"', '') for s in sub_total_schema_parts])}
        """
        prompt_fields_example["sub_total"] = json.loads("{" + ", ".join(sub_total_example_parts) + "}")

    total_schema_parts = []
    total_example_parts = []
    if "total.total_price" in labels_to_request: 
        total_schema_parts.append('"total_price": "string (The final total amount). Do not check the CASH given or change given just the total bill amount."')
        total_example_parts.append('"total_price": "108.00"')
    
    if total_schema_parts:
        prompt_fields_schema["total"] = f"""
    - total: object (e.g., {{{", ".join(total_example_parts)}}})
        - {chr(10).join([s.replace('"', '') for s in total_schema_parts])}
        """
        prompt_fields_example["total"] = json.loads("{" + ", ".join(total_example_parts) + "}")

    prompt_parts = []
    prompt_parts.append("""
    You are an expert at extracting structured information from receipt images.
    Analyze the provided receipt image and Only output JSON format in your response (nothing else, no text).
    Provide the output as a JSON object. If a field is not found or cannot be confidently extracted, omit it or set its value to null.

    **Important Notes for Extraction:**
    1. **Strictly adhere to the JSON schema and only extract the fields explicitly requested.** Do not hallucinate or include any other fields.
    2. **Accuracy for Numeric Values:** Extract all numeric values (prices, counts, totals) exactly as they appear, without adding or removing digits. For all items, do not give per quantity price - always extract total price of that quantity written. For the final bill amount "total_price": do not give the cash amount given by the customer please.
    3. **Clean Values:** For all fields, extract *only* the relevant text or numeric value. Do NOT include any surrounding descriptive words, currency symbols, quantity prefixes (like 'x', 'qty'), or punctuation unless it's part of the core name or a decimal point. For discount, please use your mind that it cannot be more than total amount.
    4. **Menu Items:** Each object in `menu_items` must represent a item entry found on the receipt. Do not combine multiple menu items into one entry, or split a single item across multiple entries. Do not miss any items in the receipt, one item can also be in 2 lines if the names are long. 
        "PB1" or text with "less ice" is a tax not a menu item, do not give it as an item answer. Moreover, if you see an item only single time - do not put it 2 times (eg - "EGG TART"). If there is no price in front of an item then please exclude it from the items. **Crucially, extract ALL instances of an item, even if it appears multiple times on the receipt (e.g., if "Coke" appears twice, extract two separate "Coke" entries).**

    Required fields:
    """)

    if "sub_total" in prompt_fields_schema: prompt_parts.append(prompt_fields_schema["sub_total"])
    if "total" in prompt_fields_schema: prompt_parts.append(prompt_fields_schema["total"])
    if "menu_items" in prompt_fields_schema: 
        prompt_parts.append(prompt_fields_schema["menu_items"])

    prompt_parts.append("\nOutput example:\n```json\n")
    prompt_parts.append(json.dumps(prompt_fields_example, indent=2))
    prompt_parts.append("\n```\nPlease provide only the JSON output within markdown triple backticks.")
    
    final_prompt = "\n".join(prompt_parts)
    final_prompt = re.sub(r'\n\s*\n', '\n\n', final_prompt).strip()
    return final_prompt

# --- Call Ollama ---
async def call_ollama_vlm(image_base64: str, prompt: str):
    url = f"{OLLAMA_HOST}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": -1 
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        full_response = response.json()
        
        generated_content = full_response.get("response", "").strip()
        if not generated_content:
            print("Warning: Ollama returned an empty response.")
            return None

        if generated_content.startswith("```json"):
            json_start_idx = generated_content.find('{')
            json_end_idx = generated_content.rfind('}') + 1
            if json_start_idx != -1 and json_end_idx != -1 and json_end_idx > json_start_idx:
                json_string = generated_content[json_start_idx:json_end_idx]
            else:
                print(f"Warning: JSON markdown block found but structure unexpected. Raw: {generated_content[:200]}")
                json_string = generated_content.replace("```json", "").replace("```", "").strip()
        elif "{" in generated_content:
            json_start_idx = generated_content.find('{')
            json_end_idx = generated_content.rfind('}') + 1
            if json_start_idx != -1 and json_end_idx != -1 and json_end_idx > json_start_idx:
                json_string = generated_content[json_start_idx:json_end_idx]
            else:
                print(f"Warning: JSON markdown block found but structure unexpected. Raw: {generated_content[:200]}")
                json_string = generated_content.replace("```json", "").replace("```", "")
        else:
            json_string = generated_content

        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print(f"Warning: Ollama did not return valid JSON. Raw response snippet: {generated_content[:500]}...")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None

# --- Post-process Ollama Output ---
def postprocess_ollama_output_to_flattened_entities(llm_output: str):
    json_str = llm_output
    print(json_str)
    # if json_str.startswith("```json"):
    #     json_str = json_str[len("```json"):]
    # if json_str.endswith("```"):
    #     json_str = json_str[:-len("```")]

    extracted_data = json_str
    # try:
    #     extracted_data = json.loads(json_str)
    # except json.JSONDecodeError as e:
    #     print(f"Could not parse JSON from Ollama output: {e}. Raw output (snippet): {llm_output[:200]}...", file=sys.stderr)
    #     return []

    flattened_extracted_entities = []
    numeric_labels = frozenset({
        "menu.cnt", "menu.price", "menu.itemsubtotal", "menu.num",
        "sub_total.subtotal_price", "sub_total.discount_price",
        "sub_total.service_price", "sub_total.tax_price", 
        "total.total_price", "total.cashprice", "total.change_price", 
        "total.creditcardprice", "total.menuqty_cnt" 
    })

    def _add_llm_entity(label, value):
        if label not in TARGET_LABELS:
            return
        if value is None or (isinstance(value, str) and value.lower().strip() in ["null", "not provided"]):
            return
        is_numeric_category = label in numeric_labels
        normalized_value = normalize_text_for_matching(str(value), is_numeric_field=is_numeric_category)
        if normalized_value:
            flattened_extracted_entities.append({"label": label, "text": normalized_value})

    sub_total = extracted_data.get("sub_total")
    if isinstance(sub_total, dict):
        _add_llm_entity("sub_total.subtotal_price", sub_total.get("subtotal_price"))
        _add_llm_entity("sub_total.discount_price", sub_total.get("discount_price"))

    total = extracted_data.get("total")
    if isinstance(total, dict):
        _add_llm_entity("total.total_price", total.get("total_price"))

    menu_items = extracted_data.get("menu_items", [])
    if isinstance(menu_items, list):
        for item in menu_items:
            if isinstance(item, dict):
                _add_llm_entity("menu.nm", item.get("name"))
                _add_llm_entity("menu.cnt", item.get("count"))
                _add_llm_entity("menu.price", item.get("price"))
                _add_llm_entity("menu.num", item.get("num"))
                _add_llm_entity("menu.itemsubtotal", item.get("item_subtotal"))

    return flattened_extracted_entities

# --- KIE Metrics Calculation ---
def calculate_kie_metrics(gt_entities: list, pred_entities: list, fuzz_threshold=65):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_predicted_entities = []
    false_negatives_list = []
    false_positives_list = []

    gt_consumed = [False] * len(gt_entities)
    pred_consumed = [False] * len(pred_entities)

    for i, gt_entity in enumerate(gt_entities):
        if gt_consumed[i]: continue
        gt_label = gt_entity["label"]
        gt_text = gt_entity["text"]
        current_fuzz_threshold = 100 if gt_label in EXACT_MATCH_LABELS else (65 if gt_label == "menu.nm" else fuzz_threshold)
        fuzzy_match_func = fuzz.ratio if gt_label in EXACT_MATCH_LABELS else fuzz.token_set_ratio

        best_match_idx = -1
        best_match_score = -1
        for j, pred_entity in enumerate(pred_entities):
            if pred_consumed[j] or gt_label != pred_entity["label"]: continue
            score = fuzzy_match_func(gt_text, pred_entity["text"])
            if score > best_match_score:
                best_match_score = score
                best_match_idx = j
        
        if best_match_idx != -1:
            if best_match_score >= current_fuzz_threshold:
                true_positives += 1
                gt_consumed[i] = True
                pred_consumed[best_match_idx] = True
                matched_predicted_entities.append(pred_entities[best_match_idx])
            else:
                false_negatives += 1
                false_negatives_list.append(gt_entity)
                gt_consumed[i] = True
                pred_consumed[best_match_idx] = True
        else:
            false_negatives += 1
            false_negatives_list.append(gt_entity)
    
    for i, pred_entity in enumerate(pred_entities):
        if not pred_consumed[i] and pred_entity["label"] != "menu.cnt":
            false_positives += 1
            false_positives_list.append(pred_entity)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision, 
        "recall": recall, 
        "f1": f1, 
        "true_positives": true_positives, 
        "false_positives": false_positives, 
        "false_negatives": false_negatives,
        "matched_predicted_entities": matched_predicted_entities,
        "false_negatives_list": false_negatives_list, 
        "false_positives_list": false_positives_list  
    }

# --- Additional Metrics Calculation ---
def compute_anls(gt_entities, pred_entities, threshold=0.5):
    total_anls = 0.0
    count = 0
    gt_consumed = [False] * len(gt_entities)
    pred_consumed = [False] * len(pred_entities)

    for i, gt_entity in enumerate(gt_entities):
        gt_label = gt_entity["label"]
        gt_text = gt_entity["text"]
        current_fuzz_threshold = 100 if gt_label in EXACT_MATCH_LABELS else 65 if gt_label == "menu.nm" else 70
        fuzzy_match_func = fuzz.ratio if gt_label in EXACT_MATCH_LABELS else fuzz.token_set_ratio

        best_match_idx = -1
        best_match_score = -1
        for j, pred_entity in enumerate(pred_entities):
            if pred_consumed[j] or gt_label != pred_entity["label"]:
                continue
            score = fuzzy_match_func(gt_text, pred_entity["text"])
            if score > best_match_score:
                best_match_score = score
                best_match_idx = j

        if best_match_idx != -1 and best_match_score >= current_fuzz_threshold:
            distance = Levenshtein.distance(gt_text, pred_entities[best_match_idx]["text"])
            max_len = max(len(gt_text), len(pred_entities[best_match_idx]["text"]))
            similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
            total_anls += similarity if similarity >= threshold else 0.0
            count += 1
            gt_consumed[i] = True
            pred_consumed[best_match_idx] = True
        else:
            total_anls += 0.0
            count += 1

    return total_anls / count if count > 0 else 0.0

def compute_exact_match(gt_entities, pred_entities):
    total_em = 0
    gt_consumed = [False] * len(gt_entities)
    pred_consumed = [False] * len(pred_entities)

    for i, gt_entity in enumerate(gt_entities):
        gt_label = gt_entity["label"]
        gt_text = gt_entity["text"]
        for j, pred_entity in enumerate(pred_entities):
            if pred_consumed[j] or gt_label != pred_entity["label"]:
                continue
            if gt_text == pred_entity["text"]:
                total_em += 1
                gt_consumed[i] = True
                pred_consumed[j] = True
                break
    return total_em / len(gt_entities) if gt_entities else 0.0

def compute_adjusted_ned(gt_entities, pred_entities):
    total_ned = 0.0
    count = 0
    gt_consumed = [False] * len(gt_entities)
    pred_consumed = [False] * len(pred_entities)

    for i, gt_entity in enumerate(gt_entities):
        gt_label = gt_entity["label"]
        gt_text = gt_entity["text"]
        min_distance = float('inf')
        best_match_idx = -1
        for j, pred_entity in enumerate(pred_entities):
            if pred_consumed[j] or gt_label != pred_entity["label"]:
                continue
            distance = Levenshtein.distance(gt_text, pred_entity["text"])
            if distance < min_distance:
                min_distance = distance
                best_match_idx = j
        if best_match_idx != -1:
            max_len = max(len(gt_text), len(pred_entities[best_match_idx]["text"]))
            total_ned += min_distance / max_len if max_len > 0 else 0.0
            count += 1
            gt_consumed[i] = True
            pred_consumed[best_match_idx] = True
        else:
            total_ned += 1.0
            count += 1

    return total_ned / count if count > 0 else 0.0

def compute_tokens_found_added(gt_entities, pred_entities):
    total_tokens_found = 0
    total_tokens_added = 0
    gt_consumed = [False] * len(gt_entities)
    pred_consumed = [False] * len(pred_entities)

    for i, gt_entity in enumerate(gt_entities):
        gt_label = gt_entity["label"]
        gt_text = gt_entity["text"]
        gt_tokens = gt_text.split()
        best_match_idx = -1
        max_common_tokens = 0
        for j, pred_entity in enumerate(pred_entities):
            if pred_consumed[j] or gt_label != pred_entity["label"]:
                continue
            pred_tokens = pred_entity["text"].split()
            common_tokens = set(pred_tokens) & set(gt_tokens)
            if len(common_tokens) > max_common_tokens:
                max_common_tokens = len(common_tokens)
                best_match_idx = j
        if best_match_idx != -1:
            pred_tokens = pred_entities[best_match_idx]["text"].split()
            total_tokens_found += max_common_tokens
            total_tokens_added += len(pred_tokens) - max_common_tokens
            gt_consumed[i] = True
            pred_consumed[best_match_idx] = True

    for j, pred_entity in enumerate(pred_entities):
        if not pred_consumed[j] and pred_entity["label"] != "menu.cnt":
            pred_tokens = pred_entity["text"].split()
            total_tokens_added += len(pred_tokens)

    return total_tokens_found, total_tokens_added

def compute_kieval_metrics(gt_entities, pred_entities):
    kie_metrics = calculate_kie_metrics(gt_entities, pred_entities)
    entity_f1 = kie_metrics["f1"]
    true_positives = kie_metrics["true_positives"]
    false_positives = kie_metrics["false_positives"]
    false_negatives = kie_metrics["false_negatives"]
    false_negatives_list = kie_metrics["false_negatives_list"]
    false_positives_list = kie_metrics["false_positives_list"]

    group_match = 1 if false_negatives == 0 and false_positives == 0 else 0
    group_precision = group_match / 1 if (true_positives + false_positives) > 0 else 0.0
    group_recall = group_match / 1 if (true_positives + false_negatives) > 0 else 0.0
    group_f1 = (2 * group_precision * group_recall) / (group_precision + group_recall) if (group_precision + group_recall) > 0 else 0.0

    correction_cost = false_negatives + false_positives
    aligned_score = 1.0 - (correction_cost / max(len(gt_entities), len(pred_entities))) if max(len(gt_entities), len(pred_entities)) > 0 else 0.0

    return {
        "entity_f1": entity_f1,
        "group_f1": group_f1,
        "aligned_score": aligned_score,
        "correction_cost": correction_cost,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "false_negatives_list": false_negatives_list,
        "false_positives_list": false_positives_list
    }

# --- Calculate All Metrics ---
def calculate_all_metrics(gt_entities, pred_entities):
    anls = compute_anls(gt_entities, pred_entities)
    em = compute_exact_match(gt_entities, pred_entities)
    ned = compute_adjusted_ned(gt_entities, pred_entities)
    tokens_found, tokens_added = compute_tokens_found_added(gt_entities, pred_entities)
    kieval = compute_kieval_metrics(gt_entities, pred_entities)

    return {
        "anls": anls,
        "em": em,
        "ned": ned,
        "tokens_found": tokens_found,
        "tokens_added": tokens_added,
        "kieval": kieval
    }

# --- Main Evaluation Function ---
async def evaluate_ollama_on_cord():
    print("--- Starting Ollama evaluation for CORD v2 ---")
    print(f"Ollama Model: {OLLAMA_MODEL_NAME}")
    print(f"Evaluating only the following TARGET_LABELS: {sorted(list(TARGET_LABELS))}")

    # Load dataset
    dataset = load_dataset("naver-clova-ix/cord-v2")
    test_dataset = dataset["validation"]
    total_samples = len(test_dataset)
    print(f"Total test samples available: {total_samples}")
    print(f"Processing a maximum of {MAX_SAMPLES_TO_PROCESS} samples.")

    # Initialize global labels
    initialize_global_labels(dataset)

    # Aggregate metrics
    all_gt_flattened_entities_list = []
    all_predicted_flattened_entities_list = []
    total_anls = 0.0
    total_em = 0.0
    total_ned = 0.0
    total_tokens_found = 0
    total_tokens_added = 0
    total_entity_f1 = 0.0
    total_group_f1 = 0.0
    total_aligned_score = 0.0
    total_correction_cost = 0
    total_time = 0.0
    processed_samples_count = 0

    # Initialize JSONL log file
    with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
        log_file.write("")

    for i, example in enumerate(test_dataset):
        if i >= MAX_SAMPLES_TO_PROCESS:
            print(f"Reached MAX_SAMPLES_TO_PROCESS ({MAX_SAMPLES_TO_PROCESS}). Stopping.")
            break

        print(f"\n--- Processing Sample {i+1}/{MAX_SAMPLES_TO_PROCESS} (Original index: {i}) ---")
        start_time = time.time()

        image = example["image"]
        ground_truth_json_str = example["ground_truth"]
        image_width, image_height = image.size

        # Extract ground truth entities
        original_words, _, gt_flattened_entities, labels_present_in_gt = \
            extract_words_bboxes_and_labels_from_gt(ground_truth_json_str, image_width, image_height)
        
        all_gt_flattened_entities_list.extend(gt_flattened_entities)

        # Create dynamic prompt
        dynamic_labels_to_request = TARGET_LABELS.intersection(labels_present_in_gt)
        prompt = create_ollama_prompt_for_cord(dynamic_labels_to_request)
        image_base64 = pil_to_base64(image)

        print(f"Calling Ollama VLM for sample {i+1}...")
        ollama_output = await call_ollama_vlm(image_base64, prompt)

        if ollama_output:
            processed_samples_count += 1
            current_predicted_flattened_entities = postprocess_ollama_output_to_flattened_entities(ollama_output)
            all_predicted_flattened_entities_list.extend(current_predicted_flattened_entities)

            # Calculate metrics for the current sample
            sample_metrics = calculate_all_metrics(gt_flattened_entities, current_predicted_flattened_entities)
            sample_time = time.time() - start_time
            total_time += sample_time

            total_anls += sample_metrics["anls"]
            total_em += sample_metrics["em"]
            total_ned += sample_metrics["ned"]
            total_tokens_found += sample_metrics["tokens_found"]
            total_tokens_added += sample_metrics["tokens_added"]
            total_entity_f1 += sample_metrics["kieval"]["entity_f1"]
            total_group_f1 += sample_metrics["kieval"]["group_f1"]
            total_aligned_score += sample_metrics["kieval"]["aligned_score"]
            total_correction_cost += sample_metrics["kieval"]["correction_cost"]

            # Log sample results
            sample_log = {
                "sample_id": i,
                "ground_truth_entities": gt_flattened_entities,
                "predicted_entities": current_predicted_flattened_entities,
                "metrics": {
                    "anls": sample_metrics["anls"],
                    "em": sample_metrics["em"],
                    "adjusted_ned": sample_metrics["ned"],
                    "tokens_found": sample_metrics["tokens_found"],
                    "tokens_added": sample_metrics["tokens_added"],
                    "kieval": {
                        "entity_f1": sample_metrics["kieval"]["entity_f1"],
                        "group_f1": sample_metrics["kieval"]["group_f1"],
                        "aligned_score": sample_metrics["kieval"]["aligned_score"],
                        "correction_cost": sample_metrics["kieval"]["correction_cost"],
                        "true_positives": sample_metrics["kieval"]["true_positives"],
                        "false_positives": sample_metrics["kieval"]["false_positives"],
                        "false_negatives": sample_metrics["kieval"]["false_negatives"]
                    }
                },
                "processing_time": sample_time
            }
            with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
                json.dump(sample_log, log_file, ensure_ascii=False)
                log_file.write("\n")

            # Print sample results
            print("\n--- Ground Truth Entities (Flattened & Normalized) ---")
            if not gt_flattened_entities:
                print("  (No relevant GT entities for this sample in TARGET_LABELS)")
            for entity in gt_flattened_entities:
                print(f"  GT: Label='{entity['label']}', Text='{entity['text']}'")
            
            print("\n--- Ollama Predicted Entities (Flattened & Normalized) ---")
            if not current_predicted_flattened_entities:
                print("  (No predicted entities for this sample)")
            for entity in current_predicted_flattened_entities:
                print(f"  Pred: Label='{entity['label']}', Text='{entity['text']}'")
            
            print(f"\nSample {i+1} Metrics: ANLS={sample_metrics['anls']:.2f}, EM={sample_metrics['em']:.2f}, "
                  f"NED={sample_metrics['ned']:.2f}, TokensFound={sample_metrics['tokens_found']}, "
                  f"TokensAdded={sample_metrics['tokens_added']}, EntityF1={sample_metrics['kieval']['entity_f1']:.2f}, "
                  f"GroupF1={sample_metrics['kieval']['group_f1']:.2f}, AlignedScore={sample_metrics['kieval']['aligned_score']:.2f}, "
                  f"CorrectionCost={sample_metrics['kieval']['correction_cost']}, Time={sample_time:.2f}s")
            
            if sample_metrics["kieval"]["false_negatives"]:
                print("--- False Negatives ---")
                for fn_entity in sample_metrics["kieval"]["false_negatives_list"]:
                    print(f"  FN: Label='{fn_entity['label']}', Text='{fn_entity['text']}'")
            
            if sample_metrics["kieval"]["false_positives"]:
                print("--- False Positives ---")
                for fp_entity in sample_metrics["kieval"]["false_positives_list"]:
                    print(f"  FP: Label='{fp_entity['label']}', Text='{fp_entity['text']}'")
        else:
            print(f"Ollama call failed or returned no output for sample {i+1}. Skipping.", file=sys.stderr)

    # Calculate overall metrics
    final_kie_metrics = calculate_kie_metrics(all_gt_flattened_entities_list, all_predicted_flattened_entities_list)
    overall_anls = compute_anls(all_gt_flattened_entities_list, all_predicted_flattened_entities_list)
    overall_em = compute_exact_match(all_gt_flattened_entities_list, all_predicted_flattened_entities_list)
    overall_ned = compute_adjusted_ned(all_gt_flattened_entities_list, all_predicted_flattened_entities_list)
    overall_tokens_found, overall_tokens_added = compute_tokens_found_added(all_gt_flattened_entities_list, all_predicted_flattened_entities_list)

    print(f"\n--- Evaluation Summary for {OLLAMA_MODEL_NAME} on CORD v2 ---")
    print("="*60)
    if processed_samples_count > 0:
        print(f"Processed Documents: {processed_samples_count}/{total_samples}")
        print(f"Total True Positives: {final_kie_metrics['true_positives']}")
        print(f"Total False Positives: {final_kie_metrics['false_positives']}")
        print(f"Total False Negatives: {final_kie_metrics['false_negatives']}")
        print(f"Overall Precision: {final_kie_metrics['precision']:.4f}")
        print(f"Overall Recall: {final_kie_metrics['recall']:.4f}")
        print(f"Overall F1 Score: {final_kie_metrics['f1']:.4f}")
        print(f"Overall Exact Match (EM): {overall_em:.4f}")
        print(f"Overall ANLS: {overall_anls:.4f}")
        print(f"Overall Adjusted NED (SCORE): {overall_ned:.4f}")
        print(f"Total Tokens Found (SCORE): {overall_tokens_found}")
        print(f"Total Tokens Added (SCORE): {overall_tokens_added}")
        print(f"KIEval Entity F1: {final_kie_metrics['f1']:.4f}")
        print(f"KIEval Group F1: {total_group_f1 / processed_samples_count:.4f}")
        print(f"KIEval Aligned Score: {total_aligned_score / processed_samples_count:.4f}")
        print(f"KIEval Total Correction Cost: {total_correction_cost}")
        print(f"Average Time per Sample: {total_time / processed_samples_count:.2f}s")
        print(f"Total samples processed: {processed_samples_count}")
    else:
        print("No samples were successfully processed for evaluation.")
    print("="*60)

# --- Run the script ---
if __name__ == "__main__":
    import asyncio
    asyncio.run(evaluate_ollama_on_cord())