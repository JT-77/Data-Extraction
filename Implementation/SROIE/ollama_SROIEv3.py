import os
import json
import base64
import requests
from PIL import Image
from io import BytesIO
from thefuzz import fuzz
import re
from datetime import datetime
from dateutil.parser import parse as dateutil_parse
from dateutil.parser import ParserError
import asyncio
import time
import Levenshtein
import logging

# --- Configuration ---
# Default Ollama API address. Ensure Ollama is running on this host/port.
OLLAMA_HOST = "http://localhost:11434"
# OLLAMA_MODEL = "qwen2.5vl:latest"
# OLLAMA_MODEL = "mistral-small3.2:latest"
OLLAMA_MODEL = "llama3.2-vision:latest"

# Path to your SROIE 2019 dataset.
SROIE_ROOT = "../datasets/SROIE2019"
SROIE_TEST_DATA_PATH = os.path.join(SROIE_ROOT, "test")

# Expected entities for SROIE (for evaluation)
SROIE_ENTITY_KEYS = ["company", "date", "address", "total"]

# Setup logging
LOG_FILE = "new_results/llama_sroie_evaluation_log.jsonl"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions for Ollama API Interaction ---

def encode_image_to_base64(image_path):
    """Encodes an image to base64 string."""
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def call_ollama_vlm(image_base64, prompt, model_name=OLLAMA_MODEL):
    """
    Calls the Ollama API for a multimodal model.
    Returns parsed JSON response or None on error.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
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
                json_string = generated_content.replace("```json", "").replace("```", "").strip()
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

# --- Dataset Loading and Ground Truth Parsing ---

def load_sroie_document(doc_id):
    """
    Loads a single SROIE 2019 document's image path and ground truth JSON annotations.
    """
    img_path = os.path.join(SROIE_TEST_DATA_PATH, "img", f"{doc_id}.jpg")
    json_path = os.path.join(SROIE_TEST_DATA_PATH, "entities", f"{doc_id}.txt")

    if not os.path.exists(img_path) or not os.path.exists(json_path):
        print(f"Error: SROIE document {doc_id} not found at {img_path} or {json_path}")
        return None, None

    try:
        with open(json_path, 'r', encoding='utf8') as f:
            ground_truth_text = f.read()
            ground_truth = json.loads(ground_truth_text)
        
        normalized_ground_truth = {k.lower(): str(v).strip().lower() for k, v in ground_truth.items()}
        return img_path, normalized_ground_truth
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from {json_path}. Skipping.")
        return None, None
    except Exception as e:
        print(f"Error loading SROIE document {doc_id}: {e}")
        return None, None

# --- Prompt Definition ---

def get_sroie_prompt():
    """
    Generates a precise prompt for Ollama VLM to extract SROIE 2019 entities.
    """
    return """
Analyze the attached receipt image and find the below information ->
    1. "company or store name": the name of the company or store written on the top of the receipt.
    2. "date": The date of the transaction in the same format as available in text. Do not include any time with the date.
        For eg - if date is "01/01/16 10:03" then only return "01/01/16" etc.
    3. "address": The full address of the company or the store where transaction has happened.
    4. "total": The total amount of the transaction (including taxes/gst).

    Please always respond with JSON only with above 4 fields, do not return anything else. Extract the exact value from the text, do not format or modify the data.
```
"""

# --- Evaluation Metrics ---

def normalize_string_for_date_parsing(date_str):
    return date_str.replace('-', ' ').replace('.', ' ').replace(',', ' ').replace('/', ' ').strip()

def compute_anls(pred_str, gt_str, threshold=0.5):
    """Compute ANLS (Average Normalized Levenshtein Similarity)."""
    if not pred_str or not gt_str:
        return 0.0
    distance = Levenshtein.distance(pred_str.lower(), gt_str.lower())
    max_len = max(len(pred_str), len(gt_str))
    similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
    return similarity if similarity >= threshold else 0.0

def compute_exact_match(pred_str, gt_str):
    """Compute Exact Match (EM) score."""
    return 1.0 if pred_str.lower() == gt_str.lower() else 0.0

def compute_adjusted_ned(pred_str, gt_str):
    """Compute Adjusted Normalized Edit Distance (NED) for SCORE metric."""
    if not pred_str or not gt_str:
        return 1.0 if pred_str != gt_str else 0.0
    distance = Levenshtein.distance(pred_str.lower(), gt_str.lower())
    max_len = max(len(pred_str), len(gt_str))
    return distance / max_len if max_len > 0 else 0.0

def compute_tokens_found_added(pred_str, gt_str):
    """Compute TokensFound and TokensAdded for SCORE metric."""
    pred_tokens = pred_str.lower().split()
    gt_tokens = gt_str.lower().split()
    common_tokens = set(pred_tokens) & set(gt_tokens)
    tokens_found = len(common_tokens)
    tokens_added = len(pred_tokens) - len(common_tokens)
    return tokens_found, tokens_added

def compute_kieval_metrics(pred_dict, gt_dict, entity_keys):
    """
    Compute KIEval Entity/Group F1 and Aligned metrics.
    Simplified for SROIE: treats entities as a single group.
    """
    entity_matches = 0
    entity_total_pred = 0
    entity_total_gt = 0
    correction_cost = 0

    for key in entity_keys:
        gt_value = gt_dict.get(key, "")
        pred_value = pred_dict.get(key, "")
        
        entity_total_gt += 1 if gt_value else 0
        entity_total_pred += 1 if pred_value else 0
        
        if gt_value and pred_value:
            if key in ["company", "address"]:
                if fuzz.ratio(gt_value, pred_value) >= 80:
                    entity_matches += 1
                else:
                    correction_cost += 1  # Substitution cost
            elif key == "date":
                gt_date = None
                pred_date = None
                try:
                    gt_date = dateutil_parse(normalize_string_for_date_parsing(gt_value), fuzzy=True)
                    pred_date = dateutil_parse(normalize_string_for_date_parsing(pred_value), fuzzy=True)
                    if gt_date.date() == pred_date.date():
                        entity_matches += 1
                    else:
                        correction_cost += 1
                except ParserError:
                    correction_cost += 1
            elif key == "total":
                def clean_amount(amount_str):
                    cleaned = re.sub(r'[^\d.]', '', str(amount_str))
                    try:
                        return float(cleaned)
                    except ValueError:
                        return None
                gt_total = clean_amount(gt_value)
                pred_total = clean_amount(pred_value)
                if gt_total is not None and pred_total is not None and abs(gt_total - pred_total) < 0.01:
                    entity_matches += 1
                else:
                    correction_cost += 1
        elif gt_value and not pred_value:
            correction_cost += 1  # Deletion cost
        elif pred_value and not gt_value:
            correction_cost += 1  # Addition cost

    # Entity F1
    precision = entity_matches / entity_total_pred if entity_total_pred > 0 else 0.0
    recall = entity_matches / entity_total_gt if entity_total_gt > 0 else 0.0
    entity_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Group F1 (simplified: treat all entities as one group)
    group_match = 1 if entity_matches == len([k for k in entity_keys if gt_dict.get(k)]) else 0
    group_precision = group_match / 1 if entity_total_pred > 0 else 0.0
    group_recall = group_match / 1 if entity_total_gt > 0 else 0.0
    group_f1 = (2 * group_precision * group_recall) / (group_precision + group_recall) if (group_precision + group_recall) > 0 else 0.0

    # Aligned metric (based on correction cost)
    aligned_score = 1.0 - (correction_cost / max(entity_total_gt, entity_total_pred)) if max(entity_total_gt, entity_total_pred) > 0 else 0.0

    return {
        "entity_f1": entity_f1,
        "group_f1": group_f1,
        "aligned_score": aligned_score,
        "correction_cost": correction_cost
    }

def postprocess_sroie_output(llm_output_json, ground_truth_json, entity_keys):
    """
    Compares LLM's extracted JSON against ground truth for SROIE entities.
    Calculates KIE F1, SCORE, ANLS, Exact Match, and logs results.
    """
    tp = 0
    fp = 0
    fn = 0
    anls_scores = []
    em_scores = []
    ned_scores = []
    tokens_found_total = 0
    tokens_added_total = 0

    key_remapping = {
        "company_name": "company",
        "company_or_store_name": "company",
        "company or store name": "company",
        "invoice_date": "date",
        "store_address": "address",
        "grand_total": "total",
        "amount_total": "total",
    }

    normalized_llm_output = {}
    if isinstance(llm_output_json, dict):
        for k, v in llm_output_json.items():
            standardized_key = key_remapping.get(k.lower(), k.lower())
            val = str(v).strip().lower()
            if val == "null" or val == "":
                normalized_llm_output[standardized_key] = ""
            else:
                normalized_llm_output[standardized_key] = val

    sample_log = {
        "llm_output": normalized_llm_output,
        "ground_truth": ground_truth_json,
        "metrics": {}
    }

    for key in entity_keys:
        gt_value = ground_truth_json.get(key, "")
        llm_value = normalized_llm_output.get(key, "")

        # Compute ANLS and Exact Match
        anls = compute_anls(llm_value, gt_value)
        em = compute_exact_match(llm_value, gt_value)
        anls_scores.append(anls)
        em_scores.append(em)

        # Compute SCORE metrics
        ned = compute_adjusted_ned(llm_value, gt_value)
        tokens_found, tokens_added = compute_tokens_found_added(llm_value, gt_value)
        ned_scores.append(ned)
        tokens_found_total += tokens_found
        tokens_added_total += tokens_added

        sample_log["metrics"][key] = {
            "anls": anls,
            "exact_match": em,
            "adjusted_ned": ned,
            "tokens_found": tokens_found,
            "tokens_added": tokens_added
        }

        # Existing KIE F1 logic
        if gt_value == "":
            if llm_value != "":
                fp += 1
                sample_log["metrics"][key]["kie_result"] = "FP"
            continue

        if llm_value != "":
            is_match = False
            if key in ["company", "address"]:
                if fuzz.ratio(gt_value, llm_value) >= 80:
                    is_match = True
            elif key == "date":
                gt_date_obj = None
                llm_date_obj = None
                try:
                    gt_date_obj = dateutil_parse(normalize_string_for_date_parsing(gt_value), fuzzy=True, dayfirst=False)
                except ParserError:
                    pass
                try:
                    llm_date_obj = dateutil_parse(normalize_string_for_date_parsing(llm_value), fuzzy=True, dayfirst=False)
                except ParserError:
                    pass
                if gt_date_obj:
                    try:
                        gt_date_obj = dateutil_parse(normalize_string_for_date_parsing(gt_value), fuzzy=True, dayfirst=True)
                    except ParserError:
                        try:
                            gt_date_obj = dateutil_parse(normalize_string_for_date_parsing(gt_value), fuzzy=True, dayfirst=False)
                        except ParserError:
                            gt_date_obj = None
                if llm_date_obj:
                    try:
                        llm_date_obj = dateutil_parse(normalize_string_for_date_parsing(llm_value), fuzzy=True, dayfirst=True)
                    except ParserError:
                        try:
                            llm_date_obj = dateutil_parse(normalize_string_for_date_parsing(llm_value), fuzzy=True, dayfirst=False)
                        except ParserError:
                            llm_date_obj = None
                if gt_date_obj and llm_date_obj:
                    is_match = (gt_date_obj.date() == llm_date_obj.date())
            elif key == "total":
                def clean_amount(amount_str):
                    if isinstance(amount_str, (int, float)):
                        return float(amount_str)
                    cleaned = re.sub(r'[^\d.]', '', amount_str)
                    try:
                        return float(cleaned)
                    except ValueError:
                        return None
                gt_total_float = clean_amount(gt_value)
                llm_total_float = clean_amount(llm_value)
                epsilon = 0.01
                if gt_total_float is not None and llm_total_float is not None and \
                   abs(gt_total_float - llm_total_float) < epsilon:
                    is_match = True
                elif fuzz.ratio(gt_value, llm_value) == 100:
                    is_match = True

            if is_match:
                tp += 1
                sample_log["metrics"][key]["kie_result"] = "TP"
            else:
                fp += 1
                sample_log["metrics"][key]["kie_result"] = "FP"
        else:
            fn += 1
            sample_log["metrics"][key]["kie_result"] = "FN"

    # Compute KIEval metrics
    kieval_metrics = compute_kieval_metrics(normalized_llm_output, ground_truth_json, entity_keys)
    sample_log["metrics"]["kieval"] = kieval_metrics

    return tp, fp, fn, anls_scores, em_scores, ned_scores, tokens_found_total, tokens_added_total, sample_log

# --- Main Evaluation Loop ---

async def evaluate_sroie(num_samples=None):
    """Evaluates the Ollama VLM on the SROIE 2019 dataset."""
    print(f"\n--- Starting evaluation for SROIE 2019 with {OLLAMA_MODEL} ---")

    img_dir = os.path.join(SROIE_TEST_DATA_PATH, "img")
    
    all_doc_ids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg")]
    if num_samples:
        all_doc_ids = all_doc_ids[:min(num_samples, len(all_doc_ids))]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_anls = []
    total_em = []
    total_ned = []
    total_tokens_found = 0
    total_tokens_added = 0
    processed_docs = 0
    total_time = 0.0

    prompt = get_sroie_prompt()

    # Initialize JSONL log file
    with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
        log_file.write("")  # Clear file

    print(f"Total documents to process: {len(all_doc_ids)}")

    for i, doc_id in enumerate(all_doc_ids):
        print(f"\nProcessing document {i+1}/{len(all_doc_ids)}: {doc_id}.jpg")
        start_time = time.time()

        img_path, ground_truth_json = load_sroie_document(doc_id)
        if img_path is None:
            continue

        image_base64 = encode_image_to_base64(img_path)
        if image_base64 is None:
            print(f"Skipping {doc_id} due to image encoding error.")
            continue

        llm_output_json = call_ollama_vlm(image_base64, prompt, OLLAMA_MODEL)
        print(f"LLM output - {llm_output_json}")

        if llm_output_json is None:
            print(f"Skipping {doc_id} due to invalid/no LLM output.")
            continue

        tp, fp, fn, anls_scores, em_scores, ned_scores, tokens_found, tokens_added, sample_log = \
            postprocess_sroie_output(llm_output_json, ground_truth_json, SROIE_ENTITY_KEYS)
        
        sample_time = time.time() - start_time
        total_time += sample_time
        processed_docs += 1
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_anls.extend(anls_scores)
        total_em.extend(em_scores)
        total_ned.extend(ned_scores)
        total_tokens_found += tokens_found
        total_tokens_added += tokens_added

        # Log sample results
        sample_log["doc_id"] = doc_id
        sample_log["processing_time"] = sample_time
        with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
            json.dump(sample_log, log_file, ensure_ascii=False)
            log_file.write("\n")

        print(f"  Current Doc Metrics: TP={tp}, FP={fp}, FN={fn}, ANLS={sum(anls_scores)/len(anls_scores):.4f}, "
              f"EM={sum(em_scores)/len(em_scores):.4f}, NED={sum(ned_scores)/len(ned_scores):.4f}, "
              f"TokensFound={tokens_found}, TokensAdded={tokens_added}, Time={sample_time:.2f}s")

    # --- Aggregate and Print Final Results ---
    print("\n" + "="*50)
    print(f"--- {OLLAMA_MODEL} Overall Evaluation Results for SROIE 2019 ---")
    print("="*50)
    if processed_docs > 0:
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_anls = sum(total_anls) / len(total_anls) if total_anls else 0.0
        avg_em = sum(total_em) / len(total_em) if total_em else 0.0
        avg_ned = sum(total_ned) / len(total_ned) if total_ned else 0.0
        avg_time_per_sample = total_time / processed_docs if processed_docs > 0 else 0.0

        print(f"Processed Documents: {processed_docs}/{len(all_doc_ids)}")
        print(f"Total True Positives: {total_tp}")
        print(f"Total False Positives: {total_fp}")
        print(f"Total False Negatives: {total_fn}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")
        print(f"Overall F1 Score: {f1:.4f}")
        print(f"Average ANLS: {avg_anls:.4f}")
        print(f"Average Exact Match: {avg_em:.4f}")
        print(f"Average Adjusted NED (SCORE): {avg_ned:.4f}")
        print(f"Total Tokens Found (SCORE): {total_tokens_found}")
        print(f"Total Tokens Added (SCORE): {total_tokens_added}")
        print(f"Average Time per Sample: {avg_time_per_sample:.2f}s")
        
        # Aggregate KIEval metrics
        entity_f1_total = 0.0
        group_f1_total = 0.0
        aligned_score_total = 0.0
        correction_cost_total = 0
        with open(LOG_FILE, 'r', encoding='utf-8') as log_file:
            for line in log_file:
                sample = json.loads(line)
                kieval = sample["metrics"].get("kieval", {})
                entity_f1_total += kieval.get("entity_f1", 0.0)
                group_f1_total += kieval.get("group_f1", 0.0)
                aligned_score_total += kieval.get("aligned_score", 0.0)
                correction_cost_total += kieval.get("correction_cost", 0)
        
        print(f"KIEval Entity F1: {entity_f1_total / processed_docs:.4f}")
        print(f"KIEval Group F1: {group_f1_total / processed_docs:.4f}")
        print(f"KIEval Aligned Score: {aligned_score_total / processed_docs:.4f}")
        print(f"KIEval Total Correction Cost: {correction_cost_total}")
    else:
        print("No documents were successfully processed for evaluation.")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(evaluate_sroie())