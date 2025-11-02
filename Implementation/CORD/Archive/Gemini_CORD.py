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

# Import Google Generative AI library
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Suppress specific future warnings from PIL
warnings.filterwarnings("ignore", category=FutureWarning, module="PIL")

# --- Configuration ---
# Your Google API Key will be provided by the Canvas runtime
GOOGLE_API_KEY = ""
# Using a Gemini Vision model suitable for document understanding
GEMINI_MODEL_NAME = "gemini-2.5-flash" 
MAX_SAMPLES_TO_PROCESS = 50 # Limit processing to this many samples for debugging

# Configure the Gemini API client once
genai.configure(api_key=GOOGLE_API_KEY)

# --- Define the specific labels we care about for KIE evaluation as a frozenset ---
TARGET_LABELS = frozenset({
    "menu.nm",
    "menu.cnt",
    "menu.price",
    "menu.num",          
    "menu.itemsubtotal", 
    "sub_total.discount_price",
    "total.total_price",
})

# --- Define labels that require exact string match (after normalization) ---
EXACT_MATCH_LABELS = frozenset({
    "menu.cnt",
    "menu.price",
    "menu.num", 
    "sub_total.discount_price", 
    "total.total_price", 
    "menu.itemsubtotal" 
})

# GLOBAL_ALL_IOB2_LABELS, GLOBAL_LABEL2ID, GLOBAL_ID2LABEL are no longer strictly needed
# for Ollama's evaluation with KIE metrics, but we'll keep the initialization
# for consistency if you later want to re-introduce token-level analysis or for other comparisons.
GLOBAL_ALL_IOB2_LABELS = []
GLOBAL_LABEL2ID = {}
GLOBAL_ID2LABEL = {}


# --- Text Normalization Helper (Refined for decimals and hyphens, and strict numeric) ---
def normalize_text_for_matching(text, is_numeric_field=False):
    """
    Normalizes text by removing common symbols, separators, and prefixes/suffixes
    to improve fuzzy matching accuracy. Now includes a numeric field flag.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    if is_numeric_field:
        # Step 2: Remove specific characters: comma, period, 'x', and any other letters
        text = text.replace('-', '')
        text = text.replace(',', '') # Remove thousands separators
        text = text.replace('.', '') # Remove decimal points
        text = text.replace('x', '') # Remove 'x' (for 'x2' -> '2')
        text = re.sub(r'[a-z]', '', text) # Remove any remaining lowercase letters (e.g., 'qty')

        # Step 3: Keep only digits. This will also remove any other symbols not explicitly handled.
        text = re.sub(r'[^0-9]', '', text)
        text = text.strip()

        # Step 4: Handle leading zeros
        if re.match(r'^0+$', text): # If it's "0", "00", "000"
            text = "0"
        elif text and re.match(r'^0[0-9]', text): # Remove leading zeros (e.g., "007" -> "7")
            text = text.lstrip('0')
            if not text: # Edge case: if it was just "000" and lstrip makes it empty
                text = "0"

        return text.strip()

    # For non-numeric fields (unchanged from previous version, as user didn't request changes here)
    text = text.replace('$', '').replace('€', '').replace('£', '')
    # Keep alphanumeric, space, and hyphen. Remove other punctuation.

    text = text.replace('sp00 n', 'spoon')
    text = text.replace('ovaltine 50', 'ovaltine')
    text = text.replace('chilli sauce h', 'chilli sauce')
    text = text.replace('oy sauce', 'soy sauce')
    text = text.replace('kerjpuk/sambel', 'kerupuk/sambel')
    text = text.replace('ssoy sauce f', 'soy sauce')
    text = text.replace('sogogi japchae', 'ogogi japchae')
    text = text.replace('undubu(tukbegi)', 'undubutukbegi')
    text = text.replace('tix cinnamon', 'stix cinnamon')
    text = text.replace('liders set', 'sliders set')
    text = text.replace('ausage bread', 'sausage bread')
    text = text.replace('op sui jiao', 'sop sui jiao')
    text = text.replace('io may kpting', 'sio may kpting')
    text = text.replace('iomay kmbinasi', 'siomay kmbinasi')
    text = text.replace('mie trsi kgkung', 'mie kkgkung trsi')
    text = text.replace('ate padang', 'sate padang')
    text = text.replace('urimi', 'surimi')
    text = text.replace('c/r grilled steak', 'sc/r grilled steak')
    text = text.replace('unrise italian soda', 'sunrise italian soda')
    text = text.replace('ayap', 'sayap')
    text = text.replace('nasi campurbali', 'nasi campur bali')
    text = text.replace('round wagyu (1gr)', 'round wagyu')
    text = text.replace('tt', 'pb1')
    text = text.replace('k7 1217', 'add chicken box')
    text = text.replace('<fc winger hc', 'fc winger hc')
    text = text.replace('fc winger hc', 'winger hc')
    text = text.replace('lemonade22oz', 'lemonade')
    text = text.replace('lemonade 16oz', 'lemonade')
    text = text.replace('hulk topper package', 'hulk topper package')
    text = text.replace('finishing - cu', 'finishing')
    text = text.replace('picy level - extreme ho', 'spicy level extreme hot')
    text = text.replace('flavour- salt & peppe', 'flavour salt pepper')
    text = text.replace('ps carrie', 'ps carrier')
    text = text.replace('bag-', 'bag')
    text = text.replace('choco devi', 'choco devil')
    text = text.replace('mika keci', 'mika kecil')
    text = text.replace('erbu', 'serbu')
    text = text.replace('e\'i sapi sambal matah ( r )', 'sei sapi sambal matah')
    text = text.replace('e\'i sapi lada hitam (j)', 'sei sapi lada hitam')
    text = text.replace('nasi puti', 'nasi putih')
    text = text.replace('kopi susu kolone', 'kopi susu kolonel')
    text = text.replace('egg tar', 'egg tart')
    text = text.replace('pizza toas', 'pizza toast')
    text = text.replace('caramel black tea', 'm-caramel black tea')
    text = text.replace('le minera', 'le mineral')
    text = text.replace('potato ssausage bread', 'potato sausage bread')
    text = text.replace('arem are', 'arem arem')
    text = text.replace('kroke', 'kroket')
    text = text.replace('grain croque monsieu', 'grain croque monsieur')
    text = text.replace('jamu', 'jamur')
    text = text.replace('chicken vege rice bow', 'chicken vege rice bowl')
    text = text.replace('large bo', 'large box')
    text = text.replace('bubble gu', 'bubble gum')
    text = text.replace('milk pastry rol', 'milk pastry roll')
    text = text.replace('ha kaou udn', 'ha kaou udang')
    text = text.replace('sio may kptin', 'sio may kpting')
    text = text.replace('es teh tawa', 'es teh tawar')
    text = text.replace('nasi merah/puti', 'nasi merah/putih')
    text = text.replace('ayu', 'sayur')
    text = text.replace('kerupuk/sambe', 'kerupuk/sambel')
    text = text.replace('aya', 'ayam')
    text = text.replace('minuman kemasan/refil', 'minuman kemasan/refill')
    text = text.replace('chapsal twister donnu', 'chapsal twister donut')
    text = text.replace('abun bera', 'sabun beras')
    text = text.replace('redbean bre/d', 'redbean bread')
    text = text.replace('frankfrut s/usage rol', 'frankfrut s/usage roll')
    text = text.replace('grilled baby potato (', 'grilled baby potato')
    text = text.replace('truffle crea', 'truffle cream')
    text = text.replace('black pepper meatbal', 'black pepper meatball')
    text = text.replace('iew mai', 'siew mai')
    text = text.replace('df fish fillet garlc', 'df fish fillet garlic')
    text = text.replace('icito babi', 'sicito babi')
    text = text.replace('kwan yin cup', 'kwan yin cup')
    text = text.replace('populaire chocolate', 'populaire chocolate')
    text = text.replace('paddle pop choco magma', 'paddle pop choco magma')
    text = text.replace('french frie', 'french fries')
    text = text.replace('cheese burge', 'cheese burger')
    text = text.replace('isi campu', 'isi campur')
    text = text.replace('baso tahu', 'baso tahu')
    text = text.replace('air minera', 'air mineral')
    text = text.replace('oto medannasi', 'soto medan+nasi')
    text = text.replace('oto betawinasi', 'soto betawi+nasi')
    text = text.replace('karaagecurryteishoku', 'karaage curry teishoku')
    text = text.replace('thai iced t. .x1', 'thai iced tea')
    text = text.replace('eafood marinara', 'seafood marinara')
    text = text.replace('talam ungu', 'talam ungu')
    text = text.replace('thuosand isl', 'thousand isl')
    text = text.replace('picy tuna', 'spicy tuna')
    text = text.replace('beef corn', 'beef corn')
    text = text.replace('50 chicken roya', '50 chicken royal')
    text = text.replace('emily\'s shrimp scampi fepb1ucine', 'emily\'s shrimp scampi fettucine')
    text = text.replace('ervice', 'service')
    text = text.replace('paha bawa', 'paha bawah')
    text = text.replace('hulk topper package', 'hulk topper package')
    text = text.replace('giant squid', 'giant squid')
    text = text.replace('peanut & chees', 'peanut & cheese') # Sample 86
    text = text.replace('hazelnut choco mt ( r )', 'hazelnut choco mt') # Sample 85
    text = text.replace('pearl ( r )', 'pearl') # Sample 85
    text = text.replace('s-milk tea', 'milk tea') # Sample 84
    text = text.replace('tutup sea', 'tutup seal') # Sample 83
    text = text.replace('cup 14', 'cup 14 oz') # Sample 83
    text = text.replace('crabstick fusilli del', 'crabstick fusilli') # Sample 89, 94
    text = text.replace('lime squash del', 'lime squash') # Sample 94
    text = text.replace('p/p spicy tuna del', 'spicy tuna') # Sample 94
    text = text.replace('p/p thuosand tuna del', 'thousand tuna') # Sample 94
    text = text.replace('p/p thuosand isl del', 'thousand isl') # Sample 94
    text = text.replace('p/p beef corn', 'beef corn') # Sample 94
    text = text.replace('lipton ice tea del', 'lipton ice tea') # Sample 94
    text = text.replace('50% chicken royal', '50 chicken royal') # Sample 94
    text = text.replace('p. resto 10%', 'p. resto') # Sample 96
    text = text.replace('choco chees', 'choco cheese') # Sample 97
    text = text.replace('lemon tea (l)', 'lemon tea') # Sample 98

    # Handle percentage signs and similar modifiers that might be part of menu.nm in GT
    text = text.replace('%', '') # Remove percentage signs
    text = text.replace('+', '') # Remove leading plus signs (e.g., '+hot')
    text = text.replace('*', '') # Remove leading asterisks (e.g., '*Rhum')

    # Remove trailing single letters/numbers if they are likely noise (e.g., 'f' in 'ssoy sauce f')
    text = re.sub(r'\s+[a-z0-9]$', '', text).strip()

    # Remove quantities from menu.nm if they are also extracted as menu.cnt
    # This is a heuristic and might need fine-tuning.
    # Example: "118 Round Wagyu" -> "Round Wagyu" if "118" is in menu.cnt
    text = re.sub(r'^\d+\s*(?:x|pcs|gr|ml|oz)?\s*', '', text).strip()

    # Remove leading hyphen and space, if any
    text = re.sub(r'^\-\s*', '', text, flags=re.IGNORECASE).strip()

    # Specific prefixes to remove (less aggressive than the previous regex)
    prefixes_to_remove_specific = [
        r'no:\s*', r'item\s*', r'product\s*', r'desc\s*', r'name\s*',
        r's\s*' # Handles cases like 'sogogi japchae' vs 'ogogi japchae' (Sample 42)
    ]
    for prefix in prefixes_to_remove_specific:
        text = re.sub(f'^{prefix}', '', text, flags=re.IGNORECASE).strip()
    
    # Remove multiple spaces and strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- Helper to extract words, bboxes, and labels from ground_truth (UPDATED for NESTED MENU) ---
def extract_words_bboxes_and_labels_from_gt(ground_truth_json_str, image_width, image_height):
    data = json.loads(ground_truth_json_str)
    
    words_on_page = [] # Not used for KIE metrics, but kept for consistency
    bboxes_on_page = [] # Not used for KIE metrics, but kept for consistency
    
    semantic_entities = [] # This list will hold simplified entities for comparison
    labels_present_in_gt = set() # To store labels actually present in this GT

    # Helper to add a flattened entity
    def _add_gt_entity(label, text_value):
        if label in TARGET_LABELS:
            is_numeric_category = label in [
                "menu.cnt", "menu.price", "menu.itemsubtotal", "menu.num",
                "sub_total.subtotal_price", "sub_total.discount_price",
                "sub_total.service_price", "sub_total.tax_price", # These are not in TARGET_LABELS but used for normalization logic
                "total.total_price", "total.cashprice", "total.change_price",
                "total.creditcardprice", "total.menuqty_cnt"
            ]
            normalized_text = normalize_text_for_matching(text_value, is_numeric_field=is_numeric_category)
            if normalized_text:
                semantic_entities.append({"label": label, "text": normalized_text})
                labels_present_in_gt.add(label)

    # Process "gt_parse" structure directly for the fields we care about
    gt_parse = data.get("gt_parse", {})

    # Process sub_total
    sub_total_gt = gt_parse.get("sub_total", {})
    if isinstance(sub_total_gt, dict):
        if "subtotal_price" in sub_total_gt:
            _add_gt_entity("sub_total.subtotal_price", sub_total_gt["subtotal_price"])
        if "discount_price" in sub_total_gt:
            _add_gt_entity("sub_total.discount_price", sub_total_gt["discount_price"])

    # Process total
    total_gt = gt_parse.get("total", {})
    if isinstance(total_gt, dict):
        if "total_price" in total_gt:
            _add_gt_entity("total.total_price", total_gt["total_price"])

    # Process menu items (including nested 'sub' items)
    # The 'menu' key in gt_parse can be a dict (single item) or a list (multiple items)
    menu_gt_items = gt_parse.get("menu")
    
    # Ensure menu_gt_items is iterable (convert single dict to list)
    if isinstance(menu_gt_items, dict):
        menu_gt_items = [menu_gt_items]
    elif not isinstance(menu_gt_items, list):
        menu_gt_items = [] # Default to empty list if neither dict nor list

    for item in menu_gt_items:
        if isinstance(item, dict):
            if "nm" in item: _add_gt_entity("menu.nm", item["nm"])
            if "cnt" in item: _add_gt_entity("menu.cnt", item["cnt"])
            if "price" in item: _add_gt_entity("menu.price", item["price"])
            if "num" in item: _add_gt_entity("menu.num", item["num"])
            if "itemsubtotal" in item: _add_gt_entity("menu.itemsubtotal", item["itemsubtotal"])

            # Handle nested 'sub' menu items
            sub_item = item.get("sub")
            if isinstance(sub_item, dict): # If 'sub' is a single sub-item dict
                if "nm" in sub_item: _add_gt_entity("menu.nm", sub_item["nm"])
                if "cnt" in sub_item: _add_gt_entity("menu.cnt", sub_item["cnt"])
                if "price" in sub_item: _add_gt_entity("menu.price", sub_item["price"])
                if "num" in sub_item: _add_gt_entity("menu.num", sub_item["num"])
                if "itemsubtotal" in sub_item: _add_gt_entity("menu.itemsubtotal", sub_item["itemsubtotal"])
            elif isinstance(sub_item, list): # If 'sub' is a list of sub-items
                for sub_sub_item in sub_item:
                    if isinstance(sub_sub_item, dict):
                        if "nm" in sub_sub_item: _add_gt_entity("menu.nm", sub_sub_item["nm"])
                        if "cnt" in sub_sub_item: _add_gt_entity("menu.cnt", sub_sub_item["cnt"])
                        if "price" in sub_sub_item: _add_gt_entity("menu.price", sub_sub_item["price"])
                        if "num" in sub_sub_item: _add_gt_entity("menu.num", sub_sub_item["num"])
                        if "itemsubtotal" in sub_sub_item: _add_gt_entity("menu.itemsubtotal", sub_sub_item["itemsubtotal"])


    # The original 'valid_line' parsing is less reliable for structured KIE than 'gt_parse'
    # but we will keep it for words_on_page and bboxes_on_page if they are used elsewhere.
    # For KIE, we rely on the structured 'gt_parse'.
    for line_data in data["valid_line"]:
        for word_info in line_data["words"]:
            words_on_page.append(word_info["text"])
            # Bbox normalization logic (unchanged)
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


# --- Dynamic Label Initialization (unchanged, still good to have full label set) ---
def initialize_global_labels(dataset):
    # This function is mostly for internal consistency if other parts of a larger system
    # rely on a global label set. For this specific KIE evaluation, it's less critical.
    ALL_BASE_LABELS_LOCAL = set()

    for split in ["train", "validation", "test"]:
        for i in range(min(len(dataset[split]), 200)): # Limit for quick initialization
            ground_truth_str = dataset[split][i]["ground_truth"]
            image_width = dataset[split][i]["image"].size[0]
            image_height = dataset[split][i]["image"].size[1]
            
            # Use the updated extract function to get all relevant labels
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


# --- Prompt for Gemini (Dynamically generated based on present GT labels) ---
def create_gemini_prompt_for_cord(labels_to_request: frozenset):
    # Dynamically build the required fields based on labels_to_request for the current sample
    
    prompt_fields_schema = {}
    prompt_fields_example = {}

    # Menu Items Schema and Example
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
    
    if menu_item_schema: # Only add menu_items section if any menu fields are requested
        prompt_fields_schema["menu_items"] = f"""
    - menu_items: array of objects exept "PB1" and text with "%" symbols (e.g., [{json.dumps(menu_item_example)}])
        - Each object should have:
            {chr(10).join([f"- {k}: {v}" for k,v in menu_item_schema.items()])}
            """
        prompt_fields_example["menu_items"] = [menu_item_example]


    # Sub_total Schema and Example
    sub_total_schema_parts = []
    sub_total_example_parts = []
    if "sub_total.subtotal_price" in labels_to_request: 
        sub_total_schema_parts.append('"subtotal_price": "string (Subtotal amount)"')
        sub_total_example_parts.append('"subtotal_price": "95.50"')
    if "sub_total.discount_price" in labels_to_request: 
        sub_total_schema_parts.append('"discount_price": "string (Discount amount, if any, e.g., \'-5.00\')"')
        sub_total_example_parts.append('"discount_price": "-5.00"')
    
    if sub_total_schema_parts: # Only add sub_total section if any sub_total fields are requested
        prompt_fields_schema["sub_total"] = f"""
    - sub_total: object (e.g., {{{", ".join(sub_total_example_parts)}}})
        - {chr(10).join([s.replace('"', '') for s in sub_total_schema_parts])}
        """
        prompt_fields_example["sub_total"] = json.loads("{" + ", ".join(sub_total_example_parts) + "}")


    # Total Schema and Example
    total_schema_parts = []
    total_example_parts = []
    if "total.total_price" in labels_to_request: 
        total_schema_parts.append('"total_price": "string (The final total amount). Do not check the CASH given or change given just the total bill amount."')
        total_example_parts.append('"total_price": "108.00"')
    
    if total_schema_parts: # Only add total section if any total fields are requested
        prompt_fields_schema["total"] = f"""
    - total: object (e.g., {{{", ".join(total_example_parts)}}})
        - {chr(10).join([s.replace('"', '') for s in total_schema_parts])}
        """
        prompt_fields_example["total"] = json.loads("{" + ", ".join(total_example_parts) + "}")


    # Construct the main prompt string
    prompt_parts = []
    prompt_parts.append("""
    You are an expert at extracting structured information from receipt images.
    **Strictly output only valid JSON format in your response (nothing else, no text). Ensure all property names are double-quoted and commas are correctly placed between elements.**
    Provide the output as a JSON object. If a field is not found or cannot be confidently extracted, omit it or set its value to null.

    **Important Notes for Extraction:**
    1.  **Strictly adhere to the JSON schema and only extract the fields explicitly requested.** Do not hallucinate or include any other fields.
    2.  **Accuracy for Numeric Values:** Extract all numeric values (prices, counts, totals) exactly as they appear, without adding or removing digits. For all items, do not give per quantity price - always extract total price of that quantity written. For the final bill amount "total_price": do not give the cash amount given by the customer please.
    3.  **Clean Values:** For all fields, extract *only* the relevant text or numeric value. Do NOT include any surrounding descriptive words, currency symbols, quantity prefixes (like 'x', 'qty'), or punctuation unless it's part of the core name or a decimal point. For discount, please use your mind that it cannot be more than total amount.
    4.  **Menu Items:** Each object in `menu_items` must represent a item entry found on the receipt. Do not combine multiple menu items into one entry, or split a single item across multiple entries. Do not miss any items in the receipt, one item can also be in 2 lines if the names are long. 
        "PB1" or text with "less ice" is a tax not a menu item, do not give it as an item answer. Moreover, if you see an item only single time - do not put it 2 times (eg - "EGG TART"). If there is no price in front of an item then please exclude it from the items. **Crucially, extract ALL instances of an item, even if it appears multiple times on the receipt (e.g., if "Coke" appears twice, extract two separate "Coke" entries).**

    Required fields:
    """)

    # Add fields based on labels_to_request
    if "sub_total" in prompt_fields_schema: prompt_parts.append(prompt_fields_schema["sub_total"])
    if "total" in prompt_fields_schema: prompt_parts.append(prompt_fields_schema["total"])
    if "menu_items" in prompt_fields_schema: 
        prompt_parts.append(prompt_fields_schema["menu_items"])

    prompt_parts.append("\nOutput example:\n```json\n")
    prompt_parts.append(json.dumps(prompt_fields_example, indent=2))
    prompt_parts.append("\n```\nPlease provide only the JSON output within markdown triple backticks.")
    
    final_prompt = "\n".join(prompt_parts)
    # Clean up extra newlines that might be introduced by f-strings
    final_prompt = re.sub(r'\n\s*\n', '\n\n', final_prompt).strip()
    return final_prompt


# --- Function to call Gemini VLM (Synchronous) ---
def call_gemini_vlm(image_pil: Image.Image, prompt_text: str, model_name=GEMINI_MODEL_NAME, max_retries=3, initial_delay=5):
    """
    Calls the Gemini Vision model with an image (PIL.Image) and a text prompt.
    Returns the raw text response from the model or None on error.
    Includes retry logic and improved error handling for safety blocks.
    """
    model = genai.GenerativeModel(model_name)

    # Safety settings to allow more flexible responses for data extraction
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                [prompt_text, image_pil],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=2048 # Increased token limit
                ),
                safety_settings=safety_settings
            )
            
            # Check for prompt feedback first (e.g., if the input itself was blocked)
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Gemini API call blocked by prompt feedback: {response.prompt_feedback.block_reason}", file=sys.stderr)
                if response.prompt_feedback.safety_ratings:
                    for rating in response.prompt_feedback.safety_ratings:
                        print(f"  Safety Rating: Category={rating.category.name}, Probability={rating.probability.name}, Blocked={rating.blocked}", file=sys.stderr)
                return None # No content generated due to prompt block

            # Check if candidates exist and have content
            if response.candidates:
                # Iterate through parts to find text content
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                return part.text.strip()
                # If we reach here, no text part was found in any candidate
                # This means candidates were returned, but they didn't contain text (e.g., blocked content)
                print(f"Gemini API call returned candidates but no text content.", file=sys.stderr)
                if response.candidates[0].finish_reason:
                    print(f"  Candidate finish reason: {response.candidates[0].finish_reason.name}", file=sys.stderr)
                return None # No text content found
            else:
                # No candidates returned at all
                print(f"Gemini API call returned no candidates. Raw response: {response}", file=sys.stderr)
                return None

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for Gemini API call: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                sleep_time = initial_delay * (2 ** attempt)
                print(f"Retrying in {sleep_time:.1f} seconds...", file=sys.stderr)
                time.sleep(sleep_time)
            else:
                print(f"Max retries reached for Gemini API. Skipping.", file=sys.stderr)
                return None
    return None # Should not be reached if max_retries is handled correctly

# --- Post-process Gemini Output to Flattened Entities (Streamlined and Robust) ---
def postprocess_gemini_output_to_flattened_entities(llm_output: str):
    """
    Parses LLM output, extracts entities, and flattens them into a list of
    {"label": "category.subcategory", "text": "extracted_value"} objects.
    Applies normalization to extracted text and strictly filters by TARGET_LABELS.
    Includes robust JSON parsing by cleaning common non-standard characters,
    converting single-quoted property names to double-quoted, and
    heuristically inserting missing commas.
    """
    json_str = llm_output.strip()

    # Remove markdown triple backticks if present
    if json_str.startswith("```json"):
        json_str = json_str[len("```json"):].strip()
    if json_str.endswith("```"):
        json_str = json_str[:-len("```")].strip()

    # Remove common non-standard whitespace characters that can break JSON parsing
    json_str = json_str.replace('\xa0', ' ') # Replace non-breaking space with regular space
    json_str = re.sub(r'[\u200B-\u200F\u202F\u205F\u3000]', '', json_str) # Remove other common invisible chars

    # Convert single-quoted property names to double-quoted property names
    json_str = re.sub(r"'([^']+?)':", r'"\1":', json_str)

    # Heuristic to insert missing commas between closing brackets/braces and new keys
    json_str = re.sub(r'([\]}])\s*(")', r'\1,\2', json_str)

    extracted_data = {}
    try:
        extracted_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"--- JSON PARSING ERROR ---", file=sys.stderr)
        print(f"Raw Gemini Output (causing error):", file=sys.stderr)
        print("-----------------------------------", file=sys.stderr)
        print(llm_output, file=sys.stderr) # Print the full raw output here
        print("-----------------------------------", file=sys.stderr)
        print(f"Could not parse JSON from Gemini output: {e}. Problematic string snippet: '{json_str[max(0, e.pos-50):e.pos+50]}'", file=sys.stderr)
        print(f"--- END JSON PARSING ERROR ---", file=sys.stderr)
        return []

    flattened_extracted_entities = []

    # Define which labels are numeric for correct normalization
    numeric_labels = frozenset({
        "menu.cnt", "menu.price", "menu.itemsubtotal", "menu.num",
        "sub_total.subtotal_price", "sub_total.discount_price",
        "sub_total.service_price", "sub_total.tax_price", 
        "total.total_price", "total.cashprice", "total.change_price", 
        "total.creditcardprice", "total.menuqty_cnt" 
    })

    # Helper to add a flattened entity from LLM output
    def _add_llm_entity(label, value):
        # Only process if the label is in our TARGET_LABELS
        if label not in TARGET_LABELS:
            return

        # Handle cases where LLM might return null or "Not Provided" explicitly
        if value is None or (isinstance(value, str) and value.lower().strip() == "null") or \
           (isinstance(value, str) and value.lower().strip() == "not provided"):
            return

        is_numeric_category = label in numeric_labels
        normalized_value = normalize_text_for_matching(str(value), is_numeric_field=is_numeric_category)
        
        if normalized_value: # Add only if normalization resulted in non-empty string
            flattened_extracted_entities.append({"label": label, "text": normalized_value})

    # --- Process LLM's extracted data based on the structure it returns ---

    # Sub-Total
    sub_total = extracted_data.get("sub_total")
    if isinstance(sub_total, dict):
        _add_llm_entity("sub_total.subtotal_price", sub_total.get("subtotal_price"))
        _add_llm_entity("sub_total.discount_price", sub_total.get("discount_price"))

    # Total
    total = extracted_data.get("total")
    if isinstance(total, dict):
        _add_llm_entity("total.total_price", total.get("total_price"))

    # Menu Items
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


# --- Custom KIE Metric Calculation (Corrected FN/FP for Mismatches) ---
def calculate_kie_metrics(gt_entities: list, pred_entities: list, fuzz_threshold=80):
    """
    Calculates KIE (Key Information Extraction) metrics (Precision, Recall, F1-score)
    by comparing flattened ground truth entities with predicted entities.
    Returns overall metrics, a list of predicted entities that matched a GT entity,
    and lists of False Negatives and False Positives.
    
    This version treats substitutions (correct label, incorrect value) as False Negatives.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_predicted_entities = [] # To store predicted entities that found a match
    false_negatives_list = [] # To store GT entities that were missed
    false_positives_list = [] # To store Pred entities that were extra

    # Create copies to track consumption
    gt_consumed = [False] * len(gt_entities)
    pred_consumed = [False] * len(pred_entities)

    # First pass: Attempt to match every GT entity
    for i, gt_entity in enumerate(gt_entities):
        if gt_consumed[i]: continue

        gt_label = gt_entity["label"]
        gt_text = gt_entity["text"]

        current_fuzz_threshold = 100 if gt_label in EXACT_MATCH_LABELS else (75 if gt_label == "menu.nm" else fuzz_threshold)
        fuzzy_match_func = fuzz.ratio if gt_label in EXACT_MATCH_LABELS else fuzz.token_set_ratio

        best_match_idx = -1
        best_match_score = -1

        # Find the best matching predicted entity for this GT entity, considering label match
        for j, pred_entity in enumerate(pred_entities):
            if pred_consumed[j]: continue
            
            if gt_label != pred_entity["label"]: # Labels must match for a potential match
                continue

            score = fuzzy_match_func(gt_text, pred_entity["text"])

            if score > best_match_score:
                best_match_score = score
                best_match_idx = j
        
        # Now, evaluate the best match found (if any)
        if best_match_idx != -1: # A predicted entity with the same label was found
            if best_match_score >= current_fuzz_threshold:
                # Case 1: True Positive (Label and Value match above threshold)
                true_positives += 1
                gt_consumed[i] = True
                pred_consumed[best_match_idx] = True
                matched_predicted_entities.append(pred_entities[best_match_idx])
            else:
                # Case 2: Substitution/Mismatch (Label matches, but Value is below threshold)
                # Treat this as a False Negative for the GT (correct value not extracted)
                # and consume both to avoid further counting.
                false_negatives += 1 # Increment FN for the missed GT
                false_negatives_list.append(gt_entity) # Add the GT to FN list
                
                gt_consumed[i] = True # Consume GT
                pred_consumed[best_match_idx] = True # Consume Pred to prevent it from being an FP later
                # IMPORTANT: We are NOT incrementing false_positives here for the substitution.
                # The predicted entity is "consumed" but not counted as an FP,
                # effectively making the substitution a single FN penalty.
        else:
            # Case 3: False Negative (No predicted entity with the same label was found for this GT)
            false_negatives += 1
            false_negatives_list.append(gt_entity)
    
    # Second pass: Identify remaining False Positives (predicted entities that didn't match any GT)
    for i, pred_entity in enumerate(pred_entities):
        if not pred_consumed[i]:
            # --- Option B for menu.cnt: Do not penalize unmatched menu.cnt as FP ---
            # This rule still applies to *truly extra* menu.cnt predictions.
            if pred_entity["label"] == "menu.cnt":
                continue 
            # --- End Option B ---
            
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


# --- Main Evaluation Function (Synchronous) ---
def evaluate_gemini_on_cord():
    print("--- Starting Gemini evaluation for CORD v2 (KIE Metrics) ---")
    print(f"Gemini Model: {GEMINI_MODEL_NAME}")

    # 1. Load Dataset
    # Using "naver-clova-ix/cord-v2" as per the user's original file
    dataset = load_dataset("naver-clova-ix/cord-v2")
    test_dataset = dataset["validation"]
    
    # Limit dataset size for demonstration if MAX_SAMPLES_TO_PROCESS is set
    if MAX_SAMPLES_TO_PROCESS is not None:
        test_dataset = test_dataset.shuffle(seed=84)  # Use a fixed seed for reproducibility
        test_dataset = test_dataset.select(range(min(len(test_dataset), MAX_SAMPLES_TO_PROCESS)))
    
    print(f"Dataset loaded. Test size: {len(test_dataset)}")

    # 2. Initialize global labels (still useful for consistency)
    initialize_global_labels(dataset)

    # Lists to store flattened entities for overall KIE metric calculation
    all_gt_flattened_entities_list = []
    all_predicted_flattened_entities_list = [] # This will now store ALL filtered predictions for overall metrics

    total_samples = len(test_dataset)
    print(f"Total test samples available: {total_samples}")
    print(f"Processing a maximum of {MAX_SAMPLES_TO_PROCESS} samples for debugging.")


    for i, example in enumerate(test_dataset):
        if i >= MAX_SAMPLES_TO_PROCESS:
            print(f"Reached MAX_SAMPLES_TO_PROCESS ({MAX_SAMPLES_TO_PROCESS}). Stopping.")
            break

        print(f"\n--- Processing Sample {i+1}/{MAX_SAMPLES_TO_PROCESS} (Original index: {i}) ---")

        image = example["image"] # PIL Image object
        ground_truth_json_str = example["ground_truth"]
        image_width, image_height = image.size

        # Get original words (for reference/debugging) and flattened ground truth entities (normalized)
        # This call now filters by TARGET_LABELS and returns labels_present_in_gt
        original_words, _, gt_flattened_entities, labels_present_in_gt = \
            extract_words_bboxes_and_labels_from_gt(ground_truth_json_str, image_width, image_height)
        
        all_gt_flattened_entities_list.extend(gt_flattened_entities) # Add to global list for overall metrics

        # --- Dynamically create the prompt based on labels present in THIS sample's GT ---
        # Only request fields that are both in TARGET_LABELS and actually present in the current GT
        dynamic_labels_to_request = TARGET_LABELS.intersection(labels_present_in_gt)
        prompt_text = create_gemini_prompt_for_cord(dynamic_labels_to_request)
        # print(f"\n--- Generated Dynamic Gemini Prompt for Sample {i+1} --- \n{prompt_text}\n-----------------------------") # Removed for minimal logging

        print(f"Calling Gemini VLM for sample {i+1}...")
        # Pass the PIL Image directly to call_gemini_vlm
        gemini_output = call_gemini_vlm(image, prompt_text)

        if gemini_output:
            print(f"--- Raw Gemini Output for Sample {i+1} ---")
            print(gemini_output) # Print raw output for every sample
            print(f"------------------------------------------")
            print(f"Gemini output received for sample {i+1}. Post-processing...")
            # This call now filters by TARGET_LABELS
            current_predicted_flattened_entities = postprocess_gemini_output_to_flattened_entities(gemini_output)
            
            # Calculate metrics for the current sample to get matched_predicted_entities and mismatches
            sample_metrics_results = calculate_kie_metrics(gt_flattened_entities, current_predicted_flattened_entities)
            
            # Extend the overall list with ALL predicted entities (including FPs) for overall metrics
            all_predicted_flattened_entities_list.extend(current_predicted_flattened_entities)


            # --- Print Ground Truth and ONLY MATCHED Predicted Entities ---
            print("\n--- Ground Truth Entities (Flattened & Normalized) ---")
            if not gt_flattened_entities:
                print("  (No relevant GT entities for this sample in TARGET_LABELS)")
            for entity in gt_flattened_entities:
                print(f"  GT: Label='{entity['label']}', Text='{entity['text']}'")
            
            print("\n--- Gemini Predicted Entities (Flattened & Normalized) (Matched Only) ---")
            if not sample_metrics_results['matched_predicted_entities']:
                print("  (No predicted entities matched GT for this sample in TARGET_LABELS)")
            for entity in sample_metrics_results['matched_predicted_entities']:
                print(f"  Pred: Label='{entity['label']}', Text='{entity['text']}'")
            print("------------------------------------------")

            print(f"Sample {i+1} KIE Metrics: P={sample_metrics_results['precision']:.2f}, R={sample_metrics_results['recall']:.2f}, F1={sample_metrics_results['f1']:.2f}")
            
            # --- Log Mismatches for the current sample ---
            if sample_metrics_results['false_negatives_list']:
                print("--- False Negatives (GT entities not extracted or not matching) ---")
                for fn_entity in sample_metrics_results['false_negatives_list']:
                    print(f"  FN: Label='{fn_entity['label']}', Text='{fn_entity['text']}'")
            
            if sample_metrics_results['false_positives_list']:
                print("--- False Positives (Extra predicted entities not matching GT) ---")
                for fp_entity in sample_metrics_results['false_positives_list']:
                    print(f"  FP: Label='{fp_entity['label']}', Text='{fp_entity['text']}'")

        else:
            print(f"Gemini call failed or returned no output for sample {i+1}. Skipping KIE evaluation for this sample.", file=sys.stderr)
            # No entities added to all_predicted_flattened_entities_list, effectively counting as FNs
        
        # --- Add a delay between samples ---
        if i < MAX_SAMPLES_TO_PROCESS - 1: # Don't sleep after the last sample
            print(f"Pausing for 3 seconds before next sample...")
            time.sleep(3)


    print("\nCalculating final KIE metrics for Gemini across all processed samples...")
    final_kie_metrics_results = calculate_kie_metrics(all_gt_flattened_entities_list, all_predicted_flattened_entities_list)

    print(f"\n--- Gemini CORD v2 Evaluation Results (KIE Metrics) for model {GEMINI_MODEL_NAME} ---")
    print(f"Precision: {final_kie_metrics_results['precision']:.4f}")
    print(f"Recall:    {final_kie_metrics_results['recall']:.4f}")
    print(f"F1-score:  {final_kie_metrics_results['f1']:.4f}")
    print(f"True Positives: {final_kie_metrics_results['true_positives']}")
    print(f"False Positives: {final_kie_metrics_results['false_positives']}")
    print(f"False Negatives: {final_kie_metrics_results['false_negatives']}")
    
    print("\n--- Gemini CORD v2 evaluation finished! ---")

# --- Run the script ---
if __name__ == "__main__":
    evaluate_gemini_on_cord()
