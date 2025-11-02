import os
import json
import base64
import requests
from PIL import Image
from io import BytesIO
from thefuzz import fuzz # For fuzzy string matching
import re # For cleaning total amounts
from datetime import datetime # For robust date parsing

# --- Configuration ---
# Default Ollama API address. Ensure Ollama is running on this host/port.
OLLAMA_HOST = "http://localhost:11434"
# The name of the multimodal Ollama model you have pulled (e.g., "gemma:latest", "llava", "bakllava").
# Make sure you have pulled this model using `ollama pull <model_name>`.
OLLAMA_MODEL = "qwen2.5vl:latest" # Set this to your preferred Gemma model or other VLM

# Path to your SROIE 2019 dataset.
# This assumes a structure like sroie_dataset/test/img and sroie_dataset/test/entities
SROIE_ROOT = "./datasets/SROIE2019"
SROIE_TEST_DATA_PATH = os.path.join(SROIE_ROOT, "test")

# Expected entities for SROIE (for evaluation)
SROIE_ENTITY_KEYS = ["company", "date", "address", "total"]

# --- Helper Functions for Ollama API Interaction ---

def encode_image_to_base64(image_path):
    """Encodes an image to base64 string."""
    try:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="JPEG") # Assuming JPEG for SROIE images
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
    Includes robust parsing for markdown-wrapped JSON.
    """
    url = f"{OLLAMA_HOST}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False, # We want the full response at once for evaluation
        "options": {
            "temperature": 0.1, # Lower temperature for more consistent, less creative output
            "num_predict": -1 # Generate until stop token or max context length
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

        # --- Robust Parsing for Markdown-wrapped JSON ---
        # Check if the content is wrapped in a markdown code block (e.g., ```json{...}```)
        if generated_content.startswith("```json"):
            # Find the start and end of the JSON content
            json_start_idx = generated_content.find('{')
            json_end_idx = generated_content.rfind('}') + 1
            
            if json_start_idx != -1 and json_end_idx != -1 and json_end_idx > json_start_idx:
                json_string = generated_content[json_start_idx:json_end_idx]
            else:
                # Fallback if JSON structure inside markdown is not as expected
                print(f"Warning: JSON markdown block found but structure unexpected. Raw: {generated_content[:200]}")
                json_string = generated_content.replace("```json", "").replace("```", "").strip()
        else:
            json_string = generated_content # Assume it's plain JSON

        # Attempt to parse the cleaned string as JSON
        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            print(f"Warning: Ollama did not return valid JSON after stripping markdown. Raw response snippet: {generated_content[:500]}...")
            return None 
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        print("Please ensure Ollama is running and the model is pulled (`ollama run {model_name}`).")
        return None

# --- Dataset Loading and Ground Truth Parsing ---

def load_sroie_document(doc_id):
    """
    Loads a single SROIE 2019 document's image path and its ground truth JSON annotations.
    Assumes image is in test/img and JSON in test/entities (as .txt files).
    Normalizes ground truth values for robust comparison.
    """
    img_path = os.path.join(SROIE_TEST_DATA_PATH, "img", f"{doc_id}.jpg") # Assuming .jpg images
    json_path = os.path.join(SROIE_TEST_DATA_PATH, "entities", f"{doc_id}.txt") # SROIE entities are in .txt files

    if not os.path.exists(img_path) or not os.path.exists(json_path):
        print(f"Error: SROIE document {doc_id} not found at {img_path} or {json_path}")
        return None, None

    try:
        with open(json_path, 'r', encoding='utf8') as f:
            ground_truth_text = f.read()
            ground_truth = json.loads(ground_truth_text)
        
        # Normalize ground truth values: strip whitespace and convert to lowercase
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
    Explicitly asks for JSON output.
    """
    return """
Analyze the attached receipt image and extract its complete text to find the below information ->
    1. "company or store name": the name of the company or store written on the top of the receipt.
    2. "date": The date of the transaction in the same format as available in text. Do not include any time with the date.
    3. "address": The full address of the company or the store where transaction has happened.
    4. "total": The total amount of the transaction (including taxes/gst everything).

    Please return above 4 points in a JSON format. Extract the exact value from the text, do not format or modify the data.
```
"""

# --- Core Logic for Alignment and Evaluation (KIE F1) ---

def postprocess_sroie_output(llm_output_json, ground_truth_json, entity_keys):
    """
    Compares LLM's extracted JSON against ground truth for SROIE entities.
    Calculates True Positives, False Positives, False Negatives for KIE F1.
    Uses fuzzy matching for company/address, stricter matching for date/total.
    """
    tp = 0
    fp = 0
    fn = 0

    key_remapping = {
        "company_name": "company",
        "company_or_store_name": "company",
        "invoice_date": "date",
        "store_address": "address",
        "grand_total": "total",
        "amount_total": "total",
    }

    # Normalize LLM output values, explicitly treating string "null" as empty string
    normalized_llm_output = {}
    if isinstance(llm_output_json, dict):
        for k, v in llm_output_json.items():
        # Apply key remapping first, converting to lowercase
            standardized_key = key_remapping.get(k.lower(), k.lower())
            
            val = str(v).strip().lower()
            if val == "null" or val == "": 
                normalized_llm_output[standardized_key] = ""
            else:
                normalized_llm_output[standardized_key] = val
        
    for key in entity_keys:
        gt_value = ground_truth_json.get(key) # Already normalized lowercase
        llm_value = normalized_llm_output.get(key)

        # Case 1: Ground truth does NOT have this key/value (or it's an empty string)
        if gt_value == "": 
            # If LLM extracted something non-empty for this key, it's a False Positive
            if llm_value != "":
                fp += 1
                print(f"  FP for '{key}': LLM extracted '{llm_value}', but GT is empty/null.")
            continue 

        # Case 2: Ground truth HAS a non-empty value for this key
        if llm_value != "": # If LLM extracted something non-empty (not "null" string, not empty string)
            is_match = False
            if key in ["company", "address"]:
                # Use fuzzy ratio for company and address (threshold 85)
                if fuzz.ratio(gt_value, llm_value) >= 80: 
                    is_match = True
            elif key == "date":
                # Stricter date comparison: try to parse and compare dates
                # Common SROIE date formats: DD/MM/YYYY, DD MON YYYY, YYYY-MM-DD, etc.
                date_formats = ["%d/%m/%Y", "%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%m/%d/%Y"] 
                
                gt_date_obj = None
                llm_date_obj = None

                # Try parsing GT date
                for fmt in date_formats:
                    try:
                        gt_date_obj = datetime.strptime(gt_value, fmt)
                        break
                    except ValueError:
                        pass
                
                # Try parsing LLM date
                for fmt in date_formats:
                    try:
                        llm_date_obj = datetime.strptime(llm_value, fmt)
                        break
                    except ValueError:
                        pass

                if gt_date_obj and llm_date_obj and gt_date_obj == llm_date_obj:
                    is_match = True
                elif fuzz.ratio(gt_value, llm_value) == 100: # Fallback to exact string match if parsing fails
                    is_match = True
                else:
                    # Debug info for date mismatches
                    print(f"  Date Mismatch: GT='{gt_value}' ({gt_date_obj}), LLM='{llm_value}' ({llm_date_obj})")

            elif key == "total":
                # Stricter total comparison: convert to float and compare
                def clean_amount(amount_str):
                    if isinstance(amount_str, (int, float)): # Already numeric
                        return float(amount_str)
                    # Remove currency symbols, commas, spaces, and other non-numeric chars except dot
                    cleaned = re.sub(r'[^\d.]', '', amount_str) 
                    try:
                        return float(cleaned)
                    except ValueError:
                        return None # Return None if conversion fails

                gt_total_float = clean_amount(gt_value)
                llm_total_float = clean_amount(llm_value)

                # Use a small epsilon for floating point comparison
                epsilon = 0.01 

                if gt_total_float is not None and llm_total_float is not None and \
                   abs(gt_total_float - llm_total_float) < epsilon:
                    is_match = True
                elif fuzz.ratio(gt_value, llm_value) == 100: # Fallback to exact string match if numeric fails
                    is_match = True
                else:
                    # Debug info for total mismatches
                    print(f"  Total Mismatch: GT='{gt_value}' ({gt_total_float}), LLM='{llm_value}' ({llm_total_float})")

            # Apply match result
            if is_match:
                tp += 1
                # print(f"  TP for '{key}': GT='{gt_value}', LLM='{llm_value}'") # Uncomment for TP details
            else:
                fp += 1
                print(f"  FP for '{key}': GT='{gt_value}', LLM='{llm_value}' (Fuzzy Ratio: {fuzz.ratio(gt_value, llm_value) if key in ['company', 'address'] else 'N/A - strict'})")
        else: # LLM extracted empty/null (because `llm_value` is now `""`)
            # Ground truth has a value, but LLM did not extract it or extracted empty/null (False Negative)
            fn += 1
            print(f"  FN for '{key}': GT='{gt_value}', LLM extracted empty/null.")
            
    return tp, fp, fn

# --- Main Evaluation Loop ---

async def evaluate_sroie(num_samples=None):
    """Evaluates the Ollama VLM on the SROIE 2019 dataset."""
    print(f"\n--- Starting evaluation for SROIE 2019 with {OLLAMA_MODEL} ---")

    img_dir = os.path.join(SROIE_TEST_DATA_PATH, "img")
    
    all_doc_ids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg")] # Assuming .jpg
    if num_samples:
        all_doc_ids = all_doc_ids[:min(num_samples, len(all_doc_ids))]

    total_tp = 0
    total_fp = 0
    total_fn = 0
    processed_docs = 0

    prompt = get_sroie_prompt()

    print(f"Total documents to process: {len(all_doc_ids)}")

    for i, doc_id in enumerate(all_doc_ids):
        print(f"\nProcessing document {i+1}/{len(all_doc_ids)}: {doc_id}.jpg")

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

        tp, fp, fn = postprocess_sroie_output(llm_output_json, ground_truth_json, SROIE_ENTITY_KEYS)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        processed_docs += 1

        print(f"  Current Doc Metrics: TP={tp}, FP={fp}, FN={fn}")

    # --- Aggregate and Print Final Results ---
    print("\n" + "="*50)
    print("--- Overall Evaluation Results for SROIE 2019 ---")
    print("="*50)
    if processed_docs > 0:
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Processed Documents: {processed_docs}/{len(all_doc_ids)}")
        print(f"Total True Positives: {total_tp}")
        print(f"Total False Positives: {total_fp}")
        print(f"Total False Negatives: {total_fn}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")
        print(f"Overall F1 Score: {f1:.4f}")
    else:
        print("No documents were successfully processed for evaluation.")
    print("="*50)

# To run the evaluation:
if __name__ == "__main__":
    import asyncio
    # IMPORTANT:
    # 1. Ensure Ollama is running (`ollama serve` in your terminal).
    # 2. Ensure you have pulled the specified OLLAMA_MODEL (e.g., `ollama pull gemma:latest`).
    # 3. Download the SROIE 2019 dataset and place it according to SROIE_ROOT path.
    #    (e.g., `sroie_dataset/test/img` and `sroie_dataset/test/entities` as .txt files)

    # For a full run, remove num_samples.
    # Start with a small num_samples (e.g., 5 or 10) to test your setup.
    asyncio.run(evaluate_sroie(num_samples=500)) # Adjust num_samples as needed for testing
    # If you want to run on the entire test set, remove num_samples=X
    # asyncio.run(evaluate_sroie())
