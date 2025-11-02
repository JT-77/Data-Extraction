import os
import json
import base64 # Not strictly needed for direct PIL.Image with Gemini, but good to keep in mind
import requests # No longer for Gemini API, but might be useful for other things
from PIL import Image
from io import BytesIO
import google.generativeai as genai # New import for Gemini API
from thefuzz import fuzz # For fuzzy string matching
import re # For cleaning total amounts
from datetime import datetime # For robust date parsing
import time # For potential rate limiting/retries

# --- Configuration ---
# !!! IMPORTANT: Replace with your actual Gemini API Key !!!
GOOGLE_API_KEY = ""

# Configure the Gemini API client
genai.configure(api_key=GOOGLE_API_KEY)

# The name of the Gemini model for multimodal input
GEMINI_MODEL = "gemini-2.5-flash" # This is the standard VLM model

# Path to your SROIE 2019 dataset.
# This assumes a structure like sroie_dataset/test/img and sroie_dataset/test/entities
SROIE_ROOT = "../datasets/SROIE2019"
SROIE_TEST_DATA_PATH = os.path.join(SROIE_ROOT, "test")

# Expected entities for SROIE (for evaluation)
SROIE_ENTITY_KEYS = ["company", "date", "address", "total"]

# --- Helper Functions for Gemini API Interaction ---

def call_gemini_vlm(image_path, prompt, model_name=GEMINI_MODEL, max_retries=3, initial_delay=5):
    """
    Calls the Gemini API for a multimodal model.
    Returns parsed JSON response or None on error.
    Includes robust parsing for markdown-wrapped JSON and retry logic.
    """
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error initializing Gemini model '{model_name}': {e}")
        return None

    try:
        # Gemini can take PIL.Image directly
        image_pil = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    for attempt in range(max_retries):
        try:
            # Send both the image and the text prompt to the model
            response = model.generate_content([prompt, image_pil])
            
            # Access the text from the response
            generated_content = response.text.strip()

            if not generated_content:
                print("Warning: Gemini returned an empty response.")
                return None

            # --- Robust Parsing for Markdown-wrapped JSON ---
            # Check if the content is wrapped in a markdown code block (e.g., ```json{...}```)
            if generated_content.startswith("```json"):
                json_start_idx = generated_content.find('{')
                json_end_idx = generated_content.find('```', json_start_idx) # Find closing ``` after JSON starts
                
                if json_start_idx != -1 and json_end_idx != -1 and json_end_idx > json_start_idx:
                    json_string = generated_content[json_start_idx:json_end_idx]
                else:
                    print(f"Warning: JSON markdown block found but structure unexpected. Raw: {generated_content[:200]}")
                    json_string = generated_content.replace("```json", "").replace("```", "").strip()
            else:
                json_string = generated_content # Assume it's plain JSON

            # Attempt to parse the cleaned string as JSON
            try:
                return json.loads(json_string)
            except json.JSONDecodeError:
                print(f"Warning: Gemini did not return valid JSON after stripping markdown. Raw response snippet: {generated_content[:500]}...")
                return None

        except genai.types.BlockedPromptException as e:
            print(f"Prompt was blocked by safety settings: {e}")
            return None
        except Exception as e: # Catch broader exceptions for retries (e.g., API errors, network issues)
            print(f"Attempt {attempt + 1}/{max_retries} failed for {image_path}: {e}")
            if attempt < max_retries - 1:
                sleep_time = initial_delay * (2 ** attempt) # Exponential backoff
                print(f"Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Max retries reached for {image_path}. Skipping.")
                return None
    return None # Should not be reached if max_retries > 0

# --- Dataset Loading and Ground Truth Parsing (Same as before) ---

def load_sroie_document(doc_id):
    """
    Loads a single SROIE 2019 document's image path and its ground truth JSON annotations.
    Assumes image is in test/img and JSON in test/entities (as .txt files).
    Normalizes ground truth values for robust comparison.
    """
    img_path = os.path.join(SROIE_TEST_DATA_PATH, "img", f"{doc_id}.jpg")
    json_path = os.path.join(SROIE_TEST_DATA_PATH, "entities", f"{doc_id}.txt")

    if not os.path.exists(img_path) or not os.path.exists(json_path):
        # print(f"Error: SROIE document {doc_id} not found at {img_path} or {json_path}") # Suppress for cleaner output during large runs
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

# --- Prompt Definition (Same as the optimized one for Ollama) ---

def get_sroie_prompt():
    """
    Generates a precise prompt for the VLM to extract SROIE 2019 entities.
    Explicitly asks for JSON output with specific keys.
    """
    return """
Analyze the attached receipt image and extract its complete text to find the below information ->
    1. "company or store name": the name of the company or store written on the top of the receipt.
    2. "date": The date of the transaction in the same format as available in text. Do not include any time with the date or format it.
        For eg - if date is "01/01/16 10:03" then only return "01/01/16" etc.
    3. "address": The full address of the company or the store where transaction has happened.
    4. "total": The total amount of the transaction (including taxes/gst everything).

    Please return above 4 points in a JSON format as your response. Extract the exact value from the text, do not format or modify the data.
    Respond with just only JSON response (nothing else, no other text).
```
"""

# --- Core Logic for Alignment and Evaluation (KIE F1) ---
# This function is designed to be generic for any LLM outputting JSON.

def postprocess_sroie_output(llm_output_json, ground_truth_json, entity_keys):
    """
    Compares LLM's extracted JSON against ground truth for SROIE entities.
    Calculates True Positives, False Positives, False Negatives for KIE F1.
    Uses fuzzy matching for company/address, stricter matching for date/total.
    Includes key remapping for LLM output to standard keys.
    """
    tp = 0
    fp = 0
    fn = 0

    # Define a mapping for common LLM key variations to our standardized keys
    # Add any other variations you observe from your LLM's output here.
    key_remapping = {
        "company_name": "company",
        "company_or_store_name": "company",
        "company or store name": "company",
        "invoice_date": "date",
        "transaction_date": "date",
        "store_address": "address",
        "full_address": "address",
        "grand_total": "total",
        "amount_total": "total",
    }

    # Normalize LLM output values and apply key remapping
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
    
    for key in entity_keys: # Iterate through our expected keys
        gt_value = ground_truth_json.get(key) # Already normalized lowercase
        llm_value = normalized_llm_output.get(key) # Get mapped and normalized LLM value

        # Case 1: Ground truth does NOT have this key/value (or it's an empty string)
        if gt_value == "": 
            if llm_value != "":
                fp += 1
                print(f"  FP for '{key}': LLM extracted '{llm_value}', but GT is empty/null.")
            continue 

        # Case 2: Ground truth HAS a non-empty value for this key
        if llm_value != "":
            is_match = False
            if key in ["company", "address"]:
                if fuzz.ratio(gt_value, llm_value) >= 90: 
                    is_match = True
            elif key == "date":
                date_formats = ["%d/%m/%Y", "%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"] 
                
                gt_date_obj = None
                llm_date_obj = None

                for fmt in date_formats:
                    try:
                        gt_date_obj = datetime.strptime(gt_value, fmt)
                        break
                    except ValueError:
                        pass
                
                for fmt in date_formats:
                    try:
                        llm_date_obj = datetime.strptime(llm_value, fmt)
                        break
                    except ValueError:
                        pass

                if gt_date_obj and llm_date_obj and gt_date_obj == llm_date_obj:
                    is_match = True
                elif fuzz.ratio(gt_value, llm_value) == 100:
                    is_match = True
                else:
                    print(f"  Date Mismatch: GT='{gt_value}' ({gt_date_obj}), LLM='{llm_value}' ({llm_date_obj})")

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
                else:
                    print(f"  Total Mismatch: GT='{gt_value}' ({gt_total_float}), LLM='{llm_value}' ({llm_total_float})")

            if is_match:
                tp += 1
            else:
                fp += 1
                print(f"  FP for '{key}': GT='{gt_value}', LLM='{llm_value}' (Fuzzy Ratio: {fuzz.ratio(gt_value, llm_value) if key in ['company', 'address'] else 'N/A - strict'})")
        else:
            fn += 1
            print(f"  FN for '{key}': GT='{gt_value}', LLM extracted empty/null.")
            
    return tp, fp, fn

# --- Main Evaluation Loop ---

async def evaluate_sroie(num_samples=None):
    """Evaluates the Gemini VLM on the SROIE 2019 dataset."""
    print(f"\n--- Starting evaluation for SROIE 2019 with {GEMINI_MODEL} ---")

    img_dir = os.path.join(SROIE_TEST_DATA_PATH, "img")
    
    all_doc_ids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg")]
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

        llm_output_json = call_gemini_vlm(img_path, prompt, GEMINI_MODEL)
        print(f"LLM output - {llm_output_json}")

        if llm_output_json is None:
            print(f"Skipping {doc_id} due to invalid/no LLM output from Gemini.")
            continue

        tp, fp, fn = postprocess_sroie_output(llm_output_json, ground_truth_json, SROIE_ENTITY_KEYS)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        processed_docs += 1

        print(f"  Current Doc Metrics: TP={tp}, FP={fp}, FN={fn}")

    # --- Aggregate and Print Final Results ---
    print("\n" + "="*50)
    print("--- Overall Evaluation Results for SROIE 2019 on Gemini 2.5 Flash ---")
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
    
    # !!! IMPORTANT: Replace "YOUR_GEMINI_API_KEY" with your actual key above !!!

    # For a full run, remove num_samples.
    # Start with a small num_samples (e.g., 5 or 10) to test your setup and API key.
    asyncio.run(evaluate_sroie(num_samples=50)) # Adjust num_samples as needed for testing
    # If you want to run on the entire test set, remove num_samples=X
    # asyncio.run(evaluate_sroie())
