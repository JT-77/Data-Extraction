import os
import json
import base64
import requests
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from thefuzz import fuzz
import re
from datetime import datetime
import time
import Levenshtein
import logging

# --- Configuration ---
GOOGLE_API_KEY = ""  # Replace with your actual Gemini API Key
genai.configure(api_key=GOOGLE_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"
SROIE_ROOT = "../datasets/SROIE2019"
SROIE_TEST_DATA_PATH = os.path.join(SROIE_ROOT, "test")
SROIE_ENTITY_KEYS = ["company", "date", "address", "total"]
LOG_FILE = "new_results/gemini_sroie_evaluation_log.jsonl"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Functions for Gemini API Interaction ---

def call_gemini_vlm(image_path, prompt, model_name=GEMINI_MODEL, max_retries=3, initial_delay=5):
    """
    Calls the Gemini API for a multimodal model.
    Returns parsed JSON response or None on error.
    """
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error initializing Gemini model '{model_name}': {e}")
        return None

    try:
        image_pil = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    for attempt in range(max_retries):
        try:
            response = model.generate_content([prompt, image_pil])
            generated_content = response.text.strip()

            if not generated_content:
                print("Warning: Gemini returned an empty response.")
                return None

            if generated_content.startswith("```json"):
                json_start_idx = generated_content.find('{')
                json_end_idx = generated_content.find('```', json_start_idx)
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
                print(f"Warning: Gemini did not return valid JSON. Raw response snippet: {generated_content[:500]}...")
                return None

        except genai.types.BlockedPromptException as e:
            print(f"Prompt was blocked by safety settings: {e}")
            return None
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for {image_path}: {e}")
            if attempt < max_retries - 1:
                sleep_time = initial_delay * (2 ** attempt)
                print(f"Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Max retries reached for {image_path}. Skipping.")
                return None
    return None

# --- Dataset Loading and Ground Truth Parsing ---

def load_sroie_document(doc_id):
    """
    Loads a single SROIE 2019 document's image path and its ground truth JSON annotations.
    """
    img_path = os.path.join(SROIE_TEST_DATA_PATH, "img", f"{doc_id}.jpg")
    json_path = os.path.join(SROIE_TEST_DATA_PATH, "entities", f"{doc_id}.txt")

    if not os.path.exists(img_path) or not os.path.exists(json_path):
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
    Generates a precise prompt for the VLM to extract SROIE 2019 entities.
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

# --- Evaluation Metrics ---

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
    """Compute KIEval Entity/Group F1 and Aligned metrics."""
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
                    correction_cost += 1
            elif key == "date":
                date_formats = ["%d/%m/%Y", "%d %b %Y", "%d %B %Y", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"]
                gt_date_obj = None
                pred_date_obj = None
                for fmt in date_formats:
                    try:
                        gt_date_obj = datetime.strptime(gt_value, fmt)
                        break
                    except ValueError:
                        pass
                for fmt in date_formats:
                    try:
                        pred_date_obj = datetime.strptime(pred_value, fmt)
                        break
                    except ValueError:
                        pass
                if gt_date_obj and pred_date_obj and gt_date_obj.date() == pred_date_obj.date():
                    entity_matches += 1
                else:
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
            correction_cost += 1
        elif pred_value and not gt_value:
            correction_cost += 1

    precision = entity_matches / entity_total_pred if entity_total_pred > 0 else 0.0
    recall = entity_matches / entity_total_gt if entity_total_gt > 0 else 0.0
    entity_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    group_match = 1 if entity_matches == len([k for k in entity_keys if gt_dict.get(k)]) else 0
    group_precision = group_match / 1 if entity_total_pred > 0 else 0.0
    group_recall = group_match / 1 if entity_total_gt > 0 else 0.0
    group_f1 = (2 * group_precision * group_recall) / (group_precision + group_recall) if (group_precision + group_recall) > 0 else 0.0

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
        "transaction_date": "date",
        "store_address": "address",
        "full_address": "address",
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

        # Compute new metrics
        anls = compute_anls(llm_value, gt_value)
        em = compute_exact_match(llm_value, gt_value)
        ned = compute_adjusted_ned(llm_value, gt_value)
        tokens_found, tokens_added = compute_tokens_found_added(llm_value, gt_value)
        anls_scores.append(anls)
        em_scores.append(em)
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
                print(f"  FP for '{key}': LLM extracted '{llm_value}', but GT is empty/null.")
            continue

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
                if gt_date_obj and llm_date_obj and gt_date_obj.date() == llm_date_obj.date():
                    is_match = True
                elif fuzz.ratio(gt_value, llm_value) == 100:
                    is_match = True
            elif key == "total":
                def clean_amount(amount_str):
                    cleaned = re.sub(r'[^\d.]', '', str(amount_str))
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
                print(f"  FP for '{key}': GT='{gt_value}', LLM='{llm_value}' (Fuzzy Ratio: {fuzz.ratio(gt_value, llm_value) if key in ['company', 'address'] else 'N/A - strict'})")
        else:
            fn += 1
            sample_log["metrics"][key]["kie_result"] = "FN"
            print(f"  FN for '{key}': GT='{gt_value}', LLM extracted empty/null.")

    # Compute KIEval metrics
    kieval_metrics = compute_kieval_metrics(normalized_llm_output, ground_truth_json, entity_keys)
    sample_log["metrics"]["kieval"] = kieval_metrics

    return tp, fp, fn, anls_scores, em_scores, ned_scores, tokens_found_total, tokens_added_total, sample_log

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
        log_file.write("")

    print(f"Total documents to process: {len(all_doc_ids)}")

    for i, doc_id in enumerate(all_doc_ids):
        print(f"\nProcessing document {i+1}/{len(all_doc_ids)}: {doc_id}.jpg")
        start_time = time.time()

        img_path, ground_truth_json = load_sroie_document(doc_id)
        if img_path is None:
            continue

        llm_output_json = call_gemini_vlm(img_path, prompt, GEMINI_MODEL)
        print(f"LLM output - {llm_output_json}")

        if llm_output_json is None:
            print(f"Skipping {doc_id} due to invalid/no LLM output from Gemini.")
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
    print(f"--- {GEMINI_MODEL} Overall Evaluation Results for SROIE 2019 ---")
    print("="*50)
    if processed_docs > 0:
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_anls = sum(total_anls) / len(total_anls) if total_anls else 0.0
        avg_em = sum(total_em) / len(total_em) if total_em else 0.0
        avg_ned = sum(total_ned) / len(total_ned) if total_ned else 0.0
        avg_time_per_sample = total_time / processed_docs if processed_docs > 0 else 0.0

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
        print(f"KIEval Entity F1: {entity_f1_total / processed_docs:.4f}")
        print(f"KIEval Group F1: {group_f1_total / processed_docs:.4f}")
        print(f"KIEval Aligned Score: {aligned_score_total / processed_docs:.4f}")
        print(f"KIEval Total Correction Cost: {correction_cost_total}")
        print(f"Average Time per Sample: {avg_time_per_sample:.2f}s")
    else:
        print("No documents were successfully processed for evaluation.")
    print("="*50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(evaluate_sroie(num_samples=400))