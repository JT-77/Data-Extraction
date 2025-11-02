import os, re, json, glob
import torch
from PIL import Image
from difflib import SequenceMatcher
from transformers import DonutProcessor, VisionEncoderDecoderModel
import random
import time
import Levenshtein
import logging
from thefuzz import fuzz
from datetime import datetime

# Setup logging
LOG_FILE = "donut_sroie_evaluation_log.jsonl"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the fine-tuned Donut model and processor for SROIE
model_name = "sam749/donut-base-finetuned-sroie-v2"
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Prepare generation settings
task_prompt = "<s>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
max_length = model.decoder.config.max_position_embeddings
pad_token_id = processor.tokenizer.pad_token_id
eos_token_id = processor.tokenizer.eos_token_id
bad_words_ids = [[processor.tokenizer.unk_token_id]]

# Define data directories
img_dir = "../datasets/SROIE2019/test/img"
entity_dir = "../datasets/SROIE2019/test/entities"

# Collect image file paths
image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    image_paths.extend(glob.glob(os.path.join(img_dir, ext)))


image_paths = sorted(image_paths)
import random

random.seed(18)  # for reproducibility
random.shuffle(image_paths)

image_paths = image_paths[:]

if not image_paths:
    print(f"No images found in {img_dir}. Please check the path.")
    exit()

# Fields to evaluate
target_fields = ["company", "date", "address", "total"]

# Evaluation Metrics Functions
def compute_anls(pred_str, gt_str, threshold=0.5):
    if not pred_str or not gt_str:
        return 0.0
    distance = Levenshtein.distance(pred_str.lower(), gt_str.lower())
    max_len = max(len(pred_str), len(gt_str))
    similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
    return similarity if similarity >= threshold else 0.0

def compute_exact_match(pred_str, gt_str):
    return 1.0 if pred_str.lower() == gt_str.lower() else 0.0

def compute_adjusted_ned(pred_str, gt_str):
    if not pred_str or not gt_str:
        return 1.0 if pred_str != gt_str else 0.0
    distance = Levenshtein.distance(pred_str.lower(), gt_str.lower())
    max_len = max(len(pred_str), len(gt_str))
    return distance / max_len if max_len > 0 else 0.0

def compute_tokens_found_added(pred_str, gt_str):
    pred_tokens = pred_str.lower().split()
    gt_tokens = gt_str.lower().split()
    common_tokens = set(pred_tokens) & set(gt_tokens)
    tokens_found = len(common_tokens)
    tokens_added = len(pred_tokens) - len(common_tokens)
    return tokens_found, tokens_added

def compute_kieval_metrics(pred_dict, gt_dict, entity_keys):
    entity_matches = 0
    entity_total_pred = 0
    entity_total_gt = 0
    correction_cost = 0

    for key in entity_keys:
        gt_value = gt_dict.get(key, "")
        pred_value = pred_dict.get(key, "")
        
        # Normalize for comparison (same as original F1)
        gt_norm = re.sub(r"\s+", " ", str(gt_value)).strip().lower()
        pred_norm = re.sub(r"\s+", " ", str(pred_value)).strip().lower()
        
        entity_total_gt += 1 if gt_value else 0
        entity_total_pred += 1 if pred_value else 0
        
        if gt_value and pred_value:
            if key in ["company", "address"]:
                score = fuzz.ratio(gt_norm, pred_norm)
                if score >= 60:  # Match original threshold
                    entity_matches += 1
                else:
                    correction_cost += 1
            elif key in ["date", "total"]:
                if gt_norm == pred_norm:  # Exact string match after normalization
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

# Initialize counters
sum_precision = sum_recall = sum_f1 = 0.0
sum_anls = sum_em = sum_ned = sum_tokens_found = sum_tokens_added = 0.0
sum_entity_f1 = sum_group_f1 = sum_aligned_score = sum_correction_cost = 0.0
total_time = 0.0
num_samples = 0

# Initialize JSONL log file
with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
    log_file.write("")

for img_path in image_paths:
    start_time = time.time()
    file_name = os.path.basename(img_path)
    base_name = os.path.splitext(file_name)[0]
    gt_path = os.path.join(entity_dir, base_name + ".txt")
    if not os.path.exists(gt_path):
        gt_path = os.path.join(entity_dir, base_name + ".json")
    if not os.path.exists(gt_path):
        print(f"Ground truth for `{file_name}` not found, skipping.")
        continue

    # Load ground truth
    with open(gt_path, 'r') as f:
        content = f.read().strip()
    try:
        gt_data = json.loads(content)
    except json.JSONDecodeError:
        gt_data = {}
        if content.startswith("{") and content.endswith("}"):
            fixed = content.replace("'", "\"")
            try:
                gt_data = json.loads(fixed)
            except Exception:
                gt_data = {}
        else:
            for line in content.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    gt_data[key.strip().lower()] = val.strip()
    gt_data = {k.lower(): v for k, v in gt_data.items()}

    # Load and preprocess image
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Run model
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bad_words_ids=bad_words_ids,
            use_cache=True
        )
    sequence = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    try:
        pred_data = processor.token2json(sequence)
    except Exception:
        try:
            pred_data = json.loads(sequence)
        except Exception:
            pred_data = {}
    pred_data = {k.lower(): v for k, v in pred_data.items()}

    # Compute metrics
    correct_fields = 0
    predicted_fields = 0
    anls_scores = []
    em_scores = []
    ned_scores = []
    tokens_found_total = 0
    tokens_added_total = 0
    sample_log = {
        "file_name": file_name,
        "pred_data": pred_data,
        "gt_data": gt_data,
        "metrics": {}
    }

    for field in target_fields:
        pred_val = str(pred_data.get(field, ""))
        actual_val = str(gt_data.get(field, ""))
        pv_norm = re.sub(r"\s+", " ", pred_val).strip().lower()
        av_norm = re.sub(r"\s+", " ", actual_val).strip().lower()

        # Existing F1 logic
        if field in ["company", "address"]:
            score = SequenceMatcher(None, pv_norm, av_norm).ratio() * 100
            match = score >= 60
        else:
            score = 100.0 if pv_norm == av_norm else 0.0
            match = pv_norm == av_norm
        if pred_val != "":
            predicted_fields += 1
        if match:
            correct_fields += 1

        # New metrics
        anls = compute_anls(pred_val, actual_val)
        em = compute_exact_match(pred_val, actual_val)
        ned = compute_adjusted_ned(pred_val, actual_val)
        tokens_found, tokens_added = compute_tokens_found_added(pred_val, actual_val)
        anls_scores.append(anls)
        em_scores.append(em)
        ned_scores.append(ned)
        tokens_found_total += tokens_found
        tokens_added_total += tokens_added

        sample_log["metrics"][field] = {
            "anls": anls,
            "exact_match": em,
            "adjusted_ned": ned,
            "tokens_found": tokens_found,
            "tokens_added": tokens_added,
            "match": match,
            "score": score
        }

        match_text = "Yes" if match else "No"
        # print(f"{file_name} - {field}: Predicted=\"{pred_val}\" vs Actual=\"{actual_val}\" -> Match: {match_text} (score={score:.1f}%)")

    # KIEval metrics
    kieval_metrics = compute_kieval_metrics(pred_data, gt_data, target_fields)
    sample_log["metrics"]["kieval"] = kieval_metrics

    # Compute sample time
    sample_time = time.time() - start_time
    sample_log["processing_time"] = sample_time

    # Log sample results
    with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
        json.dump(sample_log, log_file, ensure_ascii=False)
        log_file.write("\n")

    # Update aggregates
    precision = correct_fields / predicted_fields if predicted_fields > 0 else 0.0
    recall = correct_fields / len(target_fields) if len(target_fields) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    sum_precision += precision
    sum_recall += recall
    sum_f1 += f1_score
    sum_anls += sum(anls_scores)
    sum_em += sum(em_scores)
    sum_ned += sum(ned_scores)
    sum_tokens_found += tokens_found_total
    sum_tokens_added += tokens_added_total
    sum_entity_f1 += kieval_metrics["entity_f1"]
    sum_group_f1 += kieval_metrics["group_f1"]
    sum_aligned_score += kieval_metrics["aligned_score"]
    sum_correction_cost += kieval_metrics["correction_cost"]
    total_time += sample_time
    num_samples += 1

    print(f"{file_name} - Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1-score: {f1_score*100:.2f}%, "
          f"ANLS: {sum(anls_scores)/len(anls_scores):.4f}, EM: {sum(em_scores)/len(em_scores):.4f}, "
          f"NED: {sum(ned_scores)/len(em_scores):.4f}, Time: {sample_time:.2f}s")

# Compute and print average metrics
if num_samples > 0:
    avg_precision = sum_precision / num_samples
    avg_recall = sum_recall / num_samples
    avg_f1 = sum_f1 / num_samples
    avg_anls = sum_anls / (num_samples * len(target_fields))
    avg_em = sum_em / (num_samples * len(target_fields))
    avg_ned = sum_ned / (num_samples * len(target_fields))
    avg_time = total_time / num_samples

    print(f"\nAverage Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average F1-score: {avg_f1:.2f}")
    print(f"Average ANLS: {avg_anls:.4f}")
    print(f"Average Exact Match: {avg_em:.4f}")
    print(f"Average Adjusted NED (SCORE): {avg_ned:.4f}")
    print(f"Total Tokens Found (SCORE): {sum_tokens_found}")
    print(f"Total Tokens Added (SCORE): {sum_tokens_added}")
    print(f"KIEval Entity F1: {sum_entity_f1 / num_samples:.4f}")
    print(f"KIEval Group F1: {sum_group_f1 / num_samples:.4f}")
    print(f"KIEval Aligned Score: {sum_aligned_score / num_samples:.4f}")
    print(f"KIEval Total Correction Cost: {sum_correction_cost}")
    print(f"Average Time per Sample: {avg_time:.2f}s")