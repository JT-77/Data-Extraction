import re
import json
import torch
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
import time
import logging
import Levenshtein
from thefuzz import fuzz

# Setup logging
LOG_FILE = "donut_cord_evaluation_log.jsonl"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define labels that require exact string match (after normalization)
EXACT_MATCH_LABELS = frozenset({
    "menu.cnt",
    "menu.price",
    "menu.num", 
    "sub_total.discount_price", 
    "total.total_price", 
    "menu.itemsubtotal" 
})

# Load dataset
dataset = load_dataset("naver-clova-ix/cord-v2", split="test")
dataset = dataset.shuffle(seed=84)
dataset = dataset.select(range(50))

# Initialize Donut processor and model
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Prepare the decoder prompt for CORD-v2
task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

# Function to flatten a nested dictionary of predictions/annotations into key-value pairs
def flatten_dict(d, parent_key=""):
    pairs = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            pairs.extend(flatten_dict(v, parent_key=new_key))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    pairs.extend(flatten_dict(item, parent_key=new_key))
                else:
                    pairs.append((new_key, str(item)))
        else:
            pairs.append((new_key, str(v)))
    return pairs

# Normalization function (adapted from your code)
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

    # Non-numeric normalization (your rules)
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

# Metrics functions (adapted for flattened pairs, treating as entities)
def calculate_kie_metrics(gt_pairs, pred_pairs, fuzz_threshold=70):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_predicted_pairs = []
    false_negatives_list = []
    false_positives_list = []

    gt_consumed = [False] * len(gt_pairs)
    pred_consumed = [False] * len(pred_pairs)

    for i, gt_pair in enumerate(gt_pairs):
        if gt_consumed[i]: continue
        gt_label = gt_pair[0]
        gt_text = normalize_text_for_matching(gt_pair[1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS))
        current_fuzz_threshold = 80 if gt_label in EXACT_MATCH_LABELS else 75 if gt_label == "menu.nm" else fuzz_threshold
        fuzzy_match_func = fuzz.ratio if gt_label in EXACT_MATCH_LABELS else fuzz.token_set_ratio

        best_match_idx = -1
        best_match_score = -1
        for j, pred_pair in enumerate(pred_pairs):
            if pred_consumed[j] or gt_label != pred_pair[0]:
                continue
            pred_text = normalize_text_for_matching(pred_pair[1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS))
            score = fuzzy_match_func(gt_text, pred_text)
            if score > best_match_score:
                best_match_score = score
                best_match_idx = j
        
        if best_match_idx != -1:
            if best_match_score >= current_fuzz_threshold:
                true_positives += 1
                gt_consumed[i] = True
                pred_consumed[best_match_idx] = True
                matched_predicted_pairs.append(pred_pairs[best_match_idx])
            else:
                false_negatives += 1
                false_negatives_list.append(gt_pair)
                gt_consumed[i] = True
                pred_consumed[best_match_idx] = True
        else:
            false_negatives += 1
            false_negatives_list.append(gt_pair)
    
    for i, pred_pair in enumerate(pred_pairs):
        if not pred_consumed[i] and pred_pair[0] != "menu.cnt":
            false_positives += 1
            false_positives_list.append(pred_pair)

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
        "matched_predicted_entities": matched_predicted_pairs,
        "false_negatives_list": false_negatives_list, 
        "false_positives_list": false_positives_list  
    }

def compute_anls(gt_pairs, pred_pairs, threshold=0.5):
    total_anls = 0.0
    count = 0
    gt_consumed = [False] * len(gt_pairs)
    pred_consumed = [False] * len(pred_pairs)

    for i, gt_pair in enumerate(gt_pairs):
        gt_label = gt_pair[0]
        gt_text = normalize_text_for_matching(gt_pair[1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS))
        current_fuzz_threshold = 80 if gt_label in EXACT_MATCH_LABELS else 70 if gt_label == "menu.nm" else 80
        fuzzy_match_func = fuzz.ratio if gt_label in EXACT_MATCH_LABELS else fuzz.token_set_ratio

        best_match_idx = -1
        best_match_score = -1
        for j, pred_pair in enumerate(pred_pairs):
            if pred_consumed[j] or gt_label != pred_pair[0]:
                continue
            pred_text = normalize_text_for_matching(pred_pair[1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS))
            score = fuzzy_match_func(gt_text, pred_text)
            if score > best_match_score:
                best_match_score = score
                best_match_idx = j

        if best_match_idx != -1 and best_match_score >= current_fuzz_threshold:
            distance = Levenshtein.distance(gt_text, normalize_text_for_matching(pred_pairs[best_match_idx][1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS)))
            max_len = max(len(gt_text), len(normalize_text_for_matching(pred_pairs[best_match_idx][1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS))))
            similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
            total_anls += similarity if similarity >= threshold else 0.0
            count += 1
            gt_consumed[i] = True
            pred_consumed[best_match_idx] = True
        else:
            total_anls += 0.0
            count += 1

    return total_anls / count if count > 0 else 0.0

def compute_exact_match(gt_pairs, pred_pairs):
    total_em = 0
    gt_consumed = [False] * len(gt_pairs)
    pred_consumed = [False] * len(pred_pairs)

    for i, gt_pair in enumerate(gt_pairs):
        gt_label = gt_pair[0]
        gt_text = normalize_text_for_matching(gt_pair[1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS))
        for j, pred_pair in enumerate(pred_pairs):
            if pred_consumed[j] or gt_label != pred_pair[0]:
                continue
            pred_text = normalize_text_for_matching(pred_pair[1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS))
            if gt_text == pred_text:
                total_em += 1
                gt_consumed[i] = True
                pred_consumed[j] = True
                break
    return total_em / len(gt_pairs) if gt_pairs else 0.0

def compute_adjusted_ned(gt_pairs, pred_pairs):
    total_ned = 0.0
    count = 0
    gt_consumed = [False] * len(gt_pairs)
    pred_consumed = [False] * len(pred_pairs)

    for i, gt_pair in enumerate(gt_pairs):
        gt_label = gt_pair[0]
        gt_text = normalize_text_for_matching(gt_pair[1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS))
        min_distance = float('inf')
        best_match_idx = -1
        for j, pred_pair in enumerate(pred_pairs):
            if pred_consumed[j] or gt_label != pred_pair[0]:
                continue
            pred_text = normalize_text_for_matching(pred_pair[1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS))
            distance = Levenshtein.distance(gt_text, pred_text)
            if distance < min_distance:
                min_distance = distance
                best_match_idx = j
        if best_match_idx != -1:
            max_len = max(len(gt_text), len(normalize_text_for_matching(pred_pairs[best_match_idx][1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS))))
            total_ned += min_distance / max_len if max_len > 0 else 0.0
            count += 1
            gt_consumed[i] = True
            pred_consumed[best_match_idx] = True
        else:
            total_ned += 1.0
            count += 1

    return total_ned / count if count > 0 else 0.0

def compute_tokens_found_added(gt_pairs, pred_pairs):
    total_tokens_found = 0
    total_tokens_added = 0
    gt_consumed = [False] * len(gt_pairs)
    pred_consumed = [False] * len(pred_pairs)

    for i, gt_pair in enumerate(gt_pairs):
        gt_label = gt_pair[0]
        gt_text = normalize_text_for_matching(gt_pair[1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS))
        gt_tokens = gt_text.split()
        best_match_idx = -1
        max_common_tokens = 0
        for j, pred_pair in enumerate(pred_pairs):
            if pred_consumed[j] or gt_label != pred_pair[0]:
                continue
            pred_text = normalize_text_for_matching(pred_pair[1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS))
            pred_tokens = pred_text.split()
            common_tokens = set(pred_tokens) & set(gt_tokens)
            if len(common_tokens) > max_common_tokens:
                max_common_tokens = len(common_tokens)
                best_match_idx = j
        if best_match_idx != -1:
            pred_text = normalize_text_for_matching(pred_pairs[best_match_idx][1], is_numeric_field=(gt_label in EXACT_MATCH_LABELS))
            pred_tokens = pred_text.split()
            total_tokens_found += max_common_tokens
            total_tokens_added += len(pred_tokens) - max_common_tokens
            gt_consumed[i] = True
            pred_consumed[best_match_idx] = True

    for j, pred_pair in enumerate(pred_pairs):
        if not pred_consumed[j] and pred_pair[0] != "menu.cnt":
            pred_text = normalize_text_for_matching(pred_pair[1], is_numeric_field=(pred_pair[0] in EXACT_MATCH_LABELS))
            pred_tokens = pred_text.split()
            total_tokens_added += len(pred_tokens)

    return total_tokens_found, total_tokens_added

def compute_kieval_metrics(gt_pairs, pred_pairs):
    kie_metrics = calculate_kie_metrics(gt_pairs, pred_pairs)
    entity_f1 = kie_metrics["f1"]
    true_positives = kie_metrics["true_positives"]
    false_positives = kie_metrics["false_positives"]
    false_negatives = kie_metrics["false_negatives"]

    group_match = 1 if false_negatives == 0 and false_positives == 0 else 0
    group_precision = group_match / 1 if (true_positives + false_positives) > 0 else 0.0
    group_recall = group_match / 1 if (true_positives + false_negatives) > 0 else 0.0
    group_f1 = (2 * group_precision * group_recall) / (group_precision + group_recall) if (group_precision + group_recall) > 0 else 0.0

    correction_cost = false_negatives + false_positives
    aligned_score = 1.0 - (correction_cost / max(len(gt_pairs), len(pred_pairs))) if max(len(gt_pairs), len(pred_pairs)) > 0 else 0.0

    return {
        "entity_f1": entity_f1,
        "group_f1": group_f1,
        "aligned_score": aligned_score,
        "correction_cost": correction_cost,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

def calculate_all_metrics(gt_pairs, pred_pairs):
    anls = compute_anls(gt_pairs, pred_pairs)
    em = compute_exact_match(gt_pairs, pred_pairs)
    ned = compute_adjusted_ned(gt_pairs, pred_pairs)
    tokens_found, tokens_added = compute_tokens_found_added(gt_pairs, pred_pairs)
    kieval = compute_kieval_metrics(gt_pairs, pred_pairs)

    return {
        "anls": anls,
        "em": em,
        "ned": ned,
        "tokens_found": tokens_found,
        "tokens_added": tokens_added,
        "kieval": kieval
    }

# --- Evaluation Loop ---
overall_correct = 0
overall_predicted = 0
overall_actual = 0
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

print("Per-sample precision, recall, F1:")
print("Evaluating on", len(dataset), "samples...")

# Initialize JSONL log file
with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
    log_file.write("")

for idx in range(len(dataset)):
    start_time = time.time()
    example = dataset[idx]
    image = example["image"]
    gt_data = json.loads(example["ground_truth"])
    gt_fields = gt_data.get("gt_parse", gt_data)
    
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        use_cache=True
    )
    seq = processor.batch_decode(outputs)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()
    pred_fields = processor.token2json(seq)
    
    gt_pairs = flatten_dict(gt_fields)
    pred_pairs = flatten_dict(pred_fields)
    
    num_gt = len(gt_pairs)
    num_pred = len(pred_pairs)
    gt_pair_counts = {}
    for pair in gt_pairs:
        gt_pair_counts[pair] = gt_pair_counts.get(pair, 0) + 1
    correct = 0
    gt_count_copy = gt_pair_counts.copy()
    for pair in pred_pairs:
        if pair in gt_count_copy and gt_count_copy[pair] > 0:
            correct += 1
            gt_count_copy[pair] -= 1
    
    precision = correct / num_pred if num_pred > 0 else 0.0
    recall = correct / num_gt if num_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    processed_samples_count += 1
    sample_metrics = calculate_all_metrics(gt_pairs, pred_pairs)
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
    
    overall_correct += correct
    overall_predicted += num_pred
    overall_actual += num_gt

    # Log sample
    sample_log = {
        "sample_id": idx,
        "ground_truth_pairs": gt_pairs,
        "predicted_pairs": pred_pairs,
        "metrics": {
            "anls": sample_metrics["anls"],
            "em": sample_metrics["em"],
            "adjusted_ned": sample_metrics["ned"],
            "tokens_found": sample_metrics["tokens_found"],
            "tokens_added": sample_metrics["tokens_added"],
            "kieval": sample_metrics["kieval"]
        },
        "processing_time": sample_time
    }
    with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
        json.dump(sample_log, log_file, ensure_ascii=False)
        log_file.write("\n")

    print(f"Sample {idx:3d}: Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}, "
          f"ANLS={sample_metrics['anls']:.3f}, EM={sample_metrics['em']:.3f}, NED={sample_metrics['ned']:.3f}, "
          f"TokensFound={sample_metrics['tokens_found']}, TokensAdded={sample_metrics['tokens_added']}, "
          f"EntityF1={sample_metrics['kieval']['entity_f1']:.3f}, GroupF1={sample_metrics['kieval']['group_f1']:.3f}, "
          f"AlignedScore={sample_metrics['kieval']['aligned_score']:.3f}, CorrectionCost={sample_metrics['kieval']['correction_cost']}, "
          f"Time={sample_time:.2f}s")

# Compute aggregate metrics
overall_precision = overall_correct / overall_predicted if overall_predicted > 0 else 0.0
overall_recall = overall_correct / overall_actual if overall_actual > 0 else 0.0
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
avg_time = total_time / len(dataset) if len(dataset) > 0 else 0.0

print("\nOverall performance on 10 receipts:")
print("="*60)
print(f"Precision = {overall_precision:.3f}, Recall = {overall_recall:.3f}, F1-score = {overall_f1:.3f}")
print(f"Overall ANLS: {total_anls / len(dataset):.4f}" if len(dataset) > 0 else "No samples processed.")
print(f"Overall Exact Match (EM): {total_em / len(dataset):.4f}" if len(dataset) > 0 else "No samples processed.")
print(f"Overall Adjusted NED (SCORE): {total_ned / len(dataset):.4f}" if len(dataset) > 0 else "No samples processed.")
print(f"Total Tokens Found (SCORE): {total_tokens_found}")
print(f"Total Tokens Added (SCORE): {total_tokens_added}")
print(f"KIEval Entity F1: {total_entity_f1 / len(dataset):.4f}" if len(dataset) > 0 else "No samples processed.")
print(f"KIEval Group F1: {total_group_f1 / len(dataset):.4f}" if len(dataset) > 0 else "No samples processed.")
print(f"KIEval Aligned Score: {total_aligned_score / len(dataset):.4f}" if len(dataset) > 0 else "No samples processed.")
print(f"KIEval Total Correction Cost: {total_correction_cost}")
print(f"Average Time per Sample: {avg_time:.2f}s")
print("="*60)