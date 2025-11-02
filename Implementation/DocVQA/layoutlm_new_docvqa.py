from transformers import pipeline
from datasets import load_dataset
from rapidfuzz import fuzz
import string
import time
import Levenshtein
import logging
import json
import re

# Setup logging
LOG_FILE = "layoutlmv3_docvqa_evaluation_log.jsonl"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
device = 0
try:
    import torch
    if not torch.cuda.is_available():
        device = -1
except ImportError:
    device = -1

qa_pipeline = pipeline(
    "document-question-answering", 
    model="impira/layoutlm-document-qa",
    device=device
)

# Load dataset
dataset = load_dataset("nielsr/docvqa_1200_examples", split="test")
dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(100))

for col in ["answer", "words", "bounding_boxes"]:
    if col in dataset.column_names:
        dataset = dataset.remove_columns(col)

# Metrics functions
def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return " ".join([w for w in text.split() if w not in {"a", "an", "the"}])
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def compute_em(pred: str, answers: [str]) -> int:
    pred_norm = normalize_answer(pred)
    for ans in answers:
        ans_norm = normalize_answer(ans)
        if fuzz.token_set_ratio(pred_norm, ans_norm) >= 80:
            return 1
    return 0

def compute_f1(pred: str, answers: [str]) -> float:
    pred_norm = normalize_answer(pred)
    best_f1 = 0.0
    for ans in answers:
        ans_norm = normalize_answer(ans)
        if fuzz.token_set_ratio(pred_norm, ans_norm) >= 80:
            return 1.0
        else:
            best_f1 = max(best_f1, fuzz.token_set_ratio(pred_norm, ans_norm) / 100.0)
    return best_f1

def compute_anls(pred: str, answers: [str], threshold=0.5) -> float:
    pred_norm = normalize_answer(pred)
    best_nls = 0.0
    for ans in answers:
        ans_norm = normalize_answer(ans)
        dist = Levenshtein.distance(pred_norm, ans_norm)
        max_len = max(len(pred_norm), len(ans_norm))
        nls = 1.0 if max_len == 0 and dist == 0 else 1.0 - (dist / max_len) if max_len > 0 else 0.0
        if nls > best_nls:
            best_nls = nls
    return best_nls if best_nls >= threshold else 0.0

def compute_adjusted_ned(pred: str, answers: [str]):
    pred_norm = normalize_answer(pred)
    min_distance = float('inf')
    for ans in answers:
        ans_norm = normalize_answer(ans)
        distance = Levenshtein.distance(pred_norm, ans_norm)
        min_distance = min(min_distance, distance)
    max_len = max(len(pred_norm), max(len(normalize_answer(ans)) for ans in answers))
    return min_distance / max_len if max_len > 0 else 0.0

def compute_tokens_found_added(pred: str, answers: [str]):
    pred_norm = normalize_answer(pred)
    pred_tokens = pred_norm.split()
    max_common_tokens = 0
    for ans in answers:
        ans_norm = normalize_answer(ans)
        gt_tokens = ans_norm.split()
        common_tokens = set(pred_tokens) & set(gt_tokens)
        if len(common_tokens) > max_common_tokens:
            max_common_tokens = len(common_tokens)
    tokens_found = max_common_tokens
    tokens_added = len(pred_tokens) - max_common_tokens
    return tokens_found, tokens_added

def compute_kieval_metrics(pred: str, answers: [str]):
    pred_norm = normalize_answer(pred)
    normalized_gts = [normalize_answer(ans) for ans in answers]
    
    entity_matches = 0
    entity_total_pred = 1 if pred_norm else 0
    entity_total_gt = 1 if normalized_gts else 0
    correction_cost = 0

    if pred_norm and normalized_gts:
        max_score = max(fuzz.ratio(pred_norm, gt) for gt in normalized_gts)
        if max_score >= 80:
            entity_matches = 1
        else:
            correction_cost = 1
    elif pred_norm and not normalized_gts:
        correction_cost = 1
    elif not pred_norm and normalized_gts:
        correction_cost = 1

    precision = entity_matches / entity_total_pred if entity_total_pred > 0 else 0.0
    recall = entity_matches / entity_total_gt if entity_total_gt > 0 else 0.0
    entity_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    group_match = 1 if entity_matches == 1 else 0
    group_precision = group_match / 1 if entity_total_pred > 0 else 0.0
    group_recall = group_match / 1 if entity_total_gt > 0 else 0.0
    group_f1 = (2 * group_precision * group_recall) / (group_precision + group_recall) if (group_precision + group_recall) > 0 else 0.0

    aligned_score = 1.0 - (correction_cost / max(entity_total_pred, entity_total_gt)) if max(entity_total_pred, entity_total_gt) > 0 else 0.0

    return {
        "entity_f1": entity_f1,
        "group_f1": group_f1,
        "aligned_score": aligned_score,
        "correction_cost": correction_cost
    }

# Evaluation loop
em_sum = 0
f1_sum = 0.0
anls_sum = 0.0
ned_sum = 0.0
tokens_found_total = 0
tokens_added_total = 0
entity_f1_sum = 0.0
group_f1_sum = 0.0
aligned_score_sum = 0.0
correction_cost_total = 0
total_time = 0.0
total = len(dataset)

print("Evaluating on", total, "samples...")

# Initialize JSONL log file
with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
    log_file.write("")

for idx, sample in enumerate(dataset):
    start_time = time.time()
    question = sample["query"].get("en", "") if isinstance(sample["query"], dict) else sample["query"]
    answers = sample["answers"]
    image = sample["image"]

    result = qa_pipeline(image=image, question=question)
    pred_answer = result[0]['answer'] if result and 'answer' in result[0] else ""

    em = compute_em(pred_answer, answers)
    f1 = compute_f1(pred_answer, answers)
    anls = compute_anls(pred_answer, answers)
    ned = compute_adjusted_ned(pred_answer, answers)
    tokens_found, tokens_added = compute_tokens_found_added(pred_answer, answers)
    kieval = compute_kieval_metrics(pred_answer, answers)

    sample_time = time.time() - start_time
    total_time += sample_time

    em_sum += em
    f1_sum += f1
    anls_sum += anls
    ned_sum += ned
    tokens_found_total += tokens_found
    tokens_added_total += tokens_added
    entity_f1_sum += kieval["entity_f1"]
    group_f1_sum += kieval["group_f1"]
    aligned_score_sum += kieval["aligned_score"]
    correction_cost_total += kieval["correction_cost"]

    # Log sample
    sample_id = sample.get("id", idx)
    sample_log = {
        "sample_id": sample_id,
        "question": question,
        "predicted_answer": pred_answer,
        "ground_truth_answers": answers,
        "metrics": {
            "em": em,
            "f1": f1,
            "anls": anls,
            "adjusted_ned": ned,
            "tokens_found": tokens_found,
            "tokens_added": tokens_added,
            "kieval": kieval
        },
        "processing_time": sample_time
    }
    with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
        json.dump(sample_log, log_file, ensure_ascii=False)
        log_file.write("\n")

    print(f"Sample {idx} (ID: {sample_id}): EM={em}, F1={f1:.3f}, ANLS={anls:.3f}, NED={ned:.3f}, "
          f"TokensFound={tokens_found}, TokensAdded={tokens_added}, Time={sample_time:.2f}s | "
          f"Predicted: '{pred_answer}', GT: {answers}")

# Overall scores
overall_em = em_sum / total
overall_f1 = f1_sum / total
overall_anls = anls_sum / total
overall_ned = ned_sum / total
avg_time = total_time / total

print("\n--- Evaluation Summary for LayoutLMv3 ---")
print("="*60)
print(f"Overall Exact Match (EM): {overall_em:.2%}")
print(f"Overall F1 Score: {overall_f1:.2%}")
print(f"Overall ANLS: {overall_anls:.2%}")
print(f"Overall Adjusted NED (SCORE): {overall_ned:.4f}")
print(f"Total Tokens Found (SCORE): {tokens_found_total}")
print(f"Total Tokens Added (SCORE): {tokens_added_total}")
print(f"KIEval Entity F1: {entity_f1_sum / total:.4f}")
print(f"KIEval Group F1: {group_f1_sum / total:.4f}")
print(f"KIEval Aligned Score: {aligned_score_sum / total:.4f}")
print(f"KIEval Total Correction Cost: {correction_cost_total}")
print(f"Average Time per Sample: {avg_time:.2f}s")
print("="*60)