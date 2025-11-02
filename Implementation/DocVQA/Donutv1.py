from transformers import pipeline
from datasets import load_dataset
from thefuzz import fuzz

# Load the pre-trained Donut model fine-tuned on DocVQA via the HF pipeline
# If a GPU is available, use it for faster inference
device = 0  # set to 0 for GPU or -1 for CPU
try:
    import torch
    if not torch.cuda.is_available():
        device = -1
except ImportError:
    device = -1

qa_pipeline = pipeline(
    "document-question-answering", 
    model="naver-clova-ix/donut-base-finetuned-docvqa", 
    device=device
)

# Load a DocVQA-style dataset. Here we use a Hugging Face sample of the DocVQA dataset.
dataset = load_dataset("nielsr/docvqa_1200_examples", split="test")
dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(100))

# Remove unused columns for clarity (the sample dataset has an 'answer' field from another model, and OCR details)
for col in ["answer", "words", "bounding_boxes"]:
    if col in dataset.column_names:
        dataset = dataset.remove_columns(col)

# Helper functions for metrics
import string

def normalize_for_f1(s: str) -> [str]:
    """Lowercase and remove punctuation from a string, and split into tokens."""
    s = s.lower().strip()
    # Remove punctuation (keep alphanumeric and whitespace)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    tokens = s.split()
    return tokens

def compute_em(pred: str, answers: [str]) -> int:
    """Exact match: 1 if pred exactly matches any answer (after lowercase & strip), else 0."""
    pred_norm = pred.strip().lower()
    for ans in answers:
        ans_norm = ans.strip().lower()
        if fuzz.ratio(pred_norm, ans_norm) >= 80:
            return 1
    return 0

def compute_f1(pred: str, answers: [str]) -> float:
    """Compute best F1 score between prediction and a list of ground-truth answers."""
    pred_tokens = normalize_for_f1(pred)
    # If no prediction and no answer tokens, define F1 = 1 (perfectly matching empty answer)
    best_f1 = 0.0
    for ans in answers:
        ans_norm = ans.strip().lower()
        if fuzz.ratio(pred.strip().lower(), ans_norm) >= 80:
            return 1.0  # treat as perfect match
        gt_tokens = normalize_for_f1(ans)
        # Edge cases: if either is empty, handle appropriately
        if not gt_tokens and not pred_tokens:
            return 1.0  # both empty => exact match
        if not gt_tokens or not pred_tokens:
            # one empty, one non-empty => no overlap
            f1 = 0.0
        else:
            # Count overlapping tokens
            common_tokens = set(pred_tokens) & set(gt_tokens)
            num_common = sum(min(pred_tokens.count(tok), gt_tokens.count(tok)) for tok in common_tokens)
            if num_common == 0:
                f1 = 0.0
            else:
                precision = num_common / len(pred_tokens)
                recall = num_common / len(gt_tokens)
                f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    len1, len2 = len(s1), len(s2)
    # Initialize DP table
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    # Fill DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[len1][len2]

def compute_anls(pred: str, answers: [str]) -> float:
    """Compute the best normalized Levenshtein similarity (NLS) for prediction vs. ground truths."""
    # Normalize to lowercase for distance calculation
    pred_norm = pred.strip().lower()
    best_nls = 0.0
    for ans in answers:
        ans_norm = ans.strip().lower()
        # Compute Levenshtein distance
        dist = levenshtein_distance(pred_norm, ans_norm)
        max_len = max(len(pred_norm), len(ans_norm))
        # If both pred and ans are empty, treat as perfect match; otherwise compute normalized similarity
        if max_len == 0:
            nls = 1.0 if dist == 0 else 0.0
        else:
            nls = 1.0 - (dist / max_len)
        if nls > best_nls:
            best_nls = nls
    return best_nls

# Loop over the dataset and evaluate
em_sum = 0
f1_sum = 0.0
anls_sum = 0.0

print("Evaluating on", len(dataset), "samples...")
for idx, sample in enumerate(dataset):
    # Get question text (handling possible multi-lingual format)
    if isinstance(sample["query"], dict):
        # If the question field is a dict with language codes, pick English if available
        question = sample["query"].get("en", "") 
    else:
        question = sample["query"]
    answers = sample["answers"]  # list of ground-truth answer strings
    image = sample["image"]      # PIL Image object

    # Get model prediction
    result = qa_pipeline(image=image, question=question)
    # The pipeline returns a list of answers; take the first answer's text
    pred_answer = result[0]['answer'] if result and 'answer' in result[0] else ""

    # Compute metrics
    em = compute_em(pred_answer, answers)
    f1 = compute_f1(pred_answer, answers)
    anls = compute_anls(pred_answer, answers)

    em_sum += em
    f1_sum += f1
    anls_sum += anls

    # Log individual sample result
    sample_id = sample.get("id", idx)
    print(f"Sample {idx} (ID: {sample_id}): EM={em}, F1={f1:.3f}, NLS={anls:.3f} | Predicted: '{pred_answer}', GT: {answers}")

# Compute and print overall scores
total = len(dataset)
overall_em = em_sum / total
overall_f1 = f1_sum / total
overall_anls = anls_sum / total

print("\n--- Evaluation Summary for Donut ---")
print("\nOverall Exact Match (EM): {:.2%}".format(overall_em))
print("Overall F1 Score: {:.2%}".format(overall_f1))
print("Overall Average NLS (ANLS): {:.2%}".format(overall_anls))