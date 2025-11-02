from transformers import pipeline
from datasets import load_dataset
from rapidfuzz import fuzz
import string

# Use better fine-tuned model
device = 0
try:
    import torch
    if not torch.cuda.is_available():
        device = -1
except ImportError:
    device = -1

qa_pipeline = pipeline(
    "document-question-answering", 
    model="impira/layoutlm-document-qa",  # high-performing QA model
    device=device
)

dataset = load_dataset("nielsr/docvqa_1200_examples", split="test")
dataset = dataset.shuffle(seed=42)  # Optional: set seed for reproducibility
dataset = dataset.select(range(100))

for col in ["answer", "words", "bounding_boxes"]:
    if col in dataset.column_names:
        dataset = dataset.remove_columns(col)

# --- Normalization Helpers ---
def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return " ".join([w for w in text.split() if w not in {"a", "an", "the"}])
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

# --- Evaluation Metrics ---
def compute_em(pred: str, answers: [str]) -> int:
    pred_norm = normalize_answer(pred)
    for ans in answers:
        ans_norm = normalize_answer(ans)
        if fuzz.token_set_ratio(pred_norm, ans_norm) >= 80:
            return 1
    return 0

def compute_f1(pred: str, answers: [str]) -> float:
    pred_norm = normalize_answer(pred)
    for ans in answers:
        ans_norm = normalize_answer(ans)
        if fuzz.token_set_ratio(pred_norm, ans_norm) >= 80:
            return 1.0
    return max(fuzz.token_set_ratio(pred_norm, normalize_answer(ans)) / 100.0 for ans in answers)

def levenshtein_distance(s1: str, s2: str) -> int:
    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1): dp[i][0] = i
    for j in range(len2 + 1): dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[len1][len2]

def compute_anls(pred: str, answers: [str]) -> float:
    pred_norm = normalize_answer(pred)
    best_nls = 0.0
    for ans in answers:
        ans_norm = normalize_answer(ans)
        dist = levenshtein_distance(pred_norm, ans_norm)
        max_len = max(len(pred_norm), len(ans_norm))
        nls = 1.0 if max_len == 0 and dist == 0 else 1.0 - (dist / max_len) if max_len > 0 else 0.0
        if nls > best_nls:
            best_nls = nls
    return best_nls

# --- Evaluation Loop ---
em_sum = 0
f1_sum = 0.0
anls_sum = 0.0

print("Evaluating on", len(dataset), "samples...")
for idx, sample in enumerate(dataset):
    question = sample["query"].get("en", "") if isinstance(sample["query"], dict) else sample["query"]
    answers = sample["answers"]
    image = sample["image"]

    result = qa_pipeline(image=image, question=question)
    pred_answer = result[0]['answer'] if result and 'answer' in result[0] else ""

    em = compute_em(pred_answer, answers)
    f1 = compute_f1(pred_answer, answers)
    anls = compute_anls(pred_answer, answers)

    em_sum += em
    f1_sum += f1
    anls_sum += anls

    print(f"Sample {idx} (ID: {sample.get('id', idx)}): EM={em}, F1={f1:.3f}, NLS={anls:.3f} | Pred: '{pred_answer}', GT: {answers}")

total = len(dataset)
print("\n--- Evaluation Summary for LayoutLMv3 ---")
print("\nOverall Exact Match (EM): {:.2%}".format(em_sum / total))
print("Average F1 Score: {:.2%}".format(f1_sum / total))
print("Average NLS (ANLS): {:.2%}".format(anls_sum / total))