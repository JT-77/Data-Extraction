import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_dataset

# Load the fine-tuned Donut model and processor for DocVQA
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load 20 samples from the DocVQA test set using nielsr/docvqa_1200_examples
dataset = load_dataset("nielsr/docvqa_1200_examples", split="test")
samples = dataset.select(range(20))  # select first 20 samples normally

# Initialize counters for metrics
total = len(samples)
exact_matches = 0
f1_sum = 0.0
anls_sum = 0.0

for idx, sample in enumerate(samples):
    image = sample["image"]                        # PIL image of the document page
    question = sample.get("question", "")
    if isinstance(question, dict):
        question = question.get("text", "")
    gt_answer = sample.get("answer", "")
    if isinstance(gt_answer, dict):
        gt_answer = gt_answer.get("text", "")
    elif isinstance(gt_answer, list) and len(gt_answer) > 0:
        gt_answer = gt_answer[0]

    # Prepare the input prompt with question (Donut uses special tokens for QA)
    prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    inputs = processor(image, prompt, return_tensors="pt").to(device)
    
    # Generate answer (auto-regressive decoding using the Donut model)
    outputs = model.generate(**inputs, max_length=512)
    decoded = processor.decode(outputs[0], skip_special_tokens=True)
    # Post-process the decoded sequence to isolate the answer text:
    # (The decoded text may include the question before the answer; remove it if present)
    pred_answer = decoded
    if question in pred_answer:
        # Remove the question part from the decoded text
        idx_q = pred_answer.index(question) + len(question)
        pred_answer = pred_answer[idx_q:].strip()
    pred_answer = pred_answer.strip()
    
    # Normalize case for evaluation (answers are case-insensitive in DocVQA [oai_citation:8‡docvqa.org](https://www.docvqa.org/challenges/challenge-2020#:~:text=Answers%20are%20not%20case%20sensitive))
    pred_norm = pred_answer.lower()
    gt_norm = gt_answer.lower()
    
    # ANLS metric (Average Normalized Levenshtein Similarity)
    # Compute Levenshtein distance between pred_norm and gt_norm:
    if len(gt_norm) == 0 or len(pred_norm) == 0:
        nls = 1.0 if gt_norm == pred_norm else 0.0
    else:
        m, n = len(pred_norm), len(gt_norm)
        dist = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dist[i][0] = i
        for j in range(n + 1):
            dist[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if pred_norm[i-1] == gt_norm[j-1] else 1
                dist[i][j] = min(dist[i-1][j] + 1,      # deletion
                                  dist[i][j-1] + 1,      # insertion
                                  dist[i-1][j-1] + cost) # substitution
        edit_distance = dist[m][n]
        max_len = max(m, n)
        nls = 1 - edit_distance / max_len   # normalized Levenshtein similarity [oai_citation:9‡docsaid.org](https://docsaid.org/en/blog/impl-normalized-levenshtein-similarity/#:~:text=Normalized%20Levenshtein%20Similarity%20,Levenshtein%20distance%20between%20the%20two)
    
    pred_tokens = pred_norm.split()
    gt_tokens = gt_norm.split()
    common_tokens = set(pred_tokens) & set(gt_tokens)
    if len(common_tokens) == 0:
        f1 = 0.0
    else:
        prec = len(common_tokens) / len(pred_tokens)
        rec  = len(common_tokens) / len(gt_tokens)
        f1 = 2 * (prec * rec) / (prec + rec)
    
    print(f"Sample {idx}:")
    print(f"  Question: {question}")
    print(f"  Ground Truth: {gt_answer}")
    print(f"  Prediction: {pred_answer}")
    print(f"  EM: {int(pred_norm == gt_norm)}, F1: {f1:.3f}, ANLS: {nls:.3f}\n")
    
    # Exact Match (EM) metric
    if pred_norm == gt_norm:
        exact_matches += 1
    
    f1_sum += f1
    anls_sum += nls

# Calculate average metrics
em_score  = exact_matches / total
f1_score  = f1_sum / total
anls_score = anls_sum / total

print(f"Exact Match (EM): {em_score:.3f}")
print(f"Average F1 score: {f1_score:.3f}")
print(f"Average ANLS: {anls_score:.3f}")