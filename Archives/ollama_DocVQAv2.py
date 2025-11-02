import os
import json
import base64
from io import BytesIO
from PIL import Image
import requests
import numpy as np
from datasets import load_dataset
from thefuzz import fuzz # Not directly used for VQA metrics, but might be useful for other string comparisons
import Levenshtein # For Levenshtein distance for ALS
import warnings
import time
import re
import sys 

# Suppress specific future warnings from PIL and other libraries
warnings.filterwarnings("ignore", category=FutureWarning, module="PIL")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- Configuration ---
OLLAMA_HOST = "http://localhost:11434"
# OLLAMA_MODEL_NAME = "mistral-small3.2:latest"
# OLLAMA_MODEL_NAME = "qwen2.5vl:latest"
OLLAMA_MODEL_NAME = "llama3.2-vision:latest"
MAX_SAMPLES_TO_PROCESS = 100

def normalize_answer(s):
    """Lowercases, removes punctuation and articles, and extra whitespace."""
    if not isinstance(s, str): # Ensure input is a string
        return ""
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# --- Helper to convert PIL Image to base64 for Ollama ---
def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    # Use JPEG for efficiency and compatibility with most VLMs
    image.save(buffered, format="JPEG", quality=85) 
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Prompt for Ollama (for DocVQA) ---
def create_ollama_prompt_for_docvqa(question: str):
    """
    Creates a concise prompt for Ollama to answer a question based on an image.
    """
    prompt_parts = []
    prompt_parts.append(f"""
    Analyze the provided document image.
    Answer the following question concisely, based *only* on the information present in the image.
    Provide only the answer text, without any additional explanations, greetings, or formatting.
    Please do not add phrases, just need that word or number asked in the question. No phrases like this is the result, it will be held in, do not say the full form is this or that etc.
    Only give the answer.

    Question: {question}

    Answer:
    """)
    
    final_prompt = "\n".join(prompt_parts).strip()
    return final_prompt

# --- Function to call Ollama using requests.post ---
async def call_ollama_vlm(image_base64: str, prompt: str, max_retries=3, initial_delay=5):
    url = f"{OLLAMA_HOST}/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": 0.1, # Keep temperature low for factual answers
            "num_predict": 512 # Limit output length for concise answers
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300) 
            response.raise_for_status()
            full_response = response.json()
            
            if 'response' in full_response:
                return full_response['response'].strip()
            elif 'error' in full_response:
                print(f"Ollama API returned an error: {full_response['error']}", file=sys.stderr)
                return None
            else:
                print(f"Unexpected Ollama API response structure: {full_response}", file=sys.stderr)
                return None

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for Ollama API call: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                sleep_time = initial_delay * (2 ** attempt)
                print(f"Retrying in {sleep_time:.1f} seconds...", file=sys.stderr)
                time.sleep(sleep_time)
            else:
                print(f"Max retries reached for Ollama API. Skipping.", file=sys.stderr)
                return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from Ollama response: {e}. Raw response: {response.text[:500]}...", file=sys.stderr)
            return None
    return None

# --- VQA Metric Calculation (ALS and EM) ---
def calculate_vqa_metrics(predicted_answer: str, ground_truth_answers: list[str]):
    """
    Calculates fuzzy Exact Match (EM), Average Levenshtein Similarity (ALS), and fuzzy token-level F1 Score.
    """
    normalized_pred = normalize_answer(predicted_answer)
    
    ground_truth_answers = ground_truth_answers if ground_truth_answers is not None else []
    normalized_gts = [normalize_answer(gt) for gt in ground_truth_answers]

    # Fuzzy Exact Match (EM) with threshold
    em_score = 0.0
    for gt in normalized_gts:
        if fuzz.ratio(normalized_pred, gt) >= 80:
            em_score = 1.0
            break

    # Average Levenshtein Similarity (ALS)
    max_als_score = 0.0
    if not normalized_pred and not normalized_gts:
        max_als_score = 1.0
    elif not normalized_pred or not normalized_gts:
        max_als_score = 0.0
    else:
        for gt in normalized_gts:
            len_pred = len(normalized_pred)
            len_gt = len(gt)
            if len_pred == 0 and len_gt == 0:
                current_als = 1.0
            elif len_pred == 0 or len_gt == 0:
                current_als = 0.0
            else:
                lev_dist = Levenshtein.distance(normalized_pred, gt)
                current_als = 1.0 - (lev_dist / max(len_pred, len_gt))
            max_als_score = max(max_als_score, current_als)

    # Fuzzy F1 Score (token-level)
    f1_scores = []
    pred_tokens = normalized_pred.split()

    for gt in normalized_gts:
        gt_tokens = gt.split()
        matched_pred = [False] * len(pred_tokens)
        matched_gt = [False] * len(gt_tokens)
        num_common = 0

        for i, pt in enumerate(pred_tokens):
            for j, gt_t in enumerate(gt_tokens):
                if not matched_gt[j] and fuzz.ratio(pt, gt_t) >= 80:
                    matched_pred[i] = True
                    matched_gt[j] = True
                    num_common += 1
                    break

        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            f1 = 0.0
        elif num_common == 0:
            f1 = 0.0
        else:
            precision = num_common / len(pred_tokens)
            recall = num_common / len(gt_tokens)
            f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)

    max_f1 = max(f1_scores) if f1_scores else 0.0

    return {"em": em_score, "als": max_als_score, "f1": max_f1}


# --- Main Evaluation Function ---
async def evaluate_ollama_on_docvqa():
    print("--- Starting Ollama evaluation for DocVQA ---")
    print(f"Ollama Model: {OLLAMA_MODEL_NAME}")

    # 1. Load Dataset
    print(f"Loading dataset: lmms-lab/DocVQA...")
    # dataset = load_dataset("lmms-lab/DocVQA", "DocVQA")
    # test_dataset = dataset["validation"]
    test_dataset = load_dataset("nielsr/docvqa_1200_examples", split="test")
    
    # Limit dataset size for demonstration if MAX_SAMPLES_TO_PROCESS is set
    if MAX_SAMPLES_TO_PROCESS is not None:
        test_dataset = test_dataset.shuffle(seed=42)  # Use a fixed seed for reproducibility
        test_dataset = test_dataset.select(range(min(len(test_dataset), MAX_SAMPLES_TO_PROCESS)))
    
    print(f"Dataset loaded. Test size: {len(test_dataset)}")

    total_em = 0.0
    total_als = 0.0
    total_f1 = 0.0
    processed_samples_count = 0

    for i, example in enumerate(test_dataset):
        print(f"\n--- Processing Sample {i+1}/{len(test_dataset)} (Original index: {i}) ---")

        # image = example["image"]
        # question = example["question"]
        # ground_truth_answers = example["answers"] # This is a list of possible answers

        question = example["query"].get("en", "") if isinstance(example["query"], dict) else example["query"]
        ground_truth_answers = example["answers"]
        image = example["image"]

        prompt = create_ollama_prompt_for_docvqa(question)
        image_base64 = pil_to_base64(image)

        print(f"Calling Ollama VLM for sample {i+1}...")
        predicted_answer = await call_ollama_vlm(image_base64, prompt)

        if predicted_answer is not None:
            processed_samples_count += 1
            metrics = calculate_vqa_metrics(predicted_answer, ground_truth_answers)
            
            total_em += metrics["em"]
            total_als += metrics["als"]
            total_f1 += metrics["f1"]

            print(f"Question: {question}")
            print(f"Ground Truth Answers: {ground_truth_answers}")
            print(f"Predicted Answer: '{predicted_answer}'")
            print(f"Sample {i+1} VQA Metrics: EM={metrics['em']:.2f}, ALS={metrics['als']:.2f}, F1={metrics['f1']:.2f}")
        else:
            print(f"Ollama call failed or returned no output for sample {i+1}. Skipping VQA evaluation for this sample.", file=sys.stderr)

    print(f"\n--- Evaluation Summary for {OLLAMA_MODEL_NAME} ---")
    if processed_samples_count > 0:
        overall_em = total_em / processed_samples_count
        overall_als = total_als / processed_samples_count
        overall_f1 = total_f1 / processed_samples_count
        print(f"Overall Exact Match (EM): {overall_em:.4f}")
        print(f"Overall Average Levenshtein Similarity (ALS): {overall_als:.4f}")
        print(f"Overall F1 Score: {overall_f1:.4f}")
        print(f"Total samples processed: {processed_samples_count}")
    else:
        print("No samples were successfully processed for evaluation.")

# --- Run the script ---
if __name__ == "__main__":
    import asyncio
    asyncio.run(evaluate_ollama_on_docvqa())