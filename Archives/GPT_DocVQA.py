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

# Import OpenAI library
from openai import OpenAI

# Suppress specific future warnings from PIL and other libraries
warnings.filterwarnings("ignore", category=FutureWarning, module="PIL")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- Configuration ---
# Your OpenAI API Key will be provided by the Canvas runtime
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # Replace with your actual API key or set via environment variable
# Using GPT-4o model for document understanding
GPT_MODEL_NAME = "gpt-4o-mini" 
MAX_SAMPLES_TO_PROCESS = 50

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

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

# --- Helper to convert PIL Image to base64 for OpenAI ---
def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    # Use JPEG for efficiency and compatibility with most VLMs
    image.save(buffered, format="JPEG", quality=85) 
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Prompt for GPT-4o (for DocVQA) ---
def create_gpt_prompt_for_docvqa(question: str):
    """
    Creates a concise prompt for GPT-4o to answer a question based on an image.
    """
    prompt_parts = []
    prompt_parts.append(f"""
    Analyze the provided document image.
    Answer the following question concisely, based *only* on the information present in the image.
    Provide only the answer text, without any additional explanations, greetings, or formatting.
    Please do not add phrases, just need that word or number asked in the question. No phrases like "this is the result", "it will be held in", "the full form is this or that" etc.
    Only give the answer. Do not write "Answer:" in the response.

    Question: {question}
    """)
    
    final_prompt = "\n".join(prompt_parts).strip()
    return final_prompt

# --- Function to call GPT-4o VLM (Synchronous) ---
def call_gpt_vlm(image_pil: Image.Image, prompt_text: str, model_name=GPT_MODEL_NAME, max_retries=3, initial_delay=5):
    """
    Calls the GPT-4o Vision model with an image (PIL.Image) and a text prompt.
    Returns the raw text response from the model or None on error.
    Includes retry logic and improved error handling.
    """
    base64_image = pil_to_base64(image_pil)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                        ],
                    }
                ],
                max_tokens=512, # Limiting output length for concise answers
                temperature=0.1, # Keep temperature low for factual answers
            )
            
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            else:
                print(f"GPT-4o API call returned no content or unexpected structure: {response}", file=sys.stderr)
                return None

        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for GPT-4o API call: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                sleep_time = initial_delay * (2 ** attempt)
                print(f"Retrying in {sleep_time:.1f} seconds...", file=sys.stderr)
                time.sleep(sleep_time)
            else:
                print(f"Max retries reached for GPT-4o API. Skipping.", file=sys.stderr)
                return None
    return None # Should not be reached if max_retries is handled correctly


# --- VQA Metric Calculation (ALS and EM) ---
def calculate_vqa_metrics(predicted_answer: str, ground_truth_answers: list[str]):
    """
    Calculates Exact Match (EM) and Average Levenshtein Similarity (ALS) for a single QA pair.
    Handles cases where ground_truth_answers might be None or empty.
    """
    normalized_pred = normalize_answer(predicted_answer)
    
    # Ensure ground_truth_answers is an iterable list
    ground_truth_answers = ground_truth_answers if ground_truth_answers is not None else []
    normalized_gts = [normalize_answer(gt) for gt in ground_truth_answers]

    # Exact Match (EM)
    em_score = 1.0 if normalized_pred in normalized_gts and normalized_pred != "" else 0.0

    # Average Levenshtein Similarity (ALS)
    max_als_score = 0.0
    if not normalized_pred and not normalized_gts: # Both empty, perfect match
        max_als_score = 1.0
    elif not normalized_pred or not normalized_gts: # One is empty, one isn't
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

    # F1 Score Calculation (token-level)
    f1_scores = []
    pred_tokens = normalized_pred.split()

    for gt in normalized_gts:
        gt_tokens = gt.split()
        common_tokens = set(pred_tokens) & set(gt_tokens)
        num_common = sum(min(pred_tokens.count(token), gt_tokens.count(token)) for token in common_tokens)

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
def evaluate_gpt_on_docvqa():
    print("--- Starting GPT-4o evaluation for DocVQA ---")
    print(f"GPT-4o Model: {GPT_MODEL_NAME}")

    # 1. Load Dataset
    print(f"Loading dataset: nielsr/docvqa_1200_examples...")
    test_dataset = load_dataset("nielsr/docvqa_1200_examples", split="test")
    
    # Limit dataset size for demonstration if MAX_SAMPLES_TO_PROCESS is set
    if MAX_SAMPLES_TO_PROCESS is not None:
        test_dataset = test_dataset.shuffle(seed=84)  # Use a fixed seed for reproducibility
        test_dataset = test_dataset.select(range(min(len(test_dataset), MAX_SAMPLES_TO_PROCESS)))
    
    print(f"Dataset loaded. Test size: {len(test_dataset)}")

    total_em = 0.0
    total_als = 0.0
    total_f1 = 0.0
    processed_samples_count = 0

    for i, example in enumerate(test_dataset):
        print(f"\n--- Processing Sample {i+1}/{len(test_dataset)} (Original index: {i}) ---")

        question = example["query"].get("en", "") if isinstance(example["query"], dict) else example["query"]
        ground_truth_answers = example["answers"]
        image = example["image"]

        prompt = create_gpt_prompt_for_docvqa(question)
        
        print(f"Calling GPT-4o VLM for sample {i+1}...")
        predicted_answer = call_gpt_vlm(image, prompt)

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
            print(f"GPT-4o call failed or returned no output for sample {i+1}. Skipping VQA evaluation for this sample.", file=sys.stderr)

    print(f"\n--- Evaluation Summary for {GPT_MODEL_NAME} ---")
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
    evaluate_gpt_on_docvqa()
