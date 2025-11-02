import os
import json
import base64
from io import BytesIO
from PIL import Image
import requests
import numpy as np
from datasets import load_dataset
from thefuzz import fuzz
import Levenshtein
import warnings
import time
import re
import sys
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import logging

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="PIL")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- Configuration ---
GOOGLE_API_KEY = ""
GEMINI_MODEL_NAME = "gemini-2.5-flash"
MAX_SAMPLES_TO_PROCESS = 100
LOG_FILE = "new_results/gemini_docvqa_evaluation_log.jsonl"

genai.configure(api_key=GOOGLE_API_KEY)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_answer(s):
    """Lowercases, removes punctuation and articles, and extra whitespace."""
    if not isinstance(s, str):
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

# --- Prompt for Gemini ---
def create_gemini_prompt_for_docvqa(question: str):
    prompt_parts = []
    prompt_parts.append(f"""
    Analyze the provided document image.
    Answer the following question concisely, based *only* on the information present in the image.
    Provide only the answer text, without any additional explanations, greetings, or formatting.
    Please do not add phrases, just need that word or number asked in the question. No phrases like "this is the result", "it will be held in", "the full form is this or that" etc.
    Only give the answer. Do not write "Answer:" in the response.

    Question: {question}
    """)
    return "\n".join(prompt_parts).strip()

# --- Call Gemini ---
def call_gemini_vlm(image_pil: Image.Image, prompt_text: str, model_name=GEMINI_MODEL_NAME, max_retries=3, initial_delay=5):
    model = genai.GenerativeModel(model_name)
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                [prompt_text, image_pil],
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=512
                ),
                safety_settings=safety_settings
            )
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Gemini API call blocked by prompt feedback: {response.prompt_feedback.block_reason}", file=sys.stderr)
                return None
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                return part.text.strip()
            print(f"Gemini API call returned no text content.", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed for Gemini API call: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                sleep_time = initial_delay * (2 ** attempt)
                print(f"Retrying in {sleep_time:.1f} seconds...", file=sys.stderr)
                time.sleep(sleep_time)
            else:
                print(f"Max retries reached for Gemini API. Skipping.", file=sys.stderr)
                return None
    return None

# --- Metrics Calculation ---
def compute_anls(pred_str, gt_answers, threshold=0.5):
    normalized_pred = normalize_answer(pred_str)
    normalized_gts = [normalize_answer(gt) for gt in gt_answers if gt]
    if not normalized_pred or not normalized_gts:
        return 0.0
    max_similarity = 0.0
    for gt in normalized_gts:
        distance = Levenshtein.distance(normalized_pred, gt)
        max_len = max(len(normalized_pred), len(gt))
        similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
        max_similarity = max(max_similarity, similarity)
    return max_similarity if max_similarity >= threshold else 0.0

def compute_exact_match(pred_str, gt_answers):
    normalized_pred = normalize_answer(pred_str)
    normalized_gts = [normalize_answer(gt) for gt in gt_answers if gt]
    return 1.0 if normalized_pred in normalized_gts and normalized_pred != "" else 0.0

def compute_adjusted_ned(pred_str, gt_answers):
    normalized_pred = normalize_answer(pred_str)
    normalized_gts = [normalize_answer(gt) for gt in gt_answers if gt]
    if not normalized_pred or not normalized_gts:
        return 1.0 if normalized_pred != normalized_gts else 0.0
    min_distance = float('inf')
    for gt in normalized_gts:
        distance = Levenshtein.distance(normalized_pred, gt)
        min_distance = min(min_distance, distance)
    max_len = max(len(normalized_pred), max(len(gt) for gt in normalized_gts))
    return min_distance / max_len if max_len > 0 else 0.0

def compute_tokens_found_added(pred_str, gt_answers):
    normalized_pred = normalize_answer(pred_str)
    normalized_gts = [normalize_answer(gt) for gt in gt_answers if gt]
    pred_tokens = normalized_pred.split()
    max_common_tokens = 0
    for gt in normalized_gts:
        gt_tokens = gt.split()
        common_tokens = set(pred_tokens) & set(gt_tokens)
        if len(common_tokens) > max_common_tokens:
            max_common_tokens = len(common_tokens)
    tokens_found = max_common_tokens
    tokens_added = len(pred_tokens) - max_common_tokens
    return tokens_found, tokens_added

def compute_kieval_metrics(pred_str, gt_answers):
    normalized_pred = normalize_answer(pred_str)
    normalized_gts = [normalize_answer(gt) for gt in gt_answers if gt]
    
    entity_matches = 0
    entity_total_pred = 1 if normalized_pred else 0
    entity_total_gt = 1 if normalized_gts else 0
    correction_cost = 0

    if normalized_pred and normalized_gts:
        max_score = max(fuzz.ratio(normalized_pred, gt) for gt in normalized_gts)
        if max_score >= 80:
            entity_matches = 1
        else:
            correction_cost = 1
    elif normalized_pred and not normalized_gts:
        correction_cost = 1
    elif not normalized_pred and normalized_gts:
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

def calculate_vqa_metrics(predicted_answer: str, ground_truth_answers: list[str]):
    anls = compute_anls(predicted_answer, ground_truth_answers)
    em = compute_exact_match(predicted_answer, ground_truth_answers)
    ned = compute_adjusted_ned(predicted_answer, ground_truth_answers)
    tokens_found, tokens_added = compute_tokens_found_added(predicted_answer, ground_truth_answers)
    kieval = compute_kieval_metrics(predicted_answer, ground_truth_answers)

    # Original F1
    normalized_pred = normalize_answer(predicted_answer)
    normalized_gts = [normalize_answer(gt) for gt in ground_truth_answers if gt]
    f1_scores = []
    pred_tokens = normalized_pred.split()
    for gt in normalized_gts:
        gt_tokens = gt.split()
        common_tokens = set(pred_tokens) & set(gt_tokens)
        num_common = sum(min(pred_tokens.count(token), gt_tokens.count(token)) for token in common_tokens)
        if len(pred_tokens) == 0 or len(gt_tokens) == 0 or num_common == 0:
            f1 = 0.0
        else:
            precision = num_common / len(pred_tokens)
            recall = num_common / len(gt_tokens)
            f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
    max_f1 = max(f1_scores) if f1_scores else 0.0

    return {
        "em": em,
        "als": anls,
        "f1": max_f1,
        "ned": ned,
        "tokens_found": tokens_found,
        "tokens_added": tokens_added,
        "kieval": kieval
    }

# --- Main Evaluation Function ---
def evaluate_gemini_on_docvqa():
    print("--- Starting Gemini evaluation for DocVQA ---")
    print(f"Gemini Model: {GEMINI_MODEL_NAME}")

    # Load dataset
    print(f"Loading dataset: nielsr/docvqa_1200_examples...")
    test_dataset = load_dataset("nielsr/docvqa_1200_examples", split="test")
    
    if MAX_SAMPLES_TO_PROCESS is not None:
        test_dataset = test_dataset.shuffle(seed=84)
        test_dataset = test_dataset.select(range(min(len(test_dataset), MAX_SAMPLES_TO_PROCESS)))
    
    print(f"Dataset loaded. Test size: {len(test_dataset)}")

    total_em = 0.0
    total_als = 0.0
    total_f1 = 0.0
    total_ned = 0.0
    total_tokens_found = 0
    total_tokens_added = 0
    total_entity_f1 = 0.0
    total_group_f1 = 0.0
    total_aligned_score = 0.0
    total_correction_cost = 0
    total_time = 0.0
    processed_samples_count = 0

    # Initialize JSONL log file
    with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
        log_file.write("")

    for i, example in enumerate(test_dataset):
        print(f"\n--- Processing Sample {i+1}/{len(test_dataset)} (Original index: {i}) ---")
        start_time = time.time()

        question = example["query"].get("en", "") if isinstance(example["query"], dict) else example["query"]
        ground_truth_answers = example["answers"]
        image = example["image"]

        prompt = create_gemini_prompt_for_docvqa(question)
        
        print(f"Calling Gemini VLM for sample {i+1}...")
        predicted_answer = call_gemini_vlm(image, prompt)

        if predicted_answer is not None:
            processed_samples_count += 1
            metrics = calculate_vqa_metrics(predicted_answer, ground_truth_answers)
            
            sample_time = time.time() - start_time
            total_time += sample_time

            total_em += metrics["em"]
            total_als += metrics["als"]
            total_f1 += metrics["f1"]
            total_ned += metrics["ned"]
            total_tokens_found += metrics["tokens_found"]
            total_tokens_added += metrics["tokens_added"]
            total_entity_f1 += metrics["kieval"]["entity_f1"]
            total_group_f1 += metrics["kieval"]["group_f1"]
            total_aligned_score += metrics["kieval"]["aligned_score"]
            total_correction_cost += metrics["kieval"]["correction_cost"]

            # Log sample results
            sample_log = {
                "sample_id": i,
                "question": question,
                "predicted_answer": predicted_answer,
                "ground_truth_answers": ground_truth_answers,
                "metrics": {
                    "em": metrics["em"],
                    "anls": metrics["als"],
                    "f1": metrics["f1"],
                    "adjusted_ned": metrics["ned"],
                    "tokens_found": metrics["tokens_found"],
                    "tokens_added": metrics["tokens_added"],
                    "kieval": metrics["kieval"]
                },
                "processing_time": sample_time
            }
            with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
                json.dump(sample_log, log_file, ensure_ascii=False)
                log_file.write("\n")

            print(f"Question: {question}")
            print(f"Ground Truth Answers: {ground_truth_answers}")
            print(f"Predicted Answer: '{predicted_answer}'")
            print(f"Sample {i+1} Metrics: EM={metrics['em']:.2f}, ANLS={metrics['als']:.2f}, F1={metrics['f1']:.2f}, "
                  f"NED={metrics['ned']:.2f}, TokensFound={metrics['tokens_found']}, TokensAdded={metrics['tokens_added']}, "
                  f"Time={sample_time:.2f}s")
        else:
            print(f"Gemini call failed or returned no output for sample {i+1}. Skipping.", file=sys.stderr)

    print(f"\n--- Evaluation Summary for {GEMINI_MODEL_NAME} on DocVQA ---")
    print("="*60)
    if processed_samples_count > 0:
        overall_em = total_em / processed_samples_count
        overall_als = total_als / processed_samples_count
        overall_f1 = total_f1 / processed_samples_count
        overall_ned = total_ned / processed_samples_count
        avg_time = total_time / processed_samples_count
        print(f"Overall Exact Match (EM): {overall_em:.4f}")
        print(f"Overall ANLS: {overall_als:.4f}")
        print(f"Overall F1 Score: {overall_f1:.4f}")
        print(f"Overall Adjusted NED (SCORE): {overall_ned:.4f}")
        print(f"Total Tokens Found (SCORE): {total_tokens_found}")
        print(f"Total Tokens Added (SCORE): {total_tokens_added}")
        print(f"KIEval Entity F1: {total_entity_f1 / processed_samples_count:.4f}")
        print(f"KIEval Group F1: {total_group_f1 / processed_samples_count:.4f}")
        print(f"KIEval Aligned Score: {total_aligned_score / processed_samples_count:.4f}")
        print(f"KIEval Total Correction Cost: {total_correction_cost}")
        print(f"Average Time per Sample: {avg_time:.2f}s")
        print(f"Total samples processed: {processed_samples_count}")
    else:
        print("No samples were successfully processed for evaluation.")
    print("="*60)

# --- Run the script ---
if __name__ == "__main__":
    evaluate_gemini_on_docvqa()