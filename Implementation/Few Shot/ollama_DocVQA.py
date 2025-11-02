import os
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
from sklearn.cluster import DBSCAN
from collections import defaultdict

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="PIL")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Configuration
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL_NAME = "qwen2.5vl:latest"
MAX_SAMPLES_TO_PROCESS = 10
EPSILON = 0.05  # DBSCAN distance threshold (normalized)
MIN_SAMPLES = 2  # Minimum words per block

def normalize_answer(s):
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

def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Create semantic blocks using DBSCAN clustering
def create_semantic_blocks(words: list, boxes: list, image_width: int, image_height: int):
    if not words or not boxes or len(words) != len(boxes):
        print("Warning: Empty or mismatched words/boxes.", file=sys.stderr)
        return []
    
    # Normalize bounding box centers to [0,1] based on image dimensions
    centers = []
    for box in boxes:
        if len(box) != 8:
            continue
        x_center = (box[0] + box[2] + box[4] + box[6]) / 4 / image_width
        y_center = (box[1] + box[3] + box[5] + box[7]) / 4 / image_height
        centers.append([x_center, y_center])
    
    if not centers:
        return []
    
    # Cluster using DBSCAN
    clustering = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES, metric="euclidean").fit(centers)
    labels = clustering.labels_
    
    # Group words by cluster
    blocks = defaultdict(list)
    for word, label in zip(words, labels):
        if label != -1:  # Ignore noise points
            blocks[label].append(word)
    
    # Sort blocks by average y-coordinate for reading order
    block_coords = {}
    for label in blocks:
        indices = [i for i, l in enumerate(labels) if l == label]
        y_coords = [centers[i][1] for i in indices]
        block_coords[label] = sum(y_coords) / len(y_coords)
    
    sorted_blocks = []
    for label in sorted(block_coords, key=block_coords.get):
        sorted_blocks.append(" ".join(blocks[label]))
    
    return sorted_blocks

# BLOCKIE-inspired prompt with few-shot examples
def create_ollama_prompt_for_docvqa(question: str, words: list = None, boxes: list = None, image_size: tuple = None, few_shot_examples: list = None):
    # Build semantic blocks for the CURRENT (target) image
    blocks = create_semantic_blocks(words, boxes, image_size[0], image_size[1]) if words and boxes and image_size else []
    block_prompt = "\n".join([f"Block {i+1}: {block}" for i, block in enumerate(blocks)]) if blocks else "No blocks extracted."

    # Few-shot section: each example corresponds to one image in the same order they are sent
    few_shot_prompt = ""
    if few_shot_examples:
        few_shot_prompt = ["You will receive multiple images. The FIRST N images are FEW-SHOT EXAMPLES. The FINAL image is the TARGET document to answer about."]
        few_shot_prompt.append("For each example, we provide semantic text blocks and the correct answer. Learn the mapping from blocks to answers.")
        few_shot_prompt.append("")
        for i, ex in enumerate(few_shot_examples):
            ex_question = ex["question"]
            ex_answer = ex["answer"]
            ex_blocks = create_semantic_blocks(
                ex.get("words", []), ex.get("boxes", []), ex.get("image_width", 1), ex.get("image_height", 1)
            ) if ex.get("words") and ex.get("boxes") else []
            ex_block_prompt = "\n".join([f"Block {j+1}: {block}" for j, block in enumerate(ex_blocks)]) if ex_blocks else "No blocks extracted."
            few_shot_prompt.append(
                f"Example {i+1} — (corresponds to Image {i+1})\n"
                f"Blocks:\n{ex_block_prompt}\n"
                f"Question: {ex_question}\n"
                f"Answer: {ex_answer}\n"
            )
        few_shot_prompt = "\n".join(few_shot_prompt)

    prompt = f"""
{few_shot_prompt}

Now, consider the FINAL image (target document). Use ONLY the blocks and that last image to answer.

Target document semantic blocks:
{block_prompt}

Task: Answer the question with a single phrase/value exactly as it appears in the document. Do not add extra words.
Question: {question}
Answer:
""".strip()
    return prompt

# Fetch few-shot examples
def get_few_shot_examples(n_examples: int = 3, seed: int = 84):
    dataset = load_dataset("nielsr/docvqa_1200_examples", split="train")
    dataset = dataset.shuffle(seed=seed).select(range(min(n_examples, len(dataset))))
    examples = []
    for ex in dataset:
        question = ex["query"].get("en", "") if isinstance(ex["query"], dict) else ex["query"]
        answer = ex["answers"][0] if ex["answers"] else ""
        image = ex["image"]
        image_base64 = pil_to_base64(image)
        words = ex.get("words", [])
        boxes = ex.get("boxes", [])
        examples.append({
            "question": question,
            "answer": answer,
            "image_base64": image_base64,
            "words": words,
            "boxes": boxes,
            "image_width": image.width,
            "image_height": image.height,
        })
    return examples

async def call_ollama_vlm(image_base64: str, prompt: str, few_shot_images: list = None, max_retries=3, initial_delay=5):
    url = f"{OLLAMA_HOST}/api/generate"
    headers = {"Content-Type": "application/json"}
    images = []
    # Place few-shot images first, then the current (target) image last.
    if few_shot_images:
        images.extend(few_shot_images)
    images.append(image_base64)
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "images": images,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 512
        }
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            full_response = response.json()
            if 'response' in full_response:
                return full_response['response'].strip()
            else:
                print(f"Unexpected Ollama API response: {full_response}", file=sys.stderr)
                return None
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}", file=sys.stderr)
            if attempt < max_retries - 1:
                sleep_time = initial_delay * (2 ** attempt)
                print(f"Retrying in {sleep_time:.1f} seconds...", file=sys.stderr)
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Skipping.", file=sys.stderr)
                return None
    return None

def calculate_vqa_metrics(predicted_answer: str, ground_truth_answers: list[str]):
    normalized_pred = normalize_answer(predicted_answer)
    ground_truth_answers = ground_truth_answers if ground_truth_answers else []
    normalized_gts = [normalize_answer(gt) for gt in ground_truth_answers]
    
    em_score = 1.0 if normalized_pred in normalized_gts and normalized_pred != "" else 0.0
    
    max_als = 0.0
    if normalized_pred and normalized_gts:
        for gt in normalized_gts:
            lev_dist = Levenshtein.distance(normalized_pred, gt)
            current_als = 1.0 - (lev_dist / max(len(normalized_pred), len(gt)))
            max_als = max(max_als, current_als)
    
    f1_scores = []
    pred_tokens = normalized_pred.split()
    for gt in normalized_gts:
        gt_tokens = gt.split()
        common = set(pred_tokens) & set(gt_tokens)
        num_common = sum(min(pred_tokens.count(t), gt_tokens.count(t)) for t in common)
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            f1 = 0.0
        else:
            precision = num_common / len(pred_tokens)
            recall = num_common / len(gt_tokens)
            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        f1_scores.append(f1)
    max_f1 = max(f1_scores) if f1_scores else 0.0
    
    return {"em": em_score, "als": max_als, "f1": max_f1}

async def evaluate_ollama_on_docvqa(use_few_shot=True):
    print("--- Starting Ollama evaluation for DocVQA ---")
    print(f"Ollama Model: {OLLAMA_MODEL_NAME}")
    
    few_shot_examples = get_few_shot_examples(n_examples=3) if use_few_shot else None
    few_shot_images = [ex["image_base64"] for ex in few_shot_examples] if few_shot_examples else None
    
    test_dataset = load_dataset("nielsr/docvqa_1200_examples", split="test")
    if MAX_SAMPLES_TO_PROCESS:
        test_dataset = test_dataset.shuffle(seed=84).select(range(min(len(test_dataset), MAX_SAMPLES_TO_PROCESS)))
    
    total_em = total_als = total_f1 = 0.0
    processed_count = 0
    
    for i, example in enumerate(test_dataset):
        print(f"\n--- Processing Sample {i+1}/{len(test_dataset)} ---")
        question = example["query"].get("en", "") if isinstance(example["query"], dict) else example["query"]
        ground_truth_answers = example["answers"]
        image = example["image"]
        image_base64 = pil_to_base64(image)
        words = example.get("words", [])
        boxes = example.get("boxes", [])
        
        prompt = create_ollama_prompt_for_docvqa(question, words, boxes, (image.width, image.height), few_shot_examples)
        
        print(f"Calling Ollama VLM for sample {i+1}...")
        predicted_answer = await call_ollama_vlm(image_base64, prompt, few_shot_images)
        
        if predicted_answer:
            processed_count += 1
            metrics = calculate_vqa_metrics(predicted_answer, ground_truth_answers)
            total_em += metrics["em"]
            total_als += metrics["als"]
            total_f1 += metrics["f1"]
            print(f"Question: {question}")
            print(f"Ground Truth Answers: {ground_truth_answers}")
            print(f"Predicted Answer: '{predicted_answer}'")
            print(f"Sample {i+1} Metrics: EM={metrics['em']:.4f}, ALS={metrics['als']:.4f}, F1={metrics['f1']:.4f}")
        else:
            print(f"Ollama call failed for sample {i+1}.", file=sys.stderr)
    
    if processed_count > 0:
        print(f"\n--- Evaluation Summary for {OLLAMA_MODEL_NAME} ---")
        print(f"Overall EM: {total_em / processed_count:.4f}")
        print(f"Overall ALS: {total_als / processed_count:.4f}")
        print(f"Overall F1: {total_f1 / processed_count:.4f}")
        print(f"Total samples processed: {processed_count}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(evaluate_ollama_on_docvqa(use_few_shot=True))