import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from datasets import load_dataset
from thefuzz import fuzz
import Levenshtein
import warnings
import time
import re
import sys
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from sklearn.cluster import DBSCAN

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="PIL")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_SAMPLES_TO_PROCESS = 10  # Match your output
LORA_RANK = 64  # From BLOCKIE
EPOCHS = 3
EPSILON = 0.05  # DBSCAN distance threshold (normalized)
MIN_SAMPLES = 2  # Minimum words per block
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
USE_QUANTIZATION = False  # Set to True if bitsandbytes is compatible

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

# Create semantic blocks using DBSCAN
def create_semantic_blocks(words: list, boxes: list, image_width: int, image_height: int):
    if not words or not boxes or len(words) != len(boxes):
        print("Warning: Empty or mismatched words/boxes.", file=sys.stderr)
        return []
    
    centers = []
    for box in boxes:
        if len(box) != 8:
            continue
        x_center = (box[0] + box[2] + box[4] + box[6]) / 4 / image_width
        y_center = (box[1] + box[3] + box[5] + box[7]) / 4 / image_height
        centers.append([x_center, y_center])
    
    if not centers:
        return []
    
    clustering = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES, metric="euclidean").fit(centers)
    labels = clustering.labels_
    
    blocks = defaultdict(list)
    for word, label in zip(words, labels):
        if label != -1:
            blocks[label].append(word)
    
    block_coords = {}
    for label in blocks:
        indices = [i for i, l in enumerate(labels) if l == label]
        y_coords = [centers[i][1] for i in indices]
        block_coords[label] = sum(y_coords) / len(y_coords)
    
    sorted_blocks = []
    for label in sorted(block_coords, key=block_coords.get):
        sorted_blocks.append(" ".join(blocks[label]))
    
    return sorted_blocks

# BLOCKIE-inspired prompt
def create_prompt_for_docvqa(question: str, words: list = None, boxes: list = None, image_size: tuple = None):
    blocks = create_semantic_blocks(words, boxes, image_size[0], image_size[1]) if words and boxes and image_size else []
    block_prompt = "\n".join([f"Block {i+1}: {block}" for i, block in enumerate(blocks)]) if blocks else "No blocks extracted."
    
    prompt = f"""
    Analyze the document by segmenting into semantic blocks:
    {block_prompt}
    
    Using ONLY the blocks and the document image, answer the question with a single phrase or value matching the document content exactly. Do not include additional context or explanations.
    
    Question: {question}
    
    Answer:
    """
    return prompt.strip()

async def call_vlm(image: Image.Image, prompt: str, model, processor):
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    return processor.decode(outputs[0], skip_special_tokens=True).strip()

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

def fine_tune_model():
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    quantization_config = None
    if USE_QUANTIZATION:
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
        except Exception as e:
            print(f"Quantization failed: {e}. Falling back to FP16.", file=sys.stderr)
            quantization_config = None
    
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=DEVICE,
        quantization_config=quantization_config
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05
    )
    model = get_peft_model(model, lora_config)
    
    # Load DocVQA train dataset
    dataset = load_dataset("nielsr/docvqa_1200_examples", split="train")
    dataset = dataset.shuffle(seed=84).select(range(500))  # Subset for demo
    
    def preprocess(example):
        question = example["query"].get("en", "") if isinstance(example["query"], dict) else example["query"]
        answers = example["answers"][0] if example["answers"] else ""
        words = example.get("words", [])
        boxes = example.get("boxes", [])
        prompt = create_prompt_for_docvqa(question, words, boxes, (example["image"].width, example["image"].height))
        inputs = processor(text=prompt + " " + answers, images=example["image"], return_tensors="pt", padding="max_length", max_length=512)
        for k, v in inputs.items():
            inputs[k] = v.to(DEVICE)
        return inputs
    
    train_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    
    training_args = TrainingArguments(
        output_dir="./fine_tuned_qwen",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=1,  # Further reduced for M1 Pro
        gradient_accumulation_steps=2,  # To simulate batch size of 2
        learning_rate=2e-5,
        fp16=True if DEVICE == "mps" else False,
        save_steps=100,
        logging_steps=10,
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()
    model.save_pretrained("./fine_tuned_qwen")
    processor.save_pretrained("./fine_tuned_qwen")
    return model, processor

async def evaluate_on_docvqa(fine_tune=False):
    print("--- Starting evaluation on DocVQA ---")
    print(f"Device: {DEVICE}")
    if fine_tune:
        model, processor = fine_tune_model()
    else:
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map=DEVICE
        )
    
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
        words = example.get("words", [])
        boxes = example.get("boxes", [])
        
        prompt = create_prompt_for_docvqa(question, words, boxes, (image.width, image.height))
        
        print(f"Calling VLM for sample {i+1}...")
        predicted_answer = await call_vlm(image, prompt, model, processor)
        
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
    
    if processed_count > 0:
        print(f"\n--- Evaluation Summary for {MODEL_NAME} ---")
        print(f"Overall EM: {total_em / processed_count:.4f}")
        print(f"Overall ALS: {total_als / processed_count:.4f}")
        print(f"Overall F1: {total_f1 / processed_count:.4f}")
        print(f"Total samples processed: {processed_count}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(evaluate_on_docvqa(fine_tune=True))