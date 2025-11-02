import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import pandas as pd
from PIL import Image
import torch
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, Features, Value, ClassLabel, Array2D
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import warnings
from accelerate import Accelerator
accelerator = Accelerator()
import time
import logging
import Levenshtein
from thefuzz import fuzz

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="PIL")

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Configuration ---
MODEL_NAME = "microsoft/layoutlmv3-base"
OUTPUT_DIR = "./layoutlmv3_cord_v2_results"
LOGGING_DIR = "./layoutlmv3_cord_v2_logs"
LOG_FILE = "layoutlmv3_cord_evaluation_log.jsonl"

# Define labels that require exact string match (after normalization)
EXACT_MATCH_LABELS = frozenset({
    "menu.cnt",
    "menu.price",
    "menu.num", 
    "sub_total.discount_price", 
    "total.total_price", 
    "menu.itemsubtotal" 
})


# These will be populated dynamically from the dataset's ground_truth
MODEL_ID2LABEL = {}
MODEL_LABEL2ID = {}
ALL_BASE_LABELS = set()

# Initialize the LayoutLMv3 processor
processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False)

# --- Load Dataset ---
def load_cord_v2_dataset():
    print("Loading CORD v2 dataset from Hugging Face Hub (naver-clova-ix/cord-v2)...")
    dataset = load_dataset("naver-clova-ix/cord-v2")
    print("CORD v2 Dataset loaded successfully.")
    print(dataset)
    return dataset

# --- Helper Function to Extract Words, Bboxes, and Labels from ground_truth ---
def extract_words_bboxes_and_labels(ground_truth_json_str, image_width, image_height):
    data = json.loads(ground_truth_json_str)
    words = []
    bboxes = []
    ner_tags = []

    for line_data in data["valid_line"]:
        category = line_data["category"]
        ALL_BASE_LABELS.add(category)
        for i, word_info in enumerate(line_data["words"]):
            word_text = word_info["text"]
            quad = word_info["quad"]
            x_coords = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
            y_coords = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            normalized_bbox = [
                int(1000 * (x_min / image_width)),
                int(1000 * (y_min / image_height)),
                int(1000 * (x_max / image_width)),
                int(1000 * (y_max / image_height))
            ]
            normalized_bbox = [max(0, min(1000, coord)) for coord in normalized_bbox]
            if normalized_bbox[0] > normalized_bbox[2]: normalized_bbox[0], normalized_bbox[2] = normalized_bbox[2], normalized_bbox[0]
            if normalized_bbox[1] > normalized_bbox[3]: normalized_bbox[1], normalized_bbox[3] = normalized_bbox[3], normalized_bbox[1]
            words.append(word_text)
            bboxes.append(normalized_bbox)
            if i == 0:
                ner_tags.append(f"B-{category.upper()}")
            else:
                ner_tags.append(f"I-{category.upper()}")

    return words, bboxes, ner_tags

# --- Preprocessing Function for Dataset.map ---
def preprocess_function(examples):
    batch_images = examples["image"]
    batch_ground_truth_strs = examples["ground_truth"]

    all_words = []
    all_bboxes = []
    all_ner_tags = []

    for i in range(len(batch_images)):
        image = batch_images[i].convert("RGB")
        image_width, image_height = image.size
        ground_truth_str = batch_ground_truth_strs[i]

        words_on_page, bboxes_on_page, ner_tags_on_page_str = \
            extract_words_bboxes_and_labels(ground_truth_str, image_width, image_height)
        
        ner_tags_on_page_ids = [MODEL_LABEL2ID.get(tag, MODEL_LABEL2ID["O"]) for tag in ner_tags_on_page_str]

        all_words.append(words_on_page)
        all_bboxes.append(bboxes_on_page)
        all_ner_tags.append(ner_tags_on_page_ids)

    encoding = processor(
        batch_images, 
        all_words, 
        boxes=all_bboxes, 
        word_labels=all_ner_tags, 
        truncation=True, 
        padding="max_length"
    )

    return encoding

# --- Compute Metrics Function for Trainer ---
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(p.predictions, axis=2)
    true_predictions = [
        [MODEL_ID2LABEL[p_id] for (p_id, l_id) in zip(pred, lab) if l_id != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [MODEL_ID2LABEL[l_id] for (p_id, l_id) in zip(pred, lab) if l_id != -100]
        for pred, lab in zip(predictions, labels)
    ]
    
    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }
    # print("\nClassification Report:\n", classification_report(true_labels, true_predictions)) # Uncomment for detailed report

    return results

# --- Function to extract flattened entities from predictions ---
def extract_flattened_entities_from_predictions(words, predictions):
    flattened_entities = []
    current_label = None
    current_text = []
    for word, label in zip(words, predictions):
        if label.startswith("B-"):
            if current_label and current_text:
                flattened_entities.append({"label": current_label.lower(), "text": " ".join(current_text)})
            current_label = label[2:].lower()
            current_text = [word]
        elif label.startswith("I-") and current_label == label[2:].lower():
            current_text.append(word)
        else:
            if current_label and current_text:
                flattened_entities.append({"label": current_label, "text": " ".join(current_text)})
            current_label = None
            current_text = []
    if current_label and current_text:
        flattened_entities.append({"label": current_label, "text": " ".join(current_text)})
    return flattened_entities

# --- Function to extract flattened entities from ground truth ---
def extract_flattened_entities_from_gt(gt_parse):
    flattened_entities = []
    numeric_labels = frozenset({
        "menu.cnt", "menu.price", "menu.itemsubtotal", "menu.num",
        "sub_total.subtotal_price", "sub_total.discount_price",
        "sub_total.service_price", "sub_total.tax_price", 
        "total.total_price", "total.cashprice", "total.change_price", 
        "total.creditcardprice", "total.menuqty_cnt" 
    })

    def _add_entity(label, value):
        if label in TARGET_LABELS:
            is_numeric_category = label in numeric_labels
            normalized_value = normalize_text_for_matching(str(value), is_numeric_field=is_numeric_category)
            if normalized_value:
                flattened_entities.append({"label": label, "text": normalized_value})

    sub_total = gt_parse.get("sub_total", {})
    if isinstance(sub_total, dict):
        _add_entity("sub_total.subtotal_price", sub_total.get("subtotal_price"))
        _add_entity("sub_total.discount_price", sub_total.get("discount_price"))

    total = gt_parse.get("total", {})
    if isinstance(total, dict):
        _add_entity("total.total_price", total.get("total_price"))

    menu_items = gt_parse.get("menu", [])
    if isinstance(menu_items, dict):
        menu_items = [menu_items]
    if not isinstance(menu_items, list):
        menu_items = []

    for item in menu_items:
        if isinstance(item, dict):
            _add_entity("menu.nm", item.get("nm"))
            _add_entity("menu.cnt", item.get("cnt"))
            _add_entity("menu.price", item.get("price"))
            _add_entity("menu.num", item.get("num"))
            _add_entity("menu.itemsubtotal", item.get("itemsubtotal"))
            sub = item.get("sub", [])
            if isinstance(sub, dict):
                sub = [sub]
            for sub_item in sub:
                if isinstance(sub_item, dict):
                    _add_entity("menu.nm", sub_item.get("nm"))
                    _add_entity("menu.cnt", sub_item.get("cnt"))
                    _add_entity("menu.price", sub_item.get("price"))
                    _add_entity("menu.num", sub_item.get("num"))
                    _add_entity("menu.itemsubtotal", sub_item.get("itemsubtotal"))

    return flattened_entities

# --- KIE Metrics Calculation ---
def calculate_kie_metrics(gt_entities: list, pred_entities: list, fuzz_threshold=80):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_predicted_entities = []
    false_negatives_list = []
    false_positives_list = []

    gt_consumed = [False] * len(gt_entities)
    pred_consumed = [False] * len(pred_entities)

    for i, gt_entity in enumerate(gt_entities):
        if gt_consumed[i]: continue
        gt_label = gt_entity["label"]
        gt_text = gt_entity["text"]
        current_fuzz_threshold = 100 if gt_label in EXACT_MATCH_LABELS else (75 if gt_label == "menu.nm" else fuzz_threshold)
        fuzzy_match_func = fuzz.ratio if gt_label in EXACT_MATCH_LABELS else fuzz.token_set_ratio

        best_match_idx = -1
        best_match_score = -1
        for j, pred_entity in enumerate(pred_entities):
            if pred_consumed[j] or gt_label != pred_entity["label"]:
                continue
            score = fuzzy_match_func(gt_text, pred_entity["text"])
            if score > best_match_score:
                best_match_score = score
                best_match_idx = j
        
        if best_match_idx != -1:
            if best_match_score >= current_fuzz_threshold:
                true_positives += 1
                gt_consumed[i] = True
                pred_consumed[best_match_idx] = True
                matched_predicted_entities.append(pred_entities[best_match_idx])
            else:
                false_negatives += 1
                false_negatives_list.append(gt_entity)
                gt_consumed[i] = True
                pred_consumed[best_match_idx] = True
        else:
            false_negatives += 1
            false_negatives_list.append(gt_entity)
    
    for i, pred_entity in enumerate(pred_entities):
        if not pred_consumed[i] and pred_entity["label"] != "menu.cnt":
            false_positives += 1
            false_positives_list.append(pred_entity)

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
        "matched_predicted_entities": matched_predicted_entities,
        "false_negatives_list": false_negatives_list, 
        "false_positives_list": false_positives_list  
    }

# --- Additional Metrics ---
def compute_anls(gt_entities, pred_entities, threshold=0.5):
    total_anls = 0.0
    count = 0
    gt_consumed = [False] * len(gt_entities)
    pred_consumed = [False] * len(pred_entities)

    for i, gt_entity in enumerate(gt_entities):
        gt_label = gt_entity["label"]
        gt_text = gt_entity["text"]
        current_fuzz_threshold = 100 if gt_label in EXACT_MATCH_LABELS else 75 if gt_label == "menu.nm" else 80
        fuzzy_match_func = fuzz.ratio if gt_label in EXACT_MATCH_LABELS else fuzz.token_set_ratio

        best_match_idx = -1
        best_match_score = -1
        for j, pred_entity in enumerate(pred_entities):
            if pred_consumed[j] or gt_label != pred_entity["label"]:
                continue
            score = fuzzy_match_func(gt_text, pred_entity["text"])
            if score > best_match_score:
                best_match_score = score
                best_match_idx = j

        if best_match_idx != -1 and best_match_score >= current_fuzz_threshold:
            distance = Levenshtein.distance(gt_text, pred_entities[best_match_idx]["text"])
            max_len = max(len(gt_text), len(pred_entities[best_match_idx]["text"]))
            similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
            total_anls += similarity if similarity >= threshold else 0.0
            count += 1
            gt_consumed[i] = True
            pred_consumed[best_match_idx] = True
        else:
            total_anls += 0.0
            count += 1

    return total_anls / count if count > 0 else 0.0

def compute_exact_match(gt_entities, pred_entities):
    total_em = 0
    gt_consumed = [False] * len(gt_entities)
    pred_consumed = [False] * len(pred_entities)

    for i, gt_entity in enumerate(gt_entities):
        gt_label = gt_entity["label"]
        gt_text = gt_entity["text"]
        for j, pred_entity in enumerate(pred_entities):
            if pred_consumed[j] or gt_label != pred_entity["label"]:
                continue
            if gt_text == pred_entity["text"]:
                total_em += 1
                gt_consumed[i] = True
                pred_consumed[j] = True
                break
    return total_em / len(gt_entities) if gt_entities else 0.0

def compute_adjusted_ned(gt_entities, pred_entities):
    total_ned = 0.0
    count = 0
    gt_consumed = [False] * len(gt_entities)
    pred_consumed = [False] * len(pred_entities)

    for i, gt_entity in enumerate(gt_entities):
        gt_label = gt_entity["label"]
        gt_text = gt_entity["text"]
        min_distance = float('inf')
        best_match_idx = -1
        for j, pred_entity in enumerate(pred_entities):
            if pred_consumed[j] or gt_label != pred_entity["label"]:
                continue
            distance = Levenshtein.distance(gt_text, pred_entity["text"])
            if distance < min_distance:
                min_distance = distance
                best_match_idx = j
        if best_match_idx != -1:
            max_len = max(len(gt_text), len(pred_entities[best_match_idx]["text"]))
            total_ned += min_distance / max_len if max_len > 0 else 0.0
            count += 1
            gt_consumed[i] = True
            pred_consumed[best_match_idx] = True
        else:
            total_ned += 1.0
            count += 1

    return total_ned / count if count > 0 else 0.0

def compute_tokens_found_added(gt_entities, pred_entities):
    total_tokens_found = 0
    total_tokens_added = 0
    gt_consumed = [False] * len(gt_entities)
    pred_consumed = [False] * len(pred_entities)

    for i, gt_entity in enumerate(gt_entities):
        gt_label = gt_entity["label"]
        gt_text = gt_entity["text"]
        gt_tokens = gt_text.split()
        best_match_idx = -1
        max_common_tokens = 0
        for j, pred_entity in enumerate(pred_entities):
            if pred_consumed[j] or gt_label != pred_entity["label"]:
                continue
            pred_tokens = pred_entity["text"].split()
            common_tokens = set(pred_tokens) & set(gt_tokens)
            if len(common_tokens) > max_common_tokens:
                max_common_tokens = len(common_tokens)
                best_match_idx = j
        if best_match_idx != -1:
            pred_tokens = pred_entities[best_match_idx]["text"].split()
            total_tokens_found += max_common_tokens
            total_tokens_added += len(pred_tokens) - max_common_tokens
            gt_consumed[i] = True
            pred_consumed[best_match_idx] = True

    for j, pred_entity in enumerate(pred_entities):
        if not pred_consumed[j] and pred_entity["label"] != "menu.cnt":
            pred_tokens = pred_entity["text"].split()
            total_tokens_added += len(pred_tokens)

    return total_tokens_found, total_tokens_added

def compute_kieval_metrics(gt_entities, pred_entities):
    kie_metrics = calculate_kie_metrics(gt_entities, pred_entities)
    entity_f1 = kie_metrics["f1"]
    true_positives = kie_metrics["true_positives"]
    false_positives = kie_metrics["false_positives"]
    false_negatives = kie_metrics["false_negatives"]
    false_negatives_list = kie_metrics["false_negatives_list"]
    false_positives_list = kie_metrics["false_positives_list"]

    group_match = 1 if false_negatives == 0 and false_positives == 0 else 0
    group_precision = group_match / 1 if (true_positives + false_positives) > 0 else 0.0
    group_recall = group_match / 1 if (true_positives + false_negatives) > 0 else 0.0
    group_f1 = (2 * group_precision * group_recall) / (group_precision + group_recall) if (group_precision + group_recall) > 0 else 0.0

    correction_cost = false_negatives + false_positives
    aligned_score = 1.0 - (correction_cost / max(len(gt_entities), len(pred_entities))) if max(len(gt_entities), len(pred_entities)) > 0 else 0.0

    return {
        "entity_f1": entity_f1,
        "group_f1": group_f1,
        "aligned_score": aligned_score,
        "correction_cost": correction_cost,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "false_negatives_list": false_negatives_list, 
        "false_positives_list": false_positives_list  
    }

# --- Calculate All Metrics ---
def calculate_all_metrics(gt_entities, pred_entities):
    anls = compute_anls(gt_entities, pred_entities)
    em = compute_exact_match(gt_entities, pred_entities)
    ned = compute_adjusted_ned(gt_entities, pred_entities)
    tokens_found, tokens_added = compute_tokens_found_added(gt_entities, pred_entities)
    kieval = compute_kieval_metrics(gt_entities, pred_entities)

    return {
        "anls": anls,
        "em": em,
        "ned": ned,
        "tokens_found": tokens_found,
        "tokens_added": tokens_added,
        "kieval": kieval
    }

# --- Main Training and Evaluation Function ---
async def train_and_evaluate_layoutlmv3_cord():
    print("--- Starting LayoutLMv3 training and evaluation for CORD v2 ---")

    # Load Dataset
    dataset = load_cord_v2_dataset()

    # Dynamically build MODEL_ID2LABEL and MODEL_LABEL2ID
    print("\nCollecting unique labels from CORD v2 dataset for dynamic label mapping...")
    for split in ["train", "validation", "test"]:
        for i in range(min(len(dataset[split]), 200)):
            ground_truth_str = dataset[split][i]["ground_truth"]
            image_width = dataset[split][i]["image"].size[0]
            image_height = dataset[split][i]["image"].size[1]
            extract_words_bboxes_and_labels(ground_truth_str, image_width, image_height)
    
    global MODEL_ID2LABEL, MODEL_LABEL2ID
    sorted_base_labels = sorted(list(ALL_BASE_LABELS))
    all_iob2_labels = ["O"]
    for base_label in sorted_base_labels:
        all_iob2_labels.append(f"B-{base_label.upper()}")
        all_iob2_labels.append(f"I-{base_label.upper()}")
    
    MODEL_ID2LABEL = {i: label for i, label in enumerate(all_iob2_labels)}
    MODEL_LABEL2ID = {label: i for i, label in enumerate(all_iob2_labels)}
    
    print(f"\nDynamically built CORD v2 Dataset Labels:")
    print(f"  ID2LABEL: {MODEL_ID2LABEL}")
    print(f"  LABEL2ID: {MODEL_LABEL2ID}")
    print(f"  Number of labels: {len(MODEL_ID2LABEL)}")

    # Initialize Model
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(MODEL_ID2LABEL), 
        id2label=MODEL_ID2LABEL, 
        label2id=MODEL_LABEL2ID
    )
    print("Model loaded successfully. The classification head is re-initialized for the CORD v2 labels.")

    # Preprocess Dataset
    print("Applying LayoutLMv3 tokenization and label alignment to datasets...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=['image', 'ground_truth'], 
        desc="Processing documents",
    )
    
    print(f"Loaded {len(tokenized_dataset['train'])} training samples, {len(tokenized_dataset['validation'])} validation samples, and {len(tokenized_dataset['test'])} test samples.")

    # TrainingArguments
    print("\nConfiguring Training Arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10, 
        per_device_train_batch_size=2, 
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=LOGGING_DIR,
        logging_steps=50,
        eval_strategy="epoch", 
        save_strategy="epoch", 
        load_best_model_at_end=True, 
        metric_for_best_model="f1", 
        greater_is_better=True,
        report_to="none",
        fp16=torch.cuda.is_available(), 
        dataloader_num_workers=os.cpu_count() // 2, 
        gradient_accumulation_steps=4, 
        save_total_limit=3, 
    )

    # Initialize Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"], 
        tokenizer=processor, 
        compute_metrics=compute_metrics,
    )

    # Train the Model
    print("\nStarting model training...")
    train_result = trainer.train()
    print("Training complete!")

    # Evaluate the Model on the Test Set
    print("\nEvaluating the trained model on the test set (final evaluation)...")
    metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print("Test Set Evaluation Results (Best Model):")
    print(metrics)

    # Additional Evaluation for New Metrics
    print("\nStarting additional evaluation for new metrics on the test set...")
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

    # Initialize JSONL log file
    with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
        log_file.write("")

    for example in tokenized_dataset["test"]:
        start_time = time.time()
        input_ids = torch.tensor([example["input_ids"]]).to(device)
        attention_mask = torch.tensor([example["attention_mask"]]).to(device)
        bbox = torch.tensor([example["bbox"]]).to(device)
        pixel_values = torch.tensor([example["pixel_values"]]).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, bbox=bbox)
        predictions = np.argmax(outputs.logits.cpu().numpy(), axis=2)[0]
        pred_labels = [MODEL_ID2LABEL[p_id] for p_id in predictions if p_id != MODEL_LABEL2ID["O"]]  # Filter out 'O'

        # Extract flattened entities from predictions
        pred_entities = extract_flattened_entities_from_predictions(example["words"], pred_labels)

        # Extract flattened entities from ground truth (assuming gt_parse is stored or can be retrieved)
        gt_parse = json.loads(example["ground_truth"])["gt_parse"]  # Assume ground_truth is stored in the tokenized dataset
        gt_entities = extract_flattened_entities_from_gt(gt_parse)

        # Calculate metrics
        sample_metrics = calculate_all_metrics(gt_entities, pred_entities)
        sample_time = time.time() - start_time
        total_time += sample_time
        processed_samples_count += 1

        total_anls += sample_metrics["anls"]
        total_em += sample_metrics["em"]
        total_ned += sample_metrics["ned"]
        total_tokens_found += sample_metrics["tokens_found"]
        total_tokens_added += sample_metrics["tokens_added"]
        total_entity_f1 += sample_metrics["kieval"]["entity_f1"]
        total_group_f1 += sample_metrics["kieval"]["group_f1"]
        total_aligned_score += sample_metrics["kieval"]["aligned_score"]
        total_correction_cost += sample_metrics["kieval"]["correction_cost"]

        # Log sample
        sample_log = {
            "sample_id": example["id"],
            "ground_truth_entities": gt_entities,
            "predicted_entities": pred_entities,
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

    print("\nAdditional Evaluation Summary for LayoutLMv3 on CORD v2 Test Set")
    print("="*60)
    if processed_samples_count > 0:
        avg_anls = total_anls / processed_samples_count
        avg_em = total_em / processed_samples_count
        avg_ned = total_ned / processed_samples_count
        avg_time = total_time / processed_samples_count
        print(f"Average ANLS: {avg_anls:.4f}")
        print(f"Average Exact Match (EM): {avg_em:.4f}")
        print(f"Average Adjusted NED (SCORE): {avg_ned:.4f}")
        print(f"Total Tokens Found (SCORE): {total_tokens_found}")
        print(f"Total Tokens Added (SCORE): {total_tokens_added}")
        print(f"KIEval Entity F1: {total_entity_f1 / processed_samples_count:.4f}")
        print(f"KIEval Group F1: {total_group_f1 / processed_samples_count:.4f}")
        print(f"KIEval Aligned Score: {total_aligned_score / processed_samples_count:.4f}")
        print(f"KIEval Total Correction Cost: {total_correction_cost}")
        print(f"Average Time per Sample: {avg_time:.2f}s")
    else:
        print("No samples were processed for additional evaluation.")
    print("="*60)

    print("\n--- CORD v2 evaluation with LayoutLMv3 finished! ---")

# --- Run the script ---
if __name__ == "__main__":
    import asyncio
    asyncio.run(train_and_evaluate_layoutlmv3_cord())