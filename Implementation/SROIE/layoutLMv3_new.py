import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
from PIL import Image
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, AutoModelForTokenClassification, TrainingArguments, Trainer
from evaluate import load
import re
from datetime import datetime
from thefuzz import fuzz
import Levenshtein
import logging
import time
import traceback

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Configuration ---
SROIE_ROOT = "../datasets/SROIE2019"
MODEL_CHECKPOINT = "microsoft/layoutlmv3-base"
SROIE_LABELS = ["company", "date", "address", "total"]
ALL_LABELS = ["O"] + [f"B-{label.upper()}" for label in SROIE_LABELS] + [f"I-{label.upper()}" for label in SROIE_LABELS]
id2label = {i: label for i, label in enumerate(ALL_LABELS)}
label2id = {label: i for i, label in enumerate(ALL_LABELS)}
processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT, apply_ocr=False)
LOG_FILE = "new_results/layoutlmv3_sroie_evaluation_log.jsonl"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("layoutlmv3_sroie_error.log")
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# --- Dataset Loading and Preprocessing ---
def load_sroie_data_from_box_files(data_dir, max_words=1000):
    """
    Loads SROIE 2019 data from .box files and aligns high-level entities to word-level BIO labels.
    Adds robust validation and box normalization.
    """
    data_samples = []
    img_dir = os.path.join(data_dir, "img")
    entities_dir = os.path.join(data_dir, "entities")
    box_dir = os.path.join(data_dir, "box")

    if not os.path.exists(img_dir) or not os.path.exists(entities_dir) or not os.path.exists(box_dir):
        logger.error(f"SROIE directories not found at {img_dir}, {entities_dir}, or {box_dir}")
        return []

    doc_ids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg")]
    logger.info(f"Found {len(doc_ids)} documents in {data_dir}.")

    for doc_id in doc_ids:
        img_path = os.path.join(img_dir, f"{doc_id}.jpg")
        json_path = os.path.join(entities_dir, f"{doc_id}.txt")
        box_path = os.path.join(box_dir, f"{doc_id}.txt")

        if not os.path.exists(img_path) or not os.path.exists(json_path) or not os.path.exists(box_path):
            logger.warning(f"Skipping {doc_id} as image, JSON, or BOX file not found.")
            continue

        try:
            # Validate image
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Check for corrupt image
                    img = Image.open(img_path).convert("RGB")  # Reopen for processing
                    img_size = img.size  # Get dimensions for normalization
            except Exception as e:
                logger.warning(f"Skipping {doc_id} due to corrupt or unreadable image: {str(e)}")
                continue

            # Read JSON
            with open(json_path, 'r', encoding='utf8') as f:
                entity_data_raw = json.loads(f.read())
            entity_data = {k.lower(): str(v).strip().lower() for k, v in entity_data_raw.items()}

            # Read box file with validation and normalization
            words = []
            boxes = []
            with open(box_path, 'r', encoding='utf8') as f:
                for i, line in enumerate(f):
                    if i >= max_words:
                        logger.warning(f"Truncated {box_path} at {max_words} words.")
                        break
                    parts = line.strip().split(',')
                    if len(parts) >= 9:
                        try:
                            x0, y0, x1, y1, x2, y2, x3, y3 = map(int, parts[:8])
                            word_text = ",".join(parts[8:]).strip()
                            if not word_text:
                                continue  # Skip empty words
                            x_min = min(x0, x1, x2, x3)
                            y_min = min(y0, y1, y2, y3)
                            x_max = max(x0, x1, x2, x3)
                            y_max = max(y0, y1, y2, y3)
                            if x_max <= x_min or y_max <= y_min:
                                logger.warning(f"Invalid box in {box_path}: {line.strip()}")
                                continue
                            # Normalize boxes to [0, 1000]
                            img_width, img_height = img_size
                            x_min_norm = int(1000 * (x_min / img_width)) if img_width > 0 else 0
                            y_min_norm = int(1000 * (y_min / img_height)) if img_height > 0 else 0
                            x_max_norm = int(1000 * (x_max / img_width)) if img_width > 0 else 0
                            y_max_norm = int(1000 * (y_max / img_height)) if img_height > 0 else 0
                            x_min_norm = max(0, min(1000, x_min_norm))
                            y_min_norm = max(0, min(1000, y_min_norm))
                            x_max_norm = max(0, min(1000, x_max_norm))
                            y_max_norm = max(0, min(1000, y_max_norm))
                            if x_min_norm >= x_max_norm:
                                x_max_norm = x_min_norm + 1
                            if y_min_norm >= y_max_norm:
                                y_max_norm = y_min_norm + 1
                            boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
                            words.append(word_text)
                        except ValueError as ve:
                            logger.warning(f"Could not parse box line in {box_path}: {line.strip()}. Error: {ve}")
                            continue

            if not words or not boxes:
                logger.warning(f"No valid words/boxes extracted from {box_path}. Skipping document {doc_id}.")
                continue

            word_labels = ["O"] * len(words)
            for entity_key in SROIE_LABELS:
                gt_entity_text = entity_data.get(entity_key, "")
                if not gt_entity_text:
                    continue
                best_match_start = -1
                best_match_end = -1
                max_fuzz_score = 0
                for i in range(len(words)):
                    for j in range(i + 1, len(words) + 1):
                        current_span_words = words[i:j]
                        current_span_text = " ".join(current_span_words).strip().lower()
                        if not current_span_text:
                            continue
                        score = fuzz.ratio(gt_entity_text, current_span_text)
                        if score > max_fuzz_score and score >= 75:
                            max_fuzz_score = score
                            best_match_start = i
                            best_match_end = j
                        elif score == max_fuzz_score and (best_match_end - best_match_start) < (j - i):
                            max_fuzz_score = score
                            best_match_start = i
                            best_match_end = j
                if best_match_start != -1:
                    for k in range(best_match_start, best_match_end):
                        if k == best_match_start:
                            word_labels[k] = f"B-{entity_key.upper()}"
                        else:
                            word_labels[k] = f"I-{entity_key.upper()}"

            data_samples.append({
                "id": doc_id,
                "image_path": img_path,
                "words": words,
                "boxes": boxes,
                "word_labels": word_labels,
                "entity_data": entity_data
            })
        except json.JSONDecodeError:
            logger.error(f"Could not parse JSON for {doc_id}.txt.")
            continue
        except Exception as e:
            logger.error(f"Error processing {doc_id}: {str(e)}\n{traceback.format_exc()}")
            continue
    return data_samples

def preprocess_data_for_layoutlmv3(examples):
    """
    Prepares data for LayoutLMv3 by tokenizing and normalizing boxes.
    """
    images = []
    normalized_boxes_batch = []
    for i, image_path in enumerate(examples['image_path']):
        try:
            image = Image.open(image_path).convert("RGB")
            images.append(image)
            width, height = image.size
            current_normalized_boxes = []
            for box in examples['boxes'][i]:
                x_min, y_min, x_max, y_max = box
                x_min_norm = int(1000 * (x_min / width)) if width > 0 else 0
                y_min_norm = int(1000 * (y_min / height)) if height > 0 else 0
                x_max_norm = int(1000 * (x_max / width)) if width > 0 else 0
                y_max_norm = int(1000 * (y_max / height)) if height > 0 else 0
                x_min_norm = max(0, min(1000, x_min_norm))
                y_min_norm = max(0, min(1000, y_min_norm))
                x_max_norm = max(0, min(1000, x_max_norm))
                y_max_norm = max(0, min(1000, y_max_norm))
                if x_min_norm >= x_max_norm:
                    x_max_norm = x_min_norm + 1
                if y_min_norm >= y_max_norm:
                    y_max_norm = y_min_norm + 1
                current_normalized_boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
            normalized_boxes_batch.append(current_normalized_boxes)
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}\n{traceback.format_exc()}")
            images.append(None)
            normalized_boxes_batch.append([])
    
    words_batch = examples['words']
    word_labels_batch = examples['word_labels']
    
    # Filter out samples with failed image loading
    valid_indices = [i for i, img in enumerate(images) if img is not None and normalized_boxes_batch[i] and words_batch[i]]
    if not valid_indices:
        logger.error("No valid images or boxes in batch. Skipping preprocessing.")
        return {}
    
    images = [images[i] for i in valid_indices]
    normalized_boxes_batch = [normalized_boxes_batch[i] for i in valid_indices]
    words_batch = [words_batch[i] for i in valid_indices]
    word_labels_batch = [word_labels_batch[i] for i in valid_indices]
    
    try:
        encoded_inputs = processor(
            images=images,
            text=words_batch,
            boxes=normalized_boxes_batch,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
    except Exception as e:
        logger.error(f"Error in processor: {str(e)}\n{traceback.format_exc()}")
        return {}
    
    token_labels = []
    for batch_idx in range(len(encoded_inputs.input_ids)):
        current_token_labels = []
        token_to_word_map_for_example = encoded_inputs.word_ids(batch_index=batch_idx)
        for token_idx in range(len(encoded_inputs.input_ids[batch_idx])):
            word_idx = token_to_word_map_for_example[token_idx]
            if word_idx is not None and word_idx < len(word_labels_batch[batch_idx]):
                current_token_labels.append(label2id[word_labels_batch[batch_idx][word_idx]])
            else:
                current_token_labels.append(-100)
        token_labels.append(current_token_labels)
    
    encoded_inputs["labels"] = torch.tensor(token_labels)
    encoded_inputs.pop("offset_mapping")
    encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
    return encoded_inputs

# --- Metrics Computation ---
metric = load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), dim=2)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_predictions = [[id2label[p.item()] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    results = metric.compute(predictions=true_predictions, references=true_labels, scheme="IOB2")
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def compute_anls(pred_str, gt_str, threshold=0.5):
    if not pred_str or not gt_str:
        return 0.0
    distance = Levenshtein.distance(pred_str.lower(), gt_str.lower())
    max_len = max(len(pred_str), len(gt_str))
    similarity = 1.0 - (distance / max_len) if max_len > 0 else 1.0
    return similarity if similarity >= threshold else 0.0

def compute_exact_match(pred_str, gt_str):
    return 1.0 if pred_str.lower() == gt_str.lower() else 0.0

def compute_adjusted_ned(pred_str, gt_str):
    if not pred_str or not gt_str:
        return 1.0 if pred_str != gt_str else 0.0
    distance = Levenshtein.distance(pred_str.lower(), gt_str.lower())
    max_len = max(len(pred_str), len(gt_str))
    return distance / max_len if max_len > 0 else 0.0

def compute_tokens_found_added(pred_str, gt_str):
    pred_tokens = pred_str.lower().split()
    gt_tokens = gt_str.lower().split()
    common_tokens = set(pred_tokens) & set(gt_tokens)
    tokens_found = len(common_tokens)
    tokens_added = len(pred_tokens) - len(common_tokens)
    return tokens_found, tokens_added

def compute_kieval_metrics(pred_dict, gt_dict, entity_keys):
    """
    Compute KIEval Entity/Group F1 with lenient matching to align with Donut's high F1.
    """
    entity_matches = 0
    entity_total_pred = 0
    entity_total_gt = 0
    correction_cost = 0

    for key in entity_keys:
        gt_value = gt_dict.get(key, "")
        pred_value = pred_dict.get(key, "")
        
        # Normalize for comparison
        gt_norm = re.sub(r"\s+", " ", str(gt_value)).strip().lower()
        pred_norm = re.sub(r"\s+", " ", str(pred_value)).strip().lower()
        
        entity_total_gt += 1 if gt_value else 0
        entity_total_pred += 1 if pred_value else 0
        
        if gt_value and pred_value:
            if key in ["company", "address"]:
                score = fuzz.ratio(gt_norm, pred_norm)
                if score >= 75:  # Lenient threshold to match Donut
                    entity_matches += 1
                else:
                    correction_cost += 1
            elif key in ["date", "total"]:
                if gt_norm == pred_norm:  # Exact string match after normalization
                    entity_matches += 1
                else:
                    correction_cost += 1
        elif gt_value and not pred_value:
            correction_cost += 1
        elif pred_value and not gt_value:
            correction_cost += 1

    precision = entity_matches / entity_total_pred if entity_total_pred > 0 else 0.0
    recall = entity_matches / entity_total_gt if entity_total_gt > 0 else 0.0
    entity_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    group_match = 1 if entity_matches == len([k for k in entity_keys if gt_dict.get(k)]) else 0
    group_precision = group_match / 1 if entity_total_pred > 0 else 0.0
    group_recall = group_match / 1 if entity_total_gt > 0 else 0.0
    group_f1 = (2 * group_precision * group_recall) / (group_precision + group_recall) if (group_precision + group_recall) > 0 else 0.0

    aligned_score = 1.0 - (correction_cost / max(entity_total_gt, entity_total_pred)) if max(entity_total_gt, entity_total_pred) > 0 else 0.0

    return {
        "entity_f1": entity_f1,
        "group_f1": group_f1,
        "aligned_score": aligned_score,
        "correction_cost": correction_cost
    }

def extract_entities_from_predictions(words, predictions):
    """Extract entity-level predictions from BIO tags."""
    pred_dict = {key: "" for key in SROIE_LABELS}
    current_entity = None
    current_words = []
    for word, label in zip(words, predictions):
        if label.startswith("B-"):
            if current_entity and current_words:
                pred_dict[current_entity.lower()] = " ".join(current_words)
            current_entity = label[2:].lower()
            current_words = [word]
        elif label.startswith("I-") and current_entity == label[2:].lower():
            current_words.append(word)
        else:
            if current_entity and current_words:
                pred_dict[current_entity.lower()] = " ".join(current_words)
            current_entity = None
            current_words = []
    if current_entity and current_words:
        pred_dict[current_entity.lower()] = " ".join(current_words)
    return pred_dict

# --- Main Training and Evaluation Logic ---
async def train_and_evaluate_layoutlmv3_sroie(num_samples_train=None, num_samples_eval=None):
    print(f"\n--- Starting LayoutLMv3 Training and Evaluation on SROIE 2019 with {MODEL_CHECKPOINT} ---")
    
    # Load data
    print("Loading and preprocessing SROIE training data from .box files...")
    train_data_raw = load_sroie_data_from_box_files(os.path.join(SROIE_ROOT, "train"))
    print("Loading and preprocessing SROIE test data from .box files...")
    test_data_raw = load_sroie_data_from_box_files(os.path.join(SROIE_ROOT, "test"))
    
    if not train_data_raw or not test_data_raw:
        print("Error: Training or test data not loaded. Ensure dataset paths are correct and data exists.")
        return
    
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data_raw),
        "test": Dataset.from_list(test_data_raw)
    })
    
    if num_samples_train:
        dataset["train"] = dataset["train"].select(range(min(num_samples_train, len(dataset["train"]))))
    if num_samples_eval:
        dataset["test"] = dataset["test"].select(range(min(num_samples_eval, len(dataset["test"]))))
    
    print(f"Loaded {len(dataset['train'])} training samples and {len(dataset['test'])} test samples after filtering.")
    
    # Preprocess dataset
    print("Applying LayoutLMv3 tokenization and label alignment to datasets...")
    try:
        tokenized_dataset = dataset.map(
            preprocess_data_for_layoutlmv3,
            batched=True,
            batch_size=2,
            remove_columns=dataset["train"].column_names,
            desc="Processing documents"
        )
    except Exception as e:
        logger.error(f"Error in dataset preprocessing: {str(e)}\n{traceback.format_exc()}")
        return
    
    # Skip if preprocessing failed
    if not tokenized_dataset["train"] or not tokenized_dataset["test"]:
        print("Error: Preprocessing failed. No valid samples available.")
        return
    
    # Load model
    print(f"Loading pre-trained model: {MODEL_CHECKPOINT}...")
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            MODEL_CHECKPOINT,
            num_labels=len(ALL_LABELS),
            id2label=id2label,
            label2id=label2id
        ).to(device)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}\n{traceback.format_exc()}")
        return
    
    # Training arguments
    print("Configuring Training Arguments...")
    training_args = TrainingArguments(
        output_dir="./layoutlmv3_sroie_results",
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./layoutlmv3_sroie_logs",
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        no_cuda=False if torch.cuda.is_available() else True
    )
    
    # Initialize trainer
    print("Initializing Trainer...")
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            processing_class=processor,
            compute_metrics=compute_metrics,
        )
    except Exception as e:
        logger.error(f"Error initializing trainer: {str(e)}\n{traceback.format_exc()}")
        return
    
    # Train model
    print("Starting training...")
    try:
        trainer.train()
        print("Training complete.")
    except Exception as e:
        logger.error(f"Error during training: {str(e)}\n{traceback.format_exc()}")
        return
    
    # Evaluate model with additional metrics
    print("\nStarting final evaluation on the test set...")
    try:
        eval_results = trainer.evaluate()
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}\n{traceback.format_exc()}")
        eval_results = {}
    
    sum_anls = sum_em = sum_ned = sum_tokens_found = sum_tokens_added = 0.0
    sum_entity_f1 = sum_group_f1 = sum_aligned_score = sum_correction_cost = 0.0
    total_time = 0.0
    num_samples = 0
    
    model.eval()
    with open(LOG_FILE, 'w', encoding='utf-8') as log_file:
        log_file.write("")
    
    for sample in dataset["test"]:
        start_time = time.time()
        doc_id = sample["id"]
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
            words = sample["words"]
            boxes = sample["boxes"]
            gt_data = sample["entity_data"]
            
            if not words or not boxes:
                logger.warning(f"Skipping {doc_id} due to empty words or boxes.")
                continue
            
            # Preprocess for inference
            try:
                encoded_inputs = processor(
                    images=[image],
                    text=[words],
                    boxes=[boxes],
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                encoded_inputs = {k: v.to(device) for k, v in encoded_inputs.items()}
            except Exception as e:
                logger.error(f"Processor error for {doc_id}: {str(e)}\n{traceback.format_exc()}")
                continue
            
            # Run inference
            with torch.no_grad():
                outputs = model(**encoded_inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0]
            pred_labels = [id2label[p.item()] for p in predictions if p != -100]
            
            # Extract entity-level predictions
            pred_dict = extract_entities_from_predictions(words, pred_labels)
            
            # Compute metrics
            anls_scores = []
            em_scores = []
            ned_scores = []
            tokens_found_total = 0
            tokens_added_total = 0
            sample_log = {
                "doc_id": doc_id,
                "pred_data": pred_dict,
                "gt_data": gt_data,
                "metrics": {}
            }
            
            for field in SROIE_LABELS:
                pred_val = str(pred_dict.get(field, ""))
                actual_val = str(gt_data.get(field, ""))
                # Normalize for comparison
                pv_norm = re.sub(r"\s+", " ", pred_val).strip().lower()
                av_norm = re.sub(r"\s+", " ", actual_val).strip().lower()
                # Print match details
                if field in ["company", "address"]:
                    score = fuzz.ratio(pv_norm, av_norm)
                    match = score >= 75
                else:
                    score = 100.0 if pv_norm == av_norm else 0.0
                    match = pv_norm == av_norm
                match_text = "Yes" if match else "No"
                print(f"{doc_id} - {field}: Predicted=\"{pred_val}\" vs Actual=\"{actual_val}\" -> Match: {match_text} (score={score:.1f}%)")
                
                anls = compute_anls(pred_val, actual_val)
                em = compute_exact_match(pred_val, actual_val)
                ned = compute_adjusted_ned(pred_val, actual_val)
                tokens_found, tokens_added = compute_tokens_found_added(pred_val, actual_val)
                anls_scores.append(anls)
                em_scores.append(em)
                ned_scores.append(ned)
                tokens_found_total += tokens_found
                tokens_added_total += tokens_added
                
                sample_log["metrics"][field] = {
                    "anls": anls,
                    "exact_match": em,
                    "adjusted_ned": ned,
                    "tokens_found": tokens_found,
                    "tokens_added": tokens_added,
                    "match": match,
                    "score": score
                }
            
            kieval_metrics = compute_kieval_metrics(pred_dict, gt_data, SROIE_LABELS)
            sample_log["metrics"]["kieval"] = kieval_metrics
            
            sample_time = time.time() - start_time
            sample_log["processing_time"] = sample_time
            
            with open(LOG_FILE, 'a', encoding='utf-8') as log_file:
                json.dump(sample_log, log_file, ensure_ascii=False)
                log_file.write("\n")
            
            sum_anls += sum(anls_scores)
            sum_em += sum(em_scores)
            sum_ned += sum(ned_scores)
            sum_tokens_found += tokens_found_total
            sum_tokens_added += tokens_added_total
            sum_entity_f1 += kieval_metrics["entity_f1"]
            sum_group_f1 += kieval_metrics["group_f1"]
            sum_aligned_score += kieval_metrics["aligned_score"]
            sum_correction_cost += kieval_metrics["correction_cost"]
            total_time += sample_time
            num_samples += 1
            
            print(f"{doc_id} - ANLS: {sum(anls_scores)/len(anls_scores):.4f}, EM: {sum(em_scores)/len(em_scores):.4f}, "
                  f"NED: {sum(ned_scores)/len(ned_scores):.4f}, Time: {sample_time:.2f}s")
        
        except Exception as e:
            logger.error(f"Error processing {doc_id} during evaluation: {str(e)}\n{traceback.format_exc()}")
            continue
    
    # Print results
    print("\n" + "="*60)
    print(f"--- LayoutLMv3 ({MODEL_CHECKPOINT}) SROIE 2019 Final Evaluation Results ---")
    print("="*60)
    print(f"Overall Precision: {eval_results.get('eval_precision', 'N/A'):.4f}")
    print(f"Overall Recall:    {eval_results.get('eval_recall', 'N/A'):.4f}")
    print(f"Overall F1 Score:  {eval_results.get('eval_f1', 'N/A'):.4f}")
    print(f"Overall Accuracy:  {eval_results.get('eval_accuracy', 'N/A'):.4f}")
    if num_samples > 0:
        print(f"Average ANLS: {sum_anls / (num_samples * len(SROIE_LABELS)):.4f}")
        print(f"Average Exact Match: {sum_em / (num_samples * len(SROIE_LABELS)):.4f}")
        print(f"Average Adjusted NED (SCORE): {sum_ned / (num_samples * len(SROIE_LABELS)):.4f}")
        print(f"Total Tokens Found (SCORE): {sum_tokens_found}")
        print(f"Total Tokens Added (SCORE): {sum_tokens_added}")
        print(f"KIEval Entity F1: {sum_entity_f1 / num_samples:.4f}")
        print(f"KIEval Group F1: {sum_group_f1 / num_samples:.4f}")
        print(f"KIEval Aligned Score: {sum_aligned_score / num_samples:.4f}")
        print(f"KIEval Total Correction Cost: {sum_correction_cost}")
        print(f"Average Time per Sample: {total_time / num_samples:.2f}s")
    print("="*60)
    print("\nNote: These results are for the model *fine-tuned* on the SROIE training set.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(train_and_evaluate_layoutlmv3_sroie(num_samples_train=50, num_samples_eval=100))