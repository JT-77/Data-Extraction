import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import json
from PIL import Image
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, AutoModelForTokenClassification, TrainingArguments, Trainer
from evaluate import load # For seqeval metric
import re
from datetime import datetime
from thefuzz import fuzz

# --- Configuration ---
# Path to your SROIE 2019 dataset.
# This script expects the dataset to be structured like:
# SROIE_ROOT/
# └── test/
#     ├── img/       (contains .jpg image files)
#     │   ├── X00016469670.jpg
#     │   └── ...
#     ├── entities/  (contains .txt annotation files with JSON content)
#     │   ├── X00016469670.txt
#     │   └── ...
#     └── box/       (contains .txt files with word-level boxes and text)
#         ├── X00016469670.txt  <-- These are the new files you have
#         └── ...
# └── train/ (optional, for potential fine-tuning later)
#     ├── img/
#     ├── entities/
#     └── box/
SROIE_ROOT = "../datasets/SROIE2019"

MODEL_CHECKPOINT = "microsoft/layoutlmv3-base"

# Expected entity labels for SROIE 2019
SROIE_LABELS = ["company", "date", "address", "total"]
# All possible labels in BIO format + 'O' (Outside)
ALL_LABELS = ["O"] + [f"B-{label.upper()}" for label in SROIE_LABELS] + [f"I-{label.upper()}" for label in SROIE_LABELS]

# Create label mappings for the model
id2label = {i: label for i, label in enumerate(ALL_LABELS)}
label2id = {label: i for i, label in enumerate(ALL_LABELS)}

# Load the processor (tokenizer and image processor)
# We set apply_ocr=False because we are providing words and boxes explicitly from .box files.
processor = AutoProcessor.from_pretrained(MODEL_CHECKPOINT, apply_ocr=False)

# --- Dataset Loading and Preprocessing ---

def load_sroie_data_from_box_files(data_dir):
    """
    Loads SROIE 2019 data from local files, reads words and boxes from .box/.txt files,
    and aligns high-level entities to create word-level BIO labels.
    """
    data_samples = []
    img_dir = os.path.join(data_dir, "img")
    entities_dir = os.path.join(data_dir, "entities")
    box_dir = os.path.join(data_dir, "box") # New directory for word-level boxes and text

    if not os.path.exists(img_dir) or not os.path.exists(entities_dir) or not os.path.exists(box_dir):
        print(f"Error: SROIE directories not found at {img_dir}, {entities_dir}, or {box_dir}")
        return []

    # Get all document IDs (assuming image names are consistent across img, entities, box)
    doc_ids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if f.endswith(".jpg")]

    print(f"Found {len(doc_ids)} documents in {data_dir}.")

    for doc_id in doc_ids:
        img_path = os.path.join(img_dir, f"{doc_id}.jpg")
        json_path = os.path.join(entities_dir, f"{doc_id}.txt")
        box_path = os.path.join(box_dir, f"{doc_id}.txt") # Path to the .box file

        if not os.path.exists(img_path) or not os.path.exists(json_path) or not os.path.exists(box_path):
            print(f"Warning: Skipping {doc_id} as image, JSON, or BOX file not found.")
            continue

        try:
            # 1. Read high-level entities from JSON file
            with open(json_path, 'r', encoding='utf8') as f:
                entity_data_raw = json.loads(f.read())
            # Normalize entity data for robust matching (lowercase, strip whitespace)
            entity_data = {k.lower(): str(v).strip().lower() for k, v in entity_data_raw.items()}

            # 2. Read words and 8-coordinate boxes from .box file
            words = []
            boxes = [] # Will store 4-coordinate boxes
            with open(box_path, 'r', encoding='utf8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 9: # Expecting at least 8 coords + text
                        try:
                            # Parse 8 coordinates
                            x0, y0, x1, y1, x2, y2, x3, y3 = map(int, parts[:8])
                            word_text = ",".join(parts[8:]) # Text can contain commas
                            
                            # Convert 8-coordinate box to 4-coordinate [x_min, y_min, x_max, y_max]
                            # Assuming standard (x0,y0) top-left, (x2,y2) bottom-right
                            x_min = min(x0, x1, x2, x3)
                            y_min = min(y0, y1, y2, y3)
                            x_max = max(x0, x1, x2, x3)
                            y_max = max(y0, y1, y2, y3)
                            
                            # LayoutLMv3 requires x_min <= x_max and y_min <= y_max
                            # And non-zero area boxes. Filter invalid boxes.
                            if x_max > x_min and y_max > y_min:
                                words.append(word_text.strip())
                                boxes.append([x_min, y_min, x_max, y_max])
                        except ValueError:
                            print(f"Warning: Could not parse box line in {box_path}: {line.strip()}. Skipping line.")
                            continue
            
            if not words: # If no words were extracted from the box file
                print(f"Warning: No valid words/boxes extracted from {box_path}. Skipping document {doc_id}.")
                continue

            # 3. Align high-level entities to word-level labels (BIO format)
            word_labels = ["O"] * len(words) # Initialize all to "O"

            for entity_key in SROIE_LABELS:
                gt_entity_text = entity_data.get(entity_key, "")
                if not gt_entity_text:
                    continue

                # Heuristic alignment: Find entity text in the sequence of OCR'd words
                best_match_start = -1
                best_match_end = -1
                max_fuzz_score = 0

                # Iterate through all possible continuous spans of words
                for i in range(len(words)):
                    for j in range(i + 1, len(words) + 1):
                        current_span_words = words[i:j]
                        current_span_text = " ".join(current_span_words).strip().lower()
                        
                        if not current_span_text:
                            continue

                        # Calculate fuzzy ratio for comparison
                        score = fuzz.ratio(gt_entity_text, current_span_text)
                        
                        # Prioritize longer matches if scores are similar or if exact match
                        # Using a threshold (e.g., 75) for a fuzzy match
                        if score > max_fuzz_score and score >= 75: 
                            max_fuzz_score = score
                            best_match_start = i
                            best_match_end = j
                        elif score == max_fuzz_score and (best_match_end - best_match_start) < (j - i):
                            # If scores are equal, prefer a longer match (more words)
                            max_fuzz_score = score
                            best_match_start = i
                            best_match_end = j
                            
                if best_match_start != -1:
                    for k in range(best_match_start, best_match_end):
                        if k == best_match_start: # Beginning of the entity span
                            word_labels[k] = f"B-{entity_key.upper()}"
                        else: # Inside the entity span
                            word_labels[k] = f"I-{entity_key.upper()}"
                # else:
                #     print(f"Debug: Could not align GT entity '{entity_key}': '{gt_entity_text}' in {doc_id}")

            data_samples.append({
                "id": doc_id,
                "image_path": img_path,
                "words": words,
                "boxes": boxes, # These are now 4-coordinate boxes
                "word_labels": word_labels # List of strings (BIO tags)
            })
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON for {doc_id}.txt. Skipping.")
            continue
        except Exception as e:
            print(f"Warning: Error processing {doc_id}: {e}. Skipping.")
            continue
    return data_samples

# --- Preprocessing function for LayoutLMv3 processor ---
def preprocess_data_for_layoutlmv3(examples):
    """
    Takes word-level data (words, boxes, word_labels) and prepares it for LayoutLMv3.
    This involves tokenization, bounding box normalization, and aligning word labels to token labels.
    """
    images = []
    normalized_boxes_batch = []
    
    # Iterate through each example in the batch to load image and normalize boxes
    for i, image_path in enumerate(examples['image_path']):
        image = Image.open(image_path).convert("RGB")
        images.append(image)
        width, height = image.size # Get original image dimensions

        current_normalized_boxes = []
        for box in examples['boxes'][i]: 
            x_min, y_min, x_max, y_max = box

            # --- CRITICAL FIX: Normalize bounding box coordinates to 0-1000 range ---
            # Ensure width/height are not zero to prevent division by zero
            x_min_norm = int(1000 * (x_min / width)) if width > 0 else 0
            y_min_norm = int(1000 * (y_min / height)) if height > 0 else 0
            x_max_norm = int(1000 * (x_max / width)) if width > 0 else 0
            y_max_norm = int(1000 * (y_max / height)) if height > 0 else 0

            # Clamp coordinates to ensure they are strictly within 0-1000
            x_min_norm = max(0, min(1000, x_min_norm))
            y_min_norm = max(0, min(1000, y_min_norm))
            x_max_norm = max(0, min(1000, x_max_norm))
            y_max_norm = max(0, min(1000, y_max_norm))
            
            # Ensure min <= max; if a box becomes invalid after clamping (unlikely for valid inputs)
            if x_min_norm > x_max_norm: x_min_norm, x_max_norm = x_max_norm, x_min_norm
            if y_min_norm > y_max_norm: y_min_norm, y_max_norm = y_max_norm, y_min_norm

            current_normalized_boxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
        normalized_boxes_batch.append(current_normalized_boxes)
    # --- END CRITICAL FIX ---

    words_batch = examples['words'] 
    word_labels_batch = examples['word_labels'] 

    # 1. Process images, words, and normalized boxes to get tokenized inputs.
    encoded_inputs = processor(
        images=images,
        text=words_batch,
        boxes=normalized_boxes_batch, # Use the normalized boxes here
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True, # Still important for token-to-word alignment
        return_tensors="pt"
    )

    # 2. Create token-level labels based on word_labels and token_to_word_ids mapping.
    token_labels = []
    for batch_idx in range(len(encoded_inputs.input_ids)):
        current_token_labels = []
        
        # Access the word_ids mapping for the current example in the batch
        token_to_word_map_for_example = encoded_inputs.word_ids(batch_index=batch_idx) 

        for token_idx in range(len(encoded_inputs.input_ids[batch_idx])):
            word_idx = token_to_word_map_for_example[token_idx] # Use the example-specific mapping
            if word_idx is not None and word_idx < len(word_labels_batch[batch_idx]):
                current_token_labels.append(label2id[word_labels_batch[batch_idx][word_idx]])
            else:
                current_token_labels.append(-100) # Special tokens ([CLS], [SEP], [PAD])
        token_labels.append(current_token_labels)
    
    encoded_inputs["labels"] = torch.tensor(token_labels)
    
    encoded_inputs.pop("offset_mapping") # Remove offset_mapping before passing to the model

    return encoded_inputs

# --- Metrics Computation ---
metric = load("seqeval")

def compute_metrics(p):
    """
    Computes precision, recall, and F1-score for token classification.
    Handles -100 ignored labels.
    """
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), dim=2)

    # Remove ignored index (-100)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    # FIX: Convert tensor 'p' to a Python integer using .item()
    true_predictions = [[id2label[p.item()] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels, scheme="IOB2")
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# --- Main Training and Evaluation Logic ---
async def train_and_evaluate_layoutlmv3_sroie(num_samples_train=None, num_samples_eval=None):
    """
    Main function to load SROIE data, preprocess it, load LayoutLMv3,
    train it, and then evaluate its performance.
    """
    print(f"\n--- Starting LayoutLMv3 Training and Evaluation on SROIE 2019 with {MODEL_CHECKPOINT} ---")

    # 1. Load SROIE data from .box files and align labels for both train and test splits
    print("Loading and preprocessing SROIE training data from .box files...")
    train_data_raw = load_sroie_data_from_box_files(os.path.join(SROIE_ROOT, "train"))
    
    print("Loading and preprocessing SROIE test data from .box files...")
    test_data_raw = load_sroie_data_from_box_files(os.path.join(SROIE_ROOT, "test"))
    
    if not train_data_raw or not test_data_raw:
        print("Error: Training or test data not loaded. Ensure dataset paths are correct and data exists.")
        return

    # Create Hugging Face Dataset objects
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data_raw),
        "test": Dataset.from_list(test_data_raw)
    })

    # Limit samples if specified (for faster testing/debugging)
    if num_samples_train:
        dataset["train"] = dataset["train"].select(range(min(num_samples_train, len(dataset["train"]))))
    if num_samples_eval:
        dataset["test"] = dataset["test"].select(range(min(num_samples_eval, len(dataset["test"]))))
    
    print(f"Loaded {len(dataset['train'])} training samples and {len(dataset['test'])} test samples after filtering.")

    # 2. Preprocess the dataset for LayoutLMv3 (tokenization and label ID conversion)
    print("Applying LayoutLMv3 tokenization and label alignment to datasets...")
    tokenized_dataset = dataset.map(
        preprocess_data_for_layoutlmv3, 
        batched=True, 
        remove_columns=dataset["train"].column_names, # Remove original columns from both splits
        desc="Processing documents" 
    )
    
    # 3. Load the pre-trained LayoutLMv3 model
    print(f"Loading pre-trained model: {MODEL_CHECKPOINT}...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(ALL_LABELS),
        id2label=id2label,
        label2id=label2id
    )

    # 4. Define TrainingArguments for fine-tuning
    print("Configuring Training Arguments...")
    training_args = TrainingArguments(
        output_dir="./layoutlmv3_sroie_results", # Directory to save checkpoints and logs
        num_train_epochs=10, # Number of training epochs. Adjust as needed (e.g., 10-20 for full dataset)
        per_device_train_batch_size=2, # Batch size per GPU/CPU for training. Adjust based on VRAM.
        per_device_eval_batch_size=2,  # Batch size per GPU/CPU for evaluation
        learning_rate=5e-5, # Learning rate. Common values are 1e-5 to 5e-5
        warmup_ratio=0.1, # Ratio of total steps for linear warmup
        weight_decay=0.01, # Strength of weight decay
        logging_dir="./layoutlmv3_sroie_logs", # Directory for logs
        logging_steps=500, # Log training progress every N steps
        eval_strategy="epoch", # Evaluate at the end of each epoch
        save_strategy="epoch", # Save checkpoint at the end of each epoch
        load_best_model_at_end=True, # Load the best model found during training at the end
        metric_for_best_model="f1", # Metric to use for determining the best model
        greater_is_better=True, # Higher F1 is better
        report_to="none", # Disable reporting to W&B, MLflow etc.
        # no_cuda is deprecated, but use_cpu might not be available in older transformers versions
        # For robustness, we'll keep no_cuda for now, but note the warning.
        no_cuda=False if torch.cuda.is_available() else True # Use CUDA if available, else CPU
    )

    # 5. Create Trainer instance
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=processor, # Recommended way to pass the processor class
        compute_metrics=compute_metrics, # Pass the compute_metrics function here
    )

    # 6. Train the model
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # 7. Evaluate the fine-tuned model
    print("\nStarting final evaluation on the test set...")
    eval_results = trainer.evaluate()
    
    print("\n" + "="*60)
    print(f"--- LayoutLMv3 ({MODEL_CHECKPOINT}) SROIE 2019 Final Evaluation Results ---")
    print("="*60)
    print(f"Overall Precision: {eval_results.get('eval_precision', 'N/A'):.4f}")
    print(f"Overall Recall:    {eval_results.get('eval_recall', 'N/A'):.4f}")
    print(f"Overall F1 Score:  {eval_results.get('eval_f1', 'N/A'):.4f}")
    print(f"Overall Accuracy:  {eval_results.get('eval_accuracy', 'N/A'):.4f}")
    print("="*60)
    print("\nNote: These results are for the model *fine-tuned* on the SROIE training set.")

# To run the training and evaluation:
if __name__ == "__main__":
    import asyncio
    
    # IMPORTANT:
    # 1. Ensure you have the SROIE 2019 dataset downloaded and extracted
    #    according to the `SROIE_ROOT` path structure described above,
    #    including both 'train' and 'test' splits.
    # 2. Install required libraries: `pip install transformers datasets evaluate Pillow thefuzz accelerate`
    
    # For initial testing/debugging, use a small number of samples (e.g., 10-20 for train/eval).
    # This will run much faster.
    asyncio.run(train_and_evaluate_layoutlmv3_sroie(num_samples_train=50, num_samples_eval=100)) 
    
    # To run on the entire SROIE 2019 dataset (will take significant time and GPU resources):
    # asyncio.run(train_and_evaluate_layoutlmv3_sroie())