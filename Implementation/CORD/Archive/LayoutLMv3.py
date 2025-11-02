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

# Suppress specific future warnings from PIL
warnings.filterwarnings("ignore", category=FutureWarning, module="PIL")

# --- Configuration ---
MODEL_NAME = "microsoft/layoutlmv3-large"
OUTPUT_DIR = "./layoutlmv3_cord_v2_results"
LOGGING_DIR = "./layoutlmv3_cord_v2_logs"

# These will be populated dynamically from the dataset's ground_truth
MODEL_ID2LABEL = {}
MODEL_LABEL2ID = {}
ALL_BASE_LABELS = set() # To collect all unique base labels

# Initialize the LayoutLMv3 processor. CORD provides words and boxes, so no OCR needed.
processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False)

# --- 1. Load Dataset ---
def load_cord_v2_dataset():
    """
    Loads the CORD v2 dataset from Naver Clova IX on Hugging Face Hub.
    """
    print("Loading CORD v2 dataset from Hugging Face Hub (naver-clova-ix/cord-v2)...")
    dataset = load_dataset("naver-clova-ix/cord-v2")
    
    print("CORD v2 Dataset loaded successfully.")
    print(dataset)
    return dataset

# --- Helper Function to Extract Words, Bboxes, and Labels from ground_truth ---
def extract_words_bboxes_and_labels(ground_truth_json_str, image_width, image_height):
    """
    Parses the 'ground_truth' JSON string to extract words, bounding boxes,
    and assigns IOB2 labels based on the 'category' field.
    """
    data = json.loads(ground_truth_json_str)
    
    words = []
    bboxes = []
    ner_tags = [] # Will store integer IDs for IOB2 tags

    # Iterate through each 'valid_line' entry (which represents a semantic entity or group)
    for line_data in data["valid_line"]:
        category = line_data["category"] # e.g., "menu.nm", "total.total_price"
        
        # Add the base category to our global set for dynamic label mapping
        ALL_BASE_LABELS.add(category)

        # Process words within this line_data
        for i, word_info in enumerate(line_data["words"]):
            word_text = word_info["text"]
            
            # The 'quad' format is x1,y1,x2,y2,x3,y3,x4,y4 (8 coordinates)
            # We need to convert it to [x_min, y_min, x_max, y_max]
            quad = word_info["quad"]
            x_coords = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
            y_coords = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
            
            # Get min/max for 4-coordinate bbox
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            
            # Normalize bbox to 0-1000 range
            # Note: image_width/height are passed from the main preprocessing function
            normalized_bbox = [
                int(1000 * (x_min / image_width)),
                int(1000 * (y_min / image_height)),
                int(1000 * (x_max / image_width)),
                int(1000 * (y_max / image_height))
            ]
            
            # Clamp coordinates to ensure they are strictly within 0-1000
            normalized_bbox = [max(0, min(1000, coord)) for coord in normalized_bbox]
            
            # Ensure min <= max; if a box becomes invalid after clamping
            if normalized_bbox[0] > normalized_bbox[2]: normalized_bbox[0], normalized_bbox[2] = normalized_bbox[2], normalized_bbox[0]
            if normalized_bbox[1] > normalized_bbox[3]: normalized_bbox[1], normalized_bbox[3] = normalized_bbox[3], normalized_bbox[1]


            words.append(word_text)
            bboxes.append(normalized_bbox)
            
            # Assign IOB2 tag based on position within the sequence of words for this category
            if i == 0: # First word of the entity
                ner_tags.append(f"B-{category.upper()}")
            else: # Subsequent words of the entity
                ner_tags.append(f"I-{category.upper()}")
    
    return words, bboxes, ner_tags

# --- 2. Preprocessing Function for Dataset.map ---
def preprocess_function(examples):
    """
    Preprocesses a batch of examples for LayoutLMv3 token classification.
    Handles image loading, tokenization, bounding box normalization, 
    and aligning word labels to token labels (IOB2 format).
    """
    batch_images = examples["image"]
    batch_ground_truth_strs = examples["ground_truth"]

    all_words = []
    all_bboxes = []
    all_ner_tags = [] # These will be IOB2 strings (e.g., "B-MENU.NM")

    for i in range(len(batch_images)):
        image = batch_images[i].convert("RGB")
        image_width, image_height = image.size
        ground_truth_str = batch_ground_truth_strs[i]

        words_on_page, bboxes_on_page, ner_tags_on_page_str = \
            extract_words_bboxes_and_labels(ground_truth_str, image_width, image_height)
        
        # Convert string IOB2 tags to integer IDs using the global MODEL_LABEL2ID
        ner_tags_on_page_ids = [MODEL_LABEL2ID.get(tag, MODEL_LABEL2ID["O"]) for tag in ner_tags_on_page_str]

        all_words.append(words_on_page)
        all_bboxes.append(bboxes_on_page)
        all_ner_tags.append(ner_tags_on_page_ids)

    # The processor will take care of creating token-level labels from word-level labels.
    encoding = processor(
        batch_images, # Pass PIL images directly
        text=all_words, # Pass list of lists of words
        boxes=all_bboxes, # Pass list of lists of normalized bboxes
        word_labels=all_ner_tags, # Pass list of lists of integer IOB2 labels
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    return encoding

# --- 3. Define Compute Metrics Function for NER ---
def compute_metrics(p):
    """
    Computes NER metrics (Precision, Recall, F1-score) using seqeval.
    Uses the global MODEL_ID2LABEL mapping.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (-100) and convert IDs back to labels (IOB2 format)
    true_predictions = [
        [MODEL_ID2LABEL[p_id] for (p_id, l_id) in zip(pred, lab) if l_id != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [MODEL_ID2LABEL[l_id] for (p_id, l_id) in zip(pred, lab) if l_id != -100]
        for pred, lab in zip(predictions, labels)
    ]
    
    # Calculate metrics
    results = {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }
    # print("\nClassification Report:\n", classification_report(true_labels, true_predictions)) # Uncomment for detailed report

    return results

# --- Main Training and Evaluation Function ---
async def train_and_evaluate_layoutlmv3_cord():
    print("--- Starting LayoutLMv3 training and evaluation for CORD v2 ---")

    # 1. Load Dataset
    dataset = load_cord_v2_dataset()

    # --- IMPORTANT: Dynamically build MODEL_ID2LABEL and MODEL_LABEL2ID ---
    # We need to iterate through a portion of the dataset to collect all unique categories
    # to build our label mappings.
    print("\nCollecting unique labels from CORD v2 dataset for dynamic label mapping...")
    # Use a subset of the dataset to collect labels efficiently
    for split in ["train", "validation", "test"]:
        # Iterate over the first few examples or all if dataset is small
        for i in range(min(len(dataset[split]), 200)): # Check first 200 samples of each split
            ground_truth_str = dataset[split][i]["ground_truth"]
            image_width = dataset[split][i]["image"].size[0]
            image_height = dataset[split][i]["image"].size[1]
            
            # This call will populate the global ALL_BASE_LABELS set
            extract_words_bboxes_and_labels(ground_truth_str, image_width, image_height)
    
    # Now build the final IOB2 labels from the collected base labels
    global MODEL_ID2LABEL, MODEL_LABEL2ID # Modify global variables
    
    # Sort base labels for consistent ID assignment
    sorted_base_labels = sorted(list(ALL_BASE_LABELS))
    
    # Build the full IOB2 label list
    all_iob2_labels = ["O"] # Start with 'O'
    for base_label in sorted_base_labels:
        all_iob2_labels.append(f"B-{base_label.upper()}")
        all_iob2_labels.append(f"I-{base_label.upper()}")
    
    MODEL_ID2LABEL = {i: label for i, label in enumerate(all_iob2_labels)}
    MODEL_LABEL2ID = {label: i for i, label in enumerate(all_iob2_labels)}
    
    print(f"\nDynamically built CORD v2 Dataset Labels:")
    print(f"  ID2LABEL: {MODEL_ID2LABEL}")
    print(f"  LABEL2ID: {MODEL_LABEL2ID}")
    print(f"  Number of labels: {len(MODEL_ID2LABEL)}")
    # --- END IMPORTANT ---

    # 2. Initialize Processor and Model (using the now correctly defined labels)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, apply_ocr=False) 
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=len(MODEL_ID2LABEL), 
        id2label=MODEL_ID2LABEL, 
        label2id=MODEL_LABEL2ID
    )
    print("Model loaded successfully. The classification head is re-initialized for the CORD v2 labels.")

    # 3. Preprocess Dataset
    print("Applying LayoutLMv3 tokenization and label alignment to datasets...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        # Remove original columns as they are no longer needed after preprocessing
        remove_columns=['image', 'ground_truth'], 
        desc="Processing documents",
    )
    
    print(f"Loaded {len(tokenized_dataset['train'])} training samples, {len(tokenized_dataset['validation'])} validation samples, and {len(tokenized_dataset['test'])} test samples.")

    # 4. Define TrainingArguments
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

    # 5. Initialize Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"], 
        tokenizer=processor, 
        compute_metrics=compute_metrics,
    )

    # 6. Train the Model
    print("\nStarting model training...")
    train_result = trainer.train()
    print("Training complete!")

    # 7. Evaluate the Model on the Test Set (final evaluation)
    print("\nEvaluating the trained model on the test set (final evaluation)...")
    metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print("Test Set Evaluation Results (Best Model):")
    print(metrics)

    print("\n--- CORD v2 evaluation with LayoutLMv3 finished! ---")

# --- Run the script ---
if __name__ == "__main__":
    import asyncio
    
    # Ensure you have a CUDA-enabled GPU for best performance.
    # If you get OOM errors, reduce batch sizes or use gradient_accumulation_steps.
    # Set logging to INFO to see more details: transformers.logging.set_verbosity_info()
    asyncio.run(train_and_evaluate_layoutlmv3_cord())

