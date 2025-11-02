import os, re, json, glob
import torch
from PIL import Image
from difflib import SequenceMatcher
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Shuffle and select images
import random

# Load the fine-tuned Donut model and processor for SROIE
model_name = "sam749/donut-base-finetuned-sroie-v2"  # Hugging Face model fine-tuned on SROIE
processor = DonutProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

# Use GPU if available for faster inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Prepare generation settings (special start token and generation config)
task_prompt = "<s>"  # start of sequence token for document parsing task
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

max_length = model.decoder.config.max_position_embeddings  # maximum sequence length for generation
pad_token_id = processor.tokenizer.pad_token_id
eos_token_id = processor.tokenizer.eos_token_id
bad_words_ids = [[processor.tokenizer.unk_token_id]]  # prevent generating <unk> tokens

# Define data directories for images and ground truth JSON files
img_dir = "../datasets/SROIE2019/test/img"
entity_dir = "../datasets/SROIE2019/test/entities"

# Collect all image file paths (assuming .jpg/.png files)
image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png"):
    image_paths.extend(glob.glob(os.path.join(img_dir, ext)))

image_paths = sorted(image_paths)
image_paths = image_paths[:50]

if not image_paths:
    print(f"No images found in {img_dir}. Please check the path.")
    exit()

# Fields to evaluate
target_fields = ["company", "date", "address", "total"]

# Counters for overall metrics
sum_precision = sum_recall = sum_f1 = 0.0
num_samples = 0

for img_path in image_paths:
    file_name = os.path.basename(img_path)
    # Find corresponding ground truth file (.txt or .json) by image name
    base_name = os.path.splitext(file_name)[0]
    gt_path = os.path.join(entity_dir, base_name + ".txt")
    if not os.path.exists(gt_path):
        gt_path = os.path.join(entity_dir, base_name + ".json")
    if not os.path.exists(gt_path):
        print(f"Ground truth for `{file_name}` not found, skipping.")
        continue

    # Load ground truth entities (JSON content expected in the file)
    with open(gt_path, 'r') as f:
        content = f.read().strip()
    try:
        gt_data = json.loads(content)  # parse JSON string directly
    except json.JSONDecodeError:
        # If JSON parsing fails, handle possible formatting issues
        gt_data = {}
        if content:
            if content.startswith("{") and content.endswith("}"):
                # Attempt to fix quotes if JSON uses single quotes
                fixed = content.replace("'", "\"")
                try:
                    gt_data = json.loads(fixed)
                except Exception:
                    gt_data = {}
            else:
                # Parse line-based key:value pairs (if the file is not in JSON format)
                for line in content.splitlines():
                    if ":" in line:
                        key, val = line.split(":", 1)
                        gt_data[key.strip().lower()] = val.strip()
    # Normalize ground truth keys to lowercase for consistency
    gt_data = {k.lower(): v for k, v in gt_data.items()}

    # Load and preprocess the image
    image = Image.open(img_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    # Run the model to generate a prediction (JSON string output)
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bad_words_ids=bad_words_ids,
            use_cache=True
        )
    # Decode the generated tokens to text
    sequence = processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    # Remove special tokens (e.g. <s> prompt and padding/eos tokens) from the sequence
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task tag like <s_...>

    # Convert the predicted sequence (JSON string) to a Python dict
    try:
        pred_data = processor.token2json(sequence)
    except Exception:
        # Fallback to direct JSON parsing if token2json fails
        try:
            pred_data = json.loads(sequence)
        except Exception:
            pred_data = {}
    pred_data = {k.lower(): v for k, v in pred_data.items()}  # normalize keys

    # Compare predicted vs actual for each field
    correct_fields = 0
    predicted_fields = 0
    for field in target_fields:
        pred_val = str(pred_data.get(field, ""))  # predicted value (as string)
        actual_val = str(gt_data.get(field, ""))  # ground truth value (as string)
        # Normalize values: lowercase and collapse whitespace for fair comparison
        pv_norm = re.sub(r"\s+", " ", pred_val).strip().lower()
        av_norm = re.sub(r"\s+", " ", actual_val).strip().lower()
        if field in ["company", "address"]:
            score = SequenceMatcher(None, pv_norm, av_norm).ratio() * 100
            match = score >= 75
        else:  # Use exact match for date and total
            score = 100.0 if pv_norm == av_norm else 0.0
            match = pv_norm == av_norm
        if pred_val != "":  # counted as a predicted field if model provided any value
            predicted_fields += 1
        if match:
            correct_fields += 1
        # Print detailed field comparison
        match_text = "Yes" if match else "No"
        print(f"{file_name} - {field}: Predicted=\"{pred_val}\" vs Actual=\"{actual_val}\" -> Match: {match_text} (score={score:.1f}%)")

    # If the model did not predict some fields, those are considered as empty predictions
    # (predicted_fields may be < 4 if some keys missing in pred_data)
    actual_fields = len(target_fields)  # total expected fields (4)
    # Calculate precision, recall, F1 for this sample
    precision = correct_fields / predicted_fields if predicted_fields > 0 else 0.0
    recall    = correct_fields / actual_fields if actual_fields > 0 else 0.0
    f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"{file_name} - Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1-score: {f1_score*100:.2f}%\n")

    # Accumulate metrics
    sum_precision += precision
    sum_recall += recall
    sum_f1 += f1_score
    num_samples += 1

# Compute and print average metrics across all processed samples
if num_samples > 0:
    avg_precision = sum_precision / num_samples
    avg_recall   = sum_recall / num_samples
    avg_f1       = sum_f1 / num_samples
    print(f"Average Precision: {avg_precision*100:.2f}%, Average Recall: {avg_recall*100:.2f}%, Average F1-score: {avg_f1*100:.2f}%")