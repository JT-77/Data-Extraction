import os
import json
from PIL import Image
import pytesseract
from difflib import SequenceMatcher
from transformers import AutoProcessor, AutoModelForTokenClassification

# Paths (assuming dataset structure as given)
images_dir = "../datasets/SROIE2019/test/img/"
entities_dir = "../datasets/SROIE2019/test/entities/"

# Load LayoutLMv3 model and processor (fine-tuned on SROIE if available, otherwise base model)
model_name = "Theivaprakasham/layoutlmv3-finetuned-sroie"  # Use fine-tuned model on SROIE [oai_citation:7‡huggingface.co](https://huggingface.co/Theivaprakasham/layoutlmv3-finetuned-sroie#:~:text=This%20model%20is%20a%20fine,results%20on%20the%20evaluation%20set)
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Helper: fuzzy similarity ratio (0-100)
def fuzzy_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio() * 100

# Initialize counters
total_tp = total_fp = total_fn = 0
sample_metrics = []  # to store P, R, F1 for each sample

# Iterate over each image file in the test set
for filename in sorted(os.listdir(images_dir)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
        continue  # skip non-image files

    img_path = os.path.join(images_dir, filename)
    base_name, _ = os.path.splitext(filename)
    gt_path = os.path.join(entities_dir, base_name + ".txt")
    if not os.path.exists(gt_path):
        continue  # skip if no ground truth file (should not happen in SROIE test set)

    # Load image and perform OCR
    image = Image.open(img_path).convert("RGB")
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    boxes = []
    # Iterate over OCR results (level=5 gives word-level boxes)
    n_boxes = len(ocr_data["text"])
    for i in range(n_boxes):
        word = ocr_data["text"][i]
        conf = int(ocr_data["conf"][i])
        if conf > 0 and word.strip():  # consider confident, non-empty words
            x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
            # Tesseract gives upper-left corner (x,y) and width, height. Convert to (x1,y1,x2,y2) box.
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            # Normalize coordinates to 0-1000 (as expected by LayoutLM models)
            # Use image size to scale
            W, H = image.size
            norm_box = [
                int((x1 / W) * 1000),
                int((y1 / H) * 1000),
                int((x2 / W) * 1000),
                int((y2 / H) * 1000),
            ]
            words.append(word)
            boxes.append(norm_box)

    # Use processor to tokenize and encode inputs for the model
    encoding = processor(images=image, text=words, boxes=boxes, return_tensors="pt", truncation=True)
    # Perform inference
    outputs = model(**encoding)
    logits = outputs.logits  # shape: (1, seq_len, num_classes)
    predicted_ids = logits.argmax(dim=-1).squeeze().tolist()  # predicted token class indices

    # Map predicted ids to labels
    id2label = model.config.id2label  # e.g., {0: "O", 1: "B-ADDR", 2: "I-ADDR", ...}
    predicted_labels = [id2label[t] for t in predicted_ids]

    # Aggregate tokens into predicted field strings
    predicted_fields = {"company": "", "address": "", "date": "", "total": ""}
    word_ids = encoding.word_ids()  # map tokens to original word indices
    last_label = None
    last_field = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue  # skip special tokens
        label = predicted_labels[idx]
        if label == "O":
            continue  # not a field token
        # Remove BIO prefix if present
        if "-" in label:
            _, field_tag = label.split("-", 1)  # e.g., "B-ADDRESS" -> field_tag="ADDRESS"
        else:
            field_tag = label
        field_tag = field_tag.lower()
        # If starting a new field sequence
        if predicted_fields[field_tag] == "" or field_tag != last_field or label.startswith("B-"):
            # If already had a predicted value for this field and now another segment starts,
            # we append a space (assuming continuation) 
            if predicted_fields[field_tag] != "":
                predicted_fields[field_tag] += " "
            predicted_fields[field_tag] += words[word_idx]
        else:
            # Continuation (I- tag) of the same field
            predicted_fields[field_tag] += " " + words[word_idx]
        last_label = label
        last_field = field_tag

    # Read ground truth JSON for this receipt
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    # Normalize ground truth field strings (strip and lower-case)
    gt_fields = {k: (v.strip() if isinstance(v, str) else "") for k, v in gt_data.items()}
    # Also lower-case for fair comparison
    gt_fields = {k: v.lower() for k, v in gt_fields.items()}

    # Evaluate predictions for this sample
    correct = 0
    predicted_count = 0
    for field, pred_value in predicted_fields.items():
        pred_value = pred_value.strip().lower()
        if pred_value:  # if we predicted something for this field
            predicted_count += 1
            # Compute fuzzy match with ground truth
            score = fuzzy_ratio(pred_value, gt_fields.get(field, ""))
            if score >= 80:
                correct += 1
    actual_count = len(gt_fields)  # should be 4 in SROIE
    # Calculate precision, recall, F1 for this sample
    # (Avoid division by zero for precision if predicted_count is 0)
    precision = correct / predicted_count if predicted_count > 0 else 1.0 if correct == 0 else 0.0
    recall = correct / actual_count if actual_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    sample_metrics.append((filename, precision, recall, f1))

    # Update aggregate counts for overall metrics (treat each field as one item)
    total_tp += correct
    total_fp += (predicted_count - correct)
    total_fn += (actual_count - correct)

    # Print detailed results for this receipt
    print(f"{filename}:  Precision={precision*100:.1f}%,  Recall={recall*100:.1f}%,  F1={f1*100:.1f}%")
    for field in ["company", "address", "date", "total"]:
        pred_text = predicted_fields[field].strip()
        gt_text = gt_fields.get(field, "").strip()
        match = "OK" if pred_text and fuzzy_ratio(pred_text.lower(), gt_text.lower()) >= 80 else "ERR"
        print(f"  - {field.title():8}: Predicted='{pred_text}' | Actual='{gt_text}' => {match}")
    print("-" * 50)

# Compute overall Precision, Recall, F1
overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

print(f"\nOverall Precision: {overall_precision*100:.2f}%")
print(f"Overall Recall   : {overall_recall*100:.2f}%")
print(f"Overall F1-score : {overall_f1*100:.2f}%")

# (Optional) Average per-sample F1
avg_sample_f1 = sum(f for _, _, _, f in sample_metrics) / len(sample_metrics)
print(f"Average F1 (averaged across samples): {avg_sample_f1*100:.2f}%")