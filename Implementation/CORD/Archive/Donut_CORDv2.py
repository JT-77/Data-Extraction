import re
import json
import torch
from datasets import load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel

# 1. Load the CORD-v2 dataset (using the test split of 100 receipts)
dataset = load_dataset("naver-clova-ix/cord-v2", split="test")
dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(10))

# 2. Initialize Donut processor and model (fine-tuned on CORD-v2)
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Prepare the decoder prompt for CORD-v2 (task start token)
task_prompt = "<s_cord-v2>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

# Function to flatten a nested dictionary of predictions/annotations into key-value pairs
def flatten_dict(d, parent_key=""):
    pairs = []
    for k, v in d.items():
        # Compose full key path
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            # Recurse into sub-dictionary
            pairs.extend(flatten_dict(v, parent_key=new_key))
        elif isinstance(v, list):
            # For list, flatten each element (assuming elements are dicts or primitives)
            for item in v:
                if isinstance(item, dict):
                    pairs.extend(flatten_dict(item, parent_key=new_key))
                else:
                    # Primitive item in list
                    pairs.append((new_key, str(item)))
        else:
            # Base case: primitive value
            pairs.append((new_key, str(v)))
    return pairs

# 3-9. Iterate over samples, generate predictions, and compute metrics
overall_correct = 0
overall_predicted = 0
overall_actual = 0

print("Per-sample precision, recall, F1:")
for idx in range(len(dataset)):  # should be 100 samples
    example = dataset[idx]
    image = example["image"]
    gt_data = json.loads(example["ground_truth"])
    gt_fields = gt_data.get("gt_parse", gt_data)  # use gt_parse if available
    
    # Prepare image input
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    # Generate output sequence from the model
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        use_cache=True
    )
    # Decode the sequence to JSON string
    seq = processor.batch_decode(outputs)[0]
    # Remove special tokens and prompt token
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token like <s_cord-v2>
    # Convert to Python dict
    pred_fields = processor.token2json(seq)
    
    # Flatten ground truth and prediction into lists of (key, value) pairs
    gt_pairs = flatten_dict(gt_fields)
    pred_pairs = flatten_dict(pred_fields)
    # Count total pairs
    num_gt = len(gt_pairs)
    num_pred = len(pred_pairs)
    # Count correct matches (exact key & value matches)
    gt_pair_counts = {}
    for pair in gt_pairs:
        gt_pair_counts[pair] = gt_pair_counts.get(pair, 0) + 1
    correct = 0
    # Use a copy of ground truth counts to avoid matching one pair multiple times
    gt_count_copy = gt_pair_counts.copy()
    for pair in pred_pairs:
        if pair in gt_count_copy and gt_count_copy[pair] > 0:
            correct += 1
            gt_count_copy[pair] -= 1
    # Compute precision, recall, F1 for this sample
    precision = correct / num_pred if num_pred > 0 else 0.0
    recall = correct / num_gt if num_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    # Print sample metrics
    print(f"Sample {idx:3d}: Precision = {precision:.3f}, Recall = {recall:.3f}, F1 = {f1:.3f}")
    # Accumulate for overall metrics
    overall_correct += correct
    overall_predicted += num_pred
    overall_actual += num_gt

# Compute aggregate (micro-averaged) precision, recall, F1
overall_precision = overall_correct / overall_predicted if overall_predicted > 0 else 0.0
overall_recall = overall_correct / overall_actual if overall_actual > 0 else 0.0
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

print("\nOverall performance on 100 receipts:")
print(f"Precision = {overall_precision:.3f}, Recall = {overall_recall:.3f}, F1-score = {overall_f1:.3f}")