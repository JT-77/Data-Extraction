from transformers import VisionEncoderDecoderModel, DonutProcessor
from datasets import load_dataset
import torch
import re

# Load model and processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load CORD-v2 dataset (limit to 20 samples)
dataset = load_dataset("naver-clova-ix/cord-v2", split="test")
dataset = dataset.select(range(20))

task_prompt = "<s_cord-v2>"
prompt_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

def infer_and_parse(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=prompt_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        early_stopping=True,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]]
    )
    seq = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()
    return processor.token2json(seq)

# Evaluation
total_TP = total_FP = total_FN = 0

print("\nPer-sample metrics:")
print("Idx | Precision  Recall  F1-score")
print("---------------------------------")
for idx, sample in enumerate(dataset):
    gt = sample["ground_truth"]
    gt_dict = eval(gt) if isinstance(gt, str) else gt
    gt_parse = gt_dict.get("gt_parse", gt_dict)

    gt_fields = set()
    if "menu" in gt_parse:
        for item in gt_parse["menu"]:
            if not isinstance(item, dict):
                continue
            item_tuple = tuple((k, str(v)) for k, v in sorted(item.items()))
            gt_fields.add(("menu_item", item_tuple))
    if "sub_total" in gt_parse and "subtotal_price" in gt_parse["sub_total"]:
        gt_fields.add(("subtotal_price", str(gt_parse["sub_total"]["subtotal_price"])))
    if "total" in gt_parse and "total_price" in gt_parse["total"]:
        gt_fields.add(("total_price", str(gt_parse["total"]["total_price"])))
    if "sub_total" in gt_parse and "discount_price" in gt_parse["sub_total"]:
        gt_fields.add(("discount_price", str(gt_parse["sub_total"]["discount_price"])))

    pred = infer_and_parse(sample["image"])
    pred_fields = set()
    if "menu" in pred:
        pred_menu = pred["menu"] if isinstance(pred["menu"], list) else [pred["menu"]]
        for item in pred_menu:
            if not isinstance(item, dict):
                continue
            item_tuple = tuple((k, str(v)) for k, v in sorted(item.items()))
            pred_fields.add(("menu_item", item_tuple))
    if "sub_total" in pred and "subtotal_price" in pred["sub_total"]:
        pred_fields.add(("subtotal_price", str(pred["sub_total"]["subtotal_price"])))
    if "total" in pred and "total_price" in pred["total"]:
        pred_fields.add(("total_price", str(pred["total"]["total_price"])))
    if "sub_total" in pred and "discount_price" in pred["sub_total"]:
        pred_fields.add(("discount_price", str(pred["sub_total"]["discount_price"])))

    TP = len(gt_fields & pred_fields)
    FP = len(pred_fields - gt_fields)
    FN = len(gt_fields - pred_fields)
    total_TP += TP
    total_FP += FP
    total_FN += FN

    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0

    print(f"{idx:>3d} | {prec*100:6.2f}%   {rec*100:6.2f}%   {f1*100:6.2f}%")

overall_prec = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
overall_rec = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
overall_f1 = 2 * total_TP / (2 * total_TP + total_FP + total_FN) if (2 * total_TP + total_FP + total_FN) > 0 else 0.0

print("\nOverall metrics on 20 samples:")
print(f"Precision: {overall_prec*100:.2f}%,  Recall: {overall_rec*100:.2f}%,  F1-score: {overall_f1*100:.2f}%")