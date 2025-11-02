
from __future__ import annotations
"""
Fine-tune Qwen2.5-VL on DocVQA with QLoRA and evaluate (EM/F1/ANLS). Exports a LoRA adapter + optional Ollama Modelfile.

Usage examples
--------------
# Train on ~600 examples, 4-bit QLoRA, and export an Ollama Modelfile
python mew.py --use_bnb_4bit --train_samples 600 --eval_samples 100 --export_ollama

# Evaluate a saved adapter (skip training)
python mew.py --do_train false --do_eval true --output_dir outputs/qwen2_5vl_docvqa_lora
"""

import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Dataset as HFDataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info


@dataclass
class VqaRecord:
    image: Any
    question: str
    answer: str


def _first_non_empty(*values):
    for v in values:
        if v is not None:
            if isinstance(v, str) and v.strip() == "":
                continue
            return v
    return None


def normalize_sample(example: Dict[str, Any]) -> VqaRecord:
    image = _first_non_empty(example.get("image"), example.get("image_path"))
    question = _first_non_empty(example.get("question"), example.get("query"), "")
    ans = None
    for key in ("answers", "answer", "label"):
        if key in example and example[key] is not None:
            ans = example[key]
            break
    if isinstance(ans, list):
        answer = str(ans[0]) if len(ans) else ""
    else:
        answer = str(ans) if ans is not None else ""
    return VqaRecord(image=image, question=str(question), answer=answer)


class DocVQADataset(Dataset):
    def __init__(self, hf_dataset: HFDataset):
        self.records: List[VqaRecord] = [normalize_sample(hf_dataset[i]) for i in range(len(hf_dataset))]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> VqaRecord:
        return self.records[idx]


class QwenVLMDataCollator:
    def __init__(self, processor: Qwen2VLProcessor, system_prompt: str | None = None, add_system: bool = True):
        self.processor = processor
        self.system_prompt = system_prompt or (
            "You are a precise document understanding assistant. Answer with a short span (word/number/date)."
        )
        self.add_system = add_system

    def __call__(self, batch: List[VqaRecord]) -> Dict[str, torch.Tensor]:
        msgs_batch: List[List[Dict[str, Any]]] = []
        images_batch: List[List[Any]] = []
        for rec in batch:
            msgs = []
            if self.add_system and self.system_prompt:
                msgs.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
            msgs.append({"role": "user", "content": [{"type": "image", "image": rec.image}, {"type": "text", "text": rec.question}]})
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": rec.answer}]})
            msgs_batch.append(msgs)
            img_inputs, _ = process_vision_info(msgs)
            images_batch.append(img_inputs)
        prompts_text = [self.processor.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=False) for m in msgs_batch]
        full_texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in msgs_batch]
        model_inputs = self.processor(text=full_texts, images=images_batch, return_tensors="pt", padding=True)
        prompt_lens: List[int] = []
        for t in prompts_text:
            ids = self.processor.tokenizer(text=t, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            prompt_lens.append(ids.numel())
        input_ids = model_inputs["input_ids"]
        labels = input_ids.clone()
        for i, n_prompt in enumerate(prompt_lens):
            labels[i, :n_prompt] = -100
        model_inputs["labels"] = labels
        return model_inputs


def build_model_and_processor(model_id: str, use_4bit: bool):
    quant = None
    if use_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant,
    )
    processor = Qwen2VLProcessor.from_pretrained(model_id)
    return model, processor


def apply_lora(model: Qwen2VLForConditionalGeneration, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    peft_config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    return get_peft_model(model, peft_config)


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize_text(pred) == _normalize_text(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    p_tokens = _normalize_text(pred).split()
    g_tokens = _normalize_text(gold).split()
    if len(p_tokens) == 0 and len(g_tokens) == 0:
        return 1.0
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return 0.0
    common = {}
    for t in p_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    g_count = {}
    for t in g_tokens:
        g_count[t] = g_count.get(t, 0) + 1
    for t, c in common.items():
        overlap += min(c, g_count.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def _levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[lb]


def anls(pred: str, gold: str) -> float:
    p = _normalize_text(pred)
    g = _normalize_text(gold)
    if len(p) == 0 and len(g) == 0:
        return 1.0
    denom = max(len(g), len(p), 1)
    dist = _levenshtein(p, g)
    return max(0.0, 1.0 - dist / denom)


def generate_answer(model, processor, rec: VqaRecord, device: torch.device, system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": [{"type": "image", "image": rec.image}, {"type": "text", "text": rec.question}]})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=[image_inputs], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    tail = gen[:, inputs["input_ids"].shape[1]:]
    out = processor.batch_decode(tail, skip_special_tokens=True)[0]
    return out.strip()


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL DocVQA fine-tune + eval + Ollama export")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--dataset_id", default="lmms-lab/DocVQA")
    parser.add_argument("--train_samples", type=int, default=600)
    parser.add_argument("--eval_samples", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per_device_train_bs", type=int, default=2)
    parser.add_argument("--per_device_eval_bs", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--use_bnb_4bit", action="store_true")
    parser.add_argument("--output_dir", default="outputs/qwen2_5vl_docvqa_lora")
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", default=None)
    parser.add_argument("--export_ollama", action="store_true")
    parser.add_argument("--no_system", action="store_true")
    parser.add_argument("--system_prompt", default="You are a precise document understanding assistant. Answer concisely.")
    parser.add_argument("--do_train", type=lambda x: str(x).lower() != "false", default=True)
    parser.add_argument("--do_eval", type=lambda x: str(x).lower() != "false", default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print("Device:", device)

    ds = load_dataset(args.dataset_id, 'DocVQA')
    if hasattr(ds, "keys"):
        split_name = "train" if "train" in ds else ("validation" if "validation" in ds else list(ds.keys())[0])
        base = ds[split_name]
    else:
        base = ds
    print(f"Loaded {args.dataset_id} split with {len(base)} rows")

    train_hf = base.select(range(min(args.train_samples, len(base)))) if args.do_train else None
    eval_hf = base.select(range(min(args.eval_samples, len(base)))) if args.do_eval else None

    model, processor = build_model_and_processor(args.model_id, args.use_bnb_4bit)

    if args.do_train:
        model = apply_lora(model)

    collator = QwenVLMDataCollator(processor, add_system=(not args.no_system))
    train_ds = DocVQADataset(train_hf) if train_hf is not None else None
    eval_ds = DocVQADataset(eval_hf) if eval_hf is not None else None

    if args.do_train and train_ds is not None:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_bs,
            per_device_eval_batch_size=args.per_device_eval_bs,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            logging_steps=10,
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=50,
            save_steps=200,
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            fp16=False,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            report_to=([] if not args.wandb_project else ["wandb"]),
            lr_scheduler_type="constant",
        )
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
        )
        trainer.train()

        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print("Saved LoRA adapter & processor to:", args.output_dir)

        if args.push_to_hub:
            repo_id = args.hub_model_id or os.path.basename(args.output_dir.rstrip("/"))
            print("Pushing adapter to Hub repo:", repo_id)
            trainer.model.push_to_hub(repo_id)
            processor.push_to_hub(repo_id)

    if args.do_eval and eval_ds is not None:
        model.eval()
        model.to(device)
        preds: List[str] = []
        golds: List[str] = []
        for rec in eval_ds:
            with torch.no_grad():
                pred = generate_answer(model, processor, rec, device, system_prompt=(None if args.no_system else args.system_prompt))
            preds.append(pred)
            golds.append(rec.answer)

        ems, f1s, anls_vals = [], [], []
        for p, g in zip(preds, golds):
            ems.append(exact_match(p, g))
            f1s.append(f1_score(p, g))
            anls_vals.append(anls(p, g))
        em_mean = sum(ems) / max(1, len(ems))
        f1_mean = sum(f1s) / max(1, len(f1s))
        anls_mean = sum(anls_vals) / max(1, len(anls_vals))
        print({"EM": round(em_mean, 4), "F1": round(f1_mean, 4), "ANLS": round(anls_mean, 4)})

        results_dir = os.path.join(args.output_dir, "eval")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
            for p, g in zip(preds, golds):
                f.write(json.dumps({"pred": p, "gold": g}, ensure_ascii=False) + "\n")
        with open(os.path.join(results_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"EM": em_mean, "F1": f1_mean, "ANLS": anls_mean}, f, indent=2)
        print("Saved eval to:", results_dir)

    if args.export_ollama:
        modelfile_path = os.path.join(os.path.dirname(args.output_dir), "Modelfile")
        base_ollama = "qwen2.5vl:latest"
        mf = f"""
FROM {base_ollama}
ADAPTER {os.path.abspath(args.output_dir)}
SYSTEM You are a precise document understanding assistant. Answer concisely with the exact span from the document when possible.
TEMPLATE \n{{{{ .Prompt }}}}\n
PARAMETER temperature 0
""".strip()
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(mf)
        print("Wrote Ollama Modelfile:", modelfile_path)
        print("Build with: ollama create qwen2.5vl-docvqa -f", modelfile_path)


if __name__ == "__main__":
    main()
