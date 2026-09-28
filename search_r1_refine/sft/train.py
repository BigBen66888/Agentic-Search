"""Supervised fine-tuning on teacher trajectories.

Only assistant tokens contribute to the loss: the user turns carry the question
and the retrieved ``<information>`` observations, so training on them would
teach the model to hallucinate evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List

from search_r1_refine.data.schema import read_jsonl


def encode_messages(tokenizer, messages: List[Dict[str, str]], max_length: int) -> Dict[str, List[int]]:
    input_ids: List[int] = []
    labels: List[int] = []
    for message in messages:
        text = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False)
        segment = tokenizer(text, add_special_tokens=False)["input_ids"]
        if input_ids and segment and tokenizer.bos_token_id is not None and segment[0] == tokenizer.bos_token_id:
            segment = segment[1:]
        if message["role"] == "assistant":
            labels.extend(segment)
        else:
            labels.extend([-100] * len(segment))
        input_ids.extend(segment)
    eos = tokenizer.eos_token_id
    if eos is not None:
        input_ids.append(eos)
        labels.append(eos)
    input_ids, labels = input_ids[:max_length], labels[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


class SFTDataset:
    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features: List[Dict[str, List[int]]] = []
        skipped = 0
        for row in self.rows:
            messages = row.get("messages") or []
            if not messages:
                skipped += 1
                continue
            feature = encode_messages(tokenizer, messages, max_length)
            if sum(1 for x in feature["labels"] if x != -100) == 0:
                skipped += 1
                continue
            self.features.append(feature)
        print(f"[Search-R1] SFT 样本 {len(self.features)} 条，跳过 {skipped} 条", flush=True)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.features[index]


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, List[List[int]]]:
        width = max(len(x["input_ids"]) for x in batch)
        input_ids, labels, attention = [], [], []
        for item in batch:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append(item["attention_mask"] + [0] * pad)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def train(args) -> Dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = SFTDataset(args.data, tokenizer, args.max_length)
    if len(dataset) == 0:
        raise RuntimeError(f"没有可用的 SFT 样本：{args.data}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation=args.attn_impl, trust_remote_code=True)
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=2,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[],
        ddp_find_unused_parameters=False,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=Collator(tokenizer.pad_token_id))
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    stats = {"samples": len(dataset), "epochs": args.epochs, "output_dir": args.output_dir,
             "base_model": args.model}
    with open(os.path.join(args.output_dir, "sft_train_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SFT cold start")
    p.add_argument("--model", required=True, help="base model, e.g. Qwen/Qwen2.5-3B")
    p.add_argument("--data", required=True, help="sft_train.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-strategy", default="epoch")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    return p


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
