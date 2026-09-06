#!/usr/bin/env python3
"""Validated QLoRA SFT entry point for chat-format JSONL datasets."""
import argparse
import json
from pathlib import Path
import sys
import yaml

REQUIRED = {"model_name", "train_file", "valid_file", "output_dir"}

def load_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("messages"), list) or len(row["messages"]) < 2:
            raise ValueError(f"{path}:{number}: expected messages list")
        if any({"role", "content"} - set(message) for message in row["messages"]):
            raise ValueError(f"{path}:{number}: every message needs role and content")
        rows.append(row)
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    missing = REQUIRED - set(cfg)
    if missing:
        raise ValueError(f"Config missing: {sorted(missing)}")
    train, valid = load_jsonl(cfg["train_file"]), load_jsonl(cfg["valid_file"])
    if not train or not valid:
        raise ValueError("Train and validation splits must both be non-empty")
    print(json.dumps({"train_rows": len(train), "valid_rows": len(valid), "model": cfg["model_name"]}, indent=2))
    if args.dry_run:
        return
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        sys.exit(f"Missing training dependency: {exc}. Install pip install -e '.[train]'.")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    def render(row):
        return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
    train_ds = Dataset.from_list([{"text": render(row)} for row in train])
    valid_ds = Dataset.from_list([{"text": render(row)} for row in valid])
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], quantization_config=bnb, torch_dtype=torch.bfloat16, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    peft = LoraConfig(r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32), lora_dropout=cfg.get("lora_dropout", 0.05), target_modules=cfg.get("target_modules"), bias="none", task_type="CAUSAL_LM")
    train_args = SFTConfig(output_dir=cfg["output_dir"], dataset_text_field="text", max_length=cfg.get("sequence_length", 2048), per_device_train_batch_size=cfg.get("micro_batch_size", 1), per_device_eval_batch_size=1, gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16), learning_rate=cfg.get("learning_rate", 2e-4), num_train_epochs=cfg.get("epochs", 2), warmup_ratio=cfg.get("warmup_ratio", 0.03), bf16=True, gradient_checkpointing=True, logging_steps=10, eval_strategy="steps", eval_steps=cfg.get("eval_steps", 100), save_steps=cfg.get("save_steps", 100), save_total_limit=2, report_to=cfg.get("report_to", "none"))
    trainer = SFTTrainer(model=model, args=train_args, train_dataset=train_ds, eval_dataset=valid_ds, processing_class=tokenizer, peft_config=peft)
    trainer.train()
    trainer.save_model(cfg["output_dir"])

if __name__ == "__main__":
    main()
