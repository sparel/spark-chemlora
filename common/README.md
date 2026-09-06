# Shared training and evaluation

`train_qlora.py` validates configuration and JSONL chat schema before loading GPU dependencies. It uses 4-bit NF4 QLoRA, BF16 compute, gradient checkpointing, and response-only loss masking where the installed TRL version supports it.

`evaluate_json_outputs.py` compares JSONL predictions against references for parse rate, required fields, exact scalar fields, and list F1. Extend per-experiment scoring rather than pretending generic string similarity measures scientific quality.
