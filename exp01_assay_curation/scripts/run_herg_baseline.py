#!/usr/bin/env python3
"""Run a sampled prompting-only baseline against an OpenAI-compatible local endpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import requests


def rows(path, limit):
    result = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            result.append(json.loads(line))
            if len(result) >= limit:
                break
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="exp01_assay_curation/data/processed/herg_curation_v2/test.jsonl")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible server URL, excluding /chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="exp01_assay_curation/evals/baseline_outputs.jsonl")
    parser.add_argument("--metadata")
    parser.add_argument("--server-revision", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--api-key", default="local")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")

    input_path = Path(args.input)
    selected = rows(input_path, args.limit)
    if not selected:
        raise ValueError("No baseline input rows")
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    output_path = Path(args.output)
    metadata_path = Path(args.metadata) if args.metadata else output_path.with_suffix(".metadata.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    models = requests.get(args.base_url.rstrip("/").rsplit("/v1", 1)[0] + "/v1/models", timeout=30)
    models.raise_for_status()
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "base_url": args.base_url, "server_revision": args.server_revision,
        "dataset": str(input_path), "dataset_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "requested_rows": args.limit, "workers": args.workers, "temperature": 0,
        "response_format": "json_object", "max_tokens": args.max_tokens, "server_models": models.json(),
    }

    def infer(item):
        request = {"model": args.model, "messages": item["messages"][:2], "temperature": 0,
                   "max_tokens": args.max_tokens, "response_format": {"type": "json_object"}}
        response = requests.post(endpoint, headers={"Authorization": f"Bearer {args.api_key}"}, json=request, timeout=180)
        response.raise_for_status()
        return {"id": item["id"], "prediction": response.json()["choices"][0]["message"]["content"], "model": args.model}

    with output_path.open("w") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(infer, item) for item in selected]
        for future in as_completed(futures):
            handle.write(json.dumps(future.result()) + "\n")
            handle.flush()
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["written_rows"] = len(selected)
    metadata["output"] = str(output_path)
    metadata["output_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"written_rows": len(selected), "output": str(output_path), "metadata": str(metadata_path)}, indent=2))


if __name__ == "__main__":
    main()
