#!/usr/bin/env python3
"""Score JSON output validity and field agreement from JSONL records."""
import argparse
import json
from pathlib import Path


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def parse(value):
    try:
        return json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError):
        return None


def reference_value(record):
    """Return a flattened reference, including assistant targets in chat JSONL."""
    for key in ("output", "reference"):
        if key in record:
            return record[key]
    for message in reversed(record.get("messages", [])):
        if message.get("role") == "assistant":
            return message.get("content")
    return None


def f1(pred, truth):
    p, t = set(pred or []), set(truth or [])
    hit = len(p & t)
    return 2 * hit / max(len(p) + len(t), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--references", required=True)
    ap.add_argument("--fields", nargs="+", required=True)
    args = ap.parse_args()
    pred = {r["id"]: parse(r.get("output", r.get("prediction"))) for r in records(args.predictions)}
    ref = {r["id"]: parse(reference_value(r)) for r in records(args.references)}
    common = sorted(pred.keys() & ref.keys())
    valid = [key for key in common if isinstance(pred[key], dict)]
    report = {"n": len(common), "valid_json_rate": len(valid) / max(len(common), 1), "fields": {}}
    for field in args.fields:
        scores = []
        for key in valid:
            if not isinstance(ref[key], dict) or field not in ref[key]:
                continue
            a, b = pred[key].get(field), ref[key].get(field)
            scores.append(f1(a, b) if isinstance(a, list) and isinstance(b, list) else float(a == b))
        report["fields"][field] = sum(scores) / max(len(scores), 1)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
