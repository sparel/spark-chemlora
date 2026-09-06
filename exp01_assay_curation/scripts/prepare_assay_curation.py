#!/usr/bin/env python3
"""Turn joined activity/assay CSV or parquet data into chat JSONL examples."""
import argparse
import json
from pathlib import Path

import pandas as pd


def target(row):
    value = row.get("standard_value")
    unit = row.get("standard_units")
    usable = pd.notna(value) and unit in {"nM", "uM", "%"} and not bool(
        row.get("data_validity_comment")
    )
    score = pd.to_numeric(row.get("confidence_score"), errors="coerce")
    return {
        "assay_type": row.get("assay_type") or "unknown",
        "target_name": row.get("target_pref_name"),
        "target_id": row.get("target_chembl_id"),
        "target_confidence": "high" if pd.notna(score) and score >= 8 else "unknown",
        "organism": row.get("assay_organism"),
        "cell_line": row.get("assay_cell_type"),
        "endpoint": row.get("standard_type") or "other",
        "relation": row.get("standard_relation") or "unknown",
        "value": None if pd.isna(value) else value,
        "units": unit or "other",
        "qsar_usable": bool(usable),
        "exclude_reason": [] if usable else ["needs_manual_review"],
        "curation_notes": "Derived from supplied structured fields; verify ambiguous descriptions.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    df = pd.read_parquet(args.input) if args.input.endswith("parquet") else pd.read_csv(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        for index, row in df.iterrows():
            record = row.to_dict()
            prompt = "\n".join(
                [
                    "Assay description:",
                    str(record.get("description", "")),
                    "",
                    "Raw activity fields:",
                    f"standard_type: {record.get('standard_type')}",
                    f"standard_relation: {record.get('standard_relation')}",
                    f"standard_value: {record.get('standard_value')}",
                    f"standard_units: {record.get('standard_units')}",
                    f"activity_comment: {record.get('activity_comment')}",
                    "",
                    "Return a normalized assay-curation JSON record.",
                ]
            )
            example = {
                "id": f"assay_{record.get('activity_id', index)}",
                "messages": [
                    {"role": "system", "content": "You are a cheminformatics assay-curation assistant. Return valid JSON only."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(target(record), default=str)},
                ],
                "metadata": {"split_group": record.get("assay_chembl_id"), "source": "curated_input"},
            }
            out.write(json.dumps(example) + "\n")


if __name__ == "__main__":
    main()
