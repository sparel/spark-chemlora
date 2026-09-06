#!/usr/bin/env python3
"""Build leakage-resistant hERG assay-curation train/valid/test JSONL splits."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

INPUT_FIELDS = [
    "assay_description", "assay_type", "bao_label", "standard_type", "standard_relation",
    "standard_value", "standard_units", "standard_text_value", "activity_comment",
    "data_validity_comment", "potential_duplicate", "canonical_smiles",
]
ALLOWED_UNITS = {"nM", "uM", "%"}
OUTPUT_SCHEMA = """{
  "assay_type": "string",
  "target_name": "string",
  "target_id": "string",
  "target_confidence": "high | unknown",
  "organism": "string | null",
  "cell_line": "string | null",
  "endpoint": "string",
  "relation": "= | < | > | <= | >= | unknown",
  "value": "string | null",
  "units": "string | other",
  "qsar_usable": true,
  "exclude_reason": ["string"],
  "curation_notes": "string"
}"""


def value_or_none(value):
    return None if pd.isna(value) else value


def bucket(assay_id):
    return int(hashlib.sha256(str(assay_id).encode()).hexdigest()[:8], 16) % 100


def split_for(assay_id):
    number = bucket(assay_id)
    return "test" if number < 10 else "valid" if number < 20 else "train"


def qsar_admission(endpoint, relation, value, units, invalid, canonical_smiles):
    """Strict v1 numerical QSAR policy, kept separate from extraction labels."""
    reasons = []
    if endpoint != "IC50":
        reasons.append("endpoint_not_ic50")
    if relation != "=":
        reasons.append("nonexact_or_missing_relation")
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        reasons.append("missing_numeric_value")
    elif endpoint == "IC50" and numeric <= 0:
        reasons.append("nonpositive_ic50")
    if endpoint == "IC50" and units not in {"nM", "uM"}:
        reasons.append("unsupported_ic50_units")
    if invalid:
        reasons.append("data_validity_flag")
    if canonical_smiles is None or not str(canonical_smiles).strip():
        reasons.append("missing_canonical_smiles")
    return not reasons, reasons


def normalized_target(row):
    endpoint = value_or_none(row.get("standard_type")) or "other"
    relation = value_or_none(row.get("standard_relation")) or "unknown"
    units = value_or_none(row.get("standard_units")) or "other"
    value = value_or_none(row.get("standard_value"))
    invalid = value_or_none(row.get("data_validity_comment"))
    raw_units = value_or_none(row.get("standard_units"))
    usable, reason = qsar_admission(endpoint, value_or_none(row.get("standard_relation")), value, raw_units, invalid, value_or_none(row.get("canonical_smiles")))
    return {
        "assay_type": value_or_none(row.get("assay_type")) or "unknown",
        "target_name": value_or_none(row.get("target_pref_name")) or "Voltage-gated inwardly rectifying potassium channel KCNH2",
        "target_id": value_or_none(row.get("target_chembl_id")) or "CHEMBL240",
        "target_confidence": "high",
        "organism": value_or_none(row.get("target_organism")) or "Homo sapiens",
        "cell_line": value_or_none(row.get("assay_cell_type")),
        "endpoint": endpoint,
        "relation": relation,
        "value": value,
        "units": units,
        "qsar_usable": usable,
        "exclude_reason": reason,
        "curation_notes": "Normalized from supplied ChEMBL activity and assay fields; assay context requires review where ambiguous.",
    }


def prompt(row):
    fields = "\n".join(f"{name}: {value_or_none(row.get(name))}" for name in INPUT_FIELDS)
    return f"""You are curating human KCNH2/hERG (CHEMBL240) records.
Return JSON only: one flat object, no markdown, no explanation, no nesting, and no keys beyond this exact schema:
{OUTPUT_SCHEMA}

Rules: copy supported source values exactly; use null for unknown scalar fields; `qsar_usable` is true only for an exact, positive IC50 in nM/uM with no data-validity flag and a non-empty canonical SMILES; otherwise false and populate `exclude_reason` with the applicable policy reason(s).

Assay/activity record:
{fields}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activities", default="exp01_assay_curation/data/raw/chembl_herg/activities.parquet")
    parser.add_argument("--assays", default="exp01_assay_curation/data/raw/chembl_herg/assays.parquet")
    parser.add_argument("--output-dir", default="exp01_assay_curation/data/processed/herg_v1")
    args = parser.parse_args()
    activities = pd.read_parquet(args.activities)
    assays = pd.read_parquet(args.assays)
    assay_columns = [column for column in ["assay_chembl_id", "assay_cell_type", "description"] if column in assays.columns]
    merged = activities.merge(assays[assay_columns].drop_duplicates("assay_chembl_id"), on="assay_chembl_id", how="left", suffixes=("", "_assay"))
    if "assay_description" not in merged:
        merged["assay_description"] = merged.get("description")
    merged["split"] = merged["assay_chembl_id"].map(split_for)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    handles = {name: (output / f"{name}.jsonl").open("w") for name in ("train", "valid", "test")}
    try:
        for _, row in merged.iterrows():
            record = row.to_dict()
            example = {
                "id": f"chembl_activity_{record['activity_id']}",
                "messages": [
                    {"role": "system", "content": "You are a cheminformatics assay-curation assistant. Return valid JSON only."},
                    {"role": "user", "content": prompt(record)},
                    {"role": "assistant", "content": json.dumps(normalized_target(record), default=str)},
                ],
                "metadata": {
                    "source": "ChEMBL",
                    "target_chembl_id": "CHEMBL240",
                    "assay_chembl_id": record["assay_chembl_id"],
                    "split_group": record["assay_chembl_id"],
                },
            }
            handles[record["split"]].write(json.dumps(example) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    report = {
        "source_rows": len(merged),
        "unique_assays": int(merged["assay_chembl_id"].nunique()),
        "rows_by_split": merged["split"].value_counts().to_dict(),
        "assays_by_split": merged.groupby("split")["assay_chembl_id"].nunique().to_dict(),
        "endpoint_counts": merged["standard_type"].fillna("missing").value_counts().head(30).to_dict(),
        "unit_counts": merged["standard_units"].fillna("missing").value_counts().head(30).to_dict(),
        "data_validity_flagged": int(merged["data_validity_comment"].notna().sum()),
        "split_rule": "deterministic SHA-256 bucket of assay_chembl_id: 80% train, 10% validation, 10% test",
    }
    (output / "data_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
