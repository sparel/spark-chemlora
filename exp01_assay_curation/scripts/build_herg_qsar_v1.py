#!/usr/bin/env python3
"""Build the strict, auditable hERG numerical-QSAR v1 dataset.

This is deliberately narrower than the companion curation corpus: it accepts
only exact, positive IC50 observations in nM/uM with no ChEMBL validity flag.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

IC50_UNITS_TO_NM = {"nM": 1.0, "uM": 1_000.0}
QSAR_ENDPOINTS = {"IC50"}


def is_present(value: object) -> bool:
    return value is not None and not pd.isna(value)


def split_for(assay_id: object) -> str:
    bucket = int(hashlib.sha256(str(assay_id).encode()).hexdigest()[:8], 16) % 100
    return "test" if bucket < 10 else "valid" if bucket < 20 else "train"


def classify_row(row: pd.Series) -> tuple[bool, str, float | None, float | None]:
    """Return (eligible, reason, normalized_nM, pIC50)."""
    endpoint = row.get("standard_type")
    relation = row.get("standard_relation")
    value = pd.to_numeric(row.get("standard_value"), errors="coerce")
    units = row.get("standard_units")

    if endpoint not in QSAR_ENDPOINTS:
        return False, "endpoint_not_ic50", None, None
    if relation != "=":
        return False, "nonexact_or_missing_relation", None, None
    if not is_present(value) or not math.isfinite(float(value)):
        return False, "missing_or_nonfinite_value", None, None
    if is_present(row.get("data_validity_comment")):
        return False, "data_validity_flag", None, None

    if endpoint == "IC50":
        if float(value) <= 0:
            return False, "nonpositive_ic50", None, None
        factor = IC50_UNITS_TO_NM.get(units)
        if factor is None:
            return False, "unsupported_ic50_units", None, None
        value_nm = float(value) * factor
        return True, "", value_nm, 9.0 - math.log10(value_nm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activities", default="exp01_assay_curation/data/raw/chembl_herg/activities.parquet")
    parser.add_argument("--output-dir", default="exp01_assay_curation/data/processed/herg_qsar_v1")
    args = parser.parse_args()

    activities = pd.read_parquet(args.activities).copy()
    results = activities.apply(classify_row, axis=1, result_type="expand")
    results.columns = ["qsar_usable", "exclude_reason", "ic50_nM", "pIC50"]
    results["qsar_usable"] = results["qsar_usable"].astype(bool)
    data = pd.concat([activities, results], axis=1)
    data["split"] = data["assay_chembl_id"].map(split_for)

    eligible = data[data["qsar_usable"]].copy()
    # A structure is indispensable for compound-level QSAR, unlike curation.
    eligible["canonical_smiles"] = eligible["canonical_smiles"].astype("string")
    no_structure = eligible["canonical_smiles"].isna() | eligible["canonical_smiles"].str.strip().eq("")
    eligible.loc[no_structure, "qsar_usable"] = False
    eligible.loc[no_structure, "exclude_reason"] = "missing_canonical_smiles"
    strict = eligible[~no_structure].copy()

    columns = [
        "activity_id", "molecule_chembl_id", "parent_molecule_chembl_id", "canonical_smiles",
        "assay_chembl_id", "document_chembl_id", "standard_type", "standard_relation",
        "standard_value", "standard_units", "ic50_nM", "pIC50", "split",
        "data_validity_comment", "potential_duplicate",
    ]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    strict[columns].to_parquet(output / "herg_qsar_v1.parquet", index=False)
    data.to_parquet(output / "all_rows_with_qsar_admission.parquet", index=False)

    report = {
        "source_rows": len(data),
        "eligible_before_structure_gate": int(data["qsar_usable"].sum()),
        "excluded_for_missing_structure": int(no_structure.sum()),
        "strict_rows": len(strict),
        "strict_unique_molecules": int(strict["molecule_chembl_id"].nunique()),
        "strict_unique_assays": int(strict["assay_chembl_id"].nunique()),
        "rows_by_split": strict["split"].value_counts().sort_index().to_dict(),
        "endpoint_counts": strict["standard_type"].value_counts().to_dict(),
        "exclusion_counts": data.loc[~data["qsar_usable"], "exclude_reason"].value_counts().to_dict(),
        "policy": {
            "endpoints": ["IC50"],
            "relation": "= only",
            "IC50_value": "positive only",
            "IC50_units": ["nM", "uM"],
            "data_validity_comment": "must be absent",
            "structure": "canonical_smiles required",
            "censored_values": "excluded, preserved in all_rows_with_qsar_admission.parquet",
        },
    }
    (output / "data_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
