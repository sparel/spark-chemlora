#!/usr/bin/env python3
"""Fetch a reproducible raw hERG/KCNH2 activity and assay snapshot from ChEMBL."""
import argparse
import json
from pathlib import Path
from time import sleep
from urllib.parse import urljoin

import pandas as pd
import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data"
TARGET = "CHEMBL240"


def get_pages(resource, params, max_pages=None):
    url = f"{BASE}/{resource}.json"
    rows, pages = [], 0
    while url and (max_pages is None or pages < max_pages):
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        payload = response.json()
        key = {"activity": "activities", "assay": "assays"}[resource]
        rows.extend(payload.get(key, []))
        next_url = payload.get("page_meta", {}).get("next")
        url = urljoin(f"{BASE}/", next_url) if next_url else None
        params = None
        pages += 1
        if url:
            sleep(0.1)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="exp01_assay_curation/data/raw/chembl_herg")
    parser.add_argument("--max-pages", type=int, default=None, help="Use 1 for a connectivity smoke run")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    target = requests.get(f"{BASE}/target/{TARGET}.json", timeout=60)
    target.raise_for_status()
    (output / "target.json").write_text(json.dumps(target.json(), indent=2))

    activities = get_pages("activity", {"target_chembl_id": TARGET, "limit": 1000}, args.max_pages)
    pd.DataFrame(activities).to_parquet(output / "activities.parquet", index=False)

    assay_ids = sorted({row.get("assay_chembl_id") for row in activities if row.get("assay_chembl_id")})
    assays = []
    for start in range(0, len(assay_ids), 100):
        assays.extend(get_pages("assay", {"assay_chembl_id__in": ",".join(assay_ids[start:start + 100]), "limit": 1000}))
    pd.DataFrame(assays).drop_duplicates(subset=["assay_chembl_id"]).to_parquet(output / "assays.parquet", index=False)

    manifest = {
        "source": "ChEMBL data web services",
        "target_chembl_id": TARGET,
        "activity_count": len(activities),
        "assay_count": len(assays),
        "max_pages": args.max_pages,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
