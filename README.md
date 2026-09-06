# Spark ChemLoRA

Bounded LoRA/QLoRA experiments for scientific-data curation and evidence triage in pharmaceutical and agrochemical R&D.

## The governing rule

LLMs extract, normalize, summarize, and explain supplied evidence. Deterministic code and classical cheminformatics retain hard eligibility and numerical-prediction decisions. The hERG experiment demonstrates why: a model can be highly accurate yet still make the exact admission error that contaminates a QSAR dataset.

## Experiments

- `exp01_assay_curation/`: normalize hERG/ChEMBL assay and activity records into a constrained JSON evidence card.

## What this repository contains

Source code, configuration, tests, schemas, and documentation. It intentionally excludes raw and processed data, model weights, adapters, run logs, review packs, and full inference outputs. Fetch/build scripts let users reproduce public-data preparation subject to upstream licences.

Read [LICENSE](LICENSE), [NOTICE](NOTICE), [DATA_LICENSE.md](DATA_LICENSE.md), and [DATA_PROVENANCE.md](DATA_PROVENANCE.md) before use.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[train,chem]
python -m unittest discover -s common/tests
python -m unittest discover -s exp01_assay_curation/tests
```

To build the hERG corpus, fetch ChEMBL data first and review the generated records before training. Configure a base model through a local path or a Hugging Face model ID.

## hERG experiment: safety boundary

The model may emit `qsar_usable` as an extraction output, but it is **not** the final admission decision. Production code must recompute eligibility from source-level fields:

```text
endpoint == "IC50"
and relation == "="
and value is numeric and value > 0
and units in {"nM", "uM"}
and data_validity_comment is empty
and canonical_smiles is non-empty
```

This is a curation prototype, not a clinical safety-assessment system. Validate any workflow independently before use.

## Status

Research prototype. Not for clinical interpretation, safety certification, or autonomous development decisions.
