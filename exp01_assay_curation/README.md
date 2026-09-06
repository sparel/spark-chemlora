# Experiment 1: Assay curation and bioactivity normalization

## Initial scope: hERG inhibition

Use human **KCNH2 / hERG**, ChEMBL target `CHEMBL240`, as the first endpoint family. The experiment normalizes activity rows and assay context. It does **not** train a cardiac-safety predictor.

- Keep raw assay description, relation, value, units, and ChEMBL data-validity flags.
- Include `IC50`, `AC50`, `% inhibition`, and qualified textual observations initially.
- Hold out complete `assay_chembl_id` values, never random activity rows.
- Use `scripts/fetch_chembl_herg.py --max-pages 1` as a connectivity smoke run before a complete pull.


## Objective

Normalize raw activity rows plus assay descriptions into a schema fit for QSAR admission decisions and downstream SAR work.

## Data and split

Use ChEMBL-derived records. Hold out complete assay identifiers, never random activity rows. The supplied retrieval script targets the ChEMBL hERG/KCNH2 target CHEMBL240.

## Output and evaluation

Output schema includes assay type, target, endpoint, relation, value, unit, `qsar_usable`, exclusions and notes. Measure JSON/schema validity, endpoint/unit/relation accuracy, and precision/recall on `qsar_usable`. Compare against a prompting-only baseline.

## Exit criterion

A meaningful reduction in manual curation with no degradation in exclusion precision. Do not call it successful based on training loss.

## Run order

1. Run `python scripts/fetch_chembl_herg.py --max-pages 1` to verify source access, then repeat without the limit for the full snapshot.
2. Inspect `data/raw/chembl_herg/manifest.json` and sample activity/assay records.
3. Join raw records and run `python scripts/prepare_assay_curation.py ...` to create examples.
4. Inspect samples manually.
5. Use `python ../../common/scripts/train_qlora.py --config configs/train.yaml --dry-run`.
6. Train only after validation and baseline evaluation are recorded.
