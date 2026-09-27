# Business Entity Resolution Pipeline

This folder contains the full ER pipeline used for the Amazon ML Challenge 2026.

## Setup

```bash
pip install -r requirements.txt
```

## Train

From the `business_entity_resolution` directory:

```bash
python -m src.train
```

This trains a LightGBM classifier and saves `model_lgbm.joblib` with the model and F₀.₅‑optimized threshold.

## Inference

```bash
python -m src.infer
```

This produces:

- `output/matching_results.tsv` – final matches (uploaded to the leaderboard).
- `output/candidate_pairs.tsv` – candidate set from blocking.

## Validation

From the repo root:

```bash
python utils/validate_submission.py ^
    --matching output/matching_results.tsv ^
    --candidate output/candidate_pairs.tsv ^
    --test-dir dataset/test
```

The script prints `VALIDATION PASSED` or lists errors to fix.

## Structure

- `src/` – all pipeline code:
  - `blocking.py` – candidate generation / blocking.
  - `features.py` – name/address similarity features.
  - `model.py` – LightGBM model and F₀.₅ threshold tuning.
  - `train.py`, `infer.py` – training and inference entry points.
- `requirements.txt` – pinned dependencies.
- `README.md` – this file.
