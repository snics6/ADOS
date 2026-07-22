# ADOS multimodal score prediction

Video-derived multimodal features (face, pose, speech) are used to predict
ADOS-2 Module 2 scores in a multi-task setting.

## Repository layout

```
configs/          Experiment and data-split configs (YAML)
docs/             Design notes (features, protocol)
src/ados_ml/      Python package
scripts/          CLI entry points
notebooks/        Ad-hoc analysis (optional)
outputs/          Run artifacts (gitignored)
tests/            Unit tests
data/             Raw features and labels (local)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Cursor / VS Code uses `venv` via `.vscode/settings.json`.

## Cohort (current)

- IDs (including `d1_*`) that have all four task segments detected:
  birthday party, snack, pretend play, interactive play (`n=33`).
- Subject list: `configs/cohorts/four_tasks_33.yaml`

## Targets

| Target | Type | Notes |
|--------|------|-------|
| SA | regression | Social Affect |
| RRB | regression | Restricted / Repetitive Behavior |
| C2 | ordinal | Imagination / creativity; explain via pretend-play segment |
| B1 | binary (0 vs 2) | Unusual eye contact (no score-1 in this cohort) |
| B12 | ordinal | Overall quality of rapport |

## Design constraints (summary)

- Use only the four labeled task intervals.
- Predict from **child** and **child–examiner dyad** features.
- Detection / visibility quality is used only as **aggregation confidence weights**, not as predictors.
- Evaluation: 4-fold CV. Hardware target: NVIDIA A5000.
- Heads / losses / mandatory explanations: `docs/losses_and_heads.md`.

See `docs/experimental_design.md` and `docs/features.md`.

## Typical workflow

```bash
# 1. Build per-subject task-level feature tables
python scripts/extract_features.py --config configs/default.yaml

# 2. Train / evaluate (4-fold + ablations + sklearn baseline)
python scripts/train.py --config configs/default.yaml --device cuda

# 3. Print run summary JSON
python scripts/evaluate.py --run-dir outputs/runs/<run_id>
```

Outputs are written under `outputs/<run_id>/` (metrics, predictions, configs snapshot).
Features: `outputs/features/<stamp>/`. Runs: `outputs/runs/<stamp>/`.

## Data

- Features: `data/<participant_id>/` (see `data/README.md`)
- Labels: `data/ADOS2_result_2.xlsx` (Module 2 sheet)
- Participant IDs in labels may use `d1-402`; feature folders use `d1_402`. Normalize before join.
