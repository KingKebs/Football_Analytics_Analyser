---
name: machine-learning-workflow
description: Train, evaluate, and apply goal-market ML models (RandomForest, XGBoost) with recency weighting and cross-validation. Use when the user mentions ML mode, train/predict, RandomForest, XGBoost, ml-validate, BTTS, 1X2 deltas, or model persistence.
---

# Machine Learning Workflow

ML augments Poisson baselines for **Total Goals**, **1X2**, and **BTTS**. It is optional (`--ml-mode` default `off`).

Read [ReadMeDocs/ML_MODE_GUIDE.md](../../../ReadMeDocs/ML_MODE_GUIDE.md) before changing features, training, or flags.

## Preconditions

```
- [ ] Historical data exists (typically 2+ seasons)
- [ ] Sample count ≥ --ml-min-samples (default 300)
- [ ] scikit-learn installed; xgboost optional (RF-only fallback)
```

If samples are low, download more seasons or lower `--ml-min-samples` only when the user accepts higher variance.

## Modules

| File | Role |
|---|---|
| `src/ml_features.py` | Rolling form, interactions, `TRAIN_FEATURE_COLUMNS` |
| `src/ml_training.py` | Fit RF / XGB + k-fold CV |
| `src/ml_evaluation.py` | Metrics and ML−Poisson deltas |
| `src/ml_utils.py` | Safe XGBoost import |

## Train then inspect CV

```bash
python cli.py --task full-league --league E0 --ml-mode train --ml-validate
```

`--ml-validate` prints MAE/RMSE (goals) and Accuracy/LogLoss (1X2, BTTS).

## Predict (trains first if needed)

```bash
python cli.py --task full-league --leagues E0,SP1,D1 --ml-mode predict --ml-algorithms rf \
  --min-confidence 0.4 --enable-double-chance --use-parsed-all
```

## Persist models

```bash
python cli.py --task full-league --league E0 --ml-mode train --ml-validate \
  --ml-save-models --ml-models-dir models
```

Writes `models/ml_models_<LEAGUE>_<TIMESTAMP>.pkl`. Retrain after Python upgrades (pickle is version-sensitive).

## Sensitivity

| Goal | Flag |
|---|---|
| Emphasize recent form | `--ml-decay 0.75` (default 0.85) |
| Scarce data | `--ml-min-samples 150` |
| Faster run | omit `--ml-validate`; `--ml-algorithms rf` |
| Both ensembles | `--ml-algorithms rf,xgb` |

## Feature contract

Do not reorder or silently drop `TRAIN_FEATURE_COLUMNS`. Train and predict must use the same columns. Impute missing base stats; do not invent new targets without updating `ml_training.py`.

Targets: `TotalGoals = FTHG + FTAG`; 1X2 from HomeWin/Draw/AwayWin; BTTS binary.

## Output fields on each match

- ML Total Goals + model name
- ML 1X2 / BTTS probabilities
- Δ vs Poisson (positive = ML higher than baseline)

View in Streamlit **ML Predictions** tab or `data/analysis/*_formatted.txt`.

## Recovery

| Symptom | Fix |
|---|---|
| ML modules not available | `pip install -r requirements.txt` |
| Insufficient samples | Download seasons or lower `--ml-min-samples` |
| No Δ lines | Use `--ml-mode predict` |
| XGB missing | Continue with RF only |
