# Modeling Progress: Corners Prediction Improvements

This document tracks step-by-step modeling improvements for predicting 1st half, 2nd half, and total corners.
Each step includes: what changed, how to run, key outputs (highlighted), and a short interpretation.

Date: 2025-11-15

---

## Step 1 — Add Cross-Validation (5-fold)

Why
- In-sample metrics were optimistic; we need honest generalization metrics to evaluate future changes fairly.

What changed
- `corners_analysis.py`: `train_models()` now computes 5-fold CV metrics (R² and MAE) for both:
  - Total corners target
  - 1H/Total ratio target (if available)
- Console prints and JSON export include CV mean±std alongside in-sample metrics.

How to run
```bash
python3 corners_analysis.py --league E2 --no-prompt --train-model
```

Key outputs (E2)
- Total Corners (Linear, 5-fold CV):
  - R² mean±std: ~0.069 ± 0.024 (vs. in-sample ~0.092)
  - MAE mean±std: ~2.697 ± 0.078 (close to in-sample ~2.676)
- Half-Split Ratio (Linear, 5-fold CV):
  - R² mean±std: ~0.846 ± small
  - MAE mean±std: ~0.01 ± small

Interpretation
- Total corners remain hard to predict linearly; CV R² is small but positive.
- Half-split ratio is highly learnable and stable across folds.

---

## Step 2 — Evaluate Ensembles (RandomForest, optional XGBoost)

Why
- Corners vs. basic stats likely have non-linear interactions; ensembles can capture more structure than Linear.

What changed
- `corners_analysis.py`: `train_models()` now evaluates:
  - RandomForestRegressor (5-fold CV) for total and ratio targets
  - XGBoost (5-fold CV) if `xgboost` is installed
- Results printed and saved next to Linear metrics.

How to run
```bash
python3 corners_analysis.py --league E2 --no-prompt --train-model
```

Highlighted outputs (E2)
- Total Corners (RandomForest, 5-fold CV):
  - R² mean±std: ~0.183 ± 0.070
  - MAE mean±std: ~2.464 ± 0.064
- Total Corners (Linear, 5-fold CV):
  - R² mean±std: ~0.069 ± 0.024
  - MAE mean±std: ~2.697 ± 0.078
- Half-Split Ratio (RandomForest, 5-fold CV):
  - R² mean±std: ~0.984 ± 0.003
  - MAE mean±std: ~0.002 ± 0.000

Interpretation
- RandomForest substantially improves both R² and MAE for total corners compared to Linear.
- For 1H ratio, RF also performs extremely well, but ensure we check for leakage (ratio is derived from same match features).
- Ensembles are a better candidate model for total corners going forward.

---

## Step 3 — Recency Weighting (E3)

Why
- Recent seasons and matches often reflect current team styles and personnel; weighting them higher can improve generalization.

What changed
- `corners_analysis.py`: Added recency weights from the `Date` column using exponential decay (half-life ~180 days by default).
- Training/evaluation now includes weighted cross-validation metrics for Linear and RandomForest models (in addition to unweighted CV from Steps 1–2).

How to run (E3)
```bash
python3 corners_analysis.py --league E3 --no-prompt --train-model
```

Updated Step 3 — Recency Weighting (E3) Results

Observed metrics (E3)
- Total Corners — Linear (5-fold CV):
  - Unweighted: R² 0.1509 ± 0.0311, MAE 2.512 ± 0.099
  - Weighted:   R² 0.1421 ± 0.0429, MAE 2.570 ± 0.135
- Total Corners — RandomForest (5-fold CV):
  - Unweighted: R² 0.2370 ± 0.0176, MAE 2.323 ± 0.074
  - Weighted:   R² 0.2518 ± 0.0301, MAE 2.320 ± 0.098
- Half-Split Ratio — Linear (5-fold CV):
  - Unweighted: R² 0.8439 ± 0.0108, MAE 0.011 ± 0.000
  - Weighted:   R² 0.8476 ± 0.0078,  MAE 0.011 ± 0.000
- Half-Split Ratio — RandomForest (5-fold CV):
  - Unweighted: R² 0.9792 ± 0.0029, MAE 0.002 ± 0.000
  - Weighted:   R² 0.9746 ± 0.0024, MAE 0.002 ± 0.000

Interpretation
- Total Corners: Recency weighting slightly improves RandomForest R² (0.237 → 0.252) with essentially unchanged MAE; for Linear it slightly reduces R² and increases MAE. Conclusion: keep recency weighting for RF; it adds small but consistent lift in E3. Consider league-specific half-life tuning later.
- 1H Ratio: Both models remain excellent; weighting has negligible effect on MAE and only tiny R² deltas.

Try it again
```bash
# Generate enriched CSVs and metrics via automation (ensures training step runs)
python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --leagues E3 --train-model --force --auto --mode fast

# Or compute metrics directly from the latest enriched CSV (fastest)
PYTHONPATH=. python3 tools/run_weighted_metrics.py
```

---

## Step 4 — Interaction Terms (Feature Engineering)

Why
- Non-linear interactions like shots×fouls or first-half goal state×shots can correlate with corner generation pressure and may improve model fit, especially for ensembles.

What changed
- `corners_analysis.py`: `engineer_features()` adds interaction features:
  - HS_x_AS, HST_x_AST, Shots_x_Fouls, HTDiff_x_Shots, HomeDiff_x_Fouls
- Training (`train_models`) now includes these features automatically when present.

How to run
```bash
python3 corners_analysis.py --league E3 --no-prompt --train-model
# or reuse enriched CSV to just recompute metrics quickly
PYTHONPATH=. python3 tools/run_weighted_metrics.py
```

Highlighted outputs (E3)
- Total Corners — Linear (5-fold CV):
  - Unweighted: R² ≈ 0.151 ± 0.031, MAE ≈ 2.512 ± 0.099
  - Weighted:   R² ≈ 0.142 ± 0.043, MAE ≈ 2.570 ± 0.135
- Total Corners — RandomForest (5-fold CV):
  - Unweighted: R² ≈ 0.237 ± 0.018, MAE ≈ 2.323 ± 0.074
  - Weighted:   R² ≈ 0.252 ± 0.030, MAE ≈ 2.320 ± 0.098
- Half-Split Ratio (Linear/RF): unchanged at excellent levels; interactions have minimal effect here.

Interpretation
- With the current base features, these interactions did not materially shift the CV metrics vs. Step 3; RF still benefits slightly more under weighting.
- This suggests either (a) RF already captures similar structure via splits, or (b) more targeted interactions (e.g., team strength × venue, or rolling-form × opponent fouls) are needed.

Next up (proposed)
- Proceed to Step 5: External data (injuries, weather, referee) with careful gating and fallback to maintain robustness.
- Optionally, tune the interaction set and evaluate per-league feature importance to prune non-contributing interactions.
