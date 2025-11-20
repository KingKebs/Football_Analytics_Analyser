# Turning This Repository Into a Predictive Modeling Project

A practical guide for defining targets, building, validating, auditing, revising, and operating football prediction models in this repo.

## Table of Contents
1. Tasks & Targets
2. Modeling Pipeline
3. Baselines & Metrics
4. Validation Strategy
5. Model Audit Checklist
6. Model Revision Steps
7. Monitoring & Maintenance
8. Production Readiness Checklist
9. Appendix: Metric Notes

---

## 1. Tasks & Targets

| Task | Type | Output |
|------|------|--------|
| Total Goals | Count regression | Expected goals (Poisson-friendly) |
| 1X2 Outcome | 3-class classification | P(Home), P(Draw), P(Away) |
| Both Teams To Score (BTTS) | Binary classification | P(BTTS) |

---

## 2. Modeling Pipeline

### 2.1 Overview
Sequential, time-aware pipeline: ingest → clean → feature versioning → split → train & evaluate (rolling) → calibrate → package artifacts → monitor.

### 2.2 Data Ingestion & Cleaning
- Single authoritative raw match dataset (snapshot + hash).
- Immutable raw layer; transformations produce a curated layer.
- Validate row counts, date ordering, duplicate (match_id, season) pairs.

#### 2.2.1 Organize Raw → Processed (CLI)
Run:
```
python cli.py --task organize --source football-data --target data/processed
```
Produces:
- data/processed/ (structured, normalized files)
- data/processed/manifest.json (file list + SHA256 hashes)
- logs/ingest.log

Verification:
- Check ingest log has SUCCESS status.
- Assert manifest row count == concatenated processed rows.
- Store manifest hash in metrics/ or model artifact for reproducibility.

- Store per-window metrics for trend analysis.
No random K-fold; no mixing future seasons.

### 2.5 Training Procedure
- Config-driven (YAML/JSON): model_type, hyperparams, feature_set, split ranges.
- Loop:
  1. Load curated data.
  2. Build feature matrix (X), targets (y) per task.
  3. Fit model on train slice.
  4. Predict on validation slice.
  5. Record metrics + calibration stats.
- Optional ensembling: weighted average / stacking layer.
- Persist:
  - model.pkl (binary)
  - metrics.json
  - calibration_curve.json
  - feature_manifest.json

#### 2.5.1 Quick Start (Baseline Training)

1. Copy example config:
```
cp config/train_config.example.yaml config/train_config.yaml
```
2. Adjust `leagues`, `seasons`, and date split (`train_until`, `validation_from`). Ensure corresponding CSVs exist under `football-data/` (e.g. `E0_2324.csv`).
3. Run pipeline:
```
python3 train_pipeline.py --config config/train_config.yaml
```
4. Inspect artifacts created under `models/<tag>_<TIMESTAMP>/`:
   - `model.pkl` (trained RF/XGB objects)
   - `metrics.json` (validation + CV metrics)
   - `calibration_curve.json` (if calibration applied)
   - `feature_manifest.json` (feature set + data hash)
   - `validation_predictions_sample.json` (sample prediction rows)
   - `git_commit.txt` (snapshot commit hash)

5. Re-run with adjustments:
   - Change `rolling_window` for recent form sensitivity.
   - Toggle `algorithms` list (`["rf"]` only for a fast run).
   - Set `fast_mode: true` (future flag for lighter estimator counts).

6. Add a new tag for experiment tracking:
```
sed -i '' 's/tag: "baseline_v1"/tag: "exp_depth10"/' config/train_config.yaml
```

7. Compare runs:
```
diff <(jq '.goals_RMSE,.1X2_LogLoss,.BTTS_LogLoss' models/baseline_v1_*/metrics.json) \
     <(jq '.goals_RMSE,.1X2_LogLoss,.BTTS_LogLoss' models/exp_depth10_*/metrics.json)
```

#### 2.5.2 Calibration Criteria
- Examine `BTTS_ECE_before` vs `BTTS_ECE_after` in `calibration_curve.json`.
- If `BTTS_method` is `none_needed`, no action required.
- If large ECE persists, force method:
```
sed -i '' 's/method_preference: "auto"/method_preference: "isotonic"/' config/train_config.yaml
```

#### 2.5.3 Reproducibility Notes
- `hash_curated` in `feature_manifest.json` locks the raw slice; changing seasons or filtering alters this hash.
- Always commit the config and artifact directory together for audit traceability.
- Embed experiment metadata in a simple TSV for tracking:
```
echo -e "run\ttag\tgoals_RMSE\t1X2_LogLoss\tBTTS_LogLoss" > experiments.tsv
for d in models/*/; do jq -r '["'"$d"'", .goals_RMSE, .1X2_LogLoss, .BTTS_LogLoss] | @tsv' $d/metrics.json >> experiments.tsv; done
```

#### 2.5.4 Next Extensions
- Add season-aware split logic for rolling backtests.
- Integrate XGBoost hyperparameter search (Optuna / Bayesian) when `fast_mode` is false.
- Implement probability calibration for 1X2 (current pipeline calibrates BTTS only).
- Store SHAP values for feature attribution (guard by data volume).
- Add an ensemble blender that averages RF & XGB outputs with tunable weights.

### 2.6 Calibration
- Generate reliability diagram.
- If miscalibrated (ECE > threshold): apply isotonic (small data) or Platt (logistic).
- Store post-calibration wrapper separately: calibrated_model.pkl.

### 2.7 Reproducibility & Artifacts
- Deterministic seeds (where applicable; avoid shuffling time order).
- Data snapshot ID + git commit hash embedded in metrics.json.
- Hash of feature manifest to ensure alignment with model binary.

### 2.8 Governance & Docs
- Model card: purpose, data window, metrics, limitations.
- Change log: feature_set version increments with rationale.
- Deprecation: retain last 2 stable feature versions.

### 2.9 Deployment Interface (Future)
- predict_batch(matches_df, tasks=["goals","1x2","btts"]) returns structured probabilities.
- Response schema versioned; include model_version + feature_set.

### 2.10 Failure / Drift Handling
- If live log loss rises > X% over rolling 4-week baseline: trigger retrain.
- If schema mismatch detected: abort inference, raise alert.

### 2.11 Minimal Success Gate
- New model must exceed baseline on primary metric AND maintain calibration (ECE threshold) before promotion.

---

## 3. Baselines & Metrics

Baselines:
- Naive: home/away averages, last-match stats, simple Poisson.
- Existing rating-based model in repo.
- Models must beat at least one baseline on primary metrics.

Recommended metrics:
- Total Goals: MAE, RMSE, Poisson log-likelihood.
- 1X2: Log loss (primary), Brier score, calibration plot, top-1 accuracy.
- BTTS: Log loss, ROC AUC, precision/recall, calibration.
- Business layer (optional): profit, hit-rate at betting thresholds.

---

## 4. Validation Strategy

- Time-based splits (e.g., seasons): train early, validate later, final holdout.
- Use rolling / expanding window backtests (simulate production).
- Avoid random K-fold for time series.
- Keep one untouched season for final evaluation.
- Perform calibration checks (reliability diagrams; isotonic / Platt if needed).
- Slice performance: per team, venue, goal range.

---

## 5. Model Audit Checklist

Before acceptance:
- Compare to baselines on defined metrics.
- Inspect residuals (by date, team, total goals).
- Check probability calibration.
- Conduct feature importance & ablation (remove groups).
- Use explainability (SHAP / permutation) for anomalies.
- Stress test: missing data, new teams, roster changes (if relevant).

---

## 6. Model Revision Steps

1. Data integrity: validate normalize_columns & engineer_features outputs; ensure no temporal leakage.
2. Feature engineering: adjust rolling windows, interaction terms, time decay weights.
3. Hyperparameter tuning: grid / Bayesian; apply regularization.
4. Calibration: isotonic or Platt scaling if miscalibrated.
5. Ensembling: stack or weighted average with rating-based model.
6. Retraining cadence: scheduled (weekly/monthly) or triggered by performance drop.

---

## 7. Monitoring & Maintenance

- Track metrics (log loss, MAE, AUC) over time in a dashboard.
- Alert on drift or sudden degradation.
- Version: data snapshots, code commits, model artifacts.
- Automatic retraining: time-based or performance-threshold trigger.
- Maintain a model card (dataset, metrics, evaluation, limitations).

---

## 8. Production Readiness Checklist

<input></input> Clear target + metric per task  
<input></input> Time-aware train/validation/test splits + backtests  
<input></input> Baseline implemented and surpassed  
<input></input> Probability calibration verified & adjusted  
<input></input> Stable feature pipeline (ml_features.py) with tests  
<input></input> CI for feature generation + training reproducibility  
<input></input> Monitoring & retraining plan defined  

---

## 9. Appendix: Metric Notes

- Log loss: sensitive to probability calibration; primary for classification.
- Brier score: complements log loss; interpretable decomposition (reliability vs resolution).
- Poisson log-likelihood: aligns with count modeling; penalizes mis-specified variance.
- Calibration plots: detect over/under-confidence; fix with post-processing.
- Business metrics: ensure practical utility, not just statistical lift.

---

This document provides structured, actionable criteria to evolve the project from prototype toward a reliable predictive modeling system.
