# Football Analytics Analyser

This project provides tools for analyzing football data, generating predictions, and evaluating betting strategies. It includes scripts for downloading data, organizing structures, and visualizing results.

## Quick start

```bash
# Analyze a full league round for EPL (E0)
python3 automate_football_analytics_fullLeague.py --league E0 --rating-model blended --rating-blend-weight 0.3 --rating-last-n 6

# Analyze multiple leagues (e.g., EPL, La Liga, Bundesliga)
python3 automate_football_analytics_fullLeague.py --league E0,SP1,D1 --rating-model blended --rating-blend-weight 0.3 --rating-last-n 6

# Download league data
python3 download_all_tabs.py --download-football-data --leagues E0 --seasons AUTO
```

## How it works
- The `automate_football_analytics_fullLeague.py` script analyzes league data to generate predictions and betting suggestions.
- The `download_all_tabs.py` script fetches league CSVs from football-data.co.uk into `football-data/all-euro-football/`.
- The `organize_structure.py` script organizes downloaded data into a structured format for analysis.

## Output
- Analysis results: `data/full_league_suggestions_<LEAGUE>_<TIMESTAMP>.json`
- Downloaded league data: `football-data/all-euro-football/<LEAGUE>_<SEASON>.csv`
- Latest alias: `football-data/all-euro-football/<LEAGUE>.csv`

## Notes
- Install dependencies via `pip install -r requirements.txt`.
- Use `--dry-run` to preview actions without downloading or writing files.
- The project supports multiple leagues and seasons for comprehensive analysis.

## Full League ML Mode (Goal Markets)
The full league script now supports an optional Machine Learning mode for goal-based markets (Steps 1–4):
- Feature engineering (rolling form, interaction terms, ratios, basic event stats if present)
- Recency weighting (exponential decay) applied as sample weights
- Model training with RandomForest (always) and optional XGBoost (if installed)
- 5-fold cross-validation metrics for regression (Total Goals) and classification (1X2, BTTS)
- Per-fixture prediction augmentation and comparison vs Poisson baseline probabilities

### Installation Notes
Make sure the following are installed:
```bash
pip install -r requirements.txt
```
XGBoost is optional. If not installed, ML mode will fall back to RandomForest only and log an informational message.

### Enabling ML Mode
Use the new flags on `automate_football_analytics_fullLeague.py` or via the unified CLI.

Minimal train-only run (no predictions augmentation):
```bash
python3 automate_football_analytics_fullLeague.py \
  --leagues E0 \
  --ml-mode train \
  --ml-validate
```

Train + Save models:
```bash
python3 automate_football_analytics_fullLeague.py \
  --leagues E0 \
  --ml-mode train \
  --ml-validate \
  --ml-save-models \
  --ml-models-dir models
```

Predict (augment suggestions with ML outputs):
```bash
python3 automate_football_analytics_fullLeague.py \
  --leagues E0 \
  --ml-mode predict \
  --ml-validate
```

Via unified CLI:
```bash
python3 cli.py --task full-league --league E0 --ml-mode predict --ml-validate
```

### ML Flags Summary
| Flag | Purpose | Default |
|------|---------|---------|
| `--ml-mode` | off / train / predict | off |
| `--ml-validate` | Print CV metrics after training | (disabled) |
| `--ml-algorithms` | Comma list (rf,xgb) | rf,xgb |
| `--ml-decay` | Recency decay factor (0<d<=1) | 0.85 |
| `--ml-min-samples` | Minimum rows required to run ML | 300 |
| `--ml-save-models` | Persist trained models to disk | (disabled) |
| `--ml-models-dir` | Directory for saved model pickles | models |

### Output Augmentation (Predict Mode)
For each match suggestion:
```
ML Total Goals: <mean> (model RF/XGB)
ML 1X2 probs: H=.. D=.. A=.. (model RF/XGB)
ML BTTS probs: Yes=.. No=.. (model RF/XGB)
Δ1X2: H=±Δ D=±Δ A=±Δ          # Difference ML - Poisson
ΔBTTS: Yes=±Δ No=±Δ           # Difference ML - Poisson
```
Positive deltas indicate ML assigns higher probability than Poisson baseline.

### Cross-Validation Metrics (when --ml-validate)
Example log lines:
```
ML Cross-Validation Metrics:
  TotalGoals_RF: {'MAE': 1.42, 'RMSE': 1.89}
  1X2_RF: {'Accuracy': 0.53, 'LogLoss': 1.02}
  BTTS_RF: {'Accuracy': 0.61, 'LogLoss': 0.66}
```
Interpretation: Lower MAE/RMSE better (Total Goals). Higher Accuracy better (1X2, BTTS). If LogLoss is None a fold had a single class (rare in small splits).

### Fallback & Safety Behavior
- Missing historical data: ML mode skipped, Poisson-only.
- Not enough samples (`< ml_min_samples`): ML skipped, Poisson-only.
- XGBoost absent: Runs RandomForest only; metrics still printed.
- Feature columns missing (shots/fouls/corners): Automatically zero-imputed.
- Date parsing failure: Recency weights revert to positional ordering.

### Model Persistence
If `--ml-save-models` is used, a pickle file is created under `models/`:
```
models/ml_models_<LEAGUE>_<TIMESTAMP>.pkl
```
Contains: regression & classification model objects, CV metrics, metadata.

### Extending / Next Steps
Ideas to extend ML mode later:
- Add Brier score & ROC-AUC metrics
- Blend Poisson & ML probabilities for calibrated hybrid outputs
- Calibrate probabilities (Platt scaling / isotonic) on validation folds
- Add bookmaker odds features if odds feed available

### Troubleshooting
| Issue | Cause | Resolution |
|-------|-------|-----------|
| "ML modules not available" | Missing dependencies | `pip install -r requirements.txt` |
| "Insufficient samples" | Historical dataset too small | Lower `--ml-min-samples` or gather more data |
| No Δ lines printed | Predict mode not enabled | Use `--ml-mode predict` |
| LogLoss = None | Class imbalance in a fold | Increase samples or reduce folds |

---

## Example Combined Workflow
```bash
# 1. Download fresh data
python3 cli.py --task download --leagues E0
# 2. Organize (optional)
python3 cli.py --task organize --source football-data --target data
# 3. Train ML models + validate
python3 cli.py --task full-league --league E0 --ml-mode train --ml-validate
# 4. Predict with ML augmentation
python3 cli.py --task full-league --league E0 --ml-mode predict
```

## Tests
ML pipeline integrity is covered by unit tests in `tests/test_ml_pipeline.py` (feature engineering, weighting, training, prediction & evaluation).
