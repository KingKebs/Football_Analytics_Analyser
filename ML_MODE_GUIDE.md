# ML Mode Guide (Full League Goal Markets)

This guide explains the optional Machine Learning mode integrated into `automate_football_analytics_fullLeague.py`.

## 1. Overview
ML Mode augments Poisson-based analytics by training supervised models on historical match data. It produces alternative probabilities for:
- 1X2 outcomes (Home / Draw / Away)
- BTTS (Yes / No)
- Total Goals expectation (regression)

It then compares ML probabilities with Poisson baseline, highlighting differences (Δ values).

## 2. Pipeline Steps (Mapped to Steps 1–4)
| Step | Description | Implementation |
|------|-------------|----------------|
| 1 | Feature Engineering | Rolling form (goals & shots), interaction terms (HS×HST, AS×AST), ratios (Shots_Ratio, Corners_Ratio), recent stats windows |
| 2 | Recency Weighting | Exponential decay weights (default decay=0.85) applied as sample weights during training |
| 3 | Model Training & CV | RandomForest (always) + optional XGBoost; 5-fold CV for MAE/RMSE (regression) and Accuracy/LogLoss (classification) |
| 4 | Comparison to Baseline | Per-match delta ML vs Poisson probabilities for 1X2, BTTS; regression mean displayed |

## 3. Feature Set
Base features (auto-imputed if missing):
```
HS, AS, HST, AST, HF, AF, HC, AC,
HS_x_HST, AS_x_AST, Shots_Ratio, Corners_Ratio,
Home_roll_GF, Home_roll_GA, Away_roll_GF, Away_roll_GA,
Home_roll_ShotsF, Home_roll_ShotsA, Away_roll_ShotsF, Away_roll_ShotsA
```
Targets:
- TotalGoals = FTHG + FTAG
- 1X2 class from (HomeWin, Draw, AwayWin)
- BTTS (binary)

## 4. Recency Weighting
Weight_i = decay^age_index (newest match age_index=0). Weights normalized to sum=1.
- Adjust using `--ml-decay` (lower values emphasize most recent matches more strongly).

## 5. Models
RandomForest: Balanced baseline with moderate depth to reduce overfitting.
XGBoost (optional): Gradient boosting alternative for potentially improved calibration and accuracy.
Fallback behavior: If XGBoost not installed, only RF models are trained.

## 6. Cross-Validation Metrics
Printed when `--ml-validate` is set:
- TotalGoals: MAE, RMSE
- 1X2: Accuracy, LogLoss (may be None if a fold has single class)
- BTTS: Accuracy, LogLoss

Example:
```
ML Cross-Validation Metrics:
  TotalGoals_RF: {'MAE': 1.43, 'RMSE': 1.91}
  1X2_RF: {'Accuracy': 0.54, 'LogLoss': 1.02}
  BTTS_RF: {'Accuracy': 0.62, 'LogLoss': 0.66}
```
Interpretation:
- Lower MAE/RMSE → better total goals prediction.
- Higher Accuracy, lower LogLoss → better classification.

## 7. Predict Mode Output
Each match suggestion includes:
```
ML Total Goals: 2.78 (model RF)
ML 1X2 probs: H=0.55 D=0.24 A=0.21 (model RF)
ML BTTS probs: Yes=0.61 No=0.39 (model RF)
Δ1X2: H=+0.03 D=-0.02 A=-0.01
ΔBTTS: Yes=+0.04 No=-0.04
```
Δ = ML probability minus Poisson probability.
Positive Δ indicates ML model assigns higher likelihood than baseline.

## 8. Recommended Usage Patterns
Train + predict sequentially:
```bash
python3 cli.py --task full-league --league E0 --ml-mode train --ml-validate
python3 cli.py --task full-league --league E0 --ml-mode predict
```
Single combined predict run (training occurs first implicitly):
```bash
python3 cli.py --task full-league --league E0 --ml-mode predict --ml-validate
```
Save models:
```bash
python3 cli.py --task full-league --league E0 --ml-mode train --ml-validate --ml-save-models --ml-models-dir models
```

## 9. Adjusting Sensitivity
| Scenario | Adjustment |
|----------|------------|
| Very recent form matters | Lower `--ml-decay` (e.g. 0.75) |
| Data scarce | Lower `--ml-min-samples` (e.g. 150) BUT expect higher variance |
| Want faster runs | Remove `--ml-validate` (skips CV printing) |
| Only RF desired | Set `--ml-algorithms rf` |
| Emphasize ensemble diversity later | Add XGBoost and possibly LightGBM (future extension) |

## 10. Interpreting Deltas
Large positive Δ for a 1X2 side: ML sees factors (form/rolling shots) boosting that outcome beyond goal-average based Poisson.
Large negative Δ: Poisson may be overestimating; ML picks up suppression signals (low recent attacking stats).

## 11. Model Persistence
When enabled:
```
models/ml_models_<LEAGUE>_<TIMESTAMP>.pkl
```
Contents: dict with keys: regression, classification, cv_metrics, meta.
Load example (future):
```python
import pickle
with open('models/ml_models_E0_20251115_120000.pkl','rb') as f:
    saved = pickle.load(f)
```

## 12. Limitations & Future Enhancements
| Area | Current | Future Option |
|------|---------|---------------|
| Calibration | Raw probabilities | Isotonic / Platt scaling |
| Market breadth | 1X2, BTTS, Total Goals | Handicap, Over exact lines |
| Features | Rolling + simple interactions | Advanced event data (xG, possession) |
| Validation | Simple k-fold | Time series split or expanding window |
| Ensemble | RF + optional XGB | Stacking (meta learner) |

## 13. Troubleshooting
| Symptom | Cause | Solution |
|---------|-------|----------|
| "ML modules not available" | Missing `scikit-learn` | `pip install -r requirements.txt` |
| "Insufficient samples" | Historical dataset small | Lower `--ml-min-samples` or add data |
| No Δ lines | Not in predict mode | Use `--ml-mode predict` |
| LogLoss None | Fold class imbalance | Increase samples or reduce folds |
| Slow training | Large dataset + high estimators | Reduce `n_estimators` or disable XGB |

## 14. Fast Parameter Tuning (Optional)
Try smaller forests for speed:
```bash
export SKLEARN_SET_CONFIG_ENABLE=True  # (optional placeholder if you script configs)
python3 automate_football_analytics_fullLeague.py --leagues E0 --ml-mode train --ml-algorithms rf --ml-validate
```
(Not yet configurable via flags—edit `ml_training.py` directly for advanced tuning.)

## 15. Security & Stability Notes
- All processing local; no network calls in ML modules.
- Pickle files are Python-version dependent—retrain after environment upgrades for safety.

## 16. Summary
ML Mode provides data-driven refinement over Poisson baseline using recent form and engineered features. Start with default settings, inspect CV metrics, then iterate decay and algorithm choices for performance gains.

---
**Questions or improvements?** Extend `ml_features.py` or propose new metrics in issues.

