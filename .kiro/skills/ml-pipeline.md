---
name: ML Pipeline Skill
description: Manage model training, validation, feature engineering, and prediction workflow
applies_to: ["src/ml_*.py", "models/**/*.json", "test_*.py"]
---

# ML Pipeline Skill

## Components
- **Training** (`src/ml_training.py`): XGBoost/RandomForest model training
- **Features** (`src/ml_features.py`): Feature engineering and selection
- **Utilities** (`src/ml_utils.py`): Model loading, prediction, freshness checks
- **Evaluation** (`src/ml_evaluation.py`): Metrics, calibration, validation

## Training Workflow
1. Load historical data (league_data_*.csv)
2. Engineer features (team strength, home/away, recent form)
3. Split train/test
4. Train XGBoost + RandomForest
5. Evaluate and save artifacts
6. Store metadata (git commit, date, performance)

## Feature Engineering
- Team strength calculations
- Home/away advantage
- Recent form indicators
- Head-to-head stats
- Seasonal variations

## Model Validation
- Cross-validation (k-fold)
- Test set evaluation
- Calibration curves
- Signature consistency checks
- Comparison vs Poisson baseline

## Making Predictions
```python
from src.ml_utils import load_latest_model, predict

model_path = load_latest_model()
predictions = predict(model_path, fixture_data)
```

## Metadata Tracking
Models stored with:
- `model.pkl` - Trained model
- `feature_manifest.json` - Feature definitions
- `metrics.json` - Performance stats
- `git_commit.txt` - Code version
- `calibration_curve.json` - Calibration data
- `validation_predictions_sample.json` - Sample predictions

## Freshness Management
- Check `src/ml_utils.check_model_freshness()`
- Default max age: 7 days
- Triggers retraining if exceeded
- Fallback to Poisson if no model available

## Common Tasks
- **Retrain Model**: Update features, gather latest data, run training
- **Compare Predictions**: ML vs Poisson analysis
- **Debug Signature Issues**: Check prediction hashing
- **Validate Calibration**: Review calibration curves
