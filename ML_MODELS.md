# ML Models - Complete Documentation

## Table of Contents
1. [Quick Reference](#quick-reference)
2. [Predictions Fix](#predictions-fix)
3. [Signature Implementation](#signature-implementation)
4. [Technical Details](#technical-details)

---

## Quick Reference

### ML Predictions Overview
- **Purpose**: Predict 1X2, BTTS probabilities using XGBoost/RandomForest
- **Input**: Team strengths, historical matches, match features
- **Output**: Goal predictions, match outcome probabilities, DC probabilities
- **Modes**: `off` (Poisson only), `train` (learn from data), `predict` (use models)

### Running ML Analysis
```bash
# With ML predictions
python3 cli.py full-league --league E0 --ml-mode predict

# Train and predict
python3 cli.py full-league --league E0 --ml-mode train

# Poisson only (no ML)
python3 cli.py full-league --league E0 --ml-mode off
```

### ML Output Structure
```json
{
  "ml_prediction": {
    "pred_total_goals": 2.46,
    "pred_total_goals_model": "XGB",
    "prob_1x2_home": 0.34,
    "prob_1x2_draw": 0.05,
    "prob_1x2_away": 0.62,
    "prob_btts_yes": 0.46,
    "prob_btts_no": 0.54,
    "model_1x2": "RandomForest",
    "model_btts": "XGBoost"
  }
}
```

---

## Predictions Fix

### Issue: ML Predictions Signature Validation

**Problem:**
ML predictions were being validated against signature hashes that could be non-deterministic, causing false rejections of valid predictions.

**Root Cause:**
Two potential sources of non-determinism:
1. Floating-point precision in prediction values
2. Feature engineering hash inconsistencies
3. Model state or random seed not properly managed

**Solution:**
Implemented deterministic prediction signature by:

1. **Normalization**: Round all float values to 2 decimal places
2. **Serialization**: Create JSON with sorted keys for deterministic order
3. **Hashing**: Use SHA256 hash of serialized data for comparison

**Implementation:**
```python
import json
import hashlib

def compute_prediction_signature(match_id, predictions):
    """Generate deterministic signature for ML predictions."""
    payload = {
        'match': match_id,
        'total_goals': round(float(predictions['pred_total_goals']), 2),
        'prob_home': round(float(predictions['prob_1x2_home']), 2),
        'prob_draw': round(float(predictions['prob_1x2_draw']), 2),
        'prob_away': round(float(predictions['prob_1x2_away']), 2),
        'prob_btts_yes': round(float(predictions['prob_btts_yes']), 2),
        'prob_btts_no': round(float(predictions['prob_btts_no']), 2),
    }
    
    # Deterministic JSON serialization
    sig_json = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    
    # Generate hash
    sig_hash = hashlib.sha256(sig_json.encode('utf-8')).hexdigest()[:12]
    
    return sig_hash
```

**Format in Streamlit:**
```
Prediction Signature: MCI-MON|TG=2.46|1X2=0.34,0.05,0.62|BTTS=0.46,0.54|a7f2b9c3d1e5
                      │       │   │          │              │           └─ Hash
                      │       │   │          │              └─ BTTS probs
                      │       │   │          └─ 1X2 probs (H, D, A)
                      │       │   └─ Total goals
                      │       └─ Team abbreviations
                      └─ Match ID
```

**Verification:**
```python
# Verify predictions are consistent
sig1 = compute_prediction_signature("Team1_Team2", pred1)
sig2 = compute_prediction_signature("Team1_Team2", pred1)
assert sig1 == sig2  # Should always match
```

---

## Signature Implementation

### Prediction Signature Components

1. **Match Identifier**
   - Format: `HOME[:3]-AWAY[:3]` (first 3 letters of team names)
   - Example: `MCI-MON` for Manchester City vs Manchester United

2. **Prediction Values**
   - Total Goals: rounded to 2 decimals
   - 1X2 probs: [Home, Draw, Away] each to 2 decimals
   - BTTS probs: [Yes, No] each to 2 decimals

3. **Hash**
   - SHA256 of JSON-serialized payload
   - Truncated to 12 characters for readability
   - Ensures data integrity

4. **Display Format**
   ```
   HOME-AWAY|TG=X.XX|1X2=H,D,A|BTTS=Y,N|HASH
   ```

### Signature Validation Process

```python
def validate_prediction_signature(sig_string, predictions):
    """Validate a prediction signature string."""
    parts = sig_string.split('|')
    
    if len(parts) != 5:
        return False, "Invalid signature format"
    
    match_id, tg_str, x12_str, btts_str, hash_part = parts
    
    # Validate total goals
    expected_tg = round(float(predictions['pred_total_goals']), 2)
    actual_tg = float(tg_str.replace('TG=', ''))
    if abs(expected_tg - actual_tg) > 0.01:
        return False, "Total goals mismatch"
    
    # Validate 1X2 probs
    expected_hash = compute_prediction_signature(match_id, predictions)
    if expected_hash != hash_part:
        return False, "Hash mismatch"
    
    return True, "Valid"
```

### Why Determinism Matters

**Before Fix:**
```
Run 1: Signature A7F2B9C3D1E5
Run 2: Signature A7F2B9C3D1E6  ← Different! (non-deterministic)
Result: Prediction rejected as "new"
```

**After Fix:**
```
Run 1: Signature A7F2B9C3D1E5
Run 2: Signature A7F2B9C3D1E5  ← Same! (deterministic)
Result: Prediction validated, cached if previously computed
```

---

## Technical Details

### Model Training

```python
# Training pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Features for 1X2 prediction
features_1x2 = [
    'home_attack_strength',
    'away_attack_strength',
    'home_defence_strength',
    'away_defence_strength',
    'home_recent_form',
    'away_recent_form',
    'days_rest_home',
    'days_rest_away'
]

# Features for BTTS prediction
features_btts = [
    'home_attack_strength',
    'away_attack_strength',
    'home_defence_strength',
    'away_defence_strength',
    'head_to_head_btts_ratio'
]

# Training
model_1x2 = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model_1x2.fit(X_train[features_1x2], y_train_1x2)

model_btts = XGBRegressor(
    n_estimators=50,
    max_depth=5,
    random_state=42
)
model_btts.fit(X_train[features_btts], y_train_btts)
```

### Feature Engineering

```python
def compute_team_features(history_df, team, last_n=6):
    """Compute team features from historical matches."""
    played = history_df[
        (history_df['HomeTeam'] == team) | 
        (history_df['AwayTeam'] == team)
    ].tail(last_n)
    
    if played.empty:
        return {
            'attack_strength': 1.0,
            'defence_strength': 1.0,
            'recent_form': 0.0
        }
    
    goals_for, goals_against = 0, 0
    for _, row in played.iterrows():
        if row['HomeTeam'] == team:
            goals_for += row['FTHG']
            goals_against += row['FTAG']
        else:
            goals_for += row['FTAG']
            goals_against += row['FTHG']
    
    return {
        'attack_strength': goals_for / max(1, len(played)),
        'defence_strength': goals_against / max(1, len(played)),
        'recent_form': (goals_for - goals_against) / max(1, len(played))
    }
```

### Model Validation

```python
from sklearn.metrics import accuracy_score, log_loss

# Validate 1X2 predictions
y_pred = model_1x2.predict(X_test[features_1x2])
accuracy = accuracy_score(y_test_1x2, np.argmax(y_pred, axis=1))
log_loss_val = log_loss(y_test_1x2, y_pred)

print(f"1X2 Accuracy: {accuracy:.2%}")
print(f"1X2 Log Loss: {log_loss_val:.4f}")

# Validate BTTS predictions
y_pred_btts = model_btts.predict(X_test[features_btts])
btts_accuracy = accuracy_score(y_test_btts, y_pred_btts > 0.5)

print(f"BTTS Accuracy: {btts_accuracy:.2%}")
```

### Model Performance Monitoring

```python
# Track prediction stability
def compute_prediction_variance(history_of_predictions):
    """Measure variance in predictions for same match over time."""
    total_goals = [p['pred_total_goals'] for p in history_of_predictions]
    
    variance = np.var(total_goals)
    std_dev = np.std(total_goals)
    
    if std_dev > 0.5:
        print(f"Warning: High variance in predictions (std={std_dev:.2f})")
    
    return {
        'mean': np.mean(total_goals),
        'variance': variance,
        'std_dev': std_dev,
        'stability': 'HIGH' if variance < 0.1 else 'MEDIUM' if variance < 0.5 else 'LOW'
    }
```

### ML Mode Configuration

```yaml
# config/train_config.yaml
ml_config:
  mode: predict  # off, train, or predict
  algorithms:
    - xgboost
    - random_forest
  
  # Training parameters
  train:
    min_samples: 300  # Minimum matches to train
    decay_factor: 0.85  # Weight recent matches more
    test_split: 0.2
    
  # Prediction parameters
  predict:
    confidence_threshold: 0.5
    validate_signature: true
    
  # Model storage
  models_dir: models/
  save_models: true
```

---

## ML vs Poisson Comparison

The system tracks differences between ML and Poisson predictions:

```python
{
  "ml_vs_poisson": {
    "1X2": {
      "delta_home": -0.05,  # ML predicts 5% lower home win
      "delta_draw": +0.02,  # ML predicts 2% higher draw
      "delta_away": +0.03   # ML predicts 3% higher away win
    },
    "BTTS": {
      "delta_yes": -0.08,   # ML predicts 8% lower BTTS Yes
      "delta_no": +0.08
    }
  }
}
```

Interpretation:
- Positive delta: ML is more optimistic about outcome
- Negative delta: ML is more pessimistic
- Magnitude: Confidence difference from Poisson baseline

---

## Summary

### Key Features
✅ XGBoost & RandomForest ensemble predictions  
✅ Deterministic prediction signatures  
✅ Feature engineering from historical data  
✅ Comparison with Poisson baseline  
✅ Model validation and monitoring  

### Usage
```bash
# Predictions only (no model training)
python3 cli.py full-league --league E0 --ml-mode predict

# Train models from historical data
python3 cli.py full-league --league E0 --ml-mode train

# Poisson analysis (no ML)
python3 cli.py full-league --league E0 --ml-mode off
```

### Output
ML predictions appear in Streamlit under "ML Predictions" tab showing:
- Total goals predictions
- 1X2 probabilities
- BTTS probabilities
- Double Chance derived odds
- Comparison with Poisson baseline

