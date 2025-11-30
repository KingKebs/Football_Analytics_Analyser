dc# Corner Predictions: Manual vs Automated Workflow Comparison

## Overview
This document compares the manual interactive workflow using `corners_analysis.py` directly versus the automated batch workflow using `automate_corner_predictions.py`.

## Workflows

### Manual Interactive Workflow
```bash
python3 corners_analysis.py --league E0 --train-model
```

**Output includes:**
- Full league data aggregation details
- Corner correlations (top 15 features)
- Half-split estimation sample (first 5 matches)
- Team statistics (top 5 home/away teams)
- **Regression model metrics (R², MAE)**
- Corners analysis summary
- Interactive match prediction prompts
- Team listings

### Automated Batch Workflow
```bash
python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --league E2 --auto
```

**Output includes:**
- Match log parsing with league resolution
- Full `corners_analysis.py` output (correlations, stats, **model metrics**)
- Batch predictions for all scheduled fixtures
- Export to structured JSON files
- Cleanup of old files

## Key Differences

### 1. **Prediction Modes**

The automated script has two modes:

#### Fast Mode (default: `--mode fast`)
- Runs `corners_analysis.py` **once** per league to generate team statistics
- Performs local predictions using the same formula as `corners_analysis.py`
- **Much faster** for multiple fixtures (0.01s for 17 matches)
- Replicates the prediction logic without calling corners_analysis per match
- Shows compact output: Total Mean, 1H, Range

#### Full Mode (`--mode full`)
- Calls `corners_analysis.py` for **each individual match**
- Slower but shows detailed analysis per prediction
- Useful for debugging or detailed single-match analysis

### 2. **Model Training**

Both workflows now support `--train-model`:
- **Manual**: `python3 corners_analysis.py --league E2 --train-model`
- **Automated**: Automatically includes `--train-model` flag (added in latest update)

**Model Metrics Displayed:**
```
============================================================
REGRESSION MODEL METRICS
============================================================
Total Corners Model:
  R² Score:  0.0919   # ~9% variance explained - corners are noisy!
  MAE:       2.676    # Mean Absolute Error on 10-corner average

Half-Split Ratio Model:
  R² Score:  0.8463   # 84% - half-split is highly predictable
  MAE:       0.01     # Very accurate 1H/2H ratio estimation
```

### 3. **Data Science Quality Assessment**

#### Strengths ✅
- **Feature Engineering**: 14 engineered features (Total_Shots, Shots_Per_Foul, Shot_Accuracy)
- **Strong Correlations**: HC (0.67), AC (0.60) with total corners
- **Half-Split Model**: Excellent R²=0.84 for predicting 1H vs 2H distribution
- **Team Aggregations**: Venue-specific stats (home/away averages)
- **Probabilistic Betting Lines**: Normal CDF for over/under recommendations

#### Weaknesses ⚠️
- **Low R² for Total Corners**: 0.09 R² means only ~9% variance explained
  - Corners are inherently noisy/unpredictable from basic match stats
  - MAE of 2.68 on mean of 10 corners is ~27% error
- **No Train/Test Split**: Model trained and evaluated on same data (optimistic metrics)
- **Linear Model Only**: May need non-linear models (random forest, gradient boosting)
- **Missing Features**: No player data, team form, referee tendencies, weather
- **Simple Averaging**: Match predictions use team averages, not a trained model

#### Recommendations for Improvement 🚀
1. **Add cross-validation** for honest performance metrics
2. **Try ensemble models** (RandomForest, XGBoost) for non-linear patterns
3. **Incorporate recency weighting** (recent matches weighted higher)
4. **Add interaction terms** (home advantage × opponent strength)
5. **Include external data** (injuries, weather, referee stats)
6. **Time-series features** (rolling averages, momentum indicators)

## Output Comparison

### Manual Run
```
CORNER CORRELATIONS (Top 15)
HC                            :  0.6174
AC                            :  0.5345
Total_Shots                   :  0.2884
...

REGRESSION MODEL METRICS
Total Corners Model: R²=0.10, MAE=2.64
Half-Split Ratio Model: R²=0.84, MAE=0.01

Enter home and away teams to generate corner predictions...
```

### Automated Run (Fast Mode)
```
STEP 2: Running corners analysis for E2
[Shows full corners_analysis.py output including correlations and model metrics]

[FAST] [E2] [1/17] 17:00 - Burton vs Blackpool
  ✓ Total Mean 9.04  1H 3.55  Range 6.57-11.5

[FAST] [E2] [2/17] 17:00 - Leyton Orient vs Exeter
  ✓ Total Mean 10.46  1H 4.11  Range 7.64-13.28

...

✓ Exported 5 predictions to: data/corners/batch_predictions_E2_20251115.json
```

## Files Generated

### By corners_analysis.py
- `data/corners/corners_analysis_<LEAGUE>_<TIMESTAMP>.csv` - Enriched match data
- `data/corners/corners_correlations_<LEAGUE>_<TIMESTAMP>.json` - Feature correlations
- `data/corners/team_stats_<LEAGUE>_<TIMESTAMP>.json` - Aggregated team statistics
- `data/corners/model_metrics_<LEAGUE>_<TIMESTAMP>.json` - **Regression metrics (R², MAE, coefficients)**
- `data/corners/match_prediction_<LEAGUE>_<HOME>_vs_<AWAY>_<TIMESTAMP>.json` - Single match predictions

### By automate_corner_predictions.py
- All of the above, plus:
- `data/todays_fixtures_<YYYYMMDD>.csv/json` - Parsed match fixtures
- `data/corners/batch_predictions_<LEAGUE>_<TIMESTAMP>.json` - Batch predictions per league
- Auto-cleanup of old files (keeps last 7 days)

## Recent Improvements (Nov 15, 2024)

1. ✅ **Added `--train-model` to automated workflow** - Model metrics now visible in batch runs
2. ✅ **Full stdout display** - Changed from filtering to showing complete corners_analysis output
3. ✅ **Model metrics console output** - R² and MAE now printed, not just saved to JSON
4. ✅ **Better documentation** - This comparison guide

## Usage Examples

### For Single League Analysis
```bash
# Interactive with model training
python3 corners_analysis.py --league E0 --train-model

# Automated batch (fast mode)
python3 automate_corner_predictions.py --input fixtures.log --league E0 --auto
```

### For Multi-League Batch Predictions
```bash
# Multiple leagues, fast mode
python3 automate_corner_predictions.py \
  --input tmp/corners/251115_match_games.log \
  --leagues E2,E3,E1 \
  --auto \
  --mode fast

# Force re-analysis (ignore today's cached data)
python3 automate_corner_predictions.py \
  --input fixtures.log \
  --leagues E0,E1 \
  --auto \
  --force
```

## Team Name Resolution Issues

**Problem**: 12 out of 17 fixtures failed with "Failed to resolve teams"

**Cause**: Team names in match log don't exactly match names in historical data
- Log: "Accrington", "MK Dons", "Notts Co"
- Data: "Accrington Stanley", "Milton Keynes Dons", "Notts County"

**Solution**: The fuzzy matching algorithm already exists but needs improvement:
```python
def resolve(name, df):
    if name in df.index:
        return name
    lname = name.lower()
    for idx in df.index:
        if idx.lower() == lname:
            return idx
    for idx in df.index:
        if lname in idx.lower():  # Partial match
            return idx
    return None
```

Consider using `difflib.get_close_matches()` or Levenshtein distance for better fuzzy matching.

## Conclusion

The automated workflow successfully chains `parse_match_log.py` → `corners_analysis.py` for batch predictions. While the output differs in format (verbose vs compact), both workflows now:
- ✅ Train and display regression models with R²/MAE metrics
- ✅ Show full analysis (correlations, team stats, half-split estimates)
- ✅ Generate probabilistic betting recommendations
- ✅ Export structured JSON for downstream use

The main limitation is **data science quality**: the total corners model has low predictive power (R²=0.09), indicating corners are hard to predict from basic match statistics alone. The workflow is production-ready for rapid predictions, but accuracy improvements require richer feature engineering and non-linear models.

