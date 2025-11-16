# Complete Workflow Guide: Steps 1-4 Integration

## Overview
This guide shows how to run corner predictions for multiple matches across different leagues with all improvements from Steps 1-4:
- ✅ Step 1: Cross-validation for honest metrics
- ✅ Step 2: Ensemble models (RandomForest, XGBoost)
- ✅ Step 3: Recency weighting (exponential decay)
- ✅ Step 4: Interaction features (shots×fouls, etc.)

## Your Match Log Format
```
tmp/corners/251115_match_games.log
```
Contains games from multiple leagues (E2, E3) with format:
```
17:00
Burton
Blackpool

17:00
Leyton Orient
Exeter
...
```

---

## Method 1: Automated Workflow (Recommended) ⭐

**For batch predictions across multiple leagues with model training:**

```bash
python3 automate_corner_predictions.py \
  --input tmp/corners/251115_match_games.log \
  --leagues E2,E3 \
  --train-model \
  --force \
  --auto \
  --mode fast
```

**What this does:**
1. Parses your match log and identifies leagues
2. Runs corners_analysis for each league (E2, E3) with `--train-model`
   - Trains Linear, RandomForest (and XGBoost if installed)
   - Computes 5-fold CV metrics (unweighted + weighted)
   - Saves metrics to `data/corners/model_metrics_<LEAGUE>_<TIMESTAMP>.json`
3. Generates corner predictions for each match
4. Exports predictions to:
   - `data/corners/batch_predictions_E2_<TIMESTAMP>.json`
   - `data/corners/batch_predictions_E3_<TIMESTAMP>.json`
   - `data/corners/batch_predictions_E2+E3_<TIMESTAMP>.json` (combined)

**Flags explained:**
- `--train-model`: Runs model training with Steps 1-4 improvements
- `--force`: Re-runs analysis even if done today
- `--auto`: No interactive prompts per match
- `--mode fast`: Uses pre-computed team stats (faster than per-match analysis)

**Output includes:**
- Total corner predictions (mean + range)
- 1H/2H split predictions
- Suggested betting lines with probabilities (Over/Under)

---

## Method 2: League-by-League Analysis

**If you want to analyze one league at a time:**

```bash
# For E2 (League Two)
python3 corners_analysis.py --league E2 --no-prompt --train-model

# For E3 (League One)
python3 corners_analysis.py --league E3 --no-prompt --train-model
```

**What this does:**
- Loads all historical data for the league
- Engineers features (base + interaction terms from Step 4)
- Trains models with recency weighting (Step 3)
- Prints regression metrics (Steps 1-2)
- Saves team stats and correlations
- NO match predictions (use Method 1 or 3 for that)

---

## Method 3: Single Match Prediction

**For one specific match with full analysis:**

```bash
python3 corners_analysis.py \
  --league E3 \
  --home-team "Notts Co" \
  --away-team "Harrogate" \
  --no-prompt
```

**What this does:**
- Uses pre-computed team stats for the league
- Fuzzy team name matching (automatically finds "Notts County" if stats use that)
- Generates prediction JSON with:
  - Total corners prediction
  - 1H/2H split
  - Suggested betting lines
- Exports to `data/corners/match_prediction_E3_Notts_Co_vs_Harrogate_<TIMESTAMP>.json`

---

## Method 4: Quick Metrics Recomputation

**If you've already run analysis and just want to see latest metrics:**

```bash
# For E2
LEAGUE=E2 PYTHONPATH=. python3 tools/run_weighted_metrics.py

# For E3
LEAGUE=E3 PYTHONPATH=. python3 tools/run_weighted_metrics.py
```

**What this does:**
- Loads latest enriched CSV for the league
- Recomputes all CV metrics (unweighted + weighted)
- Prints regression model metrics
- Fast way to verify Steps 1-4 improvements

---

## Understanding the Output

### Model Metrics (from --train-model)
```
Total Corners Model (RandomForest, 5-fold CV):
  R² mean±std:  0.237 ± 0.018
  MAE mean±std: 2.323 ± 0.074

Total Corners Model (Weighted CV):
  RF R² mean±std:      0.252 ± 0.030
  RF MAE mean±std:     2.320 ± 0.098
```
- **Unweighted CV**: Standard cross-validation
- **Weighted CV**: Recency-weighted (Step 3) - recent matches count more
- **RF outperforms Linear** for total corners
- **MAE ~2.3**: Average error of ~2.3 corners per match

### Match Predictions
```json
{
  "home_team": "Notts Co",
  "away_team": "Harrogate",
  "pred_total_corners_mean": 10.72,
  "pred_1h_corners_mean": 4.23,
  "pred_2h_corners_mean": 6.49,
  "total_corner_lines": [
    {"line": 10.5, "p_over": 0.523, "p_under": 0.477, "recommendation": null},
    {"line": 11.5, "p_over": 0.423, "p_under": 0.577, "recommendation": null}
  ]
}
```

---

## File Locations

**Input:**
- Match log: `tmp/corners/251115_match_games.log`
- Historical data: `football-data/all-euro-football/<LEAGUE>_<SEASON>.csv`

**Output:**
- Model metrics: `data/corners/model_metrics_<LEAGUE>_<TIMESTAMP>.json`
- Team stats: `data/corners/team_stats_<LEAGUE>_<TIMESTAMP>.json`
- Batch predictions: `data/corners/batch_predictions_<LEAGUES>_<TIMESTAMP>.json`
- Single predictions: `data/corners/match_prediction_<LEAGUE>_<HOME>_vs_<AWAY>_<TIMESTAMP>.json`
- Enriched analysis: `data/corners/corners_analysis_<LEAGUE>_<TIMESTAMP>.csv`

---

## Multi-League Scenarios

### Your log has E2 and E3 matches
```bash
python3 automate_corner_predictions.py \
  --input tmp/corners/251115_match_games.log \
  --leagues E2,E3 \
  --train-model \
  --auto \
  --mode fast
```

### All Premier League + Championship
```bash
python3 automate_corner_predictions.py \
  --input tmp/corners/your_games.log \
  --leagues E0,E1 \
  --train-model \
  --auto \
  --mode fast
```

### European leagues
```bash
python3 automate_corner_predictions.py \
  --input tmp/corners/your_games.log \
  --leagues SP1,D1,I1,F1 \
  --train-model \
  --auto \
  --mode fast
```

---

## Troubleshooting

### "Team not found"
- The script uses fuzzy matching, but check team names in:
  ```bash
  python3 corners_analysis.py --league E3 --list-teams
  ```

### "No CSV files found"
- Download historical data first:
  ```bash
  python3 cli.py --task download --leagues E2,E3
  ```

### Want to see intermediate steps
- Remove `--auto` flag for interactive confirmation per match
- Remove `--mode fast` to run full per-match analysis (slower but more detailed)

### Check model improvements
- Compare metrics JSON files from different runs
- Earlier files (before Steps 1-4) won't have weighted CV metrics

---

## Performance Notes

- **fast mode**: ~0.01s per match (uses pre-computed team stats)
- **full mode**: ~2-5s per match (runs full corners_analysis per match)
- **Model training**: ~10-30s per league (one-time, cached for the day unless `--force`)

---

## Summary Commands

**Most common workflow:**
```bash
# Run everything for today's matches
python3 automate_corner_predictions.py \
  --input tmp/corners/251115_match_games.log \
  --leagues E2,E3 \
  --train-model \
  --force \
  --auto \
  --mode fast
```

**Check results:**
```bash
# Latest predictions
cat data/corners/batch_predictions_E2+E3_*.json | jq '.predictions[0]'

# Latest metrics
cat data/corners/model_metrics_E3_*.json | jq '{r2: .rf_total_r2_cv_mean, mae: .rf_total_mae_cv_mean}'
```

---

**All Steps 1-4 improvements are automatically applied when you use `--train-model` flag!** 🎉

