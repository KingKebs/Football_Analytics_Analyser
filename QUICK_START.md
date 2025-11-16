# Quick Reference: Running Predictions with Steps 1-4

## ✅ YOUR SCENARIO: Match Log with E2 & E3 Games

**Your file:** `tmp/corners/251115_match_games.log`

### RECOMMENDED COMMAND (Complete Workflow):
```bash
python3 automate_corner_predictions.py \
  --input tmp/corners/251115_match_games.log \
  --leagues E2,E3 \
  --train-model \
  --force \
  --auto \
  --mode fast
```

**This gives you:**
- ✅ Model training with Steps 1-4 improvements (CV, Ensembles, Recency, Interactions)
- ✅ Predictions for all 17 matches in your log
- ✅ Output files:
  - `data/corners/batch_predictions_E2_<TIMESTAMP>.json` (5 E2 games)
  - `data/corners/batch_predictions_E3_<TIMESTAMP>.json` (8 E3 games)
  - `data/corners/batch_predictions_E2+E3_<TIMESTAMP>.json` (combined 13 games)
  - `data/corners/model_metrics_E2_<TIMESTAMP>.json` (E2 model metrics)
  - `data/corners/model_metrics_E3_<TIMESTAMP>.json` (E3 model metrics)

---

## WHAT JUST HAPPENED (Test Run Results):

**Successfully processed:** 13 of 17 matches
- **E2 (League Two):** 5 predictions ✓
- **E3 (League One):** 8 predictions ✓

**Sample predictions generated:**

### Burton vs Blackpool (E2)
- Total corners: **9.04** (range: 6.57-11.50)
- 1H corners: **3.55**
- 2H corners: **5.49**
- **Recommendation:** OVER 7.5 (72.7% probability)

### Notts Co vs Harrogate (E3)
- Total corners: **10.72** (range: 8.12-13.31)
- 1H corners: **4.23**
- 2H corners: **6.49**

---

## FLAG MEANINGS:

| Flag | Purpose | When to Use |
|------|---------|-------------|
| `--train-model` | Train models with Steps 1-4 | First run of the day or when you want updated metrics |
| `--force` | Re-run analysis even if done today | When data has been updated |
| `--auto` | No interactive prompts | Batch processing (recommended for logs) |
| `--mode fast` | Use cached team stats | Quick predictions (recommended) |
| `--mode full` | Run full analysis per match | More detailed but slower |

---

## EXAMPLE OUTPUTS:

### 1. Model Metrics (E3)
```
Total Corners Model (RandomForest, Weighted CV):
  R² mean±std:  0.252 ± 0.030
  MAE mean±std: 2.320 ± 0.098
```
**Interpretation:** RandomForest predicts with ~2.3 corners average error

### 2. Match Prediction JSON
```json
{
  "home_team": "Burton",
  "away_team": "Blackpool",
  "pred_total_corners_mean": 9.04,
  "pred_1h_corners_mean": 3.55,
  "total_corner_lines": [
    {"line": 7.5, "p_over": 0.727, "recommendation": "OVER"},
    {"line": 10.5, "p_over": 0.269, "recommendation": "UNDER"}
  ]
}
```

---

## ALTERNATIVE WORKFLOWS:

### Just predictions (no model training)
```bash
python3 automate_corner_predictions.py \
  --input tmp/corners/251115_match_games.log \
  --leagues E2,E3 \
  --auto \
  --mode fast
```
*Uses existing team stats (faster, no CV metrics printed)*

### With model training + view metrics
```bash
python3 automate_corner_predictions.py \
  --input tmp/corners/251115_match_games.log \
  --leagues E2,E3 \
  --train-model \
  --auto \
  --mode fast

# Then view metrics
cat data/corners/model_metrics_E2_*.json | python3 -m json.tool
```

### Single league analysis
```bash
python3 corners_analysis.py --league E3 --no-prompt --train-model
```

### Single match prediction
```bash
python3 corners_analysis.py \
  --league E3 \
  --home-team "Notts Co" \
  --away-team "Harrogate" \
  --no-prompt
```

---

## STEPS 1-4 ARE AUTOMATICALLY APPLIED WHEN:
- ✅ You use `--train-model` flag
- ✅ The script loads historical data
- ✅ Features are engineered (including interaction terms)
- ✅ Models are trained with recency weighting
- ✅ Cross-validation is computed (both weighted and unweighted)
- ✅ All metrics are saved and printed

**You don't need to do anything special - it's all integrated!**

---

## FILES CREATED FROM YOUR RUN:

```
data/corners/
├── batch_predictions_E2_20251115_105359.json      # 5 E2 predictions
├── batch_predictions_E3_20251115_105359.json      # 8 E3 predictions  
├── batch_predictions_E2+E3_20251115_105359.json   # Combined 13 predictions
├── model_metrics_E2_20251115_*.json               # E2 model performance
└── model_metrics_E3_20251115_*.json               # E3 model performance
```

---

## TROUBLESHOOTING:

**"Failed to resolve teams" (like Cambridge Utd vs Barnet)**
- Team names in your log don't match historical data names
- Check available teams: `python3 corners_analysis.py --league E3 --list-teams`
- Historical data might not have these teams (e.g., Barnet not in recent E3 seasons)

**"No CSV files found"**
```bash
python3 cli.py --task download --leagues E2,E3
```

**Want to see what the script is doing?**
- Remove `--auto` for interactive mode
- Remove `2>&1 | tail` to see full output

---

## NEXT STEPS:

1. **Run with training to see Steps 1-4 metrics:**
   ```bash
   python3 automate_corner_predictions.py \
     --input tmp/corners/251115_match_games.log \
     --leagues E2,E3 \
     --train-model \
     --force \
     --auto \
     --mode fast
   ```

2. **Check the predictions:**
   ```bash
   cat data/corners/batch_predictions_E2+E3_*.json | python3 -m json.tool | less
   ```

3. **View model metrics:**
   ```bash
   LEAGUE=E2 PYTHONPATH=. python3 tools/run_weighted_metrics.py
   LEAGUE=E3 PYTHONPATH=. python3 tools/run_weighted_metrics.py
   ```

---

## Full League ML Mode (Goals) – Quick Snippet
Enable ML (RandomForest + optional XGBoost) for goal markets:
```bash
# Train + CV metrics
python3 cli.py --task full-league --league E0 --ml-mode train --ml-validate

# Predict with ML augmentation (after or implicit training)
python3 cli.py --task full-league --league E0 --ml-mode predict --ml-validate

# Save trained models
python3 cli.py --task full-league --league E0 --ml-mode train --ml-validate --ml-save-models --ml-models-dir models
```
See `ML_MODE_GUIDE.md` for full details on features, weighting, CV metrics, deltas vs Poisson, and tuning.

---

## Avoiding environment issues across shells (zsh, bash, Terminal, iTerm)
If you see errors like `ModuleNotFoundError: No module named 'pandas'`, it usually means the virtual environment isn’t activated in that shell.

Two easy fixes:

1) Use the project runner scripts (they auto-create/activate .venv and install deps on first run):
```bash
# From project root
./tools/fa automate_corner_predictions.py \
  --input tmp/corners/251115_match_games.log \
  --leagues E2,E3 \
  --train-model \
  --force \
  --auto \
  --mode fast

# Unified CLI (recommended)
./tools/fa-cli --task full-league --league E0 --ml-mode predict --ml-validate
```

2) Add a helper alias to your ~/.zshrc so any shell picks the project’s venv automatically:
```zsh
# Football Analytics Analyser helpers
fa() {
  local PROJ="$HOME/sites/Development/Football_Analytics_Analyser"
  command -v python3 >/dev/null 2>&1 || { echo "python3 not found"; return 1; }
  cd "$PROJ" || return 1
  if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv || return 1
    .venv/bin/pip install -U pip setuptools wheel || return 1
    .venv/bin/pip install -r requirements.txt || return 1
  fi
  export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
  .venv/bin/python "$@"
}

# Shortcut to the unified CLI
alias fa-cli='fa cli.py'
```
After adding this, reload your shell:
```bash
source ~/.zshrc
```
Then run:
```bash
fa automate_corner_predictions.py --input tmp/corners/251115_match_games.log --leagues E2,E3 --train-model --force --auto --mode fast
fa-cli --task full-league --league E0 --ml-mode predict --ml-validate
```

These approaches ensure a consistent Python environment (pandas, numpy, sklearn, etc.) even when switching between different shells or terminals.

---

**That's it! All Steps 1-4 improvements are working automatically.** 🎉
