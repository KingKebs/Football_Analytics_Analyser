# Understanding the Script Relationships

## Main Workflow Script vs Helper Tools

### 🎯 PRIMARY SCRIPT (Use This for Production)

```bash
python3 automate_corner_predictions.py \
  --input tmp/corners/251115_match_games.log \
  --leagues E2,E3 \
  --train-model \
  --force \
  --auto \
  --mode fast
```

**What it does (complete pipeline):**
1. ✅ Parses your match log → identifies leagues and fixtures
2. ✅ For each league (E2, E3):
   - Runs `corners_analysis.py` with `--train-model`
   - Trains models with Steps 1-4 (CV, Ensembles, Recency, Interactions)
   - **Prints all metrics to console**
   - Saves metrics to JSON files
   - Computes team statistics
3. ✅ Generates corner predictions for all matches
4. ✅ Exports predictions to JSON files

**This is your ONE-STOP script. No other scripts needed!**

---

## 🔧 Helper Tools (Optional - For Development/Debugging Only)

### 1. `tools/run_weighted_metrics.py` 

**When to use:** RARELY - only for quick metric recomputation

```bash
LEAGUE=E2 PYTHONPATH=. python3 tools/run_weighted_metrics.py
```

**What it does:**
- Loads the latest enriched CSV (already processed)
- Recomputes and prints metrics only
- Does NOT run predictions
- Does NOT process new data

**Use cases:**
- You modified the `train_models()` function and want to test changes
- You want to see metrics without running the full pipeline
- You're debugging metric calculations

**You DON'T need this for normal operations!**

---

### 2. `corners_analysis.py` (Direct Use)

**When to use:** For league analysis without automation

```bash
# Analyze one or more leagues
python3 corners_analysis.py --league E2,E3 --no-prompt --train-model

# Single match prediction
python3 corners_analysis.py --league E3 --home-team "Notts Co" --away-team "Harrogate"
```

**What it does:**
- Processes historical data for specified leagues
- Engineers features (Steps 1-4)
- Trains models and prints metrics
- Can predict individual matches

**Use cases:**
- You want league analysis without batch predictions
- You want to predict ONE specific match manually
- You're exploring team statistics

---

## 📊 Where Metrics Come From in Main Workflow

When you run the **main workflow** with `--train-model`:

```
automate_corner_predictions.py
  ├─> Runs: corners_analysis.py --league E2 --no-prompt --train-model
  │   └─> Prints: REGRESSION MODEL METRICS (E2)
  │   └─> Saves: data/corners/model_metrics_E2_*.json
  │
  ├─> Runs: corners_analysis.py --league E3 --no-prompt --train-model
  │   └─> Prints: REGRESSION MODEL METRICS (E3)
  │   └─> Saves: data/corners/model_metrics_E3_*.json
  │
  └─> Generates predictions for all matches
      └─> Saves: data/corners/batch_predictions_*.json
```

**The metrics ARE printed to your console automatically!**

---

## 🎭 Why `tools/run_weighted_metrics.py` Was Created

During development of Steps 1-4, we needed a way to:
1. Test metric calculations quickly
2. Recompute metrics after code changes
3. Verify results without running full analysis

**It was a development tool that's now superseded by the integrated workflow.**

---

## ✅ Recommended Usage Patterns

### Pattern 1: Daily Predictions (Your Use Case)
```bash
# Run once per day for your match log
python3 automate_corner_predictions.py \
  --input tmp/corners/251115_match_games.log \
  --leagues E2,E3 \
  --train-model \
  --force \
  --auto \
  --mode fast
```
**Metrics printed:** ✅ Yes, during analysis step  
**Predictions generated:** ✅ Yes, for all matches  
**Need other scripts:** ❌ No

---

### Pattern 2: Quick Predictions (Models Already Trained Today)
```bash
# Skip training if already done today
python3 automate_corner_predictions.py \
  --input tmp/corners/251115_match_games.log \
  --leagues E2,E3 \
  --auto \
  --mode fast
```
**Metrics printed:** ❌ No (uses cached stats)  
**Predictions generated:** ✅ Yes, faster  
**Need other scripts:** ❌ No

---

### Pattern 3: Single Match Exploration
```bash
# Predict one specific match
python3 corners_analysis.py \
  --league E3 \
  --home-team "Walsall" \
  --away-team "Colchester"
```
**Metrics printed:** ❌ No  
**Predictions generated:** ✅ Yes, for one match  
**Need other scripts:** ❌ No

---

### Pattern 4: League Analysis Only (No Predictions)
```bash
# Analyze E2 and E3, see team stats and metrics
python3 corners_analysis.py --league E2,E3 --no-prompt --train-model
```
**Metrics printed:** ✅ Yes  
**Predictions generated:** ❌ No  
**Need other scripts:** ❌ No

---

## 🗑️ Can I Delete `tools/run_weighted_metrics.py`?

**You CAN, but it's harmless to keep it.**

It's useful if you:
- Modify the `train_models()` function
- Want to test metric calculations quickly
- Are doing data science development work

For your daily workflow generating predictions, **you'll never need it.**

---

## 📝 Summary

**Your main command does EVERYTHING:**
```bash
python3 automate_corner_predictions.py \
  --input tmp/corners/251115_match_games.log \
  --leagues E2,E3 \
  --train-model \
  --force \
  --auto \
  --mode fast
```

**What it includes:**
- ✅ Steps 1-4 improvements (automatic)
- ✅ Model training with metrics (printed to console)
- ✅ Corner predictions for all matches
- ✅ JSON exports of everything

**You don't need any other scripts for normal operations!**

The helper tool `tools/run_weighted_metrics.py` is just a development convenience that's now redundant for your workflow.

