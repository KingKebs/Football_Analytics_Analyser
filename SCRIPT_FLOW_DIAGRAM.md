# Script Flow Diagram

## 🎯 YOUR MAIN WORKFLOW (One Command Does Everything)

```
┌─────────────────────────────────────────────────────────────────┐
│  python3 automate_corner_predictions.py                        │
│    --input tmp/corners/251115_match_games.log                  │
│    --leagues E2,E3                                             │
│    --train-model  --force  --auto  --mode fast                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ├─── Step 1: Parse match log
                              │    └─> Extract E2 and E3 fixtures
                              │
                              ├─── Step 2: For each league (E2, E3):
                              │    │
                              │    ├─> Run corners_analysis.py with --train-model
                              │    │   │
                              │    │   ├── Load historical data
                              │    │   ├── Engineer features (+ interactions)
                              │    │   ├── Train models (Linear, RF, XGBoost)
                              │    │   ├── Compute CV metrics (weighted + unweighted)
                              │    │   ├── 📊 PRINT METRICS TO CONSOLE ⭐
                              │    │   ├── Save model_metrics_*.json
                              │    │   ├── Calculate team stats
                              │    │   └── Save team_stats_*.json
                              │    │
                              │    └─> Cache stats for predictions
                              │
                              └─── Step 3: Generate predictions
                                   │
                                   ├─> For each match in E2
                                   │   └─> Predict corners (fast mode)
                                   │
                                   ├─> For each match in E3
                                   │   └─> Predict corners (fast mode)
                                   │
                                   └─> Export results
                                       ├── batch_predictions_E2_*.json
                                       ├── batch_predictions_E3_*.json
                                       └── batch_predictions_E2+E3_*.json

✅ ALL STEPS 1-4 IMPROVEMENTS AUTOMATICALLY APPLIED
✅ METRICS PRINTED DURING STEP 2 (see console output)
✅ PREDICTIONS GENERATED IN STEP 3
✅ NO OTHER SCRIPTS NEEDED
```

---

## 🔧 Alternative: Direct League Analysis (No Predictions)

```
┌─────────────────────────────────────────────────────────────────┐
│  python3 corners_analysis.py                                    │
│    --league E2,E3                                              │
│    --no-prompt                                                 │
│    --train-model                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ├─── For E2:
                              │    ├── Load data
                              │    ├── Engineer features
                              │    ├── Train models
                              │    ├── 📊 PRINT METRICS
                              │    └── Save team stats
                              │
                              └─── For E3:
                                   ├── Load data
                                   ├── Engineer features
                                   ├── Train models
                                   ├── 📊 PRINT METRICS
                                   └── Save team stats

✅ STEPS 1-4 APPLIED
✅ METRICS PRINTED
❌ NO PREDICTIONS (only analysis)
```

---

## 🛠️ Helper Tool (Development Only - OPTIONAL)

```
┌─────────────────────────────────────────────────────────────────┐
│  LEAGUE=E2 PYTHONPATH=. python3 tools/run_weighted_metrics.py  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              └─── Find latest enriched CSV
                                   └─── Load existing data
                                        └─── Recompute metrics
                                             └─── 📊 PRINT METRICS

⚠️  DEVELOPMENT/DEBUGGING TOOL ONLY
✅  Quick metric recomputation from cached data
❌  Does NOT process new data
❌  Does NOT generate predictions
❌  NOT NEEDED for normal workflow
```

---

## 📊 Where Do Metrics Appear?

### In Main Workflow Output:

```bash
$ python3 automate_corner_predictions.py --input ... --leagues E2,E3 --train-model ...

======================================================================
STEP 2: Running corners analysis for E2
======================================================================
Running: python3 corners_analysis.py --league E2 --no-prompt --train-model

✓ Loaded 1234 matches from /tmp/combined_E2.csv (E2)
✓ Features cleaned and prepared
✓ 15 new features engineered

============================================================
REGRESSION MODEL METRICS                    👈 HERE!
============================================================
Total Corners Model (Linear, in-sample):
  R² Score:  0.0919
  MAE:       2.676
Total Corners Model (Linear, 5-fold CV):
  R² mean±std:  0.0694 ± 0.0235
  MAE mean±std: 2.697 ± 0.078
Total Corners Model (RandomForest, 5-fold CV):
  R² mean±std:  0.1827 ± 0.0703
  MAE mean±std: 2.464 ± 0.064
Total Corners Model (Weighted CV):          👈 STEP 3!
  Linear R² mean±std: 0.0736 ± 0.0216
  Linear MAE mean±std: 2.734 ± 0.08
  RF R² mean±std:      0.2368 ± 0.0727
  RF MAE mean±std:     2.401 ± 0.049
...

✓ Full metrics saved to data/corners/model_metrics_E2_20251115_*.json

[continues with E3 metrics...]
[continues with predictions...]
```

**You see everything in ONE command output!**

---

## 🎓 Quick Decision Tree

```
Do you want corner predictions for matches?
│
├─ YES → Use automate_corner_predictions.py
│        ✅ Gets predictions + metrics
│
└─ NO  → Do you want league analysis only?
         │
         ├─ YES → Use corners_analysis.py --train-model
         │        ✅ Gets metrics + team stats (no predictions)
         │
         └─ NO  → Are you debugging metric calculations?
                  │
                  ├─ YES → Use tools/run_weighted_metrics.py
                  │        ✅ Quick metric recomputation
                  │
                  └─ NO  → Use automate_corner_predictions.py
                           (It's the safest all-in-one option)
```

---

## Summary

**For your daily workflow:**
```bash
# THIS IS THE ONLY COMMAND YOU NEED:
python3 automate_corner_predictions.py \
  --input tmp/corners/251115_match_games.log \
  --leagues E2,E3 \
  --train-model \
  --force \
  --auto \
  --mode fast
```

**`tools/run_weighted_metrics.py` role:**
- Development/debugging helper only
- NOT required for normal operations
- Useful for testing code changes
- Can be safely ignored for your use case

**All Steps 1-4 improvements are in the main workflow automatically!** 🎉

