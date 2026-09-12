---
name: results-visualization
description: View analysis outputs in Streamlit or CLI, and run suggestion backtests for ROI/yield. Use when the user asks to view results, open the dashboard, Streamlit, backtest, ROI, yield, formatted text, or consolidated JSON.
---

# Results Visualization

Do not launch Streamlit until at least one analysis file exists in `data/analysis/` or `data/corners/`.

Command syntax: [ReadMeDocs/QUICK_REFERENCE.md](../../../ReadMeDocs/QUICK_REFERENCE.md) (Viewing Results).

## Streamlit (preferred)

```bash
streamlit run src/streamlit_app.py --server.address localhost --server.port 8501 --browser.gatherUsageStats false
```

The app loads the **newest file by mtime** (`os.path.getmtime`), not filename sort.

| View | Glob |
|---|---|
| Full league | `data/analysis/full_league_suggestions_*.json` |
| Consolidated | `data/analysis/consolidated_full_league_*.json` |
| Corners | `data/corners/parsed_corners_predictions_*.json` |
| ML Predictions tab | fields inside the latest full-league / consolidated JSON |

New output types: add a glob next to the existing loaders and keep `max(paths, key=os.path.getmtime)`.

## CLI view / backtest

```bash
python cli.py --task view --file data/analysis/full_league_suggestions_E0_*.json
python cli.py --task backtest --file data/analysis/full_league_suggestions_E0_*.json
```

Backtest implementation: `src/analyze_suggestions_results.py`. Report ROI/yield from that script; do not invent staking math outside `algorithms.py` Kelly helpers.

## Text / JSON

```bash
cat data/analysis/consolidated_full_league_*_formatted.txt | less
```

## Empty dashboard

If Streamlit says no suggestion files:

```bash
python cli.py --task full-league --use-parsed-all --ml-mode predict --verbose
```

Then refresh the app. Confirm files landed in `data/analysis/`, not another folder.
