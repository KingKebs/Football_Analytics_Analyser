# Documentation Index

**Last Updated:** September 2026  
8 files — one canonical guide per topic.

---

## Start here

| Goal | Read |
|------|------|
| Use the tools right now | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |
| Understand the full system | [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) |
| Run today's full analysis | [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) |
| Fix an error | Check `logs/cli_*.log`, then [QUICK_REFERENCE.md §Troubleshooting](QUICK_REFERENCE.md) |

---

## Complete file list

### Core references

| File | What it covers |
|------|---------------|
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | CLI cheat-sheet — commands, flags, league codes, output files, troubleshooting |
| [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) | Architecture, capabilities, data flow, deployment notes |

### Workflows

| File | What it covers |
|------|---------------|
| [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) | Corners pipeline (Steps 1-4) **and** daily `run_analysis_workflow.py` |
| [FIXTURES_DOWNLOAD_GUIDE.md](FIXTURES_DOWNLOAD_GUIDE.md) | Automated fixtures download, scheduler, `--task download-fixtures` |

### ML & modelling

| File | What it covers |
|------|---------------|
| [ML_MODE_GUIDE.md](ML_MODE_GUIDE.md) | ML pipeline — train / predict / validate, feature engineering, model options |

### Corners

| File | What it covers |
|------|---------------|
| [CORNER_PREDICTIONS_GUIDE.md](CORNER_PREDICTIONS_GUIDE.md) | Corner market guide — theory, algorithms, outputs, interpreting predictions |

### Niche markets & staking

| File | What it covers |
|------|---------------|
| [NICHE_MARKETS_GUIDE.md](NICHE_MARKETS_GUIDE.md) | Odd/Even goals, Highest Scoring Half, cross-league parlays, parameter tuning, priority leagues |

### Data management

| File | What it covers |
|------|---------------|
| [FILE_MANAGEMENT_GUIDE.md](FILE_MANAGEMENT_GUIDE.md) | Archive, manifest, validate JSON — `data_manager.py` operations |

---

## Learning paths

**Just want to run predictions (15 min)**
1. `source venv/bin/activate`
2. Read [QUICK_REFERENCE.md](QUICK_REFERENCE.md) — Quick Start section
3. `python cli.py --task help`
4. Run your first league: `python cli.py --task full-league --leagues E0 --ml-mode predict`

**Understand the architecture (1 h)**
1. [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. Skim `cli.py` and `src/algorithms.py`

**Daily automated workflow**
1. [FIXTURES_DOWNLOAD_GUIDE.md](FIXTURES_DOWNLOAD_GUIDE.md) — set up fixture automation
2. [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) §Daily Analysis Workflow — `run_analysis_workflow.py`
3. `streamlit run src/streamlit_app.py` — view results

**ML & corners deep-dive**
1. [ML_MODE_GUIDE.md](ML_MODE_GUIDE.md)
2. [CORNER_PREDICTIONS_GUIDE.md](CORNER_PREDICTIONS_GUIDE.md)
3. Review the corner section in [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)

**Niche markets (Odd/Even, parlays)**
1. [NICHE_MARKETS_GUIDE.md](NICHE_MARKETS_GUIDE.md)
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) — DC / parlay flags

---

## Essential commands

```bash
# Setup
source venv/bin/activate

# Daily full-league run
python run_analysis_workflow.py --auto --verbose

# Single league with ML
python cli.py --task full-league --leagues E0 \
  --ml-mode predict --enable-double-chance --dc-min-prob 0.75 --verbose

# Corners
python cli.py --task corners --leagues E2,E3 --use-parsed-all \
  --corners-use-ml-prediction --verbose

# Download data
python cli.py --task download --leagues E0 --season AUTO

# Data housekeeping
python src/data_manager.py --full-cleanup --dry-run

# View results
streamlit run src/streamlit_app.py

# Logs
tail -f logs/cli_*.log
```

---

## Key entry points in code

| Surface | Path |
|---------|------|
| Unified CLI | `cli.py` |
| Rating / Poisson / Kelly | `src/algorithms.py` |
| ML train / predict | `src/ml_training.py`, `src/ml_evaluation.py` |
| ML features | `src/ml_features.py` |
| Corners engine | `src/corners_analysis.py` |
| Corners automation | `src/automate_corner_predictions.py` |
| Niche markets | `src/niche_markets/` |
| Data hygiene | `src/data_manager.py` |
| Streamlit dashboard | `src/streamlit_app.py` |
| Daily orchestration | `run_analysis_workflow.py` |
| Backtest / ROI | `src/analyze_suggestions_results.py` |
