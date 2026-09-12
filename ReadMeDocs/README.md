# Football Analytics Analyser

Python football match analysis and prediction: rating models, ML pipelines, corner markets, niche markets, and backtesting.

**Status:** Production | **Last updated:** September 2026

The canonical file list is **[INDEX.md](INDEX.md)**. Agent guidance lives in [`AGENTS.md`](../AGENTS.md) at the repo root.

---

## Start here

| Goal | Read |
|------|------|
| CLI cheat-sheet | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |
| Architecture | [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) |
| Daily / corners workflows | [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) |
| Layout and modules | [PROJECT_STRUCTURE_ANALYSIS.md](PROJECT_STRUCTURE_ANALYSIS.md) |

---

## Quick start

```bash
source venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml   # if you do not already have one

python cli.py --task help
python cli.py --task full-league --leagues E0 --ml-mode predict
```

Never pass `--leagues ALL` / `--league ALL` without `--use-parsed-all`.

---

## Common tasks

| Task | Command |
|------|---------|
| Download data | `python cli.py --task download --leagues E0,SP1 --season AUTO` |
| Organize | `python cli.py --task organize` |
| Validate | `python cli.py --task validate --check-all` |
| Full league | `python cli.py --task full-league --leagues E0` |
| Single match | `python cli.py --task single-match --home Team1 --away Team2` |
| Corners | `python cli.py --task corners --leagues E2,E3 --use-parsed-all` |
| View results | `streamlit run src/streamlit_app.py` |
| Backtest | `python cli.py --task backtest` |

Full flag lists: [QUICK_REFERENCE.md](QUICK_REFERENCE.md).

---

## Guides by topic

- **ML:** [ML_MODE_GUIDE.md](ML_MODE_GUIDE.md)
- **Corners:** [CORNER_PREDICTIONS_GUIDE.md](CORNER_PREDICTIONS_GUIDE.md), [SCRIPT_FLOW_DIAGRAM.md](SCRIPT_FLOW_DIAGRAM.md)
- **Niche markets / parlays:** [NICHE_MARKETS_GUIDE.md](NICHE_MARKETS_GUIDE.md)
- **Data hygiene:** [FILE_MANAGEMENT_GUIDE.md](FILE_MANAGEMENT_GUIDE.md)
- **Fixtures automation:** [FIXTURES_DOWNLOAD_GUIDE.md](FIXTURES_DOWNLOAD_GUIDE.md)

---

## Entry points

| Surface | Path |
|---------|------|
| Unified CLI | `cli.py` |
| Rating / Poisson / Kelly | `src/algorithms.py` |
| ML | `src/ml_training.py`, `src/ml_evaluation.py`, `src/ml_features.py` |
| Corners | `src/corners_analysis.py`, `src/automate_corner_predictions.py` |
| Niche markets | `src/niche_markets/` |
| Data hygiene | `src/data_manager.py` |
| Dashboard | `src/streamlit_app.py` |
| Daily orchestration | `run_analysis_workflow.py` |

---

## Troubleshooting

```bash
tail -50 logs/cli_*.log
python cli.py --task help
```

Then see [QUICK_REFERENCE.md](QUICK_REFERENCE.md) troubleshooting.

Data is from [football-data.co.uk](https://www.football-data.co.uk/).
