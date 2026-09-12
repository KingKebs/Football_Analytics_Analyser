# Football Analytics Analyser — Agent Guide

Python football analytics system. Default branch: `dev`.
This is **not** a Magento/PHP project. Do not apply Vaimo copyright headers, Magento module layout, or ObjectManager/repository patterns.

## Capabilities

- Multi-league match prediction (22+ league codes)
- Expected Goals (xG) and Poisson modeling
- ML integration (Linear Regression, RandomForest, XGBoost)
- Corner analysis and pattern detection
- Kelly Criterion staking
- Backtesting and ROI analysis

## Entry Points

| Surface | Path | Use |
|---|---|---|
| Unified CLI | `cli.py` | Task routing — prefer this over calling scripts directly |
| Rating models | `src/algorithms.py` | Blended, Poisson, xG, Kelly, parlays |
| ML pipeline | `src/ml_training.py`, `src/ml_evaluation.py`, `src/ml_features.py` | Train / evaluate / feature engineering |
| Corners | `src/corners_analysis.py`, `src/automate_corner_predictions.py` | Pattern analysis and match predictions |
| Data hygiene | `src/data_manager.py` | Archive, manifest, validate JSON |
| Viewer | `src/streamlit_app.py`, `src/view_suggestions.py` | Dashboards and CLI view |
| Backtest | `src/analyze_suggestions_results.py` | ROI / yield |

## Critical Rules

1. **Data first** — download and validate before analysis.
2. **ML context** — ML mode needs enough history (typically 2+ seasons; default `--ml-min-samples 300`).
3. **Recency weighting** — recent form models favor last 10–20 matches (`--last-n` default 6; `--ml-decay` default 0.85).
4. **Corner split** — 1H/2H corner predictions use statistical models, not arbitrary splits.
5. **Error recovery** — debug from `logs/cli_*.log`.
6. **Configuration** — copy `config.example.yaml` to `config.yaml` rather than inventing new config keys.
7. **Never `--leagues ALL` without `--use-parsed-all`** — that looks for a non-existent `ALL.csv`.
8. **Activate `venv`** before running Python commands on macOS/Linux.

## Supported League Codes

| Region | Codes |
|---|---|
| England | E0 (EPL), E1 (Championship), E2 (League One), E3 (League Two) |
| Germany | D1 (Bundesliga), D2 (2. Bundesliga) |
| Spain | SP1 (La Liga), SP2 (Segunda) |
| Italy | I1 (Serie A), I2 (Serie B) |
| France | F1 (Ligue 1), F2 (Ligue 2) |
| Other | N1 (Netherlands), P1 (Portugal), SC0–SC3 (Scotland), B1 (Belgium), G1 (Greece), T1 (Turkey), EC (Europe) |

## Skill Docs (read when the task matches)

- Architecture: `ReadMeDocs/SYSTEM_OVERVIEW.md`, `ReadMeDocs/PROJECT_STRUCTURE_ANALYSIS.md`
- Workflows / CLI: `ReadMeDocs/WORKFLOW_GUIDE.md`, `ReadMeDocs/QUICK_REFERENCE.md`
- Corners: `ReadMeDocs/CORNER_PREDICTIONS_GUIDE.md`, `ReadMeDocs/SCRIPT_FLOW_DIAGRAM.md`, `ReadMeDocs/SCRIPT_RELATIONSHIPS.md`
- ML: `ReadMeDocs/ML_MODE_GUIDE.md`
- Data: `ReadMeDocs/FILE_MANAGEMENT_GUIDE.md`, `ReadMeDocs/AUTOMATION_COMPARISON.md`

Project Cursor rules live in `.cursor/rules/`. Executable workflow skills live in `.cursor/skills/`.
