---
name: complete-analysis-pipeline
description: Plan and run multi-step football analysis via cli.py — download, validate, full-league, corners, niche markets, and backtest. Use when the user asks for a daily run, complete pipeline, full-league analysis, upcoming fixtures, combined markets, or sequenced CLI tasks.
---

# Complete Analysis Pipeline

Prefer `python cli.py --task <name>`. Activate `venv` first. **Data first** — do not analyze until download/validate (or parsed fixtures) exist.

## Docs to read

- Commands: [ReadMeDocs/QUICK_REFERENCE.md](../../../ReadMeDocs/QUICK_REFERENCE.md)
- Architecture: [ReadMeDocs/SYSTEM_OVERVIEW.md](../../../ReadMeDocs/SYSTEM_OVERVIEW.md)
- Module map: [ReadMeDocs/SYSTEM_OVERVIEW.md](../../../ReadMeDocs/SYSTEM_OVERVIEW.md)

## Preconditions

```
- [ ] venv activated
- [ ] Historical CSVs in football-data/ OR user asked to download
- [ ] For --use-parsed-all: todays_fixtures_YYYYMMDD in data/analysis/ or data/
- [ ] Never --league ALL / --leagues ALL without --use-parsed-all
```

## Workflow 1: Daily analysis (parsed fixtures)

```bash
python cli.py --task full-league --use-parsed-all --ml-mode predict \
  --enable-double-chance --dc-min-prob 0.75 --dc-secondary-threshold 0.80 \
  --dc-allow-multiple --verbose
```

Pin a date with `--fixtures-date YYYYMMDD`. Then view:

```bash
streamlit run src/streamlit_app.py
```

## Workflow 2: Full league study (explicit leagues)

```bash
python cli.py --task download --leagues E0,SP1,D1,F1,I1 --season AUTO
python cli.py --task organize
python cli.py --task validate --check-all
python cli.py --task full-league --leagues E0,SP1,D1 --ml-mode predict \
  --enable-double-chance --dc-min-prob 0.75 --verbose
python cli.py --task corners --use-parsed-all --corners-use-ml-prediction --verbose
python cli.py --task backtest
```

## Workflow 3: Upcoming games → analyze

```bash
python cli.py --task download-fixtures --update-today
python cli.py --task convert-upcoming --file data/raw/upcomingMatches.json --output-dir data/analysis
python cli.py --task full-league --use-parsed-all --min-confidence 0.6 --ml-mode predict \
  --enable-double-chance --dc-min-prob 0.75 --verbose
```

## Workflow 4: Corners on upcoming games

Use skill `corner-prediction`. Minimum:

```bash
python cli.py --task download-fixtures --update-today
python cli.py --task convert-upcoming --file data/raw/upcomingMatches.json --output-dir data/analysis
python cli.py --task corners --use-parsed-all --min-team-matches 5 \
  --corners-use-ml-prediction --ml-mode predict --verbose
```

## Workflow 5: Combined markets (sequential)

```bash
python cli.py --task full-league --use-parsed-all --ml-mode predict --enable-double-chance --verbose \
  && python cli.py --task corners --use-parsed-all --ml-mode predict --corners-use-ml-prediction --verbose
cd src/niche_markets && python3 strategic_parlays.py && cd ../..
streamlit run src/streamlit_app.py
```

## Defaults

| Flag | Default | Notes |
|---|---|---|
| `--rating-model` | `blended` | Also `poisson`, `xg` |
| `--blend-weight` | `0.3` | Form vs baseline |
| `--last-n` | `6` | Recent matches |
| `--min-confidence` | `0.6` | Suggestion floor |
| `--ml-mode` | `off` | Set `predict` for ML |

## Outputs

- Per league: `data/analysis/full_league_suggestions_<LEAGUE>_<TS>.json`
- Consolidated: `data/analysis/consolidated_full_league_<TS>.json`
- Corners: `data/corners/parsed_corners_predictions_<DATE>.json`

## Recovery

On failure: `tail -f logs/cli_*.log` then `grep ERROR logs/cli_*.log`.
Missing CSV → download. Empty fixtures → check `todays_fixtures_*.json` has a `League` field.
See skill `data-lifecycle` for filesystem repair.
