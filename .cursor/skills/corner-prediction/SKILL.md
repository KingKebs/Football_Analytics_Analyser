---
name: corner-prediction
description: Analyze corner patterns and predict total / 1H / 2H corner markets with statistical models and optional ML. Use when the user asks about corners, half-splits, Over/Under corner lines, automate_corner_predictions, parsed fixture corners, or --corners-use-ml-prediction.
---

# Corner Prediction

1H/2H splits are **statistical** (team ratios + trained models). Never use arbitrary 50/50 halves.

Read the method and batch workflow in:

- [ReadMeDocs/CORNER_PREDICTIONS_GUIDE.md](../../../ReadMeDocs/CORNER_PREDICTIONS_GUIDE.md)
- [ReadMeDocs/WORKFLOW_GUIDE.md](../../../ReadMeDocs/WORKFLOW_GUIDE.md)

## Preconditions

```
- [ ] League CSVs in football-data/ (download if missing)
- [ ] For parsed-all: todays_fixtures_* present
- [ ] Teams have ≥ --min-team-matches (default 5)
```

## Preferred: parsed fixtures (dynamic leagues)

```bash
python cli.py --task corners --use-parsed-all --min-team-matches 5 \
  --corners-use-ml-prediction --enable-double-chance --dc-min-prob 0.75 --verbose
```

Add `--fixtures-date YYYYMMDD` to pin a file.

## Preferred: match-log batch (Steps 1–4)

```bash
python cli.py --task corners --input tmp/corners/YYMMDD_match_games.log \
  --leagues E2,E3 --train-model --force --auto --mode fast
```

Equivalent direct script: `src/automate_corner_predictions.py` (same flags).

`--mode fast` uses cached team stats. `--mode full` rebuilds per match.

## League analysis only (no match list)

```bash
python cli.py --task analyze-corners --ml-mode predict --verbose
python cli.py --task corners --file E0_2425.csv --train-model --save-enriched
```

## Single match

```bash
python cli.py --task corners --home-team Arsenal --away-team Chelsea --corners-use-ml-prediction
```

Team names are case-sensitive; verify with `--top-n 10` if lookup fails.

## What the models produce

- Total corners (mean + range)
- 1H / 2H expected split
- Over/Under line suggestions with confidence
- Optional Monte Carlo ranges (`--corners-mc-samples`, default 1000)

MAE target after Steps 1–4 is ~2.3 corners (not ~2.7 linear-only).

## Outputs

| Pattern | Location |
|---|---|
| `parsed_corners_predictions_<DATE>.json` | `data/corners/` |
| `batch_predictions_<LEAGUES>_*.json` | `data/corners/` |
| `model_metrics_<LEAGUE>_*.json` | `data/corners/` |
| `team_stats_<LEAGUE>_*.json` | `data/corners/` |

Same-day duplicates are cleaned automatically. See [ReadMeDocs/FILE_MANAGEMENT_GUIDE.md](../../../ReadMeDocs/FILE_MANAGEMENT_GUIDE.md).

## Recovery

- Team not found → `--top-n 10` and use the exact CSV name
- No CSV → `python cli.py --task download --leagues E2,E3`
- Skip training if models already exist today: omit `--train-model`, add reuse flags from the file-management guide
