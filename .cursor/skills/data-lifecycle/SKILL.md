---
name: data-lifecycle
description: Download, validate, organize, archive, and inventory football-data.co.uk CSVs and analysis outputs. Use when the user mentions download, fixtures, organize, validate, manifest, archive, cleanup, missing ALL.csv, or data directory hygiene.
---

# Data Lifecycle

Data first: do not run `full-league` or `corners` until source files exist and validate.

Read:

- Inventory / archive: [ReadMeDocs/FILE_MANAGEMENT_GUIDE.md](../../../ReadMeDocs/FILE_MANAGEMENT_GUIDE.md)
- Fixtures: [ReadMeDocs/FIXTURES_DOWNLOAD_GUIDE.md](../../../ReadMeDocs/FIXTURES_DOWNLOAD_GUIDE.md)
- Fixtures and scheduling: [ReadMeDocs/FIXTURES_DOWNLOAD_GUIDE.md](../../../ReadMeDocs/FIXTURES_DOWNLOAD_GUIDE.md)

## Download historical results

```bash
python cli.py --task download --leagues E0,SP1,D1 --season AUTO
python cli.py --task validate --check-all
python cli.py --task organize
```

Source: football-data.co.uk → `football-data/<CODE>_<SEASON>.csv`.

## Download and convert today's fixtures

```bash
python cli.py --task download-fixtures --update-today
python cli.py --task convert-upcoming --file data/raw/upcomingMatches.json --output-dir data/analysis
```

Produces `data/analysis/todays_fixtures_YYYYMMDD.csv` and `.json`.
`--use-parsed-all` searches `data/analysis/` then `data/`.

## Hygiene (`src/data_manager.py`)

Always preview destructive actions:

```bash
python src/data_manager.py --full-cleanup --dry-run
python src/data_manager.py --manifest
python src/data_manager.py --list-leagues
python src/data_manager.py --validate
python src/data_manager.py --archive --days 7
```

Default archive threshold is 7 days (`config.example.yaml` → `data.archive_threshold_days`).

## Directory contract

| Path | Contents |
|---|---|
| `football-data/` | Raw league CSVs |
| `data/raw/` | Upcoming JSON |
| `data/analysis/` | Suggestions + parsed fixtures |
| `data/corners/` | Corner predictions and metrics |
| `data/team_stats/` | Strength matrices |
| `data/archive/` | Aged-out files |
| `data/MANIFEST.json` | Inventory |
| `logs/` | `cli_YYYYMMDD_HHMMSS.log` |
| `models/` | Persisted ML pickles |

Do not write analysis artifacts outside these trees.

## Dynamic leagues

`--use-parsed-all` extracts unique `League` values from fixtures and processes only supported codes. That avoids the `ALL.csv not found` error.

If no leagues are found, the CLI falls back to E0. Inspect the fixtures file before re-running.

## Recovery

| Symptom | Action |
|---|---|
| No CSV files found | `--task download --leagues <codes>` |
| JSON decode error | `python src/data_manager.py --validate` then re-run analysis |
| ALL.csv not found | Use `--use-parsed-all` or an explicit comma list |
| Empty fixtures | Confirm `League` / `Competition` is populated |
