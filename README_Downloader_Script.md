# Downloader Script Usage

This project includes a downloader to fetch league CSVs from football-data.co.uk into `football-data/all-euro-football/`.

## Quick start

```bash
# Download the current season for EPL and alias to E0.csv
python3 download_all_tabs.py --download-football-data --leagues E0 --seasons AUTO

# Download current + previous season for EPL
python3 download_all_tabs.py --download-football-data --leagues E0 --seasons AUTO --include-previous-season

# Download multiple leagues (current season auto-detected)
python3 download_all_tabs.py --download-football-data --leagues E0,SP1,D1 --seasons AUTO

# Force re-download even if files are fresh
python3 download_all_tabs.py --download-football-data --leagues E0 --seasons AUTO --force
```

## How it works
- `--seasons AUTO` determines the active season based on the current date (e.g., 2526 for 2025/26).
- `--include-previous-season` appends the previous season to the download list (useful for analysis and backtests).
- For each league and season, a file `<LEAGUE>_<SEASON>.csv` is saved.
- The script always refreshes the alias `<LEAGUE>.csv` to the most recent season file processed (no alias cache short-circuit).
- Per-season files use a freshness window (`--refresh-hours`, default 6h) to avoid unnecessary network calls.

## Output
- Season files: `football-data/all-euro-football/E0_2526.csv`, `E0_2425.csv`, etc.
- Latest alias: `football-data/all-euro-football/E0.csv` (copied from the most recent season file).

## Notes
- Install dependencies via `pip install -r requirements.txt` (requires `requests` and `colorama`).
- You can pass `--leagues ALL` to fetch all supported league codes defined in the script.
- Use `--dry-run` to preview actions without downloading or writing files.
