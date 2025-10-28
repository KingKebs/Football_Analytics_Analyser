# Downloader Script — README

This document explains how `download_all_tabs.py` works, the safety updates added to avoid hammering football-data.co.uk, how caching and aliasing work, and example commands you can use.

Summary of key features

- Default output directory: `football-data/all-euro-football/`.
- Default season: `2526` (current 2025/26) to avoid downloading older seasons unnecessarily.
- Per-season files are written as `<CODE>_<SEASON>.csv` (e.g. `E0_2526.csv`).
- A "latest alias" file is maintained per league as `<CODE>.csv` (e.g. `E0.csv`) which is what the analytics script expects.
- Per-league JSON cache: `.download_cache.json` lives in the output directory and records last successful download timestamps for each league.
- Dry-run mode (`--dry-run`) shows what would be downloaded and what alias would be written without performing network requests or changing files.
- Lazy import of `requests`: `requests` is only imported when a real download is attempted, so `--dry-run` works even when `requests` is not installed.

Cache and freshness behavior

- Cache file path: `<out_dir>/.download_cache.json` (default `football-data/all-euro-football/.download_cache.json`).
- Cache format (example):

  {
    "E0": {"last_download": "2025-10-27T08:26:06.123456+00:00"},
    "SP1": {"last_download": "2025-10-27T08:30:00+00:00"}
  }

- Freshness decision order when deciding whether to skip downloading a league:
  1. If `--force` is provided, the downloader ignores freshness and will (re)download.
  2. If cache entry exists for the league and `now - last_download < refresh_hours`, the league is skipped.
  3. Otherwise, the script falls back to checking the alias file's filesystem mtime (`<CODE>.csv`) and if that is fresh (`mtime age < refresh_hours`) it will skip.
  4. If neither cache nor alias are fresh, the per-season files are considered and individually downloaded if stale.

- After successful real downloads the cache is updated with an ISO8601 UTC `last_download` timestamp for the league.
- `--dry-run` will not modify cache or files.

Why this reduces hammering the remote site

- The cache allows runs of your analytics script (several times a day) to avoid repeated HTTP requests; if the cache says a league was downloaded recently the downloader will skip that league.
- By default the script targets the current season (`2526`) instead of re-downloading older seasons.
- You can increase the `--refresh-hours` window to 24 or 72 hours for even safer behavior.

Command-line flags (overview)

- `--download-football-data` (required to perform any action)
- `--seasons` (comma-separated seasons; default `2526`)
- `--leagues` (comma-separated league codes or `ALL`); default `E0`
- `--out` (output directory; default `football-data/all-euro-football`)
- `--force` (force re-download even if cache/alias are fresh)
- `--refresh-hours` (how many hours before a file is considered stale; default `6.0`)
- `--dry-run` (show what would be done without network requests or file writes)

Example usage

- Dry-run for a single league (shows what would be downloaded and alias writes):

  python3 download_all_tabs.py --download-football-data --leagues E0 --dry-run

- Dry-run and force (show what would be downloaded even if alias/cache are fresh):

  python3 download_all_tabs.py --download-football-data --leagues E0 --dry-run --force

- Real run for two leagues, with a 24-hour freshness window (safe for daily runs):

  python3 download_all_tabs.py --download-football-data --leagues E0,SP1 --refresh-hours 24

- Force re-download for a league (will perform network requests and update alias & cache):

  python3 download_all_tabs.py --download-football-data --leagues E0 --force

Notes & troubleshooting

- If `requests` isn't installed and you attempt a real download you'll see a message telling you to install `requests`. Install it with:

  python3 -m pip install requests

  or via your project requirements:

  python3 -m pip install -r requirements.txt

- To inspect alias freshness manually use `stat` (macOS example):

  stat -f "%Sm %N" -t "%Y-%m-%d %H:%M:%S" football-data/all-euro-football/E0.csv

- To view the cache (if present):

  cat football-data/all-euro-football/.download_cache.json

Recommended adjustments you might want to make

- Consider increasing the default `--refresh-hours` to `24` (one line change in the script) so analytics runs the same day never trigger network requests.
- Optionally add checksum-based overwrite protection (download to a temp file, compare, replace alias only if changed) — I can implement this for you.
- Optionally add automatic season detection (compute season `2526` from today's date) so you don't need to update `DEFAULT_SEASONS` each year.

If you'd like, I can now:

- Change the default refresh window to 24 hours,
- Implement checksum-based alias-safe replace, or
- Add automatic season detection.

Tell me which of these (if any) you want next and I'll implement and test it.

