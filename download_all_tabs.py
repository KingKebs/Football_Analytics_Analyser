"""
Download European league CSVs from football-data.co.uk into football-data/all-euro-football/

- Supports multiple league codes (E0, D1, SP1, ...)
- Supports multiple seasons (e.g., 2526, 2425, 2324)
- Caches downloads and skips re-downloading within a refresh window unless --force
- Writes season-specific files (e.g., E0_2526.csv) and also latest alias (E0.csv)

Examples:
  python download_all_tabs.py --download-football-data --leagues E0,SP1,D1 --seasons 2526,2425
  python download_all_tabs.py --download-football-data --leagues ALL          # download all supported leagues
  python download_all_tabs.py --download-football-data --force                # re-download regardless of cache
  python download_all_tabs.py --download-football-data --refresh-hours 6      # only refresh if older than 6h

Notes:
- The automate_football_analytics.py script expects latest files at:
    football-data/all-euro-football/<LEAGUE_CODE>.csv (e.g., SP1.csv)
- This downloader will maintain that convention by copying the most recent season file to that alias.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from typing import List, Dict
import json

# requests is imported lazily inside download_url_to_file so dry-run works even if requests isn't installed


# Output directories
FOOTBALL_DATA_DIR = 'football-data'
ALL_EURO_DIR = os.path.join(FOOTBALL_DATA_DIR, 'all-euro-football')

# Season codes: '2526' = 2025/26, '2425' = 2024/25, '2324' = 2023/24, ...
# NOTE: default changed to only the current season (2526) to avoid unnecessary downloads
DEFAULT_SEASONS: List[str] = ['2526']

# Supported football-data.co.uk league codes present in your repo
SUPPORTED_LEAGUES: List[str] = [
    'E0', 'E1', 'E2', 'E3',        # England
    'D1', 'D2',                    # Germany
    'SP1', 'SP2',                  # Spain
    'I1', 'I2',                    # Italy
    'F1', 'F2',                    # France
    'N1',                          # Netherlands
    'P1',                          # Portugal
    'SC0', 'SC1', 'SC2', 'SC3',    # Scotland
    'B1',                          # Belgium
    'G1',                          # Greece
    'T1',                          # Turkey
    'EC'                           # Europe (e.g., European Championship, if available)
]

# URL template for football-data downloads
BASE_URL_TMPL = 'https://www.football-data.co.uk/mmz4281/{season}/{code}.csv'


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _file_age_hours(path: str) -> float:
    if not os.path.exists(path):
        return float('inf')
    mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
    age = (_now_utc() - mtime).total_seconds() / 3600.0
    return max(0.0, age)


def download_url_to_file(url: str, dest_path: str, retries: int = 3, backoff: float = 1.0) -> bool:
    """Download URL with basic retries and exponential-ish backoff."""
    try:
        import requests
    except Exception:
        print("requests library not available: can't download URLs. Install with 'pip install requests' to enable downloads.")
        return False
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and resp.content:
                with open(dest_path, 'wb') as f:
                    f.write(resp.content)
                return True
            else:
                print(f"Attempt {attempt}: HTTP {resp.status_code} for {url}")
        except requests.RequestException as e:
            print(f"Attempt {attempt}: error downloading {url}: {e}")
        time.sleep(backoff * attempt)
    return False


def write_latest_alias(latest_src: str, alias_path: str) -> None:
    try:
        with open(latest_src, 'rb') as src, open(alias_path, 'wb') as dst:
            dst.write(src.read())
        print(f"Wrote latest alias -> {alias_path}")
    except Exception as e:
        print(f"Warning: failed to write alias {alias_path}: {e}")


def _load_cache(cache_path: str) -> Dict[str, Dict]:
    """Load per-league cache (last_download timestamps). Returns empty dict if missing or invalid."""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
    except Exception:
        print(f"Warning: failed to read cache {cache_path}, ignoring")
    return {}


def _save_cache(cache_path: str, data: Dict[str, Dict]) -> None:
    try:
        with open(cache_path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
    except Exception as e:
        print(f"Warning: failed to write cache {cache_path}: {e}")


def download_football_data(
    seasons: List[str],
    league_codes: List[str],
    out_dir: str = ALL_EURO_DIR,
    force: bool = False,
    refresh_hours: float = 6.0,
    dry_run: bool = False,
) -> Dict[str, List[str]]:
    """Download football-data CSVs for leagues×seasons.

    Returns a mapping {league_code: [saved_paths...]}
    """
    os.makedirs(out_dir, exist_ok=True)

    cache_path = os.path.join(out_dir, '.download_cache.json')
    cache = _load_cache(cache_path)

    # Expand ALL shortcut
    if len(league_codes) == 1 and league_codes[0].upper() == 'ALL':
        league_codes = SUPPORTED_LEAGUES.copy()

    results: Dict[str, List[str]] = {}

    for code in league_codes:
        code = code.upper().strip()
        if code not in SUPPORTED_LEAGUES:
            print(f"Skipping unsupported league code: {code}")
            continue

        saved: List[str] = []
        latest_season_file: str | None = None
        print(f"\n== {code}: downloading seasons {seasons} (force={force}, refresh<{refresh_hours}h, dry_run={dry_run}) ==")

        # Alias freshness check: prefer cache if available
        alias_path = os.path.join(out_dir, f"{code}.csv")
        alias_fresh = False
        if not force:
            # Check cache first
            try:
                entry = cache.get(code)
                if entry and 'last_download' in entry:
                    last_iso = entry['last_download']
                    try:
                        last_dt = datetime.fromisoformat(last_iso)
                        if last_dt.tzinfo is None:
                            # assume UTC if no tz
                            last_dt = last_dt.replace(tzinfo=timezone.utc)
                        age_h = (_now_utc() - last_dt).total_seconds() / 3600.0
                        if age_h < refresh_hours:
                            print(f"Alias fresh according to cache: {code} ({age_h:.1f}h) -> skip league {code}")
                            results[code] = [alias_path]
                            continue
                    except Exception:
                        # invalid cache timestamp, fall back to file mtime
                        pass

            except Exception:
                pass

            # Fallback to file mtime if cache not used
            if os.path.exists(alias_path):
                alias_age_h = _file_age_hours(alias_path)
                if alias_age_h < refresh_hours:
                    print(f"Alias fresh: {os.path.basename(alias_path)} ({alias_age_h:.1f}h) -> skip league {code}")
                    results[code] = [alias_path]
                    continue

        # Process per-season files
        for idx, season in enumerate(seasons):
            url = BASE_URL_TMPL.format(season=season, code=code)
            season_fname = f"{code}_{season}.csv"
            dest_path = os.path.join(out_dir, season_fname)

            # Caching: skip if exists and fresh enough
            if os.path.exists(dest_path) and not force:
                age_h = _file_age_hours(dest_path)
                if age_h < refresh_hours:
                    print(f"Fresh: {season_fname} ({age_h:.1f}h old) -> skip")
                    saved.append(dest_path)
                    if idx == 0:
                        latest_season_file = dest_path
                    continue

            # If dry-run, report and don't download
            if dry_run:
                print(f"[DRY-RUN] Would download {url} -> {season_fname}")
                saved.append(dest_path)
                if idx == 0:
                    latest_season_file = dest_path
                continue

            print(f"Downloading {url} -> {season_fname}")
            ok = download_url_to_file(url, dest_path)
            if ok:
                print(f"Saved {dest_path}")
                saved.append(dest_path)
                if idx == 0:
                    latest_season_file = dest_path
                # Update cache entry for this league
                cache.setdefault(code, {})['last_download'] = _now_utc().isoformat()
            else:
                print(f"Failed to download {url}")

        # Write latest alias (<code>.csv) from most recent season we processed
        if latest_season_file is not None:
            alias = os.path.join(out_dir, f"{code}.csv")
            if dry_run:
                print(f"[DRY-RUN] Would write alias {alias} from {latest_season_file}")
            else:
                write_latest_alias(latest_season_file, alias)
                # Also update cache after successful alias write
                cache.setdefault(code, {})['last_download'] = _now_utc().isoformat()
        else:
            print(f"No latest file available to alias for {code}")

        results[code] = saved

    # Save cache unless dry-run
    if not dry_run:
        _save_cache(cache_path, cache)
    else:
        print("Dry-run mode: cache not modified.")

    return results


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Download football-data.co.uk CSVs into all-euro-football directory')
    parser.add_argument('--download-football-data', action='store_true', help='Download football-data CSVs')
    parser.add_argument('--seasons', type=str, default=','.join(DEFAULT_SEASONS), help='Comma-separated list of seasons, e.g. 2526,2425,2324')
    parser.add_argument('--leagues', type=str, default='E0', help='Comma-separated league codes or ALL for all supported')
    parser.add_argument('--out', type=str, default=ALL_EURO_DIR, help='Output directory')
    parser.add_argument('--force', action='store_true', help='Force re-download even if files exist and are fresh')
    parser.add_argument('--refresh-hours', type=float, default=6.0, help='Only refresh files older than this many hours')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded without performing network requests or writing files')
    args = parser.parse_args(argv)

    seasons = [s.strip() for s in args.seasons.split(',') if s.strip()]
    leagues = [c.strip().upper() for c in args.leagues.split(',') if c.strip()]

    if not args.download_football_data:
        print('Nothing to do. Use --download-football-data to fetch CSVs.')
        return 0

    print(f"Out dir: {args.out}")
    print(f"Leagues: {leagues}")
    print(f"Seasons: {seasons}")
    print(f"Force:   {args.force}; Refresh-hours: {args.refresh_hours}; Dry-run: {args.dry_run}")

    results = download_football_data(
        seasons=seasons,
        league_codes=leagues,
        out_dir=args.out,
        force=args.force,
        refresh_hours=args.refresh_hours,
        dry_run=args.dry_run,
    )

    # Simple summary
    total_files = sum(len(v) for v in results.values())
    print(f"\nDone. Leagues processed: {len(results)}; files written/kept: {total_files}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
