"""
Download European league CSVs from football-data.co.uk into football-data/all-euro-football/

- Supports multiple league codes (E0, D1, SP1, ...)
- Supports multiple seasons (e.g., 2526,2425,2324)
- Caches downloads and skips re-downloading within a refresh window unless --force
- Writes season-specific files (e.g., E0_2526.csv) and also latest alias (E0.csv)

Examples:
  python download_all_tabs.py --download-football-data --leagues E0,SP1,D1 --seasons 2526,2425
  python download_all_tabs.py --download-football-data --leagues ALL          # download all supported leagues
  python download_all_tabs.py --download-football-data --force                # re-download regardless of cache
  python download_all_tabs.py --download-football-data --refresh-hours 6      # only refresh if older than 6h
  python download_all_tabs.py --download-football-data --seasons AUTO         # auto-detect current season

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
from colorama import init, Fore

# Initialize colorama
init()

# requests is imported lazily inside download_url_to_file so dry-run works even if requests isn't installed


# Output directories
FOOTBALL_DATA_DIR = 'football-data'
ALL_EURO_DIR = os.path.join(FOOTBALL_DATA_DIR, 'all-euro-football')

# Compute current season code (e.g., '2526' for 2025/26)

def current_season_code(now: datetime | None = None) -> str:
    if now is None:
        now = datetime.now(timezone.utc)
    year = now.year
    month = now.month
    if month >= 7:  # July-Dec -> season starts this calendar year
        start_year = year
        end_year = year + 1
    else:           # Jan-Jun -> season started previous calendar year
        start_year = year - 1
        end_year = year
    return f"{start_year % 100:02d}{end_year % 100:02d}"


def previous_season_code(season_code: str) -> str:
    try:
        a, b = int(season_code[:2]), int(season_code[2:])
        a_prev = (a - 1) % 100
        b_prev = a  # previous season ends at start year
        return f"{a_prev:02d}{b_prev:02d}"
    except Exception:
        return season_code


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
    try:
        import requests
    except Exception:
        print(Fore.RED + "requests library not available: can't download URLs. Install with 'pip install requests' to enable downloads." + Fore.RESET)
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
                print(Fore.RED + f"Attempt {attempt}: HTTP {resp.status_code} for {url}" + Fore.RESET)
        except requests.RequestException as e:
            print(Fore.RED + f"Attempt {attempt}: error downloading {url}: {e}" + Fore.RESET)
        time.sleep(backoff * attempt)
    return False


def write_latest_alias(latest_src: str, alias_path: str) -> None:
    try:
        with open(latest_src, 'rb') as src, open(alias_path, 'wb') as dst:
            dst.write(src.read())
        print(f"Wrote latest alias -> {alias_path}")
    except Exception as e:
        print(Fore.RED + f"Warning: failed to write alias {alias_path}: {e}" + Fore.RESET)


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
        latest_pair: tuple[str, str] | None = None  # (season, path)
        print(f"\n== {code}: downloading seasons {seasons} (force={force}, refresh<{refresh_hours}h, dry_run={dry_run}) ==")

        # Always attempt per-season file handling; don't short-circuit on alias freshness

        # Process per-season files
        for season in seasons:
            url = BASE_URL_TMPL.format(season=season, code=code)
            season_fname = f"{code}_{season}.csv"
            dest_path = os.path.join(out_dir, season_fname)

            # Caching: skip network if exists and fresh enough
            if os.path.exists(dest_path) and not force:
                age_h = _file_age_hours(dest_path)
                if age_h < refresh_hours:
                    print(f"Fresh: {season_fname} ({age_h:.1f}h old) -> skip")
                    saved.append(dest_path)
                    # track latest by season code
                    if latest_pair is None or season > latest_pair[0]:
                        latest_pair = (season, dest_path)
                    continue

            # If dry-run, report and don't download
            if dry_run:
                print(f"[DRY-RUN] Would download {url} -> {season_fname}")
                saved.append(dest_path)
                if latest_pair is None or season > latest_pair[0]:
                    latest_pair = (season, dest_path)
                continue

            print(f"Downloading {url} -> {season_fname}")
            ok = download_url_to_file(url, dest_path)
            if ok:
                print(f"Saved {dest_path}")
                saved.append(dest_path)
                if latest_pair is None or season > latest_pair[0]:
                    latest_pair = (season, dest_path)
                # Update cache entry for this league
                cache.setdefault(code, {})['last_download'] = _now_utc().isoformat()
            else:
                print(f"Failed to download {url}")

        # Write latest alias (<code>.csv) from the most recent season we processed or had cached
        if latest_pair is not None and latest_pair[1] and os.path.exists(latest_pair[1]):
            alias = os.path.join(out_dir, f"{code}.csv")
            if dry_run:
                print(f"[DRY-RUN] Would write alias {alias} from {latest_pair[1]}")
            else:
                write_latest_alias(latest_pair[1], alias)
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


def download_fixtures() -> None:
    fixtures_url = 'https://www.football-data.co.uk/fixtures.csv'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dest_dir = 'data/Results'
    dest_path = os.path.join(dest_dir, f'fixtures_{timestamp}.csv')

    print(Fore.LIGHTBLUE_EX + f"Downloading fixtures data to {dest_path}..." + Fore.RESET)

    os.makedirs(dest_dir, exist_ok=True)
    if download_url_to_file(fixtures_url, dest_path):
        print(Fore.GREEN + f"Fixtures data saved to {dest_path}" + Fore.RESET)
    else:
        print(Fore.RED + "Failed to download fixtures data." + Fore.RESET)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Download football-data.co.uk CSVs into all-euro-football directory')
    parser.add_argument('--download-football-data', action='store_true', help='Download football-data CSVs')
    parser.add_argument('--download-fixtures', action='store_true', help='Download fixtures data')
    parser.add_argument('--seasons', type=str, default='AUTO', help='Comma-separated list of seasons, e.g. 2526,2425 or AUTO for current season')
    parser.add_argument('--include-previous-season', action='store_true', help='When using AUTO seasons, also include the previous season')
    parser.add_argument('--leagues', type=str, default='E0', help='Comma-separated league codes or ALL for all supported')
    parser.add_argument('--out', type=str, default=ALL_EURO_DIR, help='Output directory')
    parser.add_argument('--force', action='store_true', help='Force re-download even if files exist and are fresh')
    parser.add_argument('--refresh-hours', type=float, default=6.0, help='Only refresh files older than this many hours')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded without performing network requests or writing files')
    args = parser.parse_args(argv)

    # Resolve seasons
    if args.seasons.strip().upper() == 'AUTO':
        cur = current_season_code()
        seasons = [cur]
        if args.include_previous_season:
            seasons.append(previous_season_code(cur))
    else:
        seasons = [s.strip() for s in args.seasons.split(',') if s.strip()]

    leagues = [c.strip().upper() for c in args.leagues.split(',') if c.strip()]

    if not args.download_football_data and not args.download_fixtures:
        print('Nothing to do. Use --download-football-data to fetch CSVs or --download-fixtures to fetch fixtures data.')
        return 0

    print(f"Out dir: {args.out}")
    print(f"Leagues: {leagues}")
    print(f"Seasons: {seasons}")
    print(f"Force:   {args.force}; Refresh-hours: {args.refresh_hours}; Dry-run: {args.dry_run}")

    if args.download_football_data:
        download_football_data(
            seasons=seasons,
            league_codes=leagues,
            out_dir=args.out,
            force=args.force,
            refresh_hours=args.refresh_hours,
            dry_run=args.dry_run,
        )

    if args.download_fixtures:
        download_fixtures()

    return 0


if __name__ == '__main__':
    sys.exit(main())
