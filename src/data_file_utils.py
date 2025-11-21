"""Compatibility helper for locating data files after reorganization.

Provides functions that search the new organized layout (data/analysis, data/processed, data/raw)
but fall back to legacy locations (data/ root or data/old csv) so existing scripts keep working.
"""
from pathlib import Path
import glob
import os
from typing import Optional, List

ROOT = Path('data')


def _candidate_paths(*parts):
    """Yield Path candidates relative to data/"""
    yield ROOT.joinpath(*parts)


def find_latest_analysis(league: Optional[str] = None) -> Optional[str]:
    """Find the most recent full league suggestions JSON.

    Looks in data/analysis/, then falls back to data/ root.
    If league is provided, filters by league code in filename.
    """
    patterns = []
    if league:
        patterns.append(f"analysis/full_league_suggestions_{league}_*.json")
        patterns.append(f"full_league_suggestions_{league}_*.json")
    else:
        patterns.append("analysis/full_league_suggestions_*.json")
        patterns.append("full_league_suggestions_*.json")

    candidates = []
    for pattern in patterns:
        candidates.extend(sorted([str(p) for p in ROOT.glob(pattern)], key=os.path.getmtime if os.path.exists else None))
    # If nothing found in data/, also search recursive (legacy behavior)
    if not candidates:
        candidates = sorted(glob.glob(str(ROOT / ('**/' + 'full_league_suggestions_*.json'))), key=os.path.getmtime)
    return candidates[-1] if candidates else None


def find_all_analysis() -> List[str]:
    """Return all analysis JSONs, new-first ordering."""
    paths = sorted([str(p) for p in ROOT.glob('analysis/*.json')])
    if not paths:
        paths = sorted([str(p) for p in ROOT.glob('full_league_suggestions_*.json')])
    return paths


def get_league_cache_path(league_code: str) -> str:
    """Return preferred path for cached league table (processed/league_tables) or legacy data/

    Scripts should use this path for reading/writing cached league_data_<code>.csv
    """
    p1 = ROOT / 'processed' / 'league_tables' / f'league_data_{league_code}.csv'
    if p1.exists():
        return str(p1)
    p2 = ROOT / f'league_data_{league_code}.csv'
    return str(p2)


def get_team_strengths_path(league_code: str) -> str:
    """Return path for home_away_team_strengths_<league>.csv in processed/team_strengths or legacy root."""
    p1 = ROOT / 'processed' / 'team_strengths' / f'home_away_team_strengths_{league_code}.csv'
    if p1.exists():
        return str(p1)
    p2 = ROOT / f'home_away_team_strengths_{league_code}.csv'
    return str(p2)


def find_latest_fixtures() -> Optional[str]:
    """Find latest todays_fixtures JSON/CSV, preferring processed/fixtures."""
    candidates = sorted([str(p) for p in ROOT.glob('processed/fixtures/todays_fixtures_*.*')], key=os.path.getmtime) if (ROOT / 'processed' / 'fixtures').exists() else []
    if not candidates:
        candidates = sorted([str(p) for p in ROOT.glob('todays_fixtures_*.*')], key=os.path.getmtime)
    return candidates[-1] if candidates else None


def find_latest_team_strengths() -> Optional[str]:
    candidates = sorted([str(p) for p in ROOT.glob('processed/team_strengths/home_away_team_strengths_*.csv')])
    if not candidates:
        candidates = sorted([str(p) for p in ROOT.glob('home_away_team_strengths_*.csv')])
    return candidates[-1] if candidates else None


def ensure_dirs_for_writing():
    """Ensure the new directory layout exists for scripts that will write outputs."""
    for sub in ['analysis', 'raw', 'processed/league_tables', 'processed/team_strengths', 'processed/fixtures', 'archive', 'temp', 'cache']:
        p = ROOT.joinpath(*sub.split('/'))
        p.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    print('data_file_utils helper - not intended as a top-level script')

