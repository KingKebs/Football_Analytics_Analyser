#!/usr/bin/env python3
"""
Safe data/ directory reorganizer for Football_Analytics_Analyser

By default this script performs a dry-run and prints planned moves.
Pass --apply to actually move files.

Usage:
  python3 organize_data_structure.py [--apply] [--days 7] [--verbose]
"""
import argparse
import os
import shutil
import glob
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path('data')

def ensure_dirs(dirs, dry_run=False):
    for d in dirs:
        p = ROOT / d
        if not p.exists():
            if dry_run:
                print(f"[DRY RUN] Would create directory: {p}")
            else:
                p.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created directory: {p}")


def move_file(src: Path, dst: Path, dry_run=True):
    dst_parent = dst.parent
    dst_parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        # same file, skip
        return False, f"skipped (already in place): {src}"

    if dst.exists():
        # avoid overwrite: append timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dst = dst.with_name(dst.stem + '_' + timestamp + dst.suffix)
    if dry_run:
        return True, f"[DRY RUN] Would move {src} → {dst}"
    else:
        shutil.move(str(src), str(dst))
        return True, f"Moved {src} → {dst}"


def plan_and_apply(apply=False, days=7, verbose=False):
    dry_run = not apply
    now = datetime.now()
    cutoff = now - timedelta(days=days)

    # Desired structure under data/
    dirs = [
        'raw',
        'processed/league_tables',
        'processed/team_strengths',
        'processed/fixtures',
        'analysis',
        'archive',
        'temp',
        'cache',
    ]

    ensure_dirs(dirs, dry_run=dry_run)

    planned_actions = []

    # Helper: move into subdir
    def move_to(subpath, src_path):
        dst = ROOT / subpath / src_path.name
        ok, msg = move_file(src_path, dst, dry_run=dry_run)
        planned_actions.append(msg)
        if verbose:
            print(msg)

    # Scan top-level files in data/
    for entry in sorted(ROOT.iterdir()):
        # skip directories we manage
        if entry.is_dir():
            # move certain dirs into archive or keep
            if entry.name in ['archived', 'old csv', 'league_analysis', 'corners', 'fixtures', 'Results']:
                # keep existing structure: move 'old csv' contents into archive
                if entry.name == 'old csv':
                    for f in sorted(entry.iterdir()):
                        if f.is_file():
                            # archive old csv contents
                            dst = ROOT / 'archive' / f.name
                            ok, msg = move_file(f, dst, dry_run=dry_run)
                            planned_actions.append(msg)
                            if verbose:
                                print(msg)
                continue
            else:
                # ignore other dirs
                continue

        # process files
        name = entry.name
        lower = name.lower()

        # JSON analysis outputs
        if lower.startswith('full_league_suggestions_') or lower.endswith('_suggestions.json') or lower.startswith('analysis_results') or lower.endswith('_results.json') or lower.endswith('_suggestions.json'):
            move_to('analysis', entry)
            continue

        # team strength CSVs
        if lower.startswith('home_away_team_strengths') or 'team_strengths' in lower:
            move_to('processed/team_strengths', entry)
            continue

        # league data CSVs
        if lower.startswith('league_data') or lower.startswith('e') and '_' in lower and lower.split('_')[0].startswith(('e','s','d','t','b')) and lower.endswith('.csv'):
            move_to('processed/league_tables', entry)
            continue

        # todays fixtures -> processed/fixtures
        if lower.startswith('todays_fixtures'):
            move_to('processed/fixtures', entry)
            continue

        # csv files not matched -> raw
        if lower.endswith('.csv'):
            move_to('raw', entry)
            continue

        # json files (other) -> analysis
        if lower.endswith('.json'):
            move_to('analysis', entry)
            continue

        # fallback: move misc files into temp
        move_to('temp', entry)

    # Archive old files from archive-like dirs (age > days)
    # Any file in processed or analysis older than cutoff -> archive
    for sub in ['processed/league_tables', 'processed/team_strengths', 'analysis', 'temp']:
        folder = ROOT / sub
        if not folder.exists():
            continue
        for f in sorted(folder.glob('*')):
            try:
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
            except Exception:
                continue
            if mtime < cutoff:
                dst = ROOT / 'archive' / f.name
                ok, msg = move_file(f, dst, dry_run=dry_run)
                planned_actions.append(msg)
                if verbose:
                    print(msg)

    # Summary
    print('\n' + '='*60)
    if dry_run:
        print('DRY RUN complete. No files were moved. To apply changes, re-run with --apply')
    else:
        print('Apply complete.')
    print(f"Planned actions ({len(planned_actions)}):")
    for a in planned_actions:
        print(' -', a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize data/ directory into a clearer structure.')
    parser.add_argument('--apply', action='store_true', help='Actually move files (default: dry-run)')
    parser.add_argument('--days', type=int, default=7, help='Days threshold to archive old files (default: 7)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    plan_and_apply(apply=args.apply, days=args.days, verbose=args.verbose)

