#!/usr/bin/env python3

"""
Script to organize files and create directory structure for Football_Analytics_Analyser
This script creates the required directories and moves files to match the project structure
based on file types and naming patterns.
"""

import os
import shutil
import glob
import sys
from datetime import datetime, timedelta

def main():
    print("Organizing Football Analytics Analyser project structure...")

    # Dry run option
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        print("DRY RUN MODE - No files will be moved\n")

    # Create main directories
    directories = [
        'data',
        'football-data',
        'tmp',
        '__pycache__',
        'venv',
        '.git',
        '.idea'
    ]

    # Create subdirectories
    subdirectories = [
        'data/old csv',
        'data/corners',
        'data/league_analysis',
        'data/fixtures',
        'data/archived',
        'football-data/all-euro-football',
        'tmp/corners'
    ]

    all_dirs = directories + subdirectories

    for directory in all_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            if not dry_run:
                print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"⚠ Warning: Could not create {directory}: {e}")

    # Move files based on patterns
    print("\nOrganizing data directory files...\n")

    # 1. Corners analysis files (should be in data/corners/)
    corner_patterns = [
        ('data/corners_analysis_*.csv', 'data/corners/'),
        ('data/corners_correlations_*.json', 'data/corners/'),
        ('data/team_stats_*.json', 'data/corners/'),
    ]

    for pattern, dest in corner_patterns:
        files = glob.glob(pattern)
        for file_path in files:
            if os.path.dirname(file_path) != dest.rstrip('/'):
                try:
                    if dry_run:
                        print(f"[DRY RUN] Would move {file_path} → {dest}")
                    else:
                        shutil.move(file_path, dest)
                        print(f"✓ {os.path.basename(file_path)} → {dest}")
                except Exception as e:
                    print(f"⚠ Could not move {file_path}: {e}")

    # 2. Full league suggestions (should be in data/league_analysis/)
    league_files = glob.glob('data/full_league_suggestions_*.json')
    cutoff_date = datetime.now() - timedelta(days=7)

    for file_path in league_files:
        try:
            file_age = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_age < cutoff_date:
                # Archive old files
                dest = 'data/archived/'
                action = 'archive'
            else:
                # Keep recent in league_analysis
                dest = 'data/league_analysis/'
                action = 'organize'

            if dry_run:
                print(f"[DRY RUN] Would {action} {file_path} → {dest}")
            else:
                shutil.move(file_path, dest)
                print(f"✓ {os.path.basename(file_path)} → {dest} ({action})")
        except Exception as e:
            print(f"⚠ Could not move {file_path}: {e}")

    # 3. Fixtures files (should be in data/fixtures/)
    fixture_files = glob.glob('data/todays_fixtures_*.*')
    for file_path in fixture_files:
        try:
            file_age = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_age < cutoff_date:
                dest = 'data/archived/'
            else:
                dest = 'data/fixtures/'

            if dry_run:
                print(f"[DRY RUN] Would move {file_path} → {dest}")
            else:
                shutil.move(file_path, dest)
                print(f"✓ {os.path.basename(file_path)} → {dest}")
        except Exception as e:
            print(f"⚠ Could not move {file_path}: {e}")

    # 4. Individual match suggestions (archive old ones)
    suggestion_files = glob.glob('data/suggestion_*.json')
    for file_path in suggestion_files:
        try:
            file_age = datetime.fromtimestamp(os.path.getmtime(file_path))
            if file_age < cutoff_date:
                dest = 'data/archived/'
                if dry_run:
                    print(f"[DRY RUN] Would archive {file_path} → {dest}")
                else:
                    shutil.move(file_path, dest)
                    print(f"✓ {os.path.basename(file_path)} → {dest} (archived)")
        except Exception as e:
            print(f"⚠ Could not move {file_path}: {e}")

    # 5. League data CSVs (keep in data/ root but organize)
    league_data_files = glob.glob('data/league_data_*.csv')
    print(f"\n✓ Found {len(league_data_files)} league data files (keeping in data/)")

    # 6. Other parsed results
    other_patterns = [
        ('data/parsed_fixtures_suggestions_*.json', 'data/archived/'),
        ('data/sample_results_*.csv', 'data/archived/'),
    ]

    for pattern, dest in other_patterns:
        files = glob.glob(pattern)
        for file_path in files:
            try:
                if dry_run:
                    print(f"[DRY RUN] Would archive {file_path} → {dest}")
                else:
                    shutil.move(file_path, dest)
                    print(f"✓ {os.path.basename(file_path)} → {dest}")
            except Exception as e:
                print(f"⚠ Could not move {file_path}: {e}")

    # Special cases
    # Move data.zip to football-data/ if it exists in root
    if os.path.isfile('data.zip'):
        try:
            if dry_run:
                print(f"[DRY RUN] Would move data.zip → football-data/")
            else:
                shutil.move('data.zip', 'football-data/')
                print("✓ data.zip → football-data/")
        except Exception as e:
            print(f"⚠ Could not move data.zip: {e}")

    # Move *.log files to tmp/
    log_files = glob.glob('*.log')
    for file_path in log_files:
        try:
            if dry_run:
                print(f"[DRY RUN] Would move {file_path} → tmp/")
            else:
                shutil.move(file_path, 'tmp/')
                print(f"✓ {file_path} → tmp/")
        except Exception as e:
            print(f"⚠ Could not move {file_path}: {e}")

    # Move *.pyc files to __pycache__/
    pyc_files = glob.glob('*.pyc')
    for file_path in pyc_files:
        try:
            if dry_run:
                print(f"[DRY RUN] Would move {file_path} → __pycache__/")
            else:
                shutil.move(file_path, '__pycache__/')
                print(f"✓ {file_path} → __pycache__/")
        except Exception as e:
            print(f"⚠ Could not move {file_path}: {e}")

    # Move *.png files to visuals/
    png_files = glob.glob('*.png')
    for file_path in png_files:
        try:
            if dry_run:
                print(f"[DRY RUN] Would move {file_path} → visuals/")
            else:
                shutil.move(file_path, 'visuals/')
                print(f"✓ {file_path} → visuals/")
        except Exception as e:
            print(f"⚠ Could not move {file_path}: {e}")

    # Summary
    print("\n" + "="*70)
    if dry_run:
        print("DRY RUN COMPLETE - No files were actually moved")
        print("\nTo apply these changes, run:")
        print("  python3 organize_structure.py")
    else:
        print("✓ Project structure organization complete!")

    print("\nDirectory structure:")
    print("  data/")
    print("    ├── corners/          (corner analysis, predictions)")
    print("    ├── league_analysis/  (full league suggestions)")
    print("    ├── fixtures/         (parsed match fixtures)")
    print("    ├── archived/         (old files >7 days)")
    print("    └── *.csv             (league data tables)")
    print("="*70)

if __name__ == '__main__':
    main()
