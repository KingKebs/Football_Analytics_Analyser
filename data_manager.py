#!/usr/bin/env python3
"""
Data Management Utility

Helps organize, clean, and manage the data directory:
- Archive old files
- Remove duplicates
- Reorganize into logical structure
- Generate manifest
- Validate file integrity
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataManager:
    """Manages data directory structure and cleanup."""

    def __init__(self, data_dir: str = 'data', dry_run: bool = False):
        self.data_dir = Path(data_dir)
        self.dry_run = dry_run
        self.manifest = {}

    def get_file_age_days(self, file_path: Path) -> int:
        """Get age of file in days."""
        mtime = file_path.stat().st_mtime
        age = datetime.now().timestamp() - mtime
        return int(age / (24 * 3600))

    def archive_old_files(self, days: int = 7, patterns: List[str] = None):
        """Move files older than N days to archive."""
        if patterns is None:
            patterns = ['full_league_suggestions_*.json', 'league_data_*.csv']

        archive_dir = self.data_dir / 'archive' / datetime.now().strftime('%Y-%m-%d')

        logger.info(f"Archiving files older than {days} days")
        count = 0

        for pattern in patterns:
            for file_path in self.data_dir.glob(pattern):
                if file_path.is_file():
                    age = self.get_file_age_days(file_path)

                    if age > days:
                        if not self.dry_run:
                            archive_dir.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(file_path), str(archive_dir / file_path.name))

                        logger.info(f"  → Archived {file_path.name} (age: {age}d)")
                        count += 1

        logger.info(f"✓ Archived {count} files")
        return count

    def remove_duplicates(self):
        """Identify and remove duplicate files."""
        logger.info("Checking for duplicates...")

        # Group files by name pattern
        patterns = {
            'full_league_suggestions': [],
            'league_data': [],
            'home_away_team_strengths': [],
        }

        for pattern in patterns:
            for file_path in self.data_dir.glob(f'{pattern}*.json'):
                if file_path.is_file():
                    patterns[pattern].append(file_path)
            for file_path in self.data_dir.glob(f'{pattern}*.csv'):
                if file_path.is_file():
                    patterns[pattern].append(file_path)

        # For each pattern, keep only the most recent
        duplicates = []
        for pattern, files in patterns.items():
            if len(files) > 1:
                files.sort(key=lambda p: p.stat().st_mtime)
                duplicates.extend(files[:-1])  # All except the latest

        logger.info(f"Found {len(duplicates)} potential duplicates")

        for dup in duplicates:
            logger.debug(f"  Duplicate: {dup.name}")

        return duplicates

    def reorganize_structure(self):
        """Reorganize data into subdirectories."""
        logger.info("Reorganizing data structure...")

        # Define subdirectories
        subdirs = {
            'raw': '*.csv',
            'analysis': 'full_league_suggestions_*.json',
            'team_stats': 'home_away_team_strengths*.csv',
            'fixtures': 'todays_fixtures_*.csv',
            'misc': 'sample_*.csv'
        }

        moves = 0

        for subdir, pattern in subdirs.items():
            target = self.data_dir / subdir

            for file_path in self.data_dir.glob(pattern):
                if file_path.is_file() and file_path.parent == self.data_dir:
                    if not self.dry_run:
                        target.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(file_path), str(target / file_path.name))

                    logger.info(f"  → {subdir}/{file_path.name}")
                    moves += 1

        logger.info(f"✓ Moved {moves} files")
        return moves

    def cleanup_pycache(self):
        """Remove __pycache__ directories."""
        logger.info("Cleaning up __pycache__ directories...")

        count = 0
        for pycache in Path('.').rglob('__pycache__'):
            if not self.dry_run:
                shutil.rmtree(pycache)
            logger.debug(f"  Removed {pycache}")
            count += 1

        logger.info(f"✓ Removed {count} __pycache__ directories")
        return count

    def generate_manifest(self):
        """Generate a manifest of all data files."""
        logger.info("Generating manifest...")

        manifest = {
            'generated': datetime.now().isoformat(),
            'categories': {
                'analysis': [],
                'raw_data': [],
                'team_stats': [],
                'fixtures': [],
                'misc': [],
            }
        }

        # Categorize files
        for file_path in self.data_dir.rglob('*'):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(self.data_dir)
            file_info = {
                'path': str(relative_path),
                'size_mb': file_path.stat().st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'age_days': self.get_file_age_days(file_path),
            }

            if 'full_league_suggestions' in file_path.name:
                manifest['categories']['analysis'].append(file_info)
            elif file_path.suffix == '.csv' and 'team_strengths' in file_path.name:
                manifest['categories']['team_stats'].append(file_info)
            elif 'fixtures' in file_path.name:
                manifest['categories']['fixtures'].append(file_info)
            elif file_path.suffix == '.csv':
                manifest['categories']['raw_data'].append(file_info)
            else:
                manifest['categories']['misc'].append(file_info)

        # Save manifest
        manifest_path = self.data_dir / 'MANIFEST.json'
        if not self.dry_run:
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

        # Print summary
        total_files = sum(len(v) for v in manifest['categories'].values())
        logger.info(f"✓ Generated manifest with {total_files} files")

        for category, files in manifest['categories'].items():
            if files:
                total_mb = sum(f['size_mb'] for f in files)
                logger.info(f"  {category}: {len(files)} files ({total_mb:.1f} MB)")

        return manifest

    def list_files_by_league(self):
        """List files grouped by league code."""
        logger.info("Files by league:")

        league_files = {}

        for file_path in self.data_dir.rglob('*'):
            if not file_path.is_file():
                continue

            # Extract league code from filename
            name = file_path.name
            league = None

            for code in ['E0', 'E1', 'E2', 'E3', 'SP1', 'SP2', 'D1', 'D2', 'F1', 'F2', 'B1', 'I1', 'I2', 'P1', 'SC0', 'SC1', 'N1', 'G1', 'T1']:
                if code in name:
                    league = code
                    break

            if league:
                if league not in league_files:
                    league_files[league] = []
                league_files[league].append(file_path.name)

        for league in sorted(league_files.keys()):
            logger.info(f"  {league}: {len(league_files[league])} files")
            for fname in sorted(league_files[league])[:3]:
                logger.debug(f"    - {fname}")
            if len(league_files[league]) > 3:
                logger.debug(f"    ... and {len(league_files[league]) - 3} more")

        return league_files

    def get_latest_analysis(self, league: str = None):
        """Find the latest analysis file for a league."""
        pattern = 'full_league_suggestions_*.json' if not league else f'full_league_suggestions_{league}_*.json'

        files = list(self.data_dir.glob(pattern))

        if not files:
            return None

        latest = max(files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Latest analysis: {latest.name}")
        return latest

    def validate_json_files(self):
        """Validate all JSON files."""
        logger.info("Validating JSON files...")

        valid = 0
        invalid = []

        for json_file in self.data_dir.rglob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    json.load(f)
                valid += 1
            except json.JSONDecodeError as e:
                invalid.append((json_file.name, str(e)))
                logger.warning(f"  Invalid: {json_file.name} - {e}")

        logger.info(f"✓ Validated {valid} JSON files")

        if invalid:
            logger.warning(f"⚠ {len(invalid)} invalid JSON files")

        return valid, invalid


def main():
    parser = argparse.ArgumentParser(
        description='Data Management Utility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_manager.py --archive --days 7
  python data_manager.py --reorganize
  python data_manager.py --manifest
  python data_manager.py --list-leagues
  python data_manager.py --validate
  python data_manager.py --cleanup-cache
  python data_manager.py --full-cleanup --dry-run
        """
    )

    parser.add_argument('--archive', action='store_true', help='Archive old files')
    parser.add_argument('--days', type=int, default=7, help='Age threshold for archiving (days)')
    parser.add_argument('--reorganize', action='store_true', help='Reorganize into subdirectories')
    parser.add_argument('--manifest', action='store_true', help='Generate manifest')
    parser.add_argument('--list-leagues', action='store_true', help='List files by league')
    parser.add_argument('--validate', action='store_true', help='Validate JSON files')
    parser.add_argument('--cleanup-cache', action='store_true', help='Remove __pycache__')
    parser.add_argument('--full-cleanup', action='store_true', help='Run all cleanup tasks')
    parser.add_argument('--dry-run', action='store_true', help='Preview without executing')
    parser.add_argument('--data-dir', default='data', help='Data directory path')

    args = parser.parse_args()

    manager = DataManager(args.data_dir, args.dry_run)

    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made\n")

    if args.archive or args.full_cleanup:
        manager.archive_old_files(args.days)

    if args.reorganize or args.full_cleanup:
        manager.reorganize_structure()

    if args.manifest or args.full_cleanup:
        manager.generate_manifest()

    if args.list_leagues or args.full_cleanup:
        manager.list_files_by_league()

    if args.validate or args.full_cleanup:
        manager.validate_json_files()

    if args.cleanup_cache or args.full_cleanup:
        manager.cleanup_pycache()

    if not any([args.archive, args.reorganize, args.manifest, args.list_leagues, args.validate, args.cleanup_cache, args.full_cleanup]):
        parser.print_help()


if __name__ == '__main__':
    main()

