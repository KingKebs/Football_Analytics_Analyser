#!/usr/bin/env python3
"""
Football Fixtures Scheduler

Automates the daily download of football fixtures and integrates with analysis.
Can be run as a cron job or scheduled task.

Usage:
    # Run daily update
    python fixtures_scheduler.py --daily-update

    # Update fixtures and run analysis
    python fixtures_scheduler.py --update-and-analyze

    # Setup as cron job (example)
    # 0 8 * * * cd /path/to/Football_Analytics_Analyser && python fixtures_scheduler.py --daily-update
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import json

# Setup logging
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"fixtures_scheduler_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FixturesScheduler:
    """Automated scheduler for fixture downloads and analysis"""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.fixtures_file = self.base_dir / 'data/raw/upcomingMatches.json'

    def download_today_fixtures(self) -> bool:
        """Download fixtures for today"""
        logger.info("Downloading today's fixtures...")

        try:
            cmd = [
                sys.executable, 'cli.py',
                '--task', 'download-fixtures',
                '--update-today',
                '--verbose'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)

            if result.returncode == 0:
                logger.info("✅ Fixtures download completed successfully")
                logger.debug(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"❌ Fixtures download failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Exception during fixtures download: {e}")
            return False

    def download_tomorrow_fixtures(self) -> bool:
        """Download fixtures for tomorrow (for next-day preparation)"""
        logger.info("Downloading tomorrow's fixtures...")

        try:
            cmd = [
                sys.executable, 'cli.py',
                '--task', 'download-fixtures',
                '--tomorrow',
                '--verbose'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)

            if result.returncode == 0:
                logger.info("✅ Tomorrow's fixtures downloaded successfully")
                return True
            else:
                logger.error(f"❌ Tomorrow's fixtures download failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Exception during tomorrow's fixtures download: {e}")
            return False

    def check_fixtures_available(self) -> bool:
        """Check if fixture file exists and has data for today"""
        if not self.fixtures_file.exists():
            logger.warning("Fixtures file doesn't exist")
            return False

        try:
            with open(self.fixtures_file) as f:
                data = json.load(f)

            # Count total fixtures
            total_fixtures = 0
            for country, leagues in data.items():
                for league, league_data in leagues.items():
                    fixtures = league_data.get('Fixtures', [])
                    total_fixtures += len(fixtures)

            logger.info(f"Found {total_fixtures} fixtures in {self.fixtures_file}")
            return total_fixtures > 0

        except Exception as e:
            logger.error(f"Error reading fixtures file: {e}")
            return False

    def run_analysis_if_fixtures_available(self) -> bool:
        """Run full league analysis if fixtures are available"""
        if not self.check_fixtures_available():
            logger.info("No fixtures available for analysis")
            return False

        logger.info("Running full league analysis on available fixtures...")

        try:
            today = datetime.now().strftime('%Y%m%d')

            cmd = [
                sys.executable, 'cli.py',
                '--task', 'full-league',
                '--leagues', 'E1,E2,E3',  # Championship, League One, League Two
                '--enable-double-chance',
                '--dc-min-prob', '0.75',
                '--dc-allow-multiple',
                '--ml-mode', 'predict',
                '--parallel-workers', '3',
                '--fixtures-date', today,
                '--verbose'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)

            if result.returncode == 0:
                logger.info("✅ Full league analysis completed successfully")
                return True
            else:
                logger.error(f"❌ Full league analysis failed: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Exception during analysis: {e}")
            return False

    def daily_update_workflow(self) -> bool:
        """Complete daily workflow: download fixtures"""
        logger.info("Starting daily fixtures update workflow...")

        success = True

        # 1. Download today's fixtures
        if not self.download_today_fixtures():
            success = False

        # 2. Download tomorrow's fixtures for preparation
        if not self.download_tomorrow_fixtures():
            logger.warning("Failed to download tomorrow's fixtures (non-critical)")

        return success

    def update_and_analyze_workflow(self) -> bool:
        """Complete workflow: download fixtures and run analysis"""
        logger.info("Starting update and analyze workflow...")

        # 1. Download today's fixtures
        if not self.download_today_fixtures():
            logger.error("Cannot proceed with analysis - fixture download failed")
            return False

        # 2. Run analysis if fixtures are available
        analysis_success = self.run_analysis_if_fixtures_available()

        # 3. Download tomorrow's fixtures (non-critical)
        self.download_tomorrow_fixtures()

        return analysis_success

    def cleanup_old_logs(self, days: int = 7):
        """Clean up log files older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            for log_file in LOG_DIR.glob("fixtures_scheduler_*.log"):
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        log_file.unlink()
                        logger.debug(f"Deleted old log: {log_file}")
                except Exception as e:
                    logger.debug(f"Error deleting {log_file}: {e}")

        except Exception as e:
            logger.warning(f"Error during log cleanup: {e}")

    def send_notification(self, success: bool, message: str):
        """Send notification about workflow completion (can be extended)"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        notification_msg = f"Fixtures Scheduler: {status} - {message}"

        logger.info(notification_msg)

        # Future enhancement: could send email, Slack, etc.
        # For now, just log


def main():
    parser = argparse.ArgumentParser(description='Automated football fixtures scheduler')

    # Workflow options
    parser.add_argument('--daily-update', action='store_true',
                       help='Run daily fixtures update (download only)')
    parser.add_argument('--update-and-analyze', action='store_true',
                       help='Download fixtures and run full analysis')

    # Maintenance options
    parser.add_argument('--cleanup-logs', type=int, default=7,
                       help='Clean up log files older than N days [default: 7]')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without executing')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable debug logging')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    scheduler = FixturesScheduler()

    # Cleanup old logs
    if args.cleanup_logs > 0:
        scheduler.cleanup_old_logs(args.cleanup_logs)

    success = False
    message = ""

    try:
        if args.dry_run:
            logger.info("DRY RUN: Would execute scheduler workflow")
            return 0

        if args.daily_update:
            success = scheduler.daily_update_workflow()
            message = "Daily fixtures update"

        elif args.update_and_analyze:
            success = scheduler.update_and_analyze_workflow()
            message = "Fixtures update and analysis"

        else:
            logger.info("No workflow specified. Use --daily-update or --update-and-analyze")
            parser.print_help()
            return 1

        # Send notification
        scheduler.send_notification(success, message)

        return 0 if success else 1

    except KeyboardInterrupt:
        logger.info("Scheduler interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Scheduler failed with exception: {e}")
        scheduler.send_notification(False, f"Exception: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
