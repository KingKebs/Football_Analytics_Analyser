#!/usr/bin/env python3
"""
Football Analytics Analyser - Setup & Initialization Script

Initializes the project structure, validates environment, and prepares for first run.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ProjectSetup:
    """Initialize and validate project structure."""

    def __init__(self):
        self.root_dir = Path.cwd()
        self.issues = []
        self.warnings = []

    def print_header(self):
        """Print setup header."""
        print("\n" + "="*80)
        print("Football Analytics Analyser - Project Setup")
        print("="*80 + "\n")

    def check_python_version(self):
        """Check Python version (3.8+)."""
        logger.info("Checking Python version...")

        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            msg = f"Python 3.8+ required, but found Python {version.major}.{version.minor}"
            self.issues.append(msg)
            logger.error(f"✗ {msg}")
            return False

        logger.info(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True

    def check_required_files(self):
        """Check that key files exist."""
        logger.info("Checking required files...")

        required = [
            'algorithms.py',
            'automate_football_analytics_fullLeague.py',
            'download_all_tabs.py',
            'requirements.txt',
        ]

        missing = []
        for fname in required:
            if not (self.root_dir / fname).exists():
                missing.append(fname)

        if missing:
            msg = f"Missing files: {', '.join(missing)}"
            self.issues.append(msg)
            logger.error(f"✗ {msg}")
            return False

        logger.info(f"✓ All required files present ({len(required)} files)")
        return True

    def check_directories(self):
        """Check directory structure."""
        logger.info("Checking directory structure...")

        dirs = [
            'data',
            'football-data',
            'logs',
            'tests',
        ]

        for dirname in dirs:
            dir_path = self.root_dir / dirname
            if not dir_path.exists():
                logger.warning(f"⚠ Creating missing directory: {dirname}/")
                dir_path.mkdir(parents=True, exist_ok=True)
                self.warnings.append(f"Created {dirname}/")

        logger.info(f"✓ Directory structure OK")
        return True

    def check_dependencies(self):
        """Check Python dependencies."""
        logger.info("Checking Python dependencies...")

        required_packages = [
            'pandas',
            'numpy',
            'scipy',
            'requests',
            'matplotlib',
            'seaborn',
            'colorama',
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if missing:
            self.warnings.append(f"Missing packages: {', '.join(missing)}")
            logger.warning(f"⚠ Missing packages: {', '.join(missing)}")
            logger.warning("  Run: pip install -r requirements.txt")
            return False

        logger.info(f"✓ All dependencies installed ({len(required_packages)} packages)")
        return True

    def check_git_status(self):
        """Check git status."""
        logger.info("Checking git status...")

        try:
            result = subprocess.run(
                ['git', 'status', '--short'],
                capture_output=True,
                text=True,
                cwd=self.root_dir
            )

            if result.returncode == 0:
                changes = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                logger.info(f"✓ Git repository OK ({changes} changes)")
                return True
            else:
                self.warnings.append("Not a git repository")
                logger.warning("⚠ Not a git repository")
                return False

        except Exception as e:
            self.warnings.append(f"Could not check git: {e}")
            logger.warning(f"⚠ Could not check git: {e}")
            return False

    def validate_scripts(self):
        """Validate key Python scripts."""
        logger.info("Validating scripts...")

        scripts = [
            'cli.py',
            'data_manager.py',
            'corners_analysis.py',
            'algorithms.py',
        ]

        errors = 0
        for script in scripts:
            script_path = self.root_dir / script
            if script_path.exists():
                try:
                    result = subprocess.run(
                        ['python3', '-m', 'py_compile', str(script_path)],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        logger.info(f"  ✓ {script}")
                    else:
                        logger.error(f"  ✗ {script}: {result.stderr}")
                        self.issues.append(f"Syntax error in {script}")
                        errors += 1
                except Exception as e:
                    logger.error(f"  ✗ {script}: {e}")
                    errors += 1

        if errors == 0:
            logger.info(f"✓ All scripts valid")
            return True
        else:
            logger.error(f"✗ {errors} scripts have errors")
            return False

    def create_logs_directory(self):
        """Create logs directory."""
        logs_dir = self.root_dir / 'logs'
        logs_dir.mkdir(exist_ok=True)
        logger.info("✓ Logs directory ready")

    def create_sample_config(self):
        """Create sample configuration file."""
        config_path = self.root_dir / 'config.example.yaml'

        if not config_path.exists():
            sample_config = """# Football Analytics Analyser - Configuration
# Copy this to config.yaml and customize

# Analysis Settings
analysis:
  league: 'E0'  # Default league
  season: 'AUTO'  # or specific season like '2024-25'
  rating_model: 'blended'  # blended | poisson | xg
  blend_weight: 0.3  # Form blending (0.0-1.0)
  last_n_matches: 6  # Recent matches to consider

# Data Settings
data:
  data_dir: 'data'
  football_data_dir: 'football-data'
  archive_old_files: true
  archive_threshold_days: 7

# Output Settings
output:
  format: 'json'  # json | csv | html | text
  save_all_results: true
  save_manifests: true

# Logging Settings
logging:
  level: 'INFO'  # DEBUG | INFO | WARNING | ERROR
  log_dir: 'logs'
  preserve_logs: true
  max_log_size_mb: 100

# Corners Analysis
corners:
  enabled: true
  estimate_half_splits: true
  save_team_stats: true

# Advanced Options
advanced:
  dry_run: false
  verbose: false
  parallel_analysis: false
"""
            with open(config_path, 'w') as f:
                f.write(sample_config)
            logger.info(f"✓ Created sample config: {config_path.name}")
        else:
            logger.info(f"✓ Config file exists: {config_path.name}")

    def print_summary(self):
        """Print setup summary."""
        print("\n" + "="*80)
        print("SETUP SUMMARY")
        print("="*80)

        if not self.issues and not self.warnings:
            print("\n✓ All checks passed! Project is ready to use.\n")
            print("Quick start:")
            print("  python cli.py --task help")
            print("  python cli.py --task download --leagues E0")
            print("  python cli.py --task full-league --league E0\n")
            return True

        if self.warnings:
            print(f"\n⚠ {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if self.issues:
            print(f"\n✗ {len(self.issues)} issue(s) to resolve:")
            for issue in self.issues:
                print(f"  - {issue}")
            print("\nSetup cannot proceed. Please fix the issues above.\n")
            return False

        return True

    def run_full_setup(self):
        """Run complete setup."""
        self.print_header()

        checks = [
            self.check_python_version,
            self.check_required_files,
            self.check_directories,
            self.check_dependencies,
            self.validate_scripts,
            self.create_logs_directory,
            self.check_git_status,
            self.create_sample_config,
        ]

        for check in checks:
            try:
                check()
            except Exception as e:
                logger.error(f"Error in {check.__name__}: {e}")
                self.issues.append(str(e))

        success = self.print_summary()
        return 0 if success else 1


def main():
    """Main entry point."""
    setup = ProjectSetup()
    exit_code = setup.run_full_setup()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

