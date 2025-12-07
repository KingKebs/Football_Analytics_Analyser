#!/usr/bin/env python3
"""
Football Analytics Analyser - Unified CLI

A single entry point for all football analytics tasks:
- Download data
- Analyze full leagues
- Analyze single matches
- View results
- Validate data
- Run corner analysis
- Manage data organization
"""

import sys
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import json

# Add src/ directory to Python path for imports
SRC_DIR = Path(__file__).parent / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Setup logging
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


class FootballAnalyticsCLI:
    """Main CLI dispatcher for football analytics tasks."""

    # Required Python files for each task (in src/ directory)
    REQUIRED_MODULES = {
        'full-league': [
            'automate_football_analytics_fullLeague.py',
            'algorithms.py',
            'ml_training.py',
            'ml_evaluation.py',
        ],
        'single-match': [
            'automate_football_analytics.py',
            'algorithms.py',
        ],
        'download': [
            'download_all_tabs.py',
        ],
        'download-fixtures': [
            'fixtures_downloader.py',
        ],
        'organize': [
            'organize_structure.py',
        ],
        'validate': [
            'check_downloaded_data.py',
        ],
        'corners': [
            'corners_analysis.py',
            'automate_corner_predictions.py',
        ],
        'analyze-corners': [
            'corners_analysis.py',
        ],
        'view': [
            'view_suggestions.py',
        ],
        'backtest': [
            'analyze_suggestions_results.py',
        ],
        'help': [],
        # Add converter script requirement for new task
        'convert-upcoming': [
            'convert_upcoming_matches.py',
        ],
    }

    def __init__(self):
        self.parser = self._build_parser()
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        self.src_dir = Path('src')

    def _check_required_files(self, task: str) -> Tuple[bool, List[str]]:
        """
        Check if all required .py files exist in src/ for the given task.

        Args:
            task: The CLI task being executed

        Returns:
            Tuple: (all_found: bool, missing_files: List[str])
        """
        required_files = self.REQUIRED_MODULES.get(task, [])
        missing = []

        for filename in required_files:
            filepath = self.src_dir / filename
            if not filepath.exists():
                missing.append(filename)
                logger.warning(f"Missing: {filepath}")

        all_found = len(missing) == 0
        return all_found, missing

    def _validate_task_files(self, task: str) -> bool:
        """
        Validate that all required files for a task exist.
        Exit with error if files are missing.

        Args:
            task: The CLI task being executed

        Returns:
            bool: True if all files found, False otherwise
        """
        all_found, missing = self._check_required_files(task)

        if not all_found:
            logger.error(f"❌ MISSING REQUIRED FILES FOR TASK: {task}")
            logger.error(f"Expected to find {len(missing)} file(s) in src/:")
            for f in missing:
                logger.error(f"  ✗ src/{f}")
            logger.error(f"\nPlease ensure all required files are in src/ directory.")
            return False

        if self.REQUIRED_MODULES.get(task):
            logger.debug(f"✓ All required files found for task: {task}")

        return True

    def _build_parser(self) -> argparse.ArgumentParser:
        """Build the main argument parser."""
        parser = argparse.ArgumentParser(
            description='Football Analytics Analyser - Unified CLI',
            epilog='Examples:\n'
                   '  %(prog)s --task full-league --league E0 --ml-mode predict\n'
                   '  %(prog)s --task corners --input tmp/corners/251115_match_games.log --leagues E2,E3 --train-model --auto --mode fast\n'
                   '  %(prog)s --task download --leagues E0,SP1\n',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

        # Main task selector
        parser.add_argument(
            '--task',
            required=True,
            choices=[
                'full-league', 'single-match', 'download', 'download-fixtures', 'organize',
                'validate', 'corners', 'analyze-corners', 'view', 'backtest', 'help',
                # New task to run convert_upcoming_matches.py
                'convert-upcoming',
            ],
            help='Task to execute'
        )

        # League and season options
        parser.add_argument('--league', default='ALL', help='League code or ALL for all supported leagues [default: ALL]')
        parser.add_argument('--leagues', help='Comma-separated league codes for batch operations (e.g., E0,SP1,D1)')
        parser.add_argument('--season', default='AUTO', help='Season (e.g., 2024-25) or AUTO for latest [default: AUTO]')

        # Converter-specific args
        parser.add_argument('--output-dir', dest='output_dir', default='data', help='Output directory for convert_upcoming_matches.py [default: data]')
        parser.add_argument('--date', help='Override date (YYYY-MM-DD) for convert_upcoming_matches.py or fixtures download')

        # Fixtures download options
        parser.add_argument('--update-today', action='store_true', help='Download fixtures for today')
        parser.add_argument('--tomorrow', action='store_true', help='Download fixtures for tomorrow')
        parser.add_argument('--no-cache', action='store_true', help='Skip cache and force fresh download from APIs')

        # Match analysis options
        parser.add_argument('--home', help='Home team name (for single match analysis)')
        parser.add_argument('--away', help='Away team name (for single match analysis)')

        # Corner predictions input log (automate_corner_predictions.py)
        parser.add_argument('--input', help='Path to match log file for corner predictions (automate_corner_predictions.py)')
        parser.add_argument('--train-model', action='store_true', help='Train corner models (Steps 1-4) before predicting')
        parser.add_argument('--force', action='store_true', help='Force re-run even if results exist')
        parser.add_argument('--auto', action='store_true', help='Run without interactive prompts')
        parser.add_argument('--mode', choices=['fast','full'], default='fast', help='Corner predictions mode (fast/full) [default: fast]')
        # Extended corner analysis flags (forwarded to corners_analysis.py)
        parser.add_argument('--home-team', help='Home team for corner match-level prediction')
        parser.add_argument('--away-team', help='Away team for corner match-level prediction')
        parser.add_argument('--top-n', type=int, default=0, help='Show top N teams by average corners (0=skip)')
        parser.add_argument('--use-parsed-all', action='store_true', help='Use parsed fixtures (todays_fixtures_*) to generate corner predictions across leagues')
        parser.add_argument('--fixtures-date', help='Date (YYYYMMDD) for parsed fixtures file')
        parser.add_argument('--min-team-matches', type=int, default=5, help='Minimum historical matches per team for corner prediction')
        parser.add_argument('--save-enriched', action='store_true', help='Save enriched engineered corner feature CSVs')

        # Corner ML flags
        parser.add_argument('--corners-save-models', action='store_true', help='Persist corner ML models')
        parser.add_argument('--corners-models-dir', default='models/corners', help='Directory for corner models')
        parser.add_argument('--corners-rf-n-estimators', type=int, default=300, help='RF trees for corners models')
        parser.add_argument('--corners-rf-max-depth', type=int, default=None, help='RF max depth (None=unlimited)')
        parser.add_argument('--corners-cv-folds', type=int, default=5, help='CV folds for corner models')
        parser.add_argument('--corners-half-life-days', type=int, default=180, help='Recency weighting half-life days')
        parser.add_argument('--corners-n-jobs', type=int, default=-1, help='Parallel jobs for RF')
        parser.add_argument('--corners-use-ml-prediction', action='store_true', help='Use ML models for match/fixture corner predictions with ranges')
        parser.add_argument('--corners-mc-samples', type=int, default=1000, help='Monte Carlo samples for corner ranges')

        # Data source options
        parser.add_argument('--file', help='Input file path or filename')
        parser.add_argument('--source', default='football-data', help='Data source directory [default: football-data]')
        parser.add_argument('--target', default='data', help='Target directory for organized data [default: data]')

        # Rating model options
        parser.add_argument('--rating-model', choices=['blended', 'poisson', 'xg'], default='blended', help='Rating model type [default: blended]')
        parser.add_argument('--blend-weight', type=float, default=0.3, help='Blend weight for form (0.0-1.0) [default: 0.3]')
        parser.add_argument('--last-n', type=int, default=6, help='Number of recent matches to consider [default: 6]')

        # ML mode forwards
        parser.add_argument('--ml-mode', choices=['off','train','predict'], default='off', help='Enable ML integration for full league goals markets')
        parser.add_argument('--ml-validate', action='store_true', help='Show cross-validation metrics for ML models')
        parser.add_argument('--ml-algorithms', default='rf,xgb', help='Algorithms to train (comma list: rf,xgb)')
        parser.add_argument('--ml-decay', type=float, default=0.85, help='Recency decay factor for ML sample weights')
        parser.add_argument('--ml-min-samples', type=int, default=300, help='Minimum samples required before ML training runs')
        parser.add_argument('--ml-save-models', action='store_true', help='Save trained ML models to disk')
        parser.add_argument('--ml-models-dir', default='models', help='Directory for saved ML models')

        # General analysis options
        parser.add_argument('--min-confidence', type=float, default=0.6, help='Minimum market probability to surface in suggestions [default: 0.6]')
        parser.add_argument('--enable-double-chance', action='store_true', help='Enable Double Chance (1X, X2, 12) market suggestions')
        parser.add_argument('--dc-min-prob', type=float, default=0.75, help='Minimum probability threshold for Double Chance selection [default: 0.75]')
        parser.add_argument('--dc-secondary-threshold', type=float, default=0.80, help='Secondary threshold for adding DC alongside 1X2 [default: 0.80]')
        parser.add_argument('--dc-allow-multiple', action='store_true', help='Allow multiple Double Chance picks where applicable')

        # Performance options
        parser.add_argument('--parallel-workers', type=int, default=1, help='Number of worker threads to process leagues in parallel [default: 1]')

        # Validation options
        parser.add_argument('--check-all', action='store_true', help='Validate all data')
        parser.add_argument('--check-data', action='store_true', help='Validate CSV integrity')
        parser.add_argument('--check-corners', action='store_true', help='Validate corner data')

        # General options
        parser.add_argument('--dry-run', action='store_true', help='Preview actions without executing')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
        parser.add_argument('--output-format', choices=['json', 'csv', 'text', 'html'], default='json', help='Output format [default: json]')
        # NOTE: --use-parsed-all already defined above for corners/full-league context; duplicate removed
        return parser

    def run(self, args=None):
        args = self.parser.parse_args(args)
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Use the leagues list if provided, otherwise fall back to the single league default
        leagues_to_run = args.leagues or args.league
        logger.info(f"Task: {args.task} | Leagues: {leagues_to_run}")

        # Validate that required files exist for this task (unless it's 'help')
        if args.task != 'help':
            if not self._validate_task_files(args.task):
                logger.error(f"Cannot proceed with task '{args.task}' - missing required files")
                return 1

        try:
            if args.task == 'full-league':
                return self.task_full_league(args)
            elif args.task == 'single-match':
                return self.task_single_match(args)
            elif args.task == 'download':
                return self.task_download(args)
            elif args.task == 'download-fixtures':
                return self.task_download_fixtures(args)
            elif args.task == 'organize':
                return self.task_organize(args)
            elif args.task == 'validate':
                return self.task_validate(args)
            elif args.task == 'corners':
                return self.task_corners(args)
            elif args.task == 'analyze-corners':
                return self.task_analyze_corners(args)
            elif args.task == 'view':
                return self.task_view(args)
            elif args.task == 'backtest':
                return self.task_backtest(args)
            elif args.task == 'help':
                return self.task_help(args)
            elif args.task == 'convert-upcoming':
                return self.task_convert_upcoming(args)
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return 1

    def task_full_league(self, args):
        """Analyze a full league round for one or more leagues."""
        leagues_to_run = [league.strip() for league in (args.leagues or args.league).split(',') if league.strip()]
        logger.info(f"Starting full league analysis for: {', '.join(leagues_to_run)}")

        try:
            from automate_football_analytics_fullLeague import main as run_full_league

            # The underlying script's main function handles multiple leagues when passed a comma-separated string
            argv = [
                '--leagues', ','.join(leagues_to_run),
                '--rating-model', args.rating_model,
                '--rating-blend-weight', str(args.blend_weight),
                '--rating-last-n', str(args.last_n),
                '--ml-mode', args.ml_mode,
                '--ml-algorithms', args.ml_algorithms,
                '--ml-decay', str(args.ml_decay),
                '--ml-min-samples', str(args.ml_min_samples),
                '--min-confidence', str(args.min_confidence),
            ]
            if args.enable_double_chance:
                argv.append('--enable-double-chance')
                argv.extend(['--dc-min-prob', str(args.dc_min_prob), '--dc-secondary-threshold', str(args.dc_secondary_threshold)])
                if args.dc_allow_multiple:
                    argv.append('--dc-allow-multiple')
            # Pass input log if provided
            if args.input:
                argv.extend(['--input-log', args.input])

            if args.ml_validate:
                argv.append('--ml-validate')
            if args.ml_save_models:
                argv.append('--ml-save-models')
                argv.extend(['--ml-models-dir', args.ml_models_dir])
            if hasattr(args, 'parallel_workers') and args.parallel_workers:
                argv.extend(['--parallel-workers', str(args.parallel_workers)])
            if args.dry_run:
                argv.append('--dry-run')
            if args.verbose:
                argv.append('--verbose')

            logger.info(f"Running automate_football_analytics_fullLeague.py with args: {' '.join(argv)}")

            if not args.dry_run:
                # We call the script's main entry point once with all leagues
                sys.argv = ['automate_football_analytics_fullLeague.py'] + argv
                run_full_league(sys.argv[1:])

            logger.info("✓ Full league analysis complete")
            return 0

        except ImportError as e:
            logger.error(f"Could not import full league script: {e}")
            return 1

    def task_single_match(self, args):
        """Analyze a single match."""
        if not args.home or not args.away:
            logger.error("Single match analysis requires --home and --away")
            return 1

        logger.info(f"Analyzing match: {args.home} vs {args.away}")

        try:
            from automate_football_analytics import main as run_single

            argv = [
                '--home', args.home,
                '--away', args.away,
                '--rating-model', args.rating_model,
                '--rating-blend-weight', str(args.blend_weight),
            ]

            if args.verbose:
                argv.append('--verbose')

            if not args.dry_run:
                sys.argv = ['automate_football_analytics.py'] + argv
                run_single(sys.argv[1:])

            logger.info("✓ Single match analysis complete")
            return 0

        except ImportError as e:
            logger.error(f"Could not import single match script: {e}")
            return 1

    def task_download(self, args):
        """Download football data."""
        leagues = args.leagues or args.league
        logger.info(f"Downloading data for leagues: {leagues}")

        try:
            from download_all_tabs import main as run_download

            argv = [
                '--download-football-data',
                '--leagues', leagues,
                '--seasons', args.season,
            ]

            if args.dry_run:
                argv.append('--dry-run')

            if args.verbose:
                argv.append('--verbose')

            if not args.dry_run:
                sys.argv = ['download_all_tabs.py'] + argv
                run_download(sys.argv[1:])

            logger.info("✓ Data download complete")
            return 0

        except ImportError as e:
            logger.error(f"Could not import download script: {e}")
            return 1

    def task_organize(self, args):
        """Organize downloaded data."""
        logger.info(f"Organizing data from {args.source} to {args.target}")

        try:
            from organize_structure import main as run_organize

            argv = [
                '--source', args.source,
                '--target', args.target,
            ]

            if args.dry_run:
                argv.append('--dry-run')

            if not args.dry_run:
                sys.argv = ['organize_structure.py'] + argv
                run_organize(sys.argv[1:])

            logger.info("✓ Data organization complete")
            return 0

        except ImportError as e:
            logger.error(f"Could not import organize script: {e}")
            return 1

    def task_validate(self, args):
        """Validate downloaded data."""
        logger.info("Starting data validation")

        try:
            from check_downloaded_data import main as run_validate

            argv = []

            if args.check_all:
                argv.append('--check-all')
            if args.check_data:
                argv.append('--check-data')
            if args.check_corners:
                argv.append('--check-corners')

            if not argv:
                argv.append('--check-all')  # Default to checking all

            if not args.dry_run:
                sys.argv = ['check_downloaded_data.py'] + argv
                run_validate(sys.argv[1:])

            logger.info("✓ Data validation complete")
            return 0

        except ImportError as e:
            logger.error(f"Could not import validation script: {e}")
            return 1

    def task_corners(self, args):
        """Corner tasks: If --input is provided, run automate_corner_predictions.py. Otherwise, fall back to CornersAnalyzer workflows."""
        # 1) Corner predictions from match log via automate_corner_predictions.py
        if args.input:
            import subprocess
            cmd = [sys.executable, 'src/automate_corner_predictions.py', '--input', args.input]
            if args.leagues:
                cmd.extend(['--leagues', args.leagues])
            if args.train_model:
                cmd.append('--train-model')
            if args.force:
                cmd.append('--force')
            if args.auto:
                cmd.append('--auto')
            if args.mode:
                cmd.extend(['--mode', args.mode])
            if args.verbose:
                cmd.append('--verbose')
            logger.info(f"Running corner predictions: {' '.join(cmd)}")
            if not args.dry_run:
                return subprocess.run(cmd, capture_output=False, text=True).returncode
            return 0

        # 2) Extended corners_analysis.py path (supports new flags)
        import subprocess
        corner_cmd = [sys.executable, 'src/corners_analysis.py']
        # Determine effective league handling; if use-parsed-all, force ALL regardless of provided league(s)
        if args.use_parsed_all:
            corner_cmd.extend(['--league', 'ALL'])
        else:
            effective_league = args.leagues if args.leagues else args.league
            corner_cmd.extend(['--league', effective_league])
        if args.file:
            corner_cmd.extend(['--file', args.file])
        # Skip passing explicit home/away when using parsed fixtures across all leagues
        if not args.use_parsed_all:
            if args.home_team:
                corner_cmd.extend(['--home-team', args.home_team])
            if args.away_team:
                corner_cmd.extend(['--away-team', args.away_team])
        if args.top_n and args.top_n > 0:
            corner_cmd.extend(['--top-n', str(args.top_n)])
        if args.train_model:
            corner_cmd.append('--train-model')
        if args.save_enriched:
            corner_cmd.append('--save-enriched')
        if args.use_parsed_all:
            corner_cmd.append('--use-parsed-all')
            if args.fixtures_date:
                corner_cmd.extend(['--fixtures-date', args.fixtures_date])
            corner_cmd.extend(['--min-team-matches', str(args.min_team_matches)])
        # Forward corner ML flags
        if args.corners_save_models:
            corner_cmd.append('--corners-save-models')
            corner_cmd.extend(['--corners-models-dir', args.corners_models_dir])
        corner_cmd.extend(['--corners-rf-n-estimators', str(args.corners_rf_n_estimators)])
        if args.corners_rf_max_depth is not None:
            corner_cmd.extend(['--corners-rf-max-depth', str(args.corners_rf_max_depth)])
        corner_cmd.extend(['--corners-cv-folds', str(args.corners_cv_folds), '--corners-half-life-days', str(args.corners_half_life_days), '--corners-n-jobs', str(args.corners_n_jobs), '--corners-mc-samples', str(args.corners_mc_samples)])
        if args.corners_use_ml_prediction:
            corner_cmd.append('--corners-use-ml-prediction')
        if args.verbose:
            corner_cmd.append('--json-summary')  # show summary when verbose for insight
        logger.info(f"Running corners_analysis.py: {' '.join(corner_cmd)}")
        if not args.dry_run:
            return subprocess.run(corner_cmd, capture_output=False, text=True).returncode
        return 0

    def task_analyze_corners(self, args):
        """Run corner analysis on all available data."""
        logger.info("Running full corners analysis on all available data")

        try:
            from corners_analysis import CornersAnalyzer
            from pathlib import Path
            import os
            from datetime import datetime

            data_dir = Path('football-data')
            csv_files = list(data_dir.glob('*.csv'))

            if not csv_files:
                logger.warning("No CSV files found in football-data/")
                return 1

            if not args.dry_run:
                for csv_file in csv_files[:1]:  # Analyze first by default
                    logger.info(f"Analyzing {csv_file.name}")
                    analyzer = CornersAnalyzer(
                        str(csv_file),
                        rf_params={
                            'n_estimators': args.corners_rf_n_estimators,
                            'random_state': 42,
                            'n_jobs': args.corners_n_jobs,
                            'max_depth': args.corners_rf_max_depth
                        },
                        cv_folds=args.corners_cv_folds,
                        half_life_days=args.corners_half_life_days,
                        mc_samples=args.corners_mc_samples
                    )

                    # Execute the proper workflow
                    if analyzer.load_data() is None:
                        logger.warning(f"Failed to load data from {csv_file.name}")
                        continue

                    if not analyzer.validate_corners_data():
                        logger.warning(f"Invalid corners data in {csv_file.name}")
                        continue

                    analyzer.clean_features()
                    analyzer.engineer_features()
                    analyzer.estimate_half_split()
                    analyzer.calculate_correlations()
                    analyzer.calculate_team_stats()

                    if args.train_model:
                        metrics = analyzer.train_models(
                            save_models=args.corners_save_models,
                            models_dir=args.corners_models_dir
                        )
                        logger.info(f"Model metrics: {metrics}")

                    if args.save_enriched:
                        out_dir = args.output_dir or 'data/corners'
                        os.makedirs(out_dir, exist_ok=True)
                        enriched_path = os.path.join(out_dir, f'enriched_{csv_file.stem}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                        analyzer.enriched_df.to_csv(enriched_path, index=False)
                        logger.info(f"Enriched feature set saved: {enriched_path}")

                    logger.info(f"✓ Analysis complete for {csv_file.name}")

            logger.info(f"✓ Analyzed {len(csv_files)} files")
            return 0

        except ImportError as e:
            logger.error(f"Could not import corners analyzer: {e}")
            return 1

    def task_view(self, args):
        """View analysis results."""
        logger.info("Viewing analysis results")

        try:
            from view_suggestions import main as run_view

            argv = []
            if args.file:
                argv.extend(['--file', args.file])

            if not args.dry_run:
                sys.argv = ['view_suggestions.py'] + argv
                run_view(sys.argv[1:])

            logger.info("✓ View complete")
            return 0

        except ImportError as e:
            logger.error(f"Could not import view script: {e}")
            return 1

    def task_backtest(self, args):
        """Run backtest analysis on results."""
        logger.info("Running backtest analysis")

        try:
            from analyze_suggestions_results import main as run_backtest

            argv = []
            if args.file:
                argv.extend(['--file', args.file])

            if not args.dry_run:
                sys.argv = ['analyze_suggestions_results.py'] + argv
                run_backtest(sys.argv[1:])

            logger.info("✓ Backtest complete")
            return 0

        except ImportError as e:
            logger.error(f"Could not import backtest script: {e}")
            return 1

    def task_help(self, args):
        """Show detailed help information."""
        print(self.parser.format_help())

        print("\n" + "="*80)
        print("DETAILED USAGE GUIDE")
        print("="*80)

        help_text = """
### FULL LEAGUE ANALYSIS
  Analyze an entire league round with ratings, probabilities, and suggestions.
  
  $ python cli.py --task full-league --league E0 --rating-model blended --last-n 6
  
  Options:
    --league              League code (E0, SP1, D1, etc.)
    --rating-model        Model type: blended | poisson | xg
    --blend-weight        Form blending weight (0.0-1.0)
    --last-n              Matches to consider for form (default: 6)

### SINGLE MATCH ANALYSIS
  Analyze a specific upcoming match.
  
  $ python cli.py --task single-match --home Arsenal --away Chelsea
  
  Options:
    --home, --away        Team names (must be exact)
    --rating-model        Model type
    --blend-weight        Form blending weight

### DOWNLOAD DATA
  Download latest football data from football-data.co.uk
  
  $ python cli.py --task download --leagues E0,SP1,D1 --season AUTO
  
  Options:
    --leagues             Comma-separated league codes
    --season              Season (e.g., 2024-25) or AUTO

### ORGANIZE DATA
  Structure downloaded CSV files for analysis.
  
  $ python cli.py --task organize --source football-data --target data
  
  Options:
    --source, --target    Directory paths

### VALIDATE DATA
  Check CSV integrity, corner data, and team coverage.
  
  $ python cli.py --task validate --check-all
  $ python cli.py --task validate --check-data --check-corners
  
  Options:
    --check-all           Validate everything
    --check-data          CSV integrity
    --check-corners       Corner data completeness

### CORNER ANALYSIS
  Analyze corner patterns in a specific file.
  
  $ python cli.py --task corners --file E0_2425.csv
  
  Options:
    --file                CSV filename or path

### ANALYZE ALL CORNERS
  Run corners analysis on all available data.
  
  $ python cli.py --task analyze-corners

### DOWNLOAD FIXTURES
  Automatically download today's fixtures from free APIs.
  
  $ python cli.py --task download-fixtures --update-today
  $ python cli.py --task download-fixtures --date 2025-11-30
  $ python cli.py --task download-fixtures --leagues "Championship,League One"
  
  Options:
    --update-today        Download fixtures for today
    --tomorrow            Download fixtures for tomorrow  
    --date YYYY-MM-DD     Download fixtures for specific date
    --no-cache           Skip cache and force fresh API calls

### VIEW RESULTS
  View generated suggestions and predictions.
  
  $ python cli.py --task view --file data/full_league_suggestions_E0_*.json

### BACKTEST
  Run backtesting on analysis results to evaluate ROI/yield.
  
  $ python cli.py --task backtest --file data/full_league_suggestions_*.json

### GENERAL OPTIONS
  --dry-run             Preview actions without executing
  --verbose             Detailed output with DEBUG logs
  --output-format       json | csv | text | html (default: json)

### EXAMPLES
  # Analyze EPL with blended model
  python cli.py --task full-league --league E0 --rating-model blended
  
  # Download multiple leagues
  python cli.py --task download --leagues E0,SP1,D1,F1 --season AUTO
  
  # Validate all data before analysis
  python cli.py --task validate --check-all
  
  # Analyze corners in EPL data
  python cli.py --task corners --file E0_2425.csv
  
  # Run full workflow (download + organize + validate + analyze)
  python cli.py --task download --leagues E0
  python cli.py --task organize --source football-data
  python cli.py --task validate --check-all
  python cli.py --task full-league --league E0

        """

        print(help_text)
        return 0

    def task_convert_upcoming(self, args):
        """Run src/convert_upcoming_matches.py with provided input/output args."""
        import subprocess
        from datetime import datetime

        input_path = args.input or 'data/raw/upcomingMatches.json'

        # Ensure the output directory exists
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

        # Pass the output directory to the script (not a specific file)
        cmd = [
            sys.executable,
            "src/convert_upcoming_matches.py",
            "--input",
            input_path,
            "--output-dir",
            args.output_dir,
        ]

        if args.date:
            cmd.extend(['--date', args.date])

        logger.info(f"Running convert_upcoming_matches.py: {' '.join(cmd)}")
        if args.dry_run:
            return 0
        return subprocess.run(cmd, capture_output=False, text=True).returncode

    def task_download_fixtures(self, args):
        """Download today's fixtures from free APIs and update upcomingMatches.json"""
        logger.info("Downloading fixtures from free APIs...")

        try:
            from fixtures_downloader import FixturesDownloader
            from datetime import datetime, timedelta

            downloader = FixturesDownloader()

            # Determine target date
            if args.update_today:
                target_date = datetime.now().strftime('%Y-%m-%d')
            elif args.tomorrow:
                target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            elif args.date:
                target_date = args.date
            else:
                target_date = datetime.now().strftime('%Y-%m-%d')
                logger.info("No date specified, using today")

            # Parse leagues from CLI args
            target_leagues = None
            if args.leagues:
                target_leagues = [league.strip() for league in args.leagues.split(',')]
            elif args.league and args.league != 'ALL':
                target_leagues = [args.league]

            # Download and update fixtures
            downloader.update_upcoming_matches(
                date=target_date,
                leagues=target_leagues,
                dry_run=args.dry_run
            )

            if not args.dry_run:
                logger.info(f"✅ Successfully updated fixtures for {target_date}")
                logger.info(f"📁 Output: {downloader.config['output_file']}")

            return 0

        except ImportError as e:
            logger.error(f"Could not import fixtures downloader: {e}")
            logger.info("Install dependencies: pip install requests beautifulsoup4")
            return 1
        except Exception as e:
            logger.error(f"Fixtures download failed: {e}")
            return 1


def main():
    """Main entry point."""
    cli = FootballAnalyticsCLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
