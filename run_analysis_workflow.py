#!/usr/bin/env python3
"""
Dynamic Analysis Workflow Orchestrator

Automated workflow for football analytics:
1. Read upcoming matches from data/raw/upcomingMatches.json
2. Auto-detect leagues and prompt for confirmation
3. Convert to parsed fixtures format
4. Run full-league analysis with ML predictions
5. Run corners analysis
6. Generate consolidated outputs for Streamlit UI

Usage:
  python run_analysis_workflow.py --date 2026-02-11
  python run_analysis_workflow.py --date 2026-02-11 --leagues E0,E2,E3  # Override auto-detection
  python run_analysis_workflow.py --auto  # Skip prompts, use defaults
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Set


class AnalysisWorkflow:
    """Orchestrates the full analysis workflow with user prompts."""

    def __init__(self, date: str = None, leagues: str = None, auto: bool = False, verbose: bool = False):
        self.date = date or datetime.now().strftime('%Y-%m-%d')
        self.date_compact = self.date.replace('-', '')
        self.manual_leagues = leagues
        self.auto = auto
        self.verbose = verbose
        self.upcoming_file = 'data/raw/upcomingMatches.json'
        self.output_dir = 'data/analysis'

    def print_header(self, text: str):
        """Print a formatted header."""
        print("\n" + "="*80)
        print(f"  {text}")
        print("="*80)

    def print_step(self, step: int, text: str):
        """Print a step indicator."""
        print(f"\n{'>'*3} STEP {step}: {text}")

    def read_upcoming_matches(self) -> dict:
        """Read and parse upcomingMatches.json."""
        self.print_step(1, "Reading upcoming matches")

        if not os.path.exists(self.upcoming_file):
            print(f"❌ Error: {self.upcoming_file} not found")
            sys.exit(1)

        with open(self.upcoming_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✅ Loaded upcoming matches from {self.upcoming_file}")
        return data

    def extract_leagues(self, data: dict) -> Set[str]:
        """Extract unique league codes from upcoming matches data."""
        self.print_step(2, "Detecting leagues from upcoming matches")

        # League mapping for detection
        LEAGUE_MAP = {
            'Premier League': 'E0',
            'Championship': 'E1',
            'League One': 'E2',
            'League Two': 'E3',
            'Ligue 1': 'F1',
            'Ligue 2': 'F2',
            'Bundesliga': 'D1',
            '2. Bundesliga': 'D2',
            'Serie A': 'I1',
            'Serie B': 'I2',
            'Eredivisie': 'N1',
            'La Liga': 'SP1',
            'Segunda División': 'SP2',
        }

        leagues = set()
        match_count = 0

        # Handle flat list format
        if isinstance(data, list):
            for match in data:
                league = match.get('league', '')
                if league:
                    leagues.add(league)
                match_count += 1
        # Handle hierarchical format
        elif isinstance(data, dict):
            for country, competitions in data.items():
                if not isinstance(competitions, dict):
                    continue
                for competition, comp_data in competitions.items():
                    league_code = LEAGUE_MAP.get(competition, '')
                    if league_code:
                        leagues.add(league_code)

                    # Count matches
                    matches = comp_data.get('matches', []) if isinstance(comp_data, dict) else []
                    match_count += len(matches)

        print(f"✅ Detected {len(leagues)} leagues: {', '.join(sorted(leagues))}")
        print(f"   Total matches: {match_count}")
        return leagues

    def confirm_leagues(self, detected_leagues: Set[str]) -> List[str]:
        """Prompt user to confirm leagues for analysis."""
        self.print_step(3, "League confirmation")

        # Use manual override if provided
        if self.manual_leagues:
            leagues = [l.strip() for l in self.manual_leagues.split(',') if l.strip()]
            print(f"📌 Using manually specified leagues: {', '.join(leagues)}")
            return leagues

        # Use detected leagues
        leagues = sorted(detected_leagues)

        if not leagues:
            print("❌ No leagues detected")
            sys.exit(1)

        print(f"\n📊 Leagues to be analyzed:")
        for league in leagues:
            print(f"   • {league}")

        if not self.auto:
            response = input(f"\n✅ Proceed with these {len(leagues)} league(s)? (y/n): ").strip().lower()
            if response != 'y':
                print("❌ Analysis cancelled by user")
                sys.exit(0)
        else:
            print("⚡ Auto-mode: Proceeding without confirmation")

        return leagues

    def convert_fixtures(self) -> bool:
        """Convert upcoming matches to parsed fixtures format."""
        self.print_step(4, "Converting fixtures to parsed format")

        cmd = [
            sys.executable,
            'cli.py',
            '--task', 'convert-upcoming',
            '--file', self.upcoming_file,
            '--output-dir', self.output_dir,
            '--date', self.date
        ]

        print(f"🔧 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Fixtures converted successfully")
            print(f"   Output: {self.output_dir}/todays_fixtures_{self.date_compact}.json")
            return True
        else:
            print(f"❌ Conversion failed")
            print(result.stderr)
            return False

    def run_full_league_analysis(self, leagues: List[str]) -> bool:
        """Run full-league analysis with ML predictions."""
        self.print_step(5, "Running full-league analysis")

        cmd = [
            sys.executable,
            'cli.py',
            '--task', 'full-league',
            '--use-parsed-all',
            '--fixtures-date', self.date_compact,
            '--leagues', ','.join(leagues),
            '--ml-mode', 'predict',
            '--enable-double-chance',
            '--dc-min-prob', '0.75',
            '--dc-secondary-threshold', '0.80'
        ]

        if self.verbose:
            cmd.append('--verbose')

        print(f"🔧 Running: {' '.join(cmd)}")
        print(f"   Leagues: {', '.join(leagues)}")
        print(f"   Date: {self.date}")
        print(f"   ML Mode: predict (with fixed feature engineering)")

        if not self.auto:
            response = input(f"\n▶ Start full-league analysis? (y/n): ").strip().lower()
            if response != 'y':
                print("⏭ Skipping full-league analysis")
                return False

        result = subprocess.run(cmd)

        if result.returncode == 0:
            print(f"✅ Full-league analysis completed")
            print(f"   Per-league outputs: data/analysis/full_league_suggestions_<LEAGUE>_{self.date_compact}_*.json")
            print(f"   Consolidated output: data/analysis/consolidated_full_league_{self.date_compact}_*.json")
            return True
        else:
            print(f"❌ Full-league analysis failed with code {result.returncode}")
            return False

    def run_corners_analysis(self, leagues: List[str]) -> bool:
        """Run corners analysis."""
        self.print_step(6, "Running corners analysis")

        cmd = [
            sys.executable,
            'cli.py',
            '--task', 'corners',
            '--use-parsed-all',
            '--fixtures-date', self.date_compact,
            '--league', 'ALL',  # Auto-detect from parsed fixtures
            '--min-team-matches', '3',
            '--corners-use-ml-prediction'
        ]

        if self.verbose:
            cmd.append('--verbose')

        print(f"🔧 Running: {' '.join(cmd)}")
        print(f"   Date: {self.date}")
        print(f"   Min team matches: 3")
        print(f"   ML prediction: enabled")

        if not self.auto:
            response = input(f"\n▶ Start corners analysis? (y/n): ").strip().lower()
            if response != 'y':
                print("⏭ Skipping corners analysis")
                return False

        result = subprocess.run(cmd)

        if result.returncode == 0:
            print(f"✅ Corners analysis completed")
            print(f"   Output: data/corners/parsed_corners_predictions_{self.date_compact}.json")
            return True
        else:
            print(f"❌ Corners analysis failed with code {result.returncode}")
            return False

    def print_summary(self, full_league_success: bool, corners_success: bool):
        """Print final summary."""
        self.print_header("WORKFLOW COMPLETE")

        print(f"\n📊 Analysis Date: {self.date}")
        print(f"\n✅ Full-League Analysis: {'SUCCESS' if full_league_success else 'FAILED/SKIPPED'}")
        if full_league_success:
            print(f"   • Per-league suggestions: data/analysis/full_league_suggestions_*_{self.date_compact}_*.json")
            print(f"   • Consolidated output: data/analysis/consolidated_full_league_{self.date_compact}_*.json")

        print(f"\n✅ Corners Analysis: {'SUCCESS' if corners_success else 'FAILED/SKIPPED'}")
        if corners_success:
            print(f"   • Corners predictions: data/corners/parsed_corners_predictions_{self.date_compact}.json")

        print(f"\n🎯 Next Steps:")
        print(f"   • View results in Streamlit UI: streamlit run src/streamlit_app.py")
        print(f"   • Or view JSON files directly in data/analysis/ and data/corners/")

    def run(self):
        """Execute the complete workflow."""
        self.print_header(f"Football Analytics Workflow - {self.date}")

        # Step 1-2: Read and detect leagues
        data = self.read_upcoming_matches()
        detected_leagues = self.extract_leagues(data)

        # Step 3: Confirm leagues
        leagues = self.confirm_leagues(detected_leagues)

        # Step 4: Convert fixtures
        if not self.convert_fixtures():
            print("\n❌ Workflow failed at fixture conversion")
            sys.exit(1)

        # Step 5: Full-league analysis
        full_league_success = self.run_full_league_analysis(leagues)

        # Step 6: Corners analysis
        corners_success = self.run_corners_analysis(leagues)

        # Summary
        self.print_summary(full_league_success, corners_success)


def main():
    parser = argparse.ArgumentParser(
        description='Dynamic Football Analytics Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for confirmation)
  python run_analysis_workflow.py --date 2026-02-11
  
  # Auto mode (no prompts, uses detected leagues)
  python run_analysis_workflow.py --date 2026-02-11 --auto
  
  # Override league detection
  python run_analysis_workflow.py --date 2026-02-11 --leagues E0,E2,E3
  
  # Use today's date
  python run_analysis_workflow.py --auto
        """
    )

    parser.add_argument('--date', help='Analysis date (YYYY-MM-DD). Default: today')
    parser.add_argument('--leagues', help='Override auto-detected leagues (comma-separated, e.g., E0,E2,E3)')
    parser.add_argument('--auto', action='store_true', help='Auto mode: skip all confirmation prompts')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    workflow = AnalysisWorkflow(
        date=args.date,
        leagues=args.leagues,
        auto=args.auto,
        verbose=args.verbose
    )

    try:
        workflow.run()
    except KeyboardInterrupt:
        print("\n\n❌ Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Workflow failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

