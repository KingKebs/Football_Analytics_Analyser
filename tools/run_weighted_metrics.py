#!/usr/bin/env python3
"""
Quick Metrics Recomputation Tool
=================================
This is a DEVELOPMENT/DEBUGGING helper script.

**You probably don't need this for normal operations!**

Purpose:
  - Quickly recompute model metrics from existing enriched CSVs
  - Test metric calculations without running full analysis
  - Development tool for Steps 1-4 improvements

When to use:
  - Testing changes to train_models() function
  - Debugging metric calculations
  - Quick verification after code changes

For normal workflow, use instead:
  python3 automate_corner_predictions.py --input <log> --leagues E2,E3 --train-model --auto --mode fast

Usage:
  LEAGUE=E2 PYTHONPATH=. python3 tools/run_weighted_metrics.py
  LEAGUE=E3 PYTHONPATH=. python3 tools/run_weighted_metrics.py
"""
import os
import glob
import pandas as pd
from datetime import datetime
from corners_analysis import CornersAnalyzer


def latest_enriched_csv(league: str) -> str:
    pattern = os.path.join('data','corners', f'corners_analysis_{league}_*.csv')
    files = glob.glob(pattern)
    if not files:
        raise SystemExit(f"No enriched CSVs found for league {league}")
    return max(files, key=os.path.getmtime)


def main():
    league = os.environ.get('LEAGUE','E3')
    csv_path = latest_enriched_csv(league)
    analyzer = CornersAnalyzer(csv_path, league)
    # Load enriched CSV directly (bypass full pipeline)
    df = pd.read_csv(csv_path)
    analyzer.df = df.copy()  # set raw df to something reasonable
    analyzer.enriched_df = df.copy()
    # Train to compute (weighted + unweighted) metrics
    metrics = analyzer.train_models()
    print(f"Saved metrics for {league}")

if __name__ == '__main__':
    main()

