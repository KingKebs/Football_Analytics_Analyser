#!/usr/bin/env python3
"""
Automate Corner Predictions Workflow
=====================================
Chains parse_match_log.py -> corners_analysis.py for batch corner predictions.

Usage:
  python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --league E2
  python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --league E2 --auto
  python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --leagues E2,E3 --auto
  python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --leagues E2,E3 --auto --reuse-today

This script:
1. Parses the match log using parse_match_log.py
2. Loads the parsed fixtures (scheduled only)
3. Runs corners_analysis.py per league (reuses if already run today unless --force)
4. Iterates through each fixture and generates corner predictions
5. Exports prediction JSON(s): combined and per-league (single file per day with overwrite option)
"""

import argparse
import json
import os
import subprocess
import sys
import math
import glob
import time
from datetime import datetime
from collections import defaultdict
import pandas as pd


def cleanup_old_daily_files(output_dir: str, keep_days: int = 7):
    """Remove old timestamped files, keeping only the last keep_days worth."""
    if not os.path.exists(output_dir):
        return

    from datetime import timedelta
    cutoff = datetime.now() - timedelta(days=keep_days)
    cutoff_ts = cutoff.timestamp()

    patterns = [
        'batch_predictions_*.json',
        'match_prediction_*.json',
        'corners_analysis_*.csv',
        'corners_correlations_*.json',
        'team_stats_*.json'
    ]

    removed_count = 0
    for pattern in patterns:
        for fpath in glob.glob(os.path.join(output_dir, pattern)):
            try:
                if os.path.getmtime(fpath) < cutoff_ts:
                    os.remove(fpath)
                    removed_count += 1
            except Exception:
                pass

    if removed_count > 0:
        print(f"🗑️  Cleaned up {removed_count} old file(s) older than {keep_days} days")


def find_today_file(pattern: str) -> str:
    """Find file matching pattern from today (YYYYMMDD in filename)."""
    today = datetime.now().strftime('%Y%m%d')
    files = glob.glob(pattern)
    for f in files:
        if today in os.path.basename(f):
            return f
    return ''


def remove_duplicate_daily_files(output_dir: str, pattern_prefix: str):
    """Remove all but the latest file matching pattern for today."""
    today = datetime.now().strftime('%Y%m%d')
    pattern = os.path.join(output_dir, f"{pattern_prefix}*{today}*.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)

    if len(files) > 1:
        # Keep the latest, remove the rest
        for f in files[:-1]:
            try:
                os.remove(f)
                print(f"  🗑️  Removed duplicate: {os.path.basename(f)}")
            except Exception:
                pass


def run_parse_match_log(input_log: str, leagues: list[str], output_dir: str = 'data') -> tuple:
    """Run parse_match_log.py and return path to outputs."""
    print(f"\n{'='*70}")
    print("STEP 1: Parsing match log")
    print(f"{'='*70}")

    leagues_arg = ",".join(leagues) if leagues else ""
    cmd = [
        'python3', 'parse_match_log.py',
        '--input', input_log,
        '--output-dir', output_dir
    ]
    if leagues_arg:
        cmd.extend(['--leagues', leagues_arg])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error parsing match log: {result.stderr}")
        return None, None

    print(result.stdout)

    # Determine output paths
    date_suffix = datetime.now().strftime("%Y%m%d")
    csv_path = os.path.join(output_dir, f"todays_fixtures_{date_suffix}.csv")
    json_path = os.path.join(output_dir, f"todays_fixtures_{date_suffix}.json")

    return csv_path, json_path


def load_parsed_fixtures(json_path: str) -> list:
    """Load fixtures from JSON output."""
    if not os.path.exists(json_path):
        print(f"Fixtures file not found: {json_path}")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        fixtures = json.load(f)

    # Filter to scheduled only (already done by parser but double-check)
    scheduled = [f for f in fixtures if f.get('status', '').lower() == 'scheduled']

    return scheduled


def group_fixtures_by_league(fixtures: list) -> dict[str, list]:
    grouped = defaultdict(list)
    for fx in fixtures:
        lg = (fx.get('league') or '').strip().upper()
        grouped[lg].append(fx)
    return grouped


def check_league_analyzed_today(league_code: str) -> bool:
    """Check if corners analysis for this league has been run today."""
    corners_dir = 'data/corners'
    if not os.path.exists(corners_dir):
        return False

    today = datetime.now().strftime("%Y%m%d")
    pattern = f"team_stats_{league_code}_{today}*.json"

    matches = glob.glob(os.path.join(corners_dir, pattern))

    return len(matches) > 0


def run_corners_analysis(league_code: str, seasons: str = None, force: bool = False, train_model: bool = False):
    """Run corners_analysis.py for the league if needed."""
    print(f"\n{'='*70}")
    print(f"STEP 2: Running corners analysis for {league_code}")
    print(f"{'='*70}")

    if not force and check_league_analyzed_today(league_code):
        print(f"✓ Corners analysis for {league_code} already run today (use --force to re-run)")
        return True

    cmd = [
        'python3', 'corners_analysis.py',
        '--league', league_code,
        '--no-prompt'
    ]

    if train_model:
        cmd.append('--train-model')

    if seasons:
        cmd.extend(['--seasons', seasons])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running corners analysis: {result.stderr}")
        return False

    # Print the full stdout for visibility
    print(result.stdout)
    if result.stderr:
        print("[corners_analysis stderr]")
        print(result.stderr)

    return True


def generate_corner_prediction(league_code: str, home_team: str, away_team: str, seasons: str = None) -> dict:
    """Generate corner prediction for a single match using corners_analysis.py."""
    cmd = [
        'python3', 'corners_analysis.py',
        '--league', league_code,
        '--home-team', home_team,
        '--away-team', away_team,
        '--no-prompt'
    ]

    if seasons:
        cmd.extend(['--seasons', seasons])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ⚠ Prediction failed: {result.stderr[:200]}")
        return None

    # Print key lines for visibility
    for line in result.stdout.split('\n'):
        if 'Home Mean:' in line or 'Total Mean:' in line or '1H Mean:' in line:
            print(f"  {line.strip()}")
        if 'Suggested Total Corner Lines' in line:
            print(f"  {line.strip()}")
        if '=>' in line and ('OVER' in line or 'UNDER' in line):
            print(f"  {line.strip()}")

    # Load the exported prediction JSON (latest)
    corners_dir = 'data/corners'
    pattern = f"match_prediction_{league_code}_{home_team.replace(' ', '_')}_vs_{away_team.replace(' ', '_')}_*.json"
    matches = sorted(glob.glob(os.path.join(corners_dir, pattern)), key=os.path.getmtime, reverse=True)

    if matches:
        with open(matches[0], 'r') as f:
            return json.load(f)
    return None


def batch_predict_fixtures_for_league(fixtures: list, league_code: str, seasons: str = None, auto_mode: bool = False):
    predictions = []
    for i, fixture in enumerate(fixtures, 1):
        home = fixture.get('home', '')
        away = fixture.get('away', '')
        time = fixture.get('time', '')
        if not home or not away:
            continue
        print(f"\n[{league_code}] [{i}/{len(fixtures)}] {time} - {home} vs {away}")
        print("-" * 60)
        if not auto_mode:
            resp = input("  Generate prediction? (Y/n/q): ").strip().lower()
            if resp == 'q':
                print("  Stopped by user.")
                break
            if resp == 'n':
                print("  Skipped.")
                continue
        pred = generate_corner_prediction(league_code, home, away, seasons)
        if pred:
            predictions.append(pred)
            print("  ✓ Prediction generated")
        else:
            print("  ✗ Prediction failed")
    return predictions


def export_batch_predictions(predictions: list, label: str, output_dir: str = 'data/corners', reuse_today: bool = False):
    if not predictions:
        return None
    os.makedirs(output_dir, exist_ok=True)

    today = datetime.now().strftime('%Y%m%d')
    safe_label = label.replace(',', '+').replace(' ', '')

    # Check if file for today already exists
    today_pattern = f"batch_predictions_{safe_label}_{today}_*.json"
    existing = find_today_file(os.path.join(output_dir, today_pattern))

    if existing and reuse_today:
        print(f"\n{'='*70}")
        print(f"✓ Reusing existing predictions from: {os.path.basename(existing)}")
        print(f"{'='*70}")
        return existing

    # Create new file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"batch_predictions_{safe_label}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            'label': label,
            'generated_at': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'predictions': predictions
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"✓ Exported {len(predictions)} predictions to: {filepath}")
    print(f"{'='*70}")

    # Clean up older files from today
    remove_duplicate_daily_files(output_dir, f"batch_predictions_{safe_label}_{today}")

    return filepath


def export_combined(predictions_by_league: dict[str, list], leagues: list[str], output_dir: str = 'data/corners', reuse_today: bool = False):
    os.makedirs(output_dir, exist_ok=True)

    today = datetime.now().strftime('%Y%m%d')
    label = '+'.join(leagues)

    # Check if combined file for today already exists
    today_pattern = f"batch_predictions_{label}_{today}_*.json"
    existing = find_today_file(os.path.join(output_dir, today_pattern))

    if existing and reuse_today:
        print(f"\n{'='*70}")
        print(f"✓ Reusing existing combined predictions from: {os.path.basename(existing)}")
        print(f"{'='*70}")
        return existing

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"batch_predictions_{label}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    totals = {lg: len(predictions_by_league.get(lg, [])) for lg in leagues}
    totals['all'] = sum(totals.values())

    blob = {
        'leagues': leagues,
        'generated_at': datetime.now().isoformat(),
        'totals': totals,
        'predictions_by_league': predictions_by_league,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"✓ Exported combined {totals['all']} predictions to: {filepath}")
    print(f"{'='*70}")

    # Clean up older files from today
    remove_duplicate_daily_files(output_dir, f"batch_predictions_{label}_{today}")

    return filepath


def find_latest_file(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        return ''
    return max(files, key=os.path.getmtime)


def load_latest_team_stats(league_code: str) -> dict:
    pattern = f"data/corners/team_stats_{league_code}_*.json"
    path = find_latest_file(pattern)
    if not path:
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def load_half_ratio(league_code: str) -> float:
    csv_path = find_latest_file(f"data/corners/corners_analysis_{league_code}_*.csv")
    if not csv_path:
        return 0.40
    try:
        import csv as _csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = _csv.DictReader(f)
            ratios = []
            for row in reader:
                val = row.get('Est_1H_Corner_Ratio')
                if val is not None and val != '':
                    try:
                        ratios.append(float(val))
                    except Exception:
                        pass
            if ratios:
                return sum(ratios)/len(ratios)
    except Exception:
        pass
    return 0.40


def quick_predict_match(league_code: str, home: str, away: str, half_ratio: float, team_stats: dict) -> dict | None:
    home_stats_raw = team_stats.get('home') if team_stats else {}
    away_stats_raw = team_stats.get('away') if team_stats else {}
    if not home_stats_raw or not away_stats_raw:
        return None
    home_df = pd.DataFrame(home_stats_raw).T
    away_df = pd.DataFrame(away_stats_raw).T
    def resolve(name, df):
        if name in df.index:
            return name
        lname = name.lower()
        for idx in df.index:
            if idx.lower() == lname:
                return idx
        for idx in df.index:
            if lname in idx.lower():
                return idx
        return None
    h_res = resolve(home, home_df)
    a_res = resolve(away, away_df)
    if not h_res or not a_res:
        return None
    h_row = home_df.loc[h_res]
    a_row = away_df.loc[a_res]
    # Prediction logic replicating corners_analysis predict_match
    pred_home = (h_row['Avg_Corners_For'] + a_row['Avg_Corners_Against']) / 2.0
    pred_away = (a_row['Avg_Corners_For'] + h_row['Avg_Corners_Against']) / 2.0
    total = pred_home + pred_away
    std_home = h_row['Std_Corners_For'] if not math.isnan(h_row['Std_Corners_For']) else 0.0
    std_away = a_row['Std_Corners_For'] if not math.isnan(a_row['Std_Corners_For']) else 0.0
    avg_std = (std_home + std_away)/2.0 if (std_home or std_away) else 2.5  # fallback variance
    pred_1h = total * half_ratio
    pred_2h = total - pred_1h
    def normal_cdf(x, mean, std):
        if std <= 0:
            return 1.0 if x >= mean else 0.0
        return 0.5 * (1 + math.erf((x - mean)/(std*math.sqrt(2))))
    market_lines = [7.5,8.5,9.5,10.5,11.5,12.5]
    total_lines = []
    for line in market_lines:
        p_over = 1 - normal_cdf(line + 0.05, total, avg_std)
        p_under = normal_cdf(line + 0.05, total, avg_std)
        rec = 'OVER' if p_over >= 0.6 else ('UNDER' if p_under >= 0.6 else None)
        total_lines.append({'line': line, 'p_over': round(p_over,3), 'p_under': round(p_under,3), 'recommendation': rec})
    half_lines_def = [3.5,4.5,5.5]
    half_std = max(avg_std*0.6, 0.1)
    first_half_lines = []
    for line in half_lines_def:
        p_over = 1 - normal_cdf(line + 0.05, pred_1h, half_std)
        p_under = normal_cdf(line + 0.05, pred_1h, half_std)
        rec = 'OVER' if p_over >= 0.6 else ('UNDER' if p_under >= 0.6 else None)
        first_half_lines.append({'line': line, 'p_over': round(p_over,3), 'p_under': round(p_under,3), 'recommendation': rec})
    return {
        'league': league_code,
        'home_team': h_res,
        'away_team': a_res,
        'input_home': home,
        'input_away': away,
        'fuzzy_used': True,
        'pred_home_corners_mean': round(pred_home,2),
        'pred_away_corners_mean': round(pred_away,2),
        'pred_total_corners_mean': round(total,2),
        'pred_total_corners_range': [round(max(total-avg_std,0),2), round(total+avg_std,2)],
        'pred_1h_corners_mean': round(pred_1h,2),
        'pred_2h_corners_mean': round(pred_2h,2),
        'half_ratio_used': round(half_ratio,4),
        'total_corner_lines': total_lines,
        'first_half_lines': first_half_lines
    }


def build_team_tables_from_csv(league_code: str):
    csv_path = find_latest_file(f"data/corners/corners_analysis_{league_code}_*.csv")
    if not csv_path:
        return None, None
    try:
        df = pd.read_csv(csv_path)
        if 'HomeTeam' not in df.columns or 'AwayTeam' not in df.columns or 'HC' not in df.columns or 'AC' not in df.columns:
            return None, None
        home = df.groupby('HomeTeam').agg({'HC':['mean','std','count'],'AC':'mean'}).round(2)
        away = df.groupby('AwayTeam').agg({'AC':['mean','std','count'],'HC':'mean'}).round(2)
        home.columns = ['Avg_Corners_For','Std_Corners_For','Matches','Avg_Corners_Against']
        away.columns = ['Avg_Corners_For','Std_Corners_For','Matches','Avg_Corners_Against']
        return home, away
    except Exception:
        return None, None


def quick_predict_match_df(league_code: str, home: str, away: str, half_ratio: float, home_df: pd.DataFrame, away_df: pd.DataFrame):
    def resolve(name, df):
        if name in df.index:
            return name
        lname = name.lower()
        for idx in df.index:
            if idx.lower() == lname:
                return idx
        for idx in df.index:
            if lname in idx.lower():
                return idx
        return None
    h_res = resolve(home, home_df)
    a_res = resolve(away, away_df)
    if not h_res or not a_res:
        return None
    h_row = home_df.loc[h_res]
    a_row = away_df.loc[a_res]
    pred_home = (h_row['Avg_Corners_For'] + a_row['Avg_Corners_Against']) / 2.0
    pred_away = (a_row['Avg_Corners_For'] + h_row['Avg_Corners_Against']) / 2.0
    total = pred_home + pred_away
    std_home = h_row['Std_Corners_For'] if not math.isnan(h_row['Std_Corners_For']) else 0.0
    std_away = a_row['Std_Corners_For'] if not math.isnan(a_row['Std_Corners_For']) else 0.0
    avg_std = (std_home + std_away)/2.0 if (std_home or std_away) else 2.5
    pred_1h = total * half_ratio
    pred_2h = total - pred_1h
    def normal_cdf(x, mean, std):
        if std <= 0:
            return 1.0 if x >= mean else 0.0
        return 0.5 * (1 + math.erf((x - mean)/(std*math.sqrt(2))))
    market_lines = [7.5,8.5,9.5,10.5,11.5,12.5]
    total_lines=[]
    for line in market_lines:
        p_over = 1 - normal_cdf(line + 0.05, total, avg_std)
        p_under = normal_cdf(line + 0.05, total, avg_std)
        rec = 'OVER' if p_over>=0.6 else ('UNDER' if p_under>=0.6 else None)
        total_lines.append({'line':line,'p_over':round(p_over,3),'p_under':round(p_under,3),'recommendation':rec})
    half_lines_def=[3.5,4.5,5.5]
    half_std=max(avg_std*0.6,0.1)
    first_half_lines=[]
    for line in half_lines_def:
        p_over = 1 - normal_cdf(line + 0.05, pred_1h, half_std)
        p_under = normal_cdf(line + 0.05, pred_1h, half_std)
        rec = 'OVER' if p_over>=0.6 else ('UNDER' if p_under>=0.6 else None)
        first_half_lines.append({'line':line,'p_over':round(p_over,3),'p_under':round(p_under,3),'recommendation':rec})
    return {
        'league': league_code,
        'home_team': h_res,
        'away_team': a_res,
        'input_home': home,
        'input_away': away,
        'fuzzy_used': True,
        'pred_home_corners_mean': round(pred_home,2),
        'pred_away_corners_mean': round(pred_away,2),
        'pred_total_corners_mean': round(total,2),
        'pred_total_corners_range': [round(max(total-avg_std,0),2), round(total+avg_std,2)],
        'pred_1h_corners_mean': round(pred_1h,2),
        'pred_2h_corners_mean': round(pred_2h,2),
        'half_ratio_used': round(half_ratio,4),
        'total_corner_lines': total_lines,
        'first_half_lines': first_half_lines
    }


def export_prediction(pred: dict) -> str | None:
    if not pred:
        return None
    os.makedirs('data/corners', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"match_prediction_{pred['league']}_{pred['home_team'].replace(' ','_')}_vs_{pred['away_team'].replace(' ','_')}_{ts}.json"
    path = os.path.join('data/corners', fname)
    with open(path,'w') as f:
        json.dump(pred,f,indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(
        description='Automate corner predictions workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --league E2
  python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --league E2 --auto
  python3 automate_corner_predictions.py --input tmp/corners/251115_match_games.log --leagues E2,E3 --auto
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Input match log file')
    parser.add_argument('--league', '-l', help='League code (e.g., E2) or comma-separated list (backwards-compatible)')
    parser.add_argument('--leagues', help='Comma-separated league codes (e.g., E2,E3)')
    parser.add_argument('--seasons', '-s', help='Season filter (e.g., 2425 or 2324,2425)')
    parser.add_argument('--auto', '-a', action='store_true', help='Auto mode (no prompts for each match)')
    parser.add_argument('--force', '-f', action='store_true', help='Force re-run corners analysis even if done today')
    parser.add_argument('--reuse-today', action='store_true', help='Reuse existing analysis data from today if available')
    parser.add_argument('--output-dir', '-o', default='data', help='Output directory for parsed fixtures')
    parser.add_argument('--mode', choices=['fast','full'], default='fast', help='Prediction mode: fast (reuse stats) or full (run per-match corners_analysis)')
    parser.add_argument('--train-model', action='store_true', help='Train regression models during corners analysis step')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Resolve leagues list
    leagues_raw = args.leagues or args.league or ''
    leagues = [c.strip().upper() for c in leagues_raw.split(',') if c.strip()]
    if not leagues:
        print("Error: Provide --league <CODE> or --leagues L1,L2")
        return 1

    print(f"\n{'='*70}")
    print("AUTOMATED CORNER PREDICTIONS WORKFLOW")
    print(f"{'='*70}")
    print(f"Input log:  {args.input}")
    print(f"Leagues:    {', '.join(leagues)}")
    print(f"Seasons:    {args.seasons or 'all'}")
    print(f"Auto mode:  {args.auto}")

    # Step 1: Parse match log with multi-league resolution
    csv_path, json_path = run_parse_match_log(args.input, leagues, args.output_dir)
    if not json_path:
        print("Failed to parse match log.")
        return 1

    fixtures = load_parsed_fixtures(json_path)
    if not fixtures:
        print(f"No scheduled fixtures found in {json_path}")
        return 1

    grouped = group_fixtures_by_league(fixtures)
    print(f"\nFound {sum(len(v) for v in grouped.values())} scheduled fixture(s) across {len(grouped)} league(s)")

    predictions_by_league: dict[str, list] = {}

    # Step 2/3: For each league -> run analysis then predict
    for lg in leagues:
        league_fixtures = grouped.get(lg, [])
        if not league_fixtures:
            print(f"\n[WARN] No fixtures for league {lg} in parsed file; skipping analysis.")
            predictions_by_league[lg] = []
            continue
        success = run_corners_analysis(lg, args.seasons, args.force, train_model=args.train_model)
        if not success:
            print(f"Failed to run corners analysis for {lg}, skipping predictions.")
            predictions_by_league[lg] = []
            continue
        if args.mode == 'fast':
            # Build team tables from latest CSV (more reliable than JSON structure)
            home_df, away_df = build_team_tables_from_csv(lg)
            if home_df is None:
                print(f"[FAST] Fallback: could not build team stats for {lg}")
                preds = []
            else:
                half_ratio = load_half_ratio(lg)
                preds = []
                start_league = time.time()
                for i, fx in enumerate(league_fixtures,1):
                    home, away, tm = fx.get('home',''), fx.get('away',''), fx.get('time','')
                    print(f"\n[FAST] [{lg}] [{i}/{len(league_fixtures)}] {tm} - {home} vs {away}")
                    pred = quick_predict_match_df(lg, home, away, half_ratio, home_df, away_df)
                    if pred:
                        export_prediction(pred)
                        preds.append(pred)
                        print(f"  ✓ Total Mean {pred['pred_total_corners_mean']} | 1H {pred['pred_1h_corners_mean']} | Range {pred['pred_total_corners_range'][0]}-{pred['pred_total_corners_range'][1]}")
                    else:
                        print("  ✗ Failed to resolve teams")
                elapsed = time.time() - start_league
                print(f"\n[FAST] Completed {len(preds)} predictions for {lg} in {elapsed:.2f}s")
        else:
            preds = batch_predict_fixtures_for_league(league_fixtures, lg, args.seasons, args.auto)
        predictions_by_league[lg] = preds
        if preds:
            export_batch_predictions(preds, label=lg, reuse_today=args.reuse_today)

    # Export combined batch
    all_count = sum(len(v) for v in predictions_by_league.values())
    if all_count:
        export_combined(predictions_by_league, leagues, reuse_today=args.reuse_today)

    # Cleanup old files (keep last 7 days)
    cleanup_old_daily_files('data/corners', keep_days=7)

    print(f"\n{'='*70}")
    print("WORKFLOW COMPLETE")
    print(f"Totals: " + ", ".join([f"{lg}={len(predictions_by_league.get(lg, []))}" for lg in leagues]) + f", all={all_count} | Mode={args.mode}")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
