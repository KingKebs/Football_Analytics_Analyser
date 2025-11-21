import os
import glob
from datetime import datetime, timedelta

def remove_old_files():
    """
    Clean up the data directory by removing old and temporary files.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    now = datetime.now()
    yesterday = now - timedelta(days=1)

    # --- Suggestions Files ---
    # Keep the latest suggestion file for each league, remove the rest.
    suggestion_files = glob.glob(os.path.join(data_dir, 'full_league_suggestions_*.json'))
    latest_suggestions = {}
    for f in suggestion_files:
        try:
            league = os.path.basename(f).split('_')[3]
            if league not in latest_suggestions or os.path.getmtime(f) > os.path.getmtime(latest_suggestions[league]):
                latest_suggestions[league] = f
        except IndexError:
            continue

    for f in suggestion_files:
        if f not in latest_suggestions.values():
            print(f"Removing old suggestion file: {f}")
            os.remove(f)

    # --- Fixture Files ---
    # Keep today's and yesterday's fixture files, remove the rest.
    fixture_files = glob.glob(os.path.join(data_dir, 'todays_fixtures_*'))
    today_str = now.strftime('%Y%m%d')
    yesterday_str = yesterday.strftime('%Y%m%d')
    for f in fixture_files:
        if today_str not in f and yesterday_str not in f:
            print(f"Removing old fixture file: {f}")
            os.remove(f)

    # --- pickMatch Files ---
    # Remove all pickMatch files
    pickmatch_files = glob.glob(os.path.join(data_dir, 'pickMatch_*.csv'))
    for f in pickmatch_files:
        print(f"Removing pickMatch file: {f}")
        os.remove(f)

    # --- Analysis Results Files ---
    # Keep the latest analysis results file, remove the rest.
    analysis_files = glob.glob(os.path.join(data_dir, 'analysis_results_*.json'))
    if analysis_files:
        latest_analysis_file = max(analysis_files, key=os.path.getmtime)
        for f in analysis_files:
            if f != latest_analysis_file:
                print(f"Removing old analysis results file: {f}")
                os.remove(f)

if __name__ == '__main__':
    remove_old_files()

