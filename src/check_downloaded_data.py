#!/usr/bin/env python3
# check_downloaded_data.py

from typing import Dict
import os
import logging
from datetime import datetime, timezone
import sys
import pandas as pd

# Default data directory (falls back to repo-local `football-data`)
EURO_FOOTBALL_DIR = os.path.join(os.path.dirname(__file__), "football-data")


def check_downloaded_data(league_code: str) -> Dict[str, object]:
    """
    Check whether league CSV and fixtures CSV are present under `EURO_FOOTBALL_DIR`.
    Returns a dict with existence, filesize and mtime (ISO) for each checked file.
    """
    league_csv = os.path.join(EURO_FOOTBALL_DIR, "{}.csv".format(league_code))
    fixtures_csv = os.path.join(EURO_FOOTBALL_DIR, "{}_fixtures.csv".format(league_code))

    def info(path):
        if os.path.exists(path):
            stat = os.stat(path)
            return {
                "exists": True,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "path": path,
            }
        return {"exists": False, "size": 0, "modified": None, "path": path}

    return {"league_csv": info(league_csv), "fixtures_csv": info(fixtures_csv)}


def load_league_table(league_code: str) -> pd.DataFrame:
    """
    Load a league CSV and return a DataFrame with at least a 'Team' column.
    This helper is defensive: if the league file contains HomeTeam/AwayTeam columns
    it will extract unique team names.
    """
    path = os.path.join(EURO_FOOTBALL_DIR, "{}.csv".format(league_code))
    if not os.path.exists(path):
        raise FileNotFoundError("League file not found: {}".format(path))
    df = pd.read_csv(path)
    if "Team" in df.columns:
        return df
    if "HomeTeam" in df.columns and "AwayTeam" in df.columns:
        teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
        return pd.DataFrame({"Team": teams})
    # Fallback: take first column as team names
    first_col = df.columns[0]
    return pd.DataFrame({"Team": df[first_col].astype(str)})


def get_next_fixtures(league_code: str, lookahead_days: int = 14, max_matches: int = 10) -> pd.DataFrame:
    """
    Load upcoming fixtures for `league_code`. If fixtures CSV exists, parse dates and
    return matches within `lookahead_days`. Otherwise simulate a next round from the league table.
    Returns a DataFrame with columns Date, HomeTeam, AwayTeam (Date is pandas.Timestamp).
    """
    fixtures_path = os.path.join(EURO_FOOTBALL_DIR, "{}_fixtures.csv".format(league_code))
    now_ts = pd.Timestamp.now(tz="UTC")
    if os.path.exists(fixtures_path):
        try:
            df = pd.read_csv(fixtures_path)
            # normalize column names to expected ones
            cols = {c.lower(): c for c in df.columns}
            # try common variants
            date_col = cols.get("date") or cols.get("kickoff") or None
            home_col = cols.get("hometeam") or cols.get("home") or None
            away_col = cols.get("awayteam") or cols.get("away") or None
            if date_col is None or home_col is None or away_col is None:
                logging.warning("Fixtures file missing required columns: {}".format(fixtures_path))
                return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            # Filter upcoming between now and lookahead window
            end_ts = now_ts + pd.Timedelta(days=lookahead_days)
            upcoming = df[(df[date_col] >= now_ts) & (df[date_col] <= end_ts)].sort_values(date_col)
            # Standardize return columns
            upcoming = upcoming.rename(columns={date_col: "Date", home_col: "HomeTeam", away_col: "AwayTeam"})
            return upcoming[["Date", "HomeTeam", "AwayTeam"]].head(max_matches).reset_index(drop=True)
        except Exception as e:
            logging.error("Failed to load fixtures file {}: {}".format(fixtures_path, e))
            return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])
    # Fallback: simulate pairings from league table
    try:
        league_df = load_league_table(league_code)
        teams = list(league_df["Team"].astype(str))
    except Exception:
        logging.warning("Could not load league table for simulation: {}".format(league_code))
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])
    if len(teams) % 2 != 0:
        teams.append("BYE")
    fixtures = []
    for i in range(0, len(teams), 2):
        if teams[i] != "BYE" and teams[i + 1] != "BYE":
            fixtures.append({"Date": now_ts, "HomeTeam": teams[i], "AwayTeam": teams[i + 1]})
            if len(fixtures) >= max_matches:
                break
    return pd.DataFrame(fixtures)


def main():
    """
    Entry point for validation or CLI usage. Allows optional league code from argv.
    """
    import sys
    league = sys.argv[1] if len(sys.argv) > 1 else "E0"
    status = check_downloaded_data(league)
    print("League CSV present: {}, Fixtures CSV present: {}".format(status['league_csv']['exists'], status['fixtures_csv']['exists']))
    next_matches = get_next_fixtures(league, lookahead_days=14, max_matches=10)
    if next_matches.empty:
        print("No upcoming fixtures found for {} (check '{}').format(league, EURO_FOOTBALL_DIR)")
    else:
        print("Upcoming matches for {}:".format(league))
        for i, r in next_matches.iterrows():
            date_str = r["Date"].strftime("%Y-%m-%d") if not pd.isna(r["Date"]) else "TBA"
            print(" Match {}: {} v {} on {}".format(i+1, r['HomeTeam'], r['AwayTeam'], date_str))

if __name__ == "__main__":
    main()
