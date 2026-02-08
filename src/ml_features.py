"""Feature engineering for goal-based ML models (Total Goals, 1X2, BTTS).

We build per-match training rows from historical match dataframe with columns:
Date, HomeTeam, AwayTeam, FTHG, FTAG and optionally shots, cards, fouls columns if present.
Gracefully handle missing advanced stats by imputing zeros.
"""
from __future__ import annotations
import logging
from typing import List, Dict
import pandas as pd
import numpy as np


ADV_STATS_MAP = {
    # common variants -> canonical
    'HS': ['HS','HomeShots','H_SHOTS'],
    'AS': ['AS','AwayShots','A_SHOTS'],
    'HST': ['HST','HomeShotsTarget','H_SHOTS_TARGET'],
    'AST': ['AST','AwayShotsTarget','A_SHOTS_TARGET'],
    'HF': ['HF','HomeFouls'],
    'AF': ['AF','AwayFouls'],
    'HC': ['HC','HomeCorners'],
    'AC': ['AC','AwayCorners'],
}

CANONICAL_COLS = ['Date','HomeTeam','AwayTeam','FTHG','FTAG']


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for tgt, variants in ADV_STATS_MAP.items():
        for v in variants:
            for c in df.columns:
                if c.lower() == v.lower():
                    col_map[c] = tgt
    # Ensure essential columns exist
    for col in CANONICAL_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required column {col} in history data")
    return df.rename(columns=col_map)


def engineer_features(history_df: pd.DataFrame, rolling_window: int = 6) -> pd.DataFrame:
    df = normalize_columns(history_df.copy())
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')

    # Create targets
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['HomeWin'] = (df['FTHG'] > df['FTAG']).astype(int)
    df['Draw'] = (df['FTHG'] == df['FTAG']).astype(int)
    df['AwayWin'] = (df['FTHG'] < df['FTAG']).astype(int)
    df['BTTS'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)

    # Ensure advanced stats present; if not create zeros
    for col in ['HS','AS','HST','AST','HF','AF','HC','AC']:
        if col not in df.columns:
            df[col] = 0

    # Interaction / ratio features
    df['HS_x_HST'] = df['HS'] * df['HST']
    df['AS_x_AST'] = df['AS'] * df['AST']
    df['Shots_Ratio'] = df['HS'] / (df['AS'] + 1)
    df['Corners_Ratio'] = (df['HC'] + 1) / (df['AC'] + 1)

    # Rolling team stats (attack / defence recent form)
    features_rows = []
    grouped = df.groupby('HomeTeam')
    # We'll compute rolling for each team and merge back; simpler approach iterate rows
    # Precompute per-team match list indices for speed
    for idx, row in df.iterrows():
        home = row['HomeTeam']; away = row['AwayTeam']
        date = row['Date']

        # Past matches for home
        past_home = df[((df['HomeTeam']==home) | (df['AwayTeam']==home)) & (df['Date'] < date)]
        past_away = df[((df['HomeTeam']==away) | (df['AwayTeam']==away)) & (df['Date'] < date)]

        def rolling_stats(past: pd.DataFrame):
            if past.empty:
                return {
                    'roll_goals_for': 0.0,
                    'roll_goals_against': 0.0,
                    'roll_shots_for': 0.0,
                    'roll_shots_against': 0.0,
                }
            recent = past.tail(rolling_window)
            gf = []; ga = []; sh_for = []; sh_against = []
            for _, r in recent.iterrows():
                if r['HomeTeam'] == home:
                    gf.append(r['FTHG']); ga.append(r['FTAG']); sh_for.append(r['HS']); sh_against.append(r['AS'])
                elif r['AwayTeam'] == home:
                    gf.append(r['FTAG']); ga.append(r['FTHG']); sh_for.append(r['AS']); sh_against.append(r['HS'])
            return {
                'roll_goals_for': np.mean(gf) if gf else 0.0,
                'roll_goals_against': np.mean(ga) if ga else 0.0,
                'roll_shots_for': np.mean(sh_for) if sh_for else 0.0,
                'roll_shots_against': np.mean(sh_against) if sh_against else 0.0,
            }

        home_roll = rolling_stats(past_home)
        away_roll = rolling_stats(past_away)

        features_rows.append({
            'Date': date,
            'HomeTeam': home,
            'AwayTeam': away,
            'FTHG': row['FTHG'],
            'FTAG': row['FTAG'],
            'TotalGoals': row['TotalGoals'],
            'HomeWin': row['HomeWin'],
            'Draw': row['Draw'],
            'AwayWin': row['AwayWin'],
            'BTTS': row['BTTS'],
            'HS': row['HS'], 'AS': row['AS'], 'HST': row['HST'], 'AST': row['AST'], 'HF': row['HF'], 'AF': row['AF'], 'HC': row['HC'], 'AC': row['AC'],
            'HS_x_HST': row['HS_x_HST'], 'AS_x_AST': row['AS_x_AST'],
            'Shots_Ratio': row['Shots_Ratio'], 'Corners_Ratio': row['Corners_Ratio'],
            'Home_roll_GF': home_roll['roll_goals_for'],
            'Home_roll_GA': home_roll['roll_goals_against'],
            'Away_roll_GF': away_roll['roll_goals_for'],
            'Away_roll_GA': away_roll['roll_goals_against'],
            'Home_roll_ShotsF': home_roll['roll_shots_for'],
            'Home_roll_ShotsA': home_roll['roll_shots_against'],
            'Away_roll_ShotsF': away_roll['roll_shots_for'],
            'Away_roll_ShotsA': away_roll['roll_shots_against'],
        })

    feat_df = pd.DataFrame(features_rows)
    logging.info(f"Engineered features rows: {len(feat_df)}")
    return feat_df


TRAIN_FEATURE_COLUMNS = [
    'HS','AS','HST','AST','HF','AF','HC','AC',
    'HS_x_HST','AS_x_AST','Shots_Ratio','Corners_Ratio',
    'Home_roll_GF','Home_roll_GA','Away_roll_GF','Away_roll_GA',
    'Home_roll_ShotsF','Home_roll_ShotsA','Away_roll_ShotsF','Away_roll_ShotsA'
]


def build_match_feature_row(latest_df: pd.DataFrame, home: str, away: str) -> Dict[str, float]:
    """Build feature vector for a match by extracting team-specific stats.

    This function properly handles the fact that teams can appear as either home or away
    in the historical data, and correctly extracts their stats regardless of position.
    """
    # Compute league averages for fallback
    league_avg = latest_df[TRAIN_FEATURE_COLUMNS].mean().to_dict()

    def get_team_stats(team: str, is_home_in_prediction: bool) -> Dict[str, float]:
        """Extract stats for a team from their last match, adjusting for home/away position."""
        rows = latest_df[(latest_df['HomeTeam']==team) | (latest_df['AwayTeam']==team)]
        if rows.empty:
            logging.debug(f"No matches found for team: {team}")
            return None

        last_match = rows.iloc[-1]
        was_home = last_match['HomeTeam'] == team

        logging.debug(f"Team {team}: found {len(rows)} matches, last match was {'HOME' if was_home else 'AWAY'}")
        logging.debug(f"  Last match columns available: {list(last_match.index)[:10]}...")
        logging.debug(f"  Sample values: HS={last_match.get('HS', 'N/A')}, Home_roll_GF={last_match.get('Home_roll_GF', 'N/A')}")

        stats = {}
        # Extract rolling stats - these are already team-specific in the feature df
        if was_home:
            # Team was home in their last match
            stats['roll_GF'] = last_match.get('Home_roll_GF', 0.0)
            stats['roll_GA'] = last_match.get('Home_roll_GA', 0.0)
            stats['roll_ShotsF'] = last_match.get('Home_roll_ShotsF', 0.0)
            stats['roll_ShotsA'] = last_match.get('Home_roll_ShotsA', 0.0)
            # Match-level stats from when they were home
            stats['Shots'] = last_match.get('HS', 0.0)
            stats['ShotsTarget'] = last_match.get('HST', 0.0)
            stats['Fouls'] = last_match.get('HF', 0.0)
            stats['Corners'] = last_match.get('HC', 0.0)
            stats['Shots_x_ShotsTarget'] = last_match.get('HS_x_HST', 0.0)
        else:
            # Team was away in their last match
            stats['roll_GF'] = last_match.get('Away_roll_GF', 0.0)
            stats['roll_GA'] = last_match.get('Away_roll_GA', 0.0)
            stats['roll_ShotsF'] = last_match.get('Away_roll_ShotsF', 0.0)
            stats['roll_ShotsA'] = last_match.get('Away_roll_ShotsA', 0.0)
            # Match-level stats from when they were away
            stats['Shots'] = last_match.get('AS', 0.0)
            stats['ShotsTarget'] = last_match.get('AST', 0.0)
            stats['Fouls'] = last_match.get('AF', 0.0)
            stats['Corners'] = last_match.get('AC', 0.0)
            stats['Shots_x_ShotsTarget'] = last_match.get('AS_x_AST', 0.0)

        return stats

    # Get stats for both teams
    home_stats = get_team_stats(home, True)
    away_stats = get_team_stats(away, False)

    # Check if we have data for both teams
    if home_stats is None or away_stats is None:
        missing_team = home if home_stats is None else away
        logging.warning(f"ML feature fallback to league average for match {home} vs {away} (missing data for {missing_team})")
        return league_avg

    # Build feature vector in the correct order expected by the model
    data = {}

    # Home team shots
    data['HS'] = home_stats['Shots']
    data['HST'] = home_stats['ShotsTarget']
    data['HF'] = home_stats['Fouls']
    data['HC'] = home_stats['Corners']
    data['HS_x_HST'] = home_stats['Shots_x_ShotsTarget']

    # Away team shots
    data['AS'] = away_stats['Shots']
    data['AST'] = away_stats['ShotsTarget']
    data['AF'] = away_stats['Fouls']
    data['AC'] = away_stats['Corners']
    data['AS_x_AST'] = away_stats['Shots_x_ShotsTarget']

    # Ratio features (computed from current match-up)
    data['Shots_Ratio'] = data['HS'] / (data['AS'] + 1)
    data['Corners_Ratio'] = (data['HC'] + 1) / (data['AC'] + 1)

    # Rolling form features
    data['Home_roll_GF'] = home_stats['roll_GF']
    data['Home_roll_GA'] = home_stats['roll_GA']
    data['Home_roll_ShotsF'] = home_stats['roll_ShotsF']
    data['Home_roll_ShotsA'] = home_stats['roll_ShotsA']

    data['Away_roll_GF'] = away_stats['roll_GF']
    data['Away_roll_GA'] = away_stats['roll_GA']
    data['Away_roll_ShotsF'] = away_stats['roll_ShotsF']
    data['Away_roll_ShotsA'] = away_stats['roll_ShotsA']

    return data
