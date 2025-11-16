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
    # Use last known values for teams; fallback zeros
    def last_team_row(team: str):
        rows = latest_df[(latest_df['HomeTeam']==team) | (latest_df['AwayTeam']==team)]
        if rows.empty:
            return None
        return rows.iloc[-1]
    h_last = last_team_row(home)
    a_last = last_team_row(away)
    data = {}
    for col in TRAIN_FEATURE_COLUMNS:
        if h_last is None or a_last is None:
            data[col] = 0.0
        else:
            # Features with prefix Home_ use home last, Away_ use away last
            if col.startswith('Home_'):
                base_col = col.replace('Home_', '')
                data[col] = float(h_last.get(col, h_last.get(base_col, 0.0)))
            elif col.startswith('Away_'):
                base_col = col.replace('Away_', '')
                data[col] = float(a_last.get(col, a_last.get(base_col, 0.0)))
            else:
                # neutral stats: use difference or ratio? keep simple average of last values
                hv = float(h_last.get(col, 0.0))
                av = float(a_last.get(col, 0.0))
                data[col] = (hv + av) / 2.0
    return data

