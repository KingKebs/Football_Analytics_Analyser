"""
Football Analytics Analyser - Full League CLI

End-to-end workflow for a full league round:
- Load league table and upcoming fixtures
- Compute team strengths (+ optional recent form blending)
- Build suggestions for all fixtures in the round
- Generate and select favorable parlays across all matches
- Save results to data/
Optional ML Mode (Steps 1-4):
- Feature engineering from historical matches
- Recency weighting
- Train RF / XGB models for Total Goals (regression), 1X2, BTTS
- 5-fold cross-validation metrics
- Compare ML predictions vs Poisson baseline
"""

import os
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import re

import numpy as np
import pandas as pd
import logging
import glob
import difflib
from data_file_utils import ensure_dirs_for_writing
import time
from multiprocessing.dummy import Pool as ThreadPool


# ML imports (lazy inside functions to avoid mandatory dependency if ML not used)
try:
    from ml_features import engineer_features, build_match_feature_row, TRAIN_FEATURE_COLUMNS
    from ml_utils import build_recency_weights, save_models, check_min_samples
    from ml_training import train_models, predict_match
    from ml_evaluation import evaluate_vs_poisson
except Exception:
    # Defer import errors until ML mode actually used
    engineer_features = build_match_feature_row = TRAIN_FEATURE_COLUMNS = None
    build_recency_weights = save_models = check_min_samples = None
    train_models = predict_match = evaluate_vs_poisson = None


# Range filter parser
_DEF_RANGE = None

def _parse_range_filter(s: str):
    if not s:
        return None
    try:
        parts = [p.strip() for p in s.split(',') if p.strip()]
        if len(parts) != 2:
            return None
        lo, hi = float(parts[0]), float(parts[1])
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)
    except Exception:
        return None

# Algorithms
from algorithms import (
    compute_basic_strengths,
    estimate_xg,
    score_probability_matrix,
    extract_markets_from_score_matrix,
    estimate_corners_and_cards,
    generate_parlays,
    prob_to_decimal_odds,
    format_bet_slip,
    match_rating,
    fit_rating_to_prob_models,
    rating_probabilities_from_rating,
    merge_form_into_strengths,
)

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')

# Paths
DATA_DIR = 'data'
OLD_CSV_SUBFOLDER = 'old csv'
EURO_FOOTBALL_DIR = os.path.join('football-data', 'all-euro-football')

# Cache window
CACHE_DURATION_HOURS = 6


class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    MAGENTA = '\033[35m'
    BLUE = '\033[34m'


# League mapping (subset)
LEAGUE_MAPPINGS: Dict[str, Dict[str, str]] = {
    'E0': {'name': 'English Premier League (EPL)', 'country': 'England', 'tier': 1},
    'E1': {'name': 'English Championship', 'country': 'England', 'tier': 2},
    'E2': {'name': 'English League One', 'country': 'England', 'tier': 3},
    'E3': {'name': 'English League Two', 'country': 'England', 'tier': 4},
    'D1': {'name': 'Bundesliga (Germany)', 'country': 'Germany', 'tier': 1},
    'D2': {'name': '2. Bundesliga (Germany)', 'country': 'Germany', 'tier': 2},
    'SP1': {'name': 'La Liga (Spain)', 'country': 'Spain', 'tier': 1},
    'SP2': {'name': 'Segunda División (Spain)', 'country': 'Spain', 'tier': 2},
    'I1': {'name': 'Serie A (Italy)', 'country': 'Italy', 'tier': 1},
    'I2': {'name': 'Serie B (Italy)', 'country': 'Italy', 'tier': 2},
    'F1': {'name': 'Ligue 1 (France)', 'country': 'France', 'tier': 1},
    'F2': {'name': 'Ligue 2 (France)', 'country': 'France', 'tier': 2},
    'P1': {'name': 'Primeira Liga (Portugal)', 'country': 'Portugal', 'tier': 1},
    'N1': {'name': 'Eredivisie (Netherlands)', 'country': 'Netherlands', 'tier': 1},
    'SC0': {'name': 'Scottish Premiership', 'country': 'Scotland', 'tier': 1},
    'SC1': {'name': 'Scottish Championship', 'country': 'Scotland', 'tier': 2},
    'SC2': {'name': 'Scottish League One', 'country': 'Scotland', 'tier': 3},
    'SC3': {'name': 'Scottish League Two', 'country': 'Scotland', 'tier': 4},
    'B1': {'name': 'Belgian Pro League', 'country': 'Belgium', 'tier': 1},
    'G1': {'name': 'Super League Greece', 'country': 'Greece', 'tier': 1},
    'T1': {'name': 'Süper Lig (Turkey)', 'country': 'Turkey', 'tier': 1},
}

# Competition name -> league code mapping (for parsed fixtures inference)
COMPETITION_TO_LEAGUE = {
    'Premier League': 'E0', 'Championship': 'E1', 'League One': 'E2', 'League Two': 'E3',
    'Bundesliga': 'D1', '2. Bundesliga': 'D2',
    'La Liga': 'SP1', 'Segunda División': 'SP2',
    'Serie A': 'I1', 'Serie B': 'I2',
    'Ligue 1': 'F1', 'Ligue 2': 'F2',
    'Primeira Liga': 'P1', 'Eredivisie': 'N1',
    'Scottish Premiership': 'SC0', 'Scottish Championship': 'SC1', 'Scottish League One': 'SC2', 'Scottish League Two': 'SC3',
    'Belgian Pro League': 'B1', 'Super League Greece': 'G1', 'Süper Lig': 'T1'
}

# Team alias expansions reused for inference (lowercased)
TEAM_ALIASES_INFERENCE = {
    'man city': 'man city', 'manchester city': 'man city',
    'man united': 'man united', 'man utd': 'man united', 'manchester united': 'man united',
    'nottm forest': "nott'm forest", 'nottingham forest': "nott'm forest", 'nottingham': "nott'm forest",
    'athletic bilbao': 'ath bilbao', 'ath bilbao': 'ath bilbao',
    'psg': 'psg',
    'inter': 'inter', 'internazionale': 'inter',
    'juventus': 'juventus', 'napoli': 'napoli'
}


def get_league_info(league_code: str) -> Dict[str, str]:
    if league_code not in LEAGUE_MAPPINGS:
        raise ValueError(f"Unknown league code: {league_code}")
    return LEAGUE_MAPPINGS[league_code]


def get_league_csv(league_code: str) -> str:
    path = os.path.join(EURO_FOOTBALL_DIR, f"{league_code}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"League CSV not found: {path}")
    return path


def aggregate_football_data(csv_path: str) -> pd.DataFrame:
    """Aggregate match-level CSV into a league table (Team, P, F, A)."""
    df_raw = pd.read_csv(csv_path, low_memory=False)
    if set(['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']).issubset(df_raw.columns):
        df_raw['FTHG'] = pd.to_numeric(df_raw['FTHG'], errors='coerce').fillna(0).astype(int)
        df_raw['FTAG'] = pd.to_numeric(df_raw['FTAG'], errors='coerce').fillna(0).astype(int)
        teams = sorted(pd.unique(df_raw[['HomeTeam', 'AwayTeam']].values.ravel('K')))
        rows = []
        for t in teams:
            home_mask = df_raw['HomeTeam'] == t
            away_mask = df_raw['AwayTeam'] == t
            p = int(home_mask.sum() + away_mask.sum())
            f = int(df_raw.loc[home_mask, 'FTHG'].sum() + df_raw.loc[away_mask, 'FTAG'].sum())
            a = int(df_raw.loc[home_mask, 'FTAG'].sum() + df_raw.loc[away_mask, 'FTHG'].sum())
            rows.append({'Team': t, 'P': p, 'F': f, 'A': a})
        return pd.DataFrame(rows)
    return df_raw


def load_league_table(league_code: str) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, f"league_data_{league_code}.csv")
    if os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path), tz=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
        if age_hours < CACHE_DURATION_HOURS:
            logging.info(f"Loading cached league table: {cache_path} ({age_hours:.1f}h old)")
            return pd.read_csv(cache_path)
    src = get_league_csv(league_code)
    info = get_league_info(league_code)
    logging.info(f"Loading {info['name']} from {src}")
    agg = aggregate_football_data(src)
    agg.to_csv(cache_path, index=False)
    logging.info(f"Saved aggregated league table to {cache_path}")
    return agg


def load_historical_matches(data_dir: str = DATA_DIR, subfolder: str = OLD_CSV_SUBFOLDER) -> pd.DataFrame:
    folder = os.path.join(data_dir, subfolder)
    files: List[str] = []
    if os.path.isdir(folder):
        files.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.csv')])
    # Also include archived historical season CSVs after reorg (data/archive)
    archive_dir = os.path.join(data_dir, 'archive')
    if os.path.isdir(archive_dir):
        for f in os.listdir(archive_dir):
            lf = f.lower()
            # heuristic: season historical files contain _data_ or end with dataset.csv; skip league_data_ cached tables
            if lf.endswith('.csv') and ('_data_' in lf or 'dataset' in lf) and not lf.startswith('league_data_'):
                files.append(os.path.join(archive_dir, f))
    if not files:
        return pd.DataFrame()
    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False)
            rename_map = {}
            for c in df.columns:
                lc = str(c).lower()
                if lc in ('hometeam', 'home_team'):
                    rename_map[c] = 'HomeTeam'
                if lc in ('awayteam', 'away_team'):
                    rename_map[c] = 'AwayTeam'
                if lc in ('fthg', 'homegoals'):
                    rename_map[c] = 'FTHG'
                if lc in ('ftag', 'awaygoals'):
                    rename_map[c] = 'FTAG'
                if lc == 'date' and c != 'Date':
                    rename_map[c] = 'Date'
            df = df.rename(columns=rename_map)
            if set(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']).issubset(df.columns):
                dfs.append(df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']])
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    all_df['Date'] = pd.to_datetime(all_df['Date'], errors='coerce', dayfirst=True)
    all_df = all_df.sort_values('Date', na_position='last')
    logging.info(f"Loaded {len(files)} historical files; combined rows: {len(all_df)}")
    return all_df


def load_upcoming_fixtures(league_code: str) -> pd.DataFrame:
    """Load upcoming fixtures for the league or simulate a round from league table."""
    fixtures_path = os.path.join(EURO_FOOTBALL_DIR, f"{league_code}_fixtures.csv")
    if os.path.exists(fixtures_path):
        try:
            df = pd.read_csv(fixtures_path)
            # Normalize
            cols = {c.lower(): c for c in df.columns}
            date_col = cols.get('date') or cols.get('kickoff')
            home_col = cols.get('hometeam') or cols.get('home')
            away_col = cols.get('awayteam') or cols.get('away')
            if date_col and home_col and away_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                now = pd.Timestamp.now(tz='UTC')
                end = now + pd.Timedelta(days=21)
                df = df[(df[date_col] >= now) & (df[date_col] <= end)].rename(columns={date_col: 'Date', home_col: 'HomeTeam', away_col: 'AwayTeam'})
                return df[['Date', 'HomeTeam', 'AwayTeam']].reset_index(drop=True)
        except Exception:
            pass
    # Fallback: simulate pairings from league table
    league_df = load_league_table(league_code)
    teams = list(league_df['Team'].astype(str))
    if len(teams) % 2 != 0:
        teams.append('BYE')
    fixtures = []
    now = pd.Timestamp.now(tz='UTC')
    for i in range(0, len(teams), 2):
        if teams[i] != 'BYE' and teams[i+1] != 'BYE':
            fixtures.append({'Date': now, 'HomeTeam': teams[i], 'AwayTeam': teams[i+1]})
            if len(fixtures) >= 10:
                break
    return pd.DataFrame(fixtures)


def load_parsed_fixtures(date_str: str = None, data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load parsed fixtures from data/todays_fixtures_*.csv|json produced by parse_match_log.py"""
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')

    # Search in multiple directories: data/, data/analysis/
    search_dirs = [data_dir, os.path.join(data_dir, 'analysis')]
    candidates = []

    for search_dir in search_dirs:
        csv_path = os.path.join(search_dir, f"todays_fixtures_{date_str}.csv")
        json_path = os.path.join(search_dir, f"todays_fixtures_{date_str}.json")
        if os.path.exists(csv_path):
            candidates.append(csv_path)
        if os.path.exists(json_path):
            candidates.append(json_path)

    if not candidates:
        # Fallback: search all directories for any fixtures files
        for search_dir in search_dirs:
            if os.path.isdir(search_dir):
                all_files = sorted(glob.glob(os.path.join(search_dir, 'todays_fixtures_*.csv')) +
                                 glob.glob(os.path.join(search_dir, 'todays_fixtures_*.json')),
                                 key=lambda p: os.path.getmtime(p), reverse=True)
                if all_files:
                    candidates.extend(all_files[:1])  # Take most recent from each dir

    if not candidates:
        return pd.DataFrame()

    path = candidates[0]
    try:
        if path.lower().endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_json(path)
    except Exception as e:
        logging.warning(f"Failed to read parsed fixtures file {path}: {e}")
        return pd.DataFrame()

    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    rename_map = {}
    if 'date' in cols:
        rename_map[cols['date']] = 'Date'
    if 'home' in cols:
        rename_map[cols['home']] = 'HomeTeam'
    if 'away' in cols:
        rename_map[cols['away']] = 'AwayTeam'
    if 'time' in cols:
        rename_map[cols['time']] = 'Time'
    if 'competition' in cols:
        rename_map[cols['competition']] = 'Competition'
    if 'league' in cols:
        rename_map[cols['league']] = 'League'
    if rename_map:
        df = df.rename(columns=rename_map)

    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception:
            pass
    else:
        df['Date'] = pd.to_datetime(datetime.now().date())

    if 'HomeTeam' not in df.columns or 'AwayTeam' not in df.columns:
        logging.warning(f"Parsed fixtures file missing HomeTeam/AwayTeam: {path}")
        return pd.DataFrame()

    for col in ['HomeTeam', 'AwayTeam', 'League', 'Competition']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = ''

    return df[['Date', 'League', 'Competition', 'HomeTeam', 'AwayTeam']]


TEAM_ALIASES = {
    # Premier League / E0 common short forms
    "nottingham": "Nott'm Forest",
    "nottm forest": "Nott'm Forest",
    "man city": "Man City",
    "man utd": "Man United",
    "manchester united": "Man United",
    "manchester city": "Man City",
    "spurs": "Tottenham",
    "wolves": "Wolves",
    "west ham": "West Ham",
    "brighton": "Brighton",
    "aston villa": "Aston Villa",
    # Spain examples (extendable)
    "ath bilbao": "Ath Bilbao",
    "real madrid": "Real Madrid",
    # Italy, Germany etc can be added similarly
}


def _canonicalize_team(name: str, team_list: List[str]) -> Tuple[str, float]:
    if not name:
        return '', 0.0
    name_norm = str(name).strip()
    low = name_norm.lower()
    # apply alias first
    if low in TEAM_ALIASES:
        target = TEAM_ALIASES[low]
        # exact match check
        for t in team_list:
            if t.lower() == target.lower():
                return t, 1.0
    # exact direct match
    for t in team_list:
        if name_norm.lower() == str(t).strip().lower():
            return t, 1.0
    # stricter fuzzy matching: high cutoff and first letter must match
    candidates = []
    import difflib as _df
    rough = _df.get_close_matches(name_norm, [str(t) for t in team_list], n=3, cutoff=0.75)
    for cand in rough:
        if cand and cand[0].lower() == name_norm[0].lower():
            score = _df.SequenceMatcher(a=name_norm.lower(), b=cand.lower()).ratio()
            if score >= 0.85:
                candidates.append((cand, score))
    if candidates:
        best = max(candidates, key=lambda x: x[1])
        return best[0], best[1]
    return '', 0.0


def map_parsed_to_canonical(parsed_df: pd.DataFrame, strengths_df: pd.DataFrame, league_code: str) -> Tuple[pd.DataFrame, List[Dict]]:
    if parsed_df is None or parsed_df.empty:
        return pd.DataFrame(), []
    if 'Team' in strengths_df.columns:
        canonical_teams = list(strengths_df['Team'].astype(str))
    else:
        return pd.DataFrame(), []

    mapped = []
    skipped: List[Dict] = []
    for _, row in parsed_df.iterrows():
        home = row.get('HomeTeam', '')
        away = row.get('AwayTeam', '')
        h_can, h_score = _canonicalize_team(home, canonical_teams)
        a_can, a_score = _canonicalize_team(away, canonical_teams)
        if h_score >= 0.7 and a_score >= 0.7:
            mapped.append({'Date': row['Date'], 'HomeTeam': h_can, 'AwayTeam': a_can})
        else:
            skipped.append({'row': row.to_dict(), 'home_match': (h_can, h_score), 'away_match': (a_can, a_score)})
    return pd.DataFrame(mapped), skipped


def build_single_match_suggestion(home_team: str, away_team: str, strengths_df: pd.DataFrame, min_confidence: float = 0.6, rating_models: Dict = None, history_df: pd.DataFrame = None, rating_model_config: Dict = None,
                                  enable_double_chance: bool = False, dc_min_prob: float = 0.75, dc_secondary_threshold: float = 0.80, dc_allow_multiple: bool = False) -> Dict:
    # xG and matrix
    xg_home, xg_away = estimate_xg(home_team, away_team, strengths_df)
    mat = score_probability_matrix(xg_home, xg_away, max_goals=6)

    # Optional rating-based override/blend for 1X2
    external_probs = None
    if rating_models and rating_model_config and rating_model_config.get('model', 'none') != 'none' and history_df is not None:
        try:
            r = match_rating(home_team, away_team, history_df, int(rating_model_config.get('last_n', 6)))
            rng = rating_model_config.get('range_filter')
            if rng and not (rng[0] < r <= rng[1]):
                r = None
            if r is not None:
                pH_r, pD_r, pA_r = rating_probabilities_from_rating(r, rating_models)
                if rating_model_config.get('model') == 'goal_supremacy':
                    external_probs = {'1X2': {'Home': pH_r, 'Draw': pD_r, 'Away': pA_r}}
                elif rating_model_config.get('model') == 'blended':
                    # derive Poisson 1X2
                    poisson_all = extract_markets_from_score_matrix(mat, min_confidence=0.0)
                    p1 = poisson_all.get('1X2', {})
                    w = float(rating_model_config.get('blend_weight', 0.3))
                    pH_p, pD_p, pA_p = float(p1.get('Home', 0.0)), float(p1.get('Draw', 0.0)), float(p1.get('Away', 0.0))
                    bH = (1 - w) * pH_p + w * pH_r
                    bD = (1 - w) * pD_p + w * pD_r
                    bA = (1 - w) * pA_p + w * pA_r
                    s = bH + bD + bA
                    if s > 0:
                        external_probs = {'1X2': {'Home': bH / s, 'Draw': bD / s, 'Away': bA / s}}
        except Exception:
            pass

    markets = extract_markets_from_score_matrix(mat, min_confidence=min_confidence, external_probs=external_probs)
    corners = estimate_corners_and_cards(xg_home, xg_away)

    picks = []
    one_x_two = markets.get('1X2', {})
    dc_markets = markets.get('DC', {}) if enable_double_chance else {}

    best_1x2 = None
    if one_x_two:
        best_1x2 = max(one_x_two.items(), key=lambda x: x[1])

    # Double Chance decision logic
    chosen_dc = None
    if enable_double_chance and dc_markets:
        # Evaluate highest DC option
        best_dc = max(dc_markets.items(), key=lambda x: x[1])
        if best_1x2:
            b_prob = float(best_1x2[1])
            # Prefer DC if straight 1X2 below threshold but DC strong
            if b_prob < dc_min_prob and float(best_dc[1]) >= dc_min_prob:
                chosen_dc = best_dc
            # Add DC alongside strong straight pick if very high probability
            elif b_prob >= dc_min_prob and float(best_dc[1]) >= dc_secondary_threshold:
                chosen_dc = best_dc if dc_allow_multiple else best_dc  # will add both below
        else:
            # No straight pick surfaced (e.g., min_confidence filtering), but DC maybe available
            if float(best_dc[1]) >= dc_min_prob:
                chosen_dc = best_dc

    # Add picks respecting selection logic
    if best_1x2 and (not chosen_dc or (chosen_dc and dc_allow_multiple)):
        picks.append({'market': '1X2', 'selection': best_1x2[0], 'prob': float(best_1x2[1]), 'odds': prob_to_decimal_odds(float(best_1x2[1]))})
    if chosen_dc:
        picks.append({'market': 'Double Chance', 'selection': chosen_dc[0], 'prob': float(chosen_dc[1]), 'odds': prob_to_decimal_odds(float(chosen_dc[1]))})

    # BTTS
    btts = markets.get('BTTS', {})
    if float(btts.get('No', 0)) > 0.6:
        picks.append({'market': 'BTTS', 'selection': 'No', 'prob': float(btts['No']), 'odds': prob_to_decimal_odds(float(btts['No']))})
    elif float(btts.get('Yes', 0)) > 0.6:
        picks.append({'market': 'BTTS', 'selection': 'Yes', 'prob': float(btts['Yes']), 'odds': prob_to_decimal_odds(float(btts['Yes']))})

    # Over/Under 2.5
    ou = markets.get('OU', {})
    if float(ou.get('Under2.5', 0)) > 0.7:
        p = float(ou['Under2.5'])
        picks.append({'market': 'Over/Under 2.5', 'selection': 'Under2.5', 'prob': p, 'odds': prob_to_decimal_odds(p)})
    elif float(ou.get('Over2.5', 0)) > 0.7:
        p = float(ou['Over2.5'])
        picks.append({'market': 'Over/Under 2.5', 'selection': 'Over2.5', 'prob': p, 'odds': prob_to_decimal_odds(p)})

    return {
        'home': home_team,
        'away': away_team,
        'xg_home': float(xg_home),
        'xg_away': float(xg_away),
        'markets': markets,
        'corners_cards': corners,
        'picks': picks,
        'score_matrix': mat.to_dict(),
    }


def build_league_suggestions(fixtures_df: pd.DataFrame, strengths_df: pd.DataFrame, min_confidence: float = 0.6, rating_models: Dict = None, history_df: pd.DataFrame = None, rating_model_config: Dict = None,
                             enable_double_chance: bool = False, dc_min_prob: float = 0.75, dc_secondary_threshold: float = 0.80, dc_allow_multiple: bool = False) -> List[Dict]:
    suggestions: List[Dict] = []
    for _, row in fixtures_df.iterrows():
        home = str(row['HomeTeam'])
        away = str(row['AwayTeam'])
        s = build_single_match_suggestion(home, away, strengths_df, min_confidence=min_confidence, rating_models=rating_models, history_df=history_df, rating_model_config=rating_model_config,
                                          enable_double_chance=enable_double_chance, dc_min_prob=dc_min_prob, dc_secondary_threshold=dc_secondary_threshold, dc_allow_multiple=dc_allow_multiple)
        suggestions.append(s)
    return suggestions


def select_favorable_parlays(parlays: List[Dict], min_prob: float = 0.5, min_odds: float = 2.0) -> List[Dict]:
    favorable = [p for p in parlays if p['probability'] > min_prob and p['decimal_odds'] > min_odds]
    return sorted(favorable, key=lambda x: (x['probability'], x['decimal_odds']), reverse=True)[:10]


# Ensure parse_input_log defined early
try:
    parse_input_log
except NameError:
    def parse_input_log(path: str) -> pd.DataFrame:
        if not path or not os.path.exists(path):
            return pd.DataFrame()
        try:
            if path.lower().endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_json(path)
        except Exception:
            return pd.DataFrame()
        rename_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc in ('hometeam','home_team','home'): rename_map[c] = 'HomeTeam'
            elif lc in ('awayteam','away_team','away'): rename_map[c] = 'AwayTeam'
            elif lc == 'date': rename_map[c] = 'Date'
        if rename_map:
            df = df.rename(columns=rename_map)
        if 'HomeTeam' not in df.columns or 'AwayTeam' not in df.columns:
            return pd.DataFrame()
        if 'Date' in df.columns:
            try: df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            except Exception: pass
        else:
            df['Date'] = pd.Timestamp.now()
        return df[['Date','HomeTeam','AwayTeam']]


def analyze_parsed_fixtures_all(parsed_df: pd.DataFrame, min_confidence: float, rating_models: Dict, history_df: pd.DataFrame, rating_model_config: Dict,
                                enable_double_chance: bool = False, dc_min_prob: float = 0.75, dc_secondary_threshold: float = 0.80, dc_allow_multiple: bool = False) -> Tuple[List[Dict], List[Dict]]:
    # Build global team index once
    strengths_cache: Dict[str, pd.DataFrame] = {}
    team_index: Dict[str, set] = {}
    for code in LEAGUE_MAPPINGS.keys():
        try:
            ldf = load_league_table(code)
            sdf = compute_basic_strengths(ldf)
            strengths_cache[code] = sdf
            team_index[code] = set(str(t).strip().lower() for t in sdf['Team'].astype(str))
        except Exception:
            strengths_cache[code] = pd.DataFrame()
            team_index[code] = set()

    def infer_league(home: str, away: str, competition: str, raw_league: str) -> str:
        # 1) Use provided league field if valid
        if raw_league and raw_league in strengths_cache:
            return raw_league
        # 2) Map competition name
        if competition:
            comp_code = COMPETITION_TO_LEAGUE.get(competition.strip(), '')
            if comp_code in strengths_cache:
                return comp_code
        h_norm = TEAM_ALIASES_INFERENCE.get(home.lower(), home.lower())
        a_norm = TEAM_ALIASES_INFERENCE.get(away.lower(), away.lower())
        # 3) Exact presence search
        candidates = []
        for code, names in team_index.items():
            score = 0
            if h_norm in names: score += 2
            if a_norm in names: score += 2
            if score: candidates.append((code, score))
        if candidates:
            return max(candidates, key=lambda x: x[1])[0]
        # 4) Fuzzy match fallback
        import difflib as _df
        best_code = ''
        best_score = 0.0
        for code, names in team_index.items():
            names_list = list(names)
            h_match = _df.get_close_matches(h_norm, names_list, n=1, cutoff=0.85)
            a_match = _df.get_close_matches(a_norm, names_list, n=1, cutoff=0.85)
            local_score = 0.0
            if h_match:
                local_score += _df.SequenceMatcher(a=h_norm, b=h_match[0]).ratio()
            if a_match:
                local_score += _df.SequenceMatcher(a=a_norm, b=a_match[0]).ratio()
            if local_score > best_score:
                best_score = local_score
                best_code = code
        return best_code if best_score >= 1.3 else ''  # requires at least moderate combined similarity

    def canonicalize(team: str, league_code: str) -> Tuple[str, float]:
        """Attempt to canonicalize team name within a league; returns (canonical, score)."""
        raw = (team or '').strip()
        if not raw or league_code not in strengths_cache or strengths_cache[league_code].empty:
            return '', 0.0
        sdf = strengths_cache[league_code]
        teams_list = list(sdf['Team'].astype(str))
        # direct exact match
        for t in teams_list:
            if raw.lower() == t.lower():
                return t, 1.0
        # alias dictionary (reuse TEAM_ALIASES + inference aliases)
        alias_low = raw.lower()
        if alias_low in TEAM_ALIASES:
            target = TEAM_ALIASES[alias_low]
            for t in teams_list:
                if t.lower() == target.lower():
                    return t, 0.95
        if alias_low in TEAM_ALIASES_INFERENCE:
            target = TEAM_ALIASES_INFERENCE[alias_low]
            for t in teams_list:
                if t.lower() == target.lower():
                    return t, 0.9
        # fuzzy match
        import difflib as _df
        candidates = _df.get_close_matches(raw, teams_list, n=1, cutoff=0.82)
        if candidates:
            score = _df.SequenceMatcher(a=raw.lower(), b=candidates[0].lower()).ratio()
            return candidates[0], score
        return '', 0.0

    suggestions: List[Dict] = []
    skipped: List[Dict] = []
    for _, row in parsed_df.iterrows():
        home_raw = str(row.get('HomeTeam') or row.get('home') or '').strip()
        away_raw = str(row.get('AwayTeam') or row.get('away') or '').strip()
        competition = str(row.get('Competition') or row.get('competition') or '').strip()
        raw_league = str(row.get('League') or row.get('league') or '').strip().upper()
        if not home_raw or not away_raw:
            skipped.append({'row': row.to_dict(), 'reason': 'missing_teams'})
            continue
        league_code = infer_league(home_raw, away_raw, competition, raw_league)
        if not league_code or league_code not in strengths_cache or strengths_cache[league_code].empty:
            skipped.append({'row': row.to_dict(), 'reason': 'unresolved_league'})
            continue
        # canonicalize teams within inferred league
        home_can, h_score = canonicalize(home_raw, league_code)
        away_can, a_score = canonicalize(away_raw, league_code)
        if h_score < 0.75 or a_score < 0.75:
            skipped.append({'row': row.to_dict(), 'reason': 'team_not_in_strengths', 'league_code': league_code, 'home_score': h_score, 'away_score': a_score})
            continue
        strengths_df = strengths_cache[league_code]
        try:
            s = build_single_match_suggestion(home_can, away_can, strengths_df, min_confidence=min_confidence, rating_models=rating_models, history_df=history_df, rating_model_config=rating_model_config,
                                              enable_double_chance=enable_double_chance, dc_min_prob=dc_min_prob, dc_secondary_threshold=dc_secondary_threshold, dc_allow_multiple=dc_allow_multiple)
            s['league_code'] = league_code
            s['competition'] = competition
            suggestions.append(s)
        except Exception as e:
            skipped.append({'row': row.to_dict(), 'reason': 'build_error', 'error': str(e), 'league_code': league_code})
    return suggestions, skipped


def main_full_league(bankroll: float = 100.0, league_code: str = 'E0', use_parsed_all: bool = False, min_confidence: float = 0.6, risk_profile: str = 'moderate', rating_model: str = 'none', rating_last_n: int = 6, min_sample_for_rating: int = 30, rating_blend_weight: float = 0.3, rating_range_filter=None,
                     ml_mode: str = 'off', ml_validate: bool = False, ml_algorithms: List[str] = None, ml_decay: float = 0.85, ml_min_samples: int = 300, ml_save_models: bool = False, ml_models_dir: str = 'models', input_log: str = None, fixtures_date: str = None,
                     enable_double_chance: bool = False, dc_min_prob: float = 0.75, dc_secondary_threshold: float = 0.80, dc_allow_multiple: bool = False, ml_models_shared: Dict = None, ml_feature_df_shared: pd.DataFrame = None):
    logging.info("Starting full league analytics workflow")
    logging.info(f"Bankroll={bankroll}, League={league_code}, Rating={rating_model}, ML_Mode={ml_mode}")

    # Load league and strengths
    try:
        league_df = load_league_table(league_code)
    except Exception as e:
        logging.error(f"Failed to load league table for {league_code}: {e}")
        return
    strengths_df = compute_basic_strengths(league_df)

    # History and form blending
    history_df = load_historical_matches()
    if not history_df.empty:
        try:
            strengths_df = merge_form_into_strengths(strengths_df, history_df, last_n=6, decay=0.6, alpha=0.4)
        except Exception:
            logging.warning("Failed to blend recent form; continuing with season strengths")

    # Rating models (optional)
    rating_models = None
    if rating_model and rating_model != 'none' and not history_df.empty:
        try:
            logging.info("Fitting rating-to-probability models from historical data...")
            rating_models = fit_rating_to_prob_models(history_df, last_n=rating_last_n, min_sample_for_rating=min_sample_for_rating)
            logging.info(f"Rating model fitted (samples={rating_models.get('sample_size', 0)})")
        except Exception as e:
            logging.warning(f"Failed to fit rating models: {e}")

    rating_model_config = {
        'model': rating_model,
        'last_n': rating_last_n,
        'blend_weight': rating_blend_weight,
        'range_filter': rating_range_filter,
    }

    # --- ML Training Phase (optional) ---
    ml_models = None
    ml_feature_df = None
    # Prefer shared precomputed models/feature frame when supplied (to avoid redoing expensive work per league)
    if ml_feature_df_shared is not None and ml_models_shared is not None:
        ml_feature_df = ml_feature_df_shared
        ml_models = ml_models_shared
        logging.debug("Using shared ML feature dataframe and models for this league")
    elif ml_mode in ('train','predict'):
        if engineer_features is None:
            logging.warning("ML modules not available - ensure dependencies installed (scikit-learn, optional xgboost)")
        elif history_df.empty:
            logging.warning("No historical data available for ML training; skipping ML mode")
        else:
            try:
                ml_feature_df = engineer_features(history_df)
                weights = build_recency_weights(ml_feature_df['Date'].values, decay=ml_decay) if build_recency_weights else None
                if not check_min_samples(ml_feature_df, ml_min_samples):
                    logging.warning(f"Insufficient samples for ML (need >= {ml_min_samples}, have {len(ml_feature_df)}); skipping ML")
                else:
                    algos = ml_algorithms or ['rf','xgb']
                    ml_models = train_models(ml_feature_df, weights=weights, algorithms=algos)
                    if ml_validate:
                        logging.info("ML Cross-Validation Metrics:")
                        for k,v in ml_models.get('cv_metrics', {}).items():
                            logging.info(f"  {k}: {v}")
                    if ml_save_models:
                        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                        path = os.path.join(ml_models_dir, f"ml_models_{league_code}_{ts}.pkl")
                        try:
                            save_models(ml_models, path)
                        except Exception as e:
                            logging.warning(f"Failed to save ML models: {e}")
            except Exception as e:
                logging.warning(f"ML training failed: {e}")

    # --- Source fixtures logic with input log override ---
    parsed_fixtures = load_parsed_fixtures(date_str=fixtures_date)
    log_df = parse_input_log(input_log) if input_log else pd.DataFrame()

    if not log_df.empty:
        # Filter only those matches whose teams exist in strengths for this league (with canonicalization)
        league_teams = list(str(t) for t in strengths_df['Team'].astype(str))
        matched_rows = []
        skipped_rows = []
        for _, r in log_df.iterrows():
            h_raw, a_raw = str(r['HomeTeam']).strip(), str(r['AwayTeam']).strip()
            h_can, h_score = _canonicalize_team(h_raw, league_teams)
            a_can, a_score = _canonicalize_team(a_raw, league_teams)
            if h_score >= 0.7 and a_score >= 0.7:
                matched_rows.append({'Date': r['Date'], 'HomeTeam': h_can, 'AwayTeam': a_can})
            else:
                skipped_rows.append({'HomeTeam': h_raw, 'AwayTeam': a_raw, 'home_match': (h_can, h_score), 'away_match': (a_can, a_score)})
        if matched_rows:
            fixtures_df = pd.DataFrame(matched_rows)
            suggestions = build_league_suggestions(fixtures_df, strengths_df, min_confidence=min_confidence, rating_models=rating_models, history_df=history_df, rating_model_config=rating_model_config,
                                                   enable_double_chance=enable_double_chance, dc_min_prob=dc_min_prob, dc_secondary_threshold=dc_secondary_threshold, dc_allow_multiple=dc_allow_multiple)
            logging.info(f"Input log matches used: {len(fixtures_df)} (skipped {len(skipped_rows)})")
        else:
            logging.warning("No valid matches from input log for this league; falling back to standard fixture logic")
            fixtures_df = None
    elif use_parsed_all and not parsed_fixtures.empty:
        suggestions, skipped = analyze_parsed_fixtures_all(parsed_fixtures, min_confidence=min_confidence, rating_models=rating_models, history_df=history_df, rating_model_config=rating_model_config,
                                                           enable_double_chance=enable_double_chance, dc_min_prob=dc_min_prob, dc_secondary_threshold=dc_secondary_threshold, dc_allow_multiple=dc_allow_multiple)
        if skipped:
            logging.info(f"Skipped {len(skipped)} parsed fixtures (team/league inference failures)")
            # breakdown by reason
            reason_counts = {}
            for sk in skipped:
                r = sk.get('reason','unknown')
                reason_counts[r] = reason_counts.get(r, 0) + 1
            logging.info("Skip reasons: " + ', '.join(f"{k}={v}" for k,v in reason_counts.items()))
        logging.info(f"Parsed fixture inference: success={len(suggestions)} skipped={len(skipped)}")
        fixtures_df = None
    else:
        fixtures_df = None
        if not parsed_fixtures.empty:
            mapped, skipped = map_parsed_to_canonical(parsed_fixtures, strengths_df, league_code)
            if not mapped.empty:
                fixtures_df = mapped
                if skipped:
                    logging.info(f"Skipped {len(skipped)} parsed fixtures due to poor team matching")
        if fixtures_df is None or fixtures_df.empty:
            fixtures_df = load_upcoming_fixtures(league_code)
        if fixtures_df.empty:
            logging.warning("No fixtures available to analyze")
            return
        suggestions = build_league_suggestions(fixtures_df, strengths_df, min_confidence=min_confidence, rating_models=rating_models, history_df=history_df, rating_model_config=rating_model_config,
                                               enable_double_chance=enable_double_chance, dc_min_prob=dc_min_prob, dc_secondary_threshold=dc_secondary_threshold, dc_allow_multiple=dc_allow_multiple)

    # --- ML Prediction Augmentation ---
    if ml_models and ml_mode == 'predict' and ml_feature_df is not None:
        enriched = []
        for s in suggestions:
            home, away = s['home'], s['away']
            feat_row_dict = build_match_feature_row(ml_feature_df, home, away)
            feature_vector = np.array([feat_row_dict[c] for c in TRAIN_FEATURE_COLUMNS], dtype=float).reshape(1,-1)
            ml_pred = predict_match(ml_models, feature_vector)
            # Recompute baseline markets directly from xG (avoids reconstructing matrix from dict)
            mat = score_probability_matrix(s['xg_home'], s['xg_away'], max_goals=6)
            baseline_full = extract_markets_from_score_matrix(mat, min_confidence=0.0)
            comparison = evaluate_vs_poisson(baseline_full, ml_pred)
            s['ml_prediction'] = ml_pred
            s['ml_vs_poisson'] = comparison
            enriched.append(s)
        suggestions = enriched

    # Print suggestions (extended output for ML)
    print(f"\n{Colors.CYAN}📊 Full League Match Suggestions for {league_code}:{Colors.RESET}")
    for i, s in enumerate(suggestions, 1):
        home, away = s['home'], s['away']
        print(f"\n{Colors.YELLOW}Match {i}:{Colors.RESET} {Colors.BOLD}{home}{Colors.RESET} v {Colors.BOLD}{away}{Colors.RESET}")
        print(f"  Estimated xG: {Colors.GREEN}{s['xg_home']:.2f}{Colors.RESET} - {Colors.GREEN}{s['xg_away']:.2f}{Colors.RESET}")
        if s['picks']:
            print(f"  {Colors.CYAN}Suggested Picks:{Colors.RESET}")
            for p in s['picks']:
                print(f"    {p['market']} {p['selection']}: {Colors.GREEN}{p['prob']*100:.1f}%{Colors.RESET} (odds {Colors.MAGENTA}{p['odds']:.2f}{Colors.RESET})")
        else:
            print(f"  {Colors.YELLOW}No high-confidence picks available{Colors.RESET}")
        if 'ml_prediction' in s:
            mp = s['ml_prediction']
            comp = s.get('ml_vs_poisson', {})
            # Compute ML Double Chance probabilities from ML 1X2 probs if available
            ml_dc_line = ''
            if all(k in mp for k in ['prob_1x2_home','prob_1x2_draw','prob_1x2_away']):
                pH_ml = mp['prob_1x2_home']; pD_ml = mp['prob_1x2_draw']; pA_ml = mp['prob_1x2_away']
                ml_dc_line = f"  {Colors.MAGENTA}ML DC probs:{Colors.RESET} 1X={pH_ml+pD_ml:.2f} X2={pD_ml+pA_ml:.2f} 12={pH_ml+pA_ml:.2f}"
            if ml_dc_line:
                print(ml_dc_line)
            print(f"  {Colors.MAGENTA}ML Total Goals:{Colors.RESET} {mp.get('pred_total_goals',0):.2f} (model {mp.get('pred_total_goals_model','-')})")
            print(f"  {Colors.MAGENTA}ML 1X2 probs:{Colors.RESET} H={mp.get('prob_1x2_home',0):.2f} D={mp.get('prob_1x2_draw',0):.2f} A={mp.get('prob_1x2_away',0):.2f} (model {mp.get('model_1x2','-')})")
            print(f"  {Colors.MAGENTA}ML BTTS probs:{Colors.RESET} Yes={mp.get('prob_btts_yes',0):.2f} No={mp.get('prob_btts_no',0):.2f} (model {mp.get('model_btts','-')})")
            if comp:
                c1 = comp.get('1X2', {})
                if c1:
                    print(f"    Δ1X2: H={c1.get('delta_home',0):+.2f} D={c1.get('delta_draw',0):+.2f} A={c1.get('delta_away',0):+.2f}")
                cb = comp.get('BTTS', {})
                if cb:
                    print(f"    ΔBTTS: Yes={cb.get('delta_yes',0):+.2f} No={cb.get('delta_no',0):+.2f}")

    # Parlays across all suggestions (unchanged)
    all_predictions = []
    for s in suggestions:
        for p in s['picks']:
            all_predictions.append((s['home'], s['away'], p['prob'], p['odds']))
    parlays = generate_parlays(all_predictions, min_size=2, max_size=3, max_results=50)
    favorable_parlays = select_favorable_parlays(parlays)

    print(f"\n{Colors.CYAN}🎲 Top Favorable Parlays:{Colors.RESET}")
    for p in favorable_parlays:
        slip = format_bet_slip(p, bankroll=bankroll)
        print(f"- Legs ({p['size']}): {p['legs']}")
        print(f"  Prob: {Colors.GREEN}{p['probability']*100:.2f}%{Colors.RESET}, Odds: {Colors.MAGENTA}{p['decimal_odds']:.2f}{Colors.RESET}, Stake: {Colors.YELLOW}{slip['stake_suggestion']}{Colors.RESET}, Return: {Colors.GREEN}{slip['potential_return']}{Colors.RESET}")

    # Save results
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    # Ensure new dirs exist and prefer data/analysis/ for suggestions
    ensure_dirs_for_writing()
    out_dir = os.path.join('data', 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'full_league_suggestions_{league_code}_{timestamp}.json')
    with open(out_path, 'w') as f:
        json.dump({'suggestions': suggestions, 'favorable_parlays': favorable_parlays}, f, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    logging.info(f"Saved results to: {out_path}")


def main_full_league_multiple(bankroll: float = 100.0, leagues: List[str] = None, use_parsed_all: bool = False, min_confidence: float = 0.6, risk_profile: str = 'moderate', rating_model: str = 'none', rating_last_n: int = 6, min_sample_for_rating: int = 30, rating_blend_weight: float = 0.3, rating_range_filter=None,
                              ml_mode: str = 'off', ml_validate: bool = False, ml_algorithms: List[str] = None, ml_decay: float = 0.85, ml_min_samples: int = 300, ml_save_models: bool = False, ml_models_dir: str = 'models', input_log: str = None, fixtures_date: str = None,
                              enable_double_chance: bool = False, dc_min_prob: float = 0.75, dc_secondary_threshold: float = 0.80, dc_allow_multiple: bool = False, parallel_workers: int = 1,
                              shared_ml_models: Dict = None, shared_ml_feature_df: pd.DataFrame = None):
    """Process multiple leagues; optionally runs leagues in parallel.

    Adds timing logs per-league and total time. Uses a ThreadPool when parallel_workers>1 to avoid pickling issues.
    """
    if leagues is None:
        leagues = ['E0']

    start_all = time.perf_counter()
    league_times = {}

    def _process_one(league_code: str):
        logging.info(f"Processing league: {league_code}")
        t0 = time.perf_counter()
        try:
            main_full_league(
                bankroll=bankroll,
                league_code=league_code,
                use_parsed_all=use_parsed_all,
                min_confidence=min_confidence,
                risk_profile=risk_profile,
                rating_model=rating_model,
                rating_last_n=rating_last_n,
                min_sample_for_rating=min_sample_for_rating,
                rating_blend_weight=rating_blend_weight,
                rating_range_filter=rating_range_filter,
                ml_mode=ml_mode,
                ml_validate=ml_validate,
                ml_algorithms=ml_algorithms,
                ml_decay=ml_decay,
                ml_min_samples=ml_min_samples,
                ml_save_models=ml_save_models,
                ml_models_dir=ml_models_dir,
                input_log=input_log,
                fixtures_date=fixtures_date,
                enable_double_chance=enable_double_chance,
                dc_min_prob=dc_min_prob,
                dc_secondary_threshold=dc_secondary_threshold,
                dc_allow_multiple=dc_allow_multiple,
                ml_models_shared=shared_ml_models,
                ml_feature_df_shared=shared_ml_feature_df
             )
        finally:
            t1 = time.perf_counter()
            elapsed = t1 - t0
            league_times[league_code] = elapsed
            logging.info(f"League {league_code} finished in {elapsed:.2f}s")

    if parallel_workers and parallel_workers > 1 and len(leagues) > 1:
        # Use threads to avoid pickling ML objects; threads can improve throughput for I/O-bound parts.
        pool = ThreadPool(min(parallel_workers, len(leagues)))
        pool.map(_process_one, leagues)
        pool.close()
        pool.join()
    else:
        for league_code in leagues:
            _process_one(league_code)

    total_elapsed = time.perf_counter() - start_all
    logging.info("Per-league timings: " + ', '.join(f"{k}={v:.2f}s" for k,v in league_times.items()))
    logging.info(f"Total full-league processing time: {total_elapsed:.2f}s")

    return total_elapsed


def extract_leagues_from_parsed_fixtures(fixtures_date: str = None, data_dir: str = DATA_DIR) -> List[str]:
    """Extract unique league codes from parsed fixtures file efficiently."""
    try:
        parsed_fixtures = load_parsed_fixtures(date_str=fixtures_date, data_dir=data_dir)
        if parsed_fixtures.empty:
            return []

        # Extract unique leagues efficiently
        leagues = parsed_fixtures['League'].dropna().str.strip()
        unique_leagues = sorted([l for l in leagues.unique() if l])

        logging.info(f"Extracted {len(unique_leagues)} leagues from parsed fixtures: {','.join(unique_leagues)}")
        return unique_leagues
    except Exception as e:
        logging.warning(f"Failed to extract leagues from parsed fixtures: {e}")
        return []


def main(argv=None):
    """Entry point compatible with CLI wrapper. Accepts argv (list of args) or uses sys.argv[1:]."""
    import sys
    parser = argparse.ArgumentParser(description='Football analytics assistant (full league)')
    parser.add_argument('--bankroll', type=float, default=100.0)
    parser.add_argument('--use-parsed-all', action='store_true')
    parser.add_argument('--min-confidence', type=float, default=0.6)
    parser.add_argument('--risk-profile', type=str, default='moderate', choices=['conservative','moderate','aggressive'])
    parser.add_argument('--rating-model', type=str, default='none', choices=['none','goal_supremacy','blended'])
    parser.add_argument('--rating-last-n', type=int, default=6)
    parser.add_argument('--min-sample-for-rating', type=int, default=30)
    parser.add_argument('--rating-blend-weight', type=float, default=0.3)
    parser.add_argument('--league', type=str)
    parser.add_argument('--leagues', type=str, default='E0')
    parser.add_argument('--rating-range-filter', type=str)
    parser.add_argument('--ml-mode', type=str, default='off', choices=['off','train','predict'])
    parser.add_argument('--ml-validate', action='store_true')
    parser.add_argument('--ml-algorithms', type=str, default='rf,xgb')
    parser.add_argument('--ml-decay', type=float, default=0.85)
    parser.add_argument('--ml-min-samples', type=int, default=300)
    parser.add_argument('--ml-save-models', action='store_true')
    parser.add_argument('--ml-models-dir', type=str, default='models')
    parser.add_argument('--fixtures-date', type=str)
    parser.add_argument('--input-log', type=str)
    parser.add_argument('--enable-double-chance', action='store_true')
    parser.add_argument('--dc-min-prob', type=float, default=0.75)
    parser.add_argument('--dc-secondary-threshold', type=float, default=0.80)
    parser.add_argument('--dc-allow-multiple', action='store_true')
    parser.add_argument('--parallel-workers', type=int, default=1, help='Number of worker threads to process leagues in parallel (default 1)')
    parser.add_argument('--verbose', action='store_true')

    parsed = parser.parse_args(argv)

    leagues_arg = parsed.leagues
    if parsed.league and parsed.league not in leagues_arg.split(','):
        leagues_arg = parsed.league
    leagues_to_run = [l.strip() for l in leagues_arg.split(',') if l.strip()]

    # Dynamic league extraction when using parsed fixtures with 'ALL'
    if parsed.use_parsed_all and leagues_to_run == ['ALL']:
        extracted_leagues = extract_leagues_from_parsed_fixtures(fixtures_date=parsed.fixtures_date)
        if extracted_leagues:
            leagues_to_run = extracted_leagues
            logging.info(f"Dynamically using leagues from parsed fixtures: {','.join(leagues_to_run)}")
        else:
            logging.warning("No leagues found in parsed fixtures, falling back to default")
            leagues_to_run = ['E0']

    ml_algos_list = [a.strip() for a in parsed.ml_algorithms.split(',') if a.strip()]

    # Precompute shared ML assets if requested to avoid repeating per-league
    shared_ml_feature_df = None
    shared_ml_models = None
    if parsed.ml_mode in ('train','predict'):
        history_df_global = load_historical_matches()
        if not history_df_global.empty and engineer_features is not None:
            try:
                shared_ml_feature_df = engineer_features(history_df_global)
                # optionally train shared models once if enough samples
                if check_min_samples and check_min_samples(shared_ml_feature_df, parsed.ml_min_samples):
                    shared_ml_models = train_models(shared_ml_feature_df, weights=(build_recency_weights(shared_ml_feature_df['Date'].values, decay=parsed.ml_decay) if build_recency_weights else None), algorithms=[a.strip() for a in parsed.ml_algorithms.split(',') if a.strip()])
                    if parsed.ml_save_models:
                        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                        try:
                            save_models(shared_ml_models, os.path.join(parsed.ml_models_dir, f"ml_models_shared_{ts}.pkl"))
                        except Exception:
                            logging.warning("Failed to persist shared ML models")
            except Exception as e:
                logging.warning(f"Failed to precompute shared ML assets: {e}")

    return main_full_league_multiple(
        bankroll=parsed.bankroll,
        leagues=leagues_to_run,
        use_parsed_all=parsed.use_parsed_all,
        min_confidence=parsed.min_confidence,
        risk_profile=parsed.risk_profile,
        rating_model=parsed.rating_model,
        rating_last_n=parsed.rating_last_n,
        min_sample_for_rating=parsed.min_sample_for_rating,
        rating_blend_weight=parsed.rating_blend_weight,
        rating_range_filter=_parse_range_filter(parsed.rating_range_filter) if parsed.rating_range_filter else None,
        ml_mode=parsed.ml_mode,
        ml_validate=parsed.ml_validate,
        ml_algorithms=ml_algos_list,
        ml_decay=parsed.ml_decay,
        ml_min_samples=parsed.ml_min_samples,
        ml_save_models=parsed.ml_save_models,
        ml_models_dir=parsed.ml_models_dir,
        input_log=parsed.input_log,
        fixtures_date=parsed.fixtures_date,
        enable_double_chance=parsed.enable_double_chance,
        dc_min_prob=parsed.dc_min_prob,
        dc_secondary_threshold=parsed.dc_secondary_threshold,
        dc_allow_multiple=parsed.dc_allow_multiple,
        parallel_workers=parsed.parallel_workers,
        shared_ml_models=shared_ml_models,
        shared_ml_feature_df=shared_ml_feature_df
    )

# End of file
