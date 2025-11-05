"""
Football Analytics Algorithms Module

This module contains the 11 core algorithms that power the football betting analysis system:
1. Team Strength Calculation Algorithm
2. Expected Goals (xG) Estimation Algorithm
3. Poisson Probability Distribution
4. Score Probability Matrix Algorithm
5. Market Probability Extraction
6. Recent Form Algorithm (Exponential Decay)
7. Form-Season Strength Blending
8. Kelly Criterion Staking Algorithm
9. Parlay Generation Algorithm
10. Smart Caching Algorithm
11. Heuristic Corner/Cards Estimation

All algorithms are mathematically sound and designed for conservative betting strategies.
"""

import math
import os
import json
import itertools
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import logging

# --- ALGORITHM 1: TEAM STRENGTH CALCULATION ---

def compute_basic_strengths(league_df: pd.DataFrame) -> pd.DataFrame:
    """
    Algorithm 1: Team Strength Calculation

    Computes relative attack/defence strengths normalized against league averages.

    Formula:
        Attack_Strength = (Goals_For / Games_Played) / League_Average_Attack
        Defence_Strength = (Goals_Against / Games_Played) / League_Average_Defence

    Args:
        league_df: DataFrame with columns Team, P (played), F (for), A (against)

    Returns:
        Enhanced DataFrame with Attack_Str and Defence_Str columns
    """
    df = league_df.copy()

    # Harmonize column names if scrapings have slightly different names
    col_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc.startswith('team'):
            col_map[c] = 'Team'
        elif lc == 'p' or lc.startswith('pl') or 'played' in lc:
            col_map[c] = 'P'
        elif lc == 'f' or ('for' in lc and 'gf' not in lc):
            col_map[c] = 'F'
        elif lc == 'a' or 'against' in lc or 'ga' in lc:
            col_map[c] = 'A'
    df = df.rename(columns=col_map)

    # Ensure required columns
    if not set(['Team', 'P', 'F', 'A']).issubset(df.columns):
        raise ValueError('League dataframe must have Team, P, F, A columns (or equivalents)')

    # Calculate base strengths (goals per game)
    df['Attack'] = df['F'] / df['P']
    df['Defence'] = df['A'] / df['P']

    # Normalize against league averages
    league_avg_attack = df['Attack'].mean()
    league_avg_defence = df['Defence'].mean()
    df['Attack_Str'] = df['Attack'] / league_avg_attack
    df['Defence_Str'] = df['Defence'] / league_avg_defence

    logging.info(f"Computed strengths for {len(df)} teams")
    return df


# --- ALGORITHM 2: EXPECTED GOALS (xG) ESTIMATION ---

def estimate_xg(home_team: str, away_team: str, strengths_df: pd.DataFrame,
                home_advantage: float = 1.12) -> Tuple[float, float]:
    """
    Algorithm 2: Expected Goals (xG) Multiplicative Model

    Estimates expected goals using normalized team strengths and home advantage.

    Formula:
        xG_Home = League_Avg × Home_Attack_Str × Away_Defence_Str × Home_Advantage
        xG_Away = League_Avg × Away_Attack_Str × Home_Defence_Str

    Args:
        home_team: Home team name
        away_team: Away team name
        strengths_df: DataFrame with team strengths
        home_advantage: Multiplier for home team (typical 1.1-1.2)

    Returns:
        Tuple of (xG_home, xG_away)
    """
    league_avg_goals = strengths_df['Attack'].mean()

    try:
        home = strengths_df[strengths_df['Team'] == home_team].iloc[0]
        away = strengths_df[strengths_df['Team'] == away_team].iloc[0]
    except IndexError:
        raise ValueError('One or both team names not found in strengths dataframe')

    # Use adjusted strengths if available, otherwise base strengths
    if 'Attack_Str_Adj' in strengths_df.columns and 'Defence_Str_Adj' in strengths_df.columns:
        home_attack_str = home['Attack_Str_Adj']
        away_defence_str = away['Defence_Str_Adj']
        away_attack_str = away['Attack_Str_Adj']
        home_defence_str = home['Defence_Str_Adj']
    else:
        home_attack_str = home['Attack_Str']
        away_defence_str = away['Defence_Str']
        away_attack_str = away['Attack_Str']
        home_defence_str = home['Defence_Str']

    # Multiplicative xG model
    xg_home = league_avg_goals * home_attack_str * away_defence_str * home_advantage
    xg_away = league_avg_goals * away_attack_str * home_defence_str

    # Prevent zero/too-small lambdas for Poisson stability
    xg_home = max(0.05, xg_home)
    xg_away = max(0.05, xg_away)

    return float(xg_home), float(xg_away)


# --- ALGORITHM 3: POISSON PROBABILITY DISTRIBUTION ---

def poisson_prob(lam: float, k: int) -> float:
    """
    Algorithm 3: Poisson Probability Mass Function

    Calculates probability of exactly k goals given expected goals λ.
    Goals in football follow Poisson distribution - this is the mathematical foundation.

    Formula: P(X = k) = (λ^k × e^(-λ)) / k!

    Args:
        lam: Expected goals (λ parameter)
        k: Actual goals scored

    Returns:
        Probability of scoring exactly k goals
    """
    if lam < 0:
        return 0.0
    try:
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    except OverflowError:
        return 0.0


# --- ALGORITHM 4: SCORE PROBABILITY MATRIX ---

def score_probability_matrix(xg_home: float, xg_away: float, max_goals: int = 6) -> pd.DataFrame:
    """
    Algorithm 4: Score Probability Matrix Generation

    Creates matrix of all possible match score probabilities using Poisson distribution.

    Args:
        xg_home: Expected goals for home team
        xg_away: Expected goals for away team
        max_goals: Maximum goals to consider (default 6)

    Returns:
        DataFrame matrix where [i,j] = P(Home=i goals, Away=j goals)
    """
    idx = range(0, max_goals + 1)
    mat = pd.DataFrame(index=idx, columns=idx, dtype=float)

    for hg in idx:
        ph = poisson_prob(xg_home, hg)
        for ag in idx:
            pa = poisson_prob(xg_away, ag)
            mat.loc[hg, ag] = ph * pa

    # Normalize for any small rounding errors
    mat = mat / mat.values.sum()
    return mat


# --- ALGORITHM 5: MARKET PROBABILITY EXTRACTION ---

def extract_markets_from_score_matrix(mat: pd.DataFrame, min_confidence: float = 0.6, external_probs: Dict[str, Dict[str, float]] = None) -> Dict[str, Dict[str, float]]:
    """
    Algorithm 5: Market Probability Extraction with Confidence Filtering

    Converts score probability matrix into betting market probabilities.
    Filters for high-confidence picks to favor low-risk markets.

    Markets calculated:
    - 1X2: Home Win (h>a), Draw (h=a), Away Win (h<a)
    - Over/Under: Goals thresholds (1.5, 2.5)
    - BTTS: Both Teams To Score (both h>0 and a>0)

    Args:
        mat: Score probability matrix from Algorithm 4
        min_confidence: Minimum prob for market selection (default 0.6)
        external_probs: Optional externally computed probabilities to override specific markets e.g., {'1X2': {'Home': pH, 'Draw': pD, 'Away': pA}}

    Returns:
        Dictionary of market probabilities
    """
    max_goals = mat.shape[0] - 1

    # Initialize market probabilities
    home_win = draw = away_win = 0.0
    under_1_5 = over_1_5 = under_2_5 = over_2_5 = 0.0
    btts_yes = btts_no = 0.0

    # Sum probabilities across all score combinations
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = mat.loc[hg, ag]

            # 1X2 Market
            if hg > ag:
                home_win += p
            elif hg == ag:
                draw += p
            else:
                away_win += p

            # Over/Under Markets
            total_goals = hg + ag
            if total_goals <= 1:
                under_1_5 += p
            else:
                over_1_5 += p
            if total_goals <= 2:
                under_2_5 += p
            else:
                over_2_5 += p

            # BTTS Market
            if hg > 0 and ag > 0:
                btts_yes += p
            else:
                btts_no += p

    markets = {
        '1X2': {'Home': home_win, 'Draw': draw, 'Away': away_win},
        'OU': {'Under1.5': under_1_5, 'Over1.5': over_1_5,
               'Under2.5': under_2_5, 'Over2.5': over_2_5},
        'BTTS': {'Yes': btts_yes, 'No': btts_no}
    }

    # If external 1X2 probabilities are provided, override the Poisson-derived 1X2
    if external_probs and isinstance(external_probs, dict) and '1X2' in external_probs:
        ext = external_probs.get('1X2') or {}
        pH = float(ext.get('Home', 0.0))
        pD = float(ext.get('Draw', 0.0))
        pA = float(ext.get('Away', 0.0))
        # Clamp and renormalize to be safe
        pH = min(max(pH, 0.001), 0.999)
        pD = min(max(pD, 0.001), 0.999)
        pA = min(max(pA, 0.001), 0.999)
        s = pH + pD + pA
        if s > 0:
            markets['1X2'] = {'Home': pH / s, 'Draw': pD / s, 'Away': pA / s}

    # Filter and log high-confidence markets
    selected = {}
    for market, options in markets.items():
        selected[market] = {k: v for k, v in options.items() if v >= min_confidence}
        if selected[market]:
            logging.info(f"Selected {market} markets: {selected[market]}")
    return selected


# --- ALGORITHM 6: RECENT FORM (EXPONENTIAL DECAY) ---

def compute_recent_form(history_df: pd.DataFrame, teams: List[str],
                        last_n: int = 6, decay: float = 0.6) -> Dict[str, Dict[str, float]]:
    """
    Algorithm 6: Recent Form with Exponential Decay

    Calculates recent form using exponentially weighted average of last N matches.
    Recent matches get higher weight than older ones.

    Exponential Decay Formula: Weight_i = decay^i (where i=0 is most recent)

    Args:
        history_df: Historical match data
        teams: List of team names
        last_n: Number of recent matches to consider
        decay: Decay factor (0 < decay <= 1)

    Returns:
        Dictionary of team form scores {team: {'attack': val, 'defence': val}}
    """
    if history_df.empty:
        return {t: {'attack': 1.0, 'defence': 1.0} for t in teams}

    form = {}
    for team in teams:
        # Get matches where team played
        played = history_df[(history_df['HomeTeam'] == team) |
                            (history_df['AwayTeam'] == team)].copy()

        if played.empty:
            form[team] = {'attack': 1.0, 'defence': 1.0}
            continue

        # Take last N matches
        last = played.tail(last_n)

        # Exponential weights: [1.0, 0.6, 0.36, 0.216, ...]
        weights = np.array([decay ** i for i in range(len(last)-1, -1, -1)], dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(last)) / len(last)

        # Extract goals for/against
        goals_for = []
        goals_against = []
        for _, row in last.iterrows():
            if row['HomeTeam'] == team:
                gf, ga = row['FTHG'], row['FTAG']
            else:
                gf, ga = row['FTAG'], row['FTHG']
            goals_for.append(gf)
            goals_against.append(ga)

        # Calculate weighted averages
        goals_for = np.array(goals_for, dtype=float)
        goals_against = np.array(goals_against, dtype=float)
        wg = float(np.dot(weights, goals_for))
        wga = float(np.dot(weights, goals_against))

        form[team] = {'attack': max(0.01, wg), 'defence': max(0.01, wga)}

    return form


# --- ALGORITHM 7: FORM-SEASON STRENGTH BLENDING ---

def merge_form_into_strengths(strengths_df: pd.DataFrame, history_df: pd.DataFrame,
                              last_n: int = 6, decay: float = 0.7, alpha: float = 0.35) -> pd.DataFrame:
    """
    Algorithm 7: Form-Season Strength Blending with Tuned Weights

    Combines season-long statistics with recent form using adjusted alpha and decay.

    Formula: Final_Strength = (1-α) × Season_Strength + α × Recent_Form_Strength

    Args:
        strengths_df: Season strength data
        history_df: Historical match data
        last_n: Number of recent matches
        decay: Form decay factor (default 0.7)
        alpha: Weight for recent form (default 0.35)

    Returns:
        Enhanced DataFrame with adjusted strength columns
    """
    df = strengths_df.copy()
    teams = list(df['Team'])
    form = compute_recent_form(history_df, teams, last_n=last_n, decay=decay)

    league_avg_attack = df['Attack'].mean()
    league_avg_defence = df['Defence'].mean()

    adj_attack_str = []
    adj_defence_str = []

    for _, row in df.iterrows():
        team = row['Team']
        season_att = row['Attack']
        season_def = row['Defence']

        if team in form:
            # Normalize recent form against league averages
            f_att_str = form[team]['attack'] / league_avg_attack if league_avg_attack > 0 else 1.0
            f_def_str = form[team]['defence'] / league_avg_defence if league_avg_defence > 0 else 1.0

            # Blend season and form: (1-α) × season + α × form
            season_att_str = season_att / league_avg_attack if league_avg_attack > 0 else 1.0
            season_def_str = season_def / league_avg_defence if league_avg_defence > 0 else 1.0

            combined_att = (1.0 - alpha) * season_att_str + alpha * f_att_str
            combined_def = (1.0 - alpha) * season_def_str + alpha * f_def_str

            # Log adjustments for visualization
            att_diff = combined_att - season_att_str
            def_diff = combined_def - season_def_str
            logging.debug(f"{team}: Attack adj {att_diff:.3f}, Defence adj {def_diff:.3f}")
        else:
            combined_att = season_att / league_avg_attack if league_avg_attack > 0 else 1.0
            combined_def = season_def / league_avg_defence if league_avg_defence > 0 else 1.0

        adj_attack_str.append(combined_att)
        adj_defence_str.append(combined_def)

    df['Attack_Str_Adj'] = adj_attack_str
    df['Defence_Str_Adj'] = adj_defence_str

    logging.info(f"Blended strengths (alpha={alpha}, decay={decay}): {len(df)} teams")
    return df


# --- ALGORITHM 8: KELLY CRITERION STAKING ---

def kelly_fraction(prob: float, odds: float, f_max: float = 0.015, risk_multiplier: float = 0.5) -> float:
    """
    Algorithm 8: Kelly Criterion for Optimal Bet Sizing with Risk Adjustment

    Calculates optimal fraction with conservative caps and risk scaling.

    Formula: f = risk_multiplier × (bp - q) / b, capped at f_max

    Args:
        prob: Probability of winning (0-1)
        odds: Decimal odds
        f_max: Maximum fraction cap (default 1.5%)
        risk_multiplier: Scales down stake for risk (default 0.5)

    Returns:
        Fraction of bankroll to bet
    """
    if prob <= 0 or odds <= 1:
        logging.debug("Invalid prob or odds; stake=0")
        return 0.0

    b = odds - 1.0
    q = 1.0 - prob
    f = risk_multiplier * (b * prob - q) / b

    if f <= 0:
        logging.debug("No positive EV; stake=0")
        return 0.0

    stake = min(f, f_max)
    logging.info(f"Kelly stake: {stake:.4f} (prob={prob:.2f}, odds={odds:.2f})")
    return stake


# --- ALGORITHM 9: PARLAY GENERATION ---

def generate_parlays(predictions: List[Tuple[str, str, float, float]],
                     min_size: int = 2, max_size: int = 4, max_results: int = 50) -> List[Dict]:
    """
    Algorithm 9: Parlay Generation with Value Ranking

    Creates combination bets from multiple single predictions.
    Uses value metric to rank and select best combinations.

    Value Metric: (probability × odds - 1) - (1 - probability)
    Combined Probability: P₁ × P₂ × ... × Pₙ (independence assumption)
    Combined Odds: O₁ × O₂ × ... × Oₙ

    Args:
        predictions: List of (home_team, away_team, prob, odds) tuples
        min_size: Minimum legs in parlay
        max_size: Maximum legs in parlay
        max_results: Maximum parlays to return

    Returns:
        List of parlay dictionaries with probabilities and odds
    """
    def value_metric(pred):
        p, o = pred[2], pred[3]
        return p * (o - 1) - (1 - p)

    # Sort predictions by value
    preds_sorted = sorted(predictions, key=value_metric, reverse=True)

    parlays = []
    for r in range(min_size, min(max_size, len(preds_sorted)) + 1):
        for combo in itertools.combinations(preds_sorted, r):
            teams = [f"{c[0]} v {c[1]}" for c in combo]
            probs = [c[2] for c in combo]
            odds = [c[3] for c in combo]

            # Calculate combined probability and odds
            combined_prob = float(np.prod(probs))
            combined_odds = float(np.prod(odds))
            expected_return = combined_prob * combined_odds

            parlays.append({
                'legs': teams,
                'size': r,
                'probability': combined_prob,
                'decimal_odds': combined_odds,
                'expected_return': expected_return
            })

    # Sort by expected return and probability
    parlays = sorted(parlays, key=lambda p: (p['expected_return'], p['probability']), reverse=True)
    return parlays[:max_results]


# --- ALGORITHM 10: SMART CACHING ---

def is_data_fresh(last_fetch_file: str, cache_hours: int = 6) -> bool:
    """
    Algorithm 10: Smart Caching with Timestamp Validation

    Prevents excessive API calls by checking data freshness.
    Uses timestamp comparison and graceful degradation.

    Args:
        last_fetch_file: Path to timestamp file
        cache_hours: Cache duration in hours

    Returns:
        True if cached data is still fresh
    """
    if not os.path.exists(last_fetch_file):
        return False

    try:
        with open(last_fetch_file, 'r') as f:
            data = json.load(f)
        last_fetch = datetime.fromisoformat(data['last_fetch'])
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        time_diff = now - last_fetch
        is_fresh = time_diff.total_seconds() < (cache_hours * 3600)

        if is_fresh:
            hours_ago = time_diff.total_seconds() / 3600
            logging.info(f"Using cached data (fetched {hours_ago:.1f}h ago)")
        return is_fresh
    except (json.JSONDecodeError, KeyError, ValueError):
        return False


def update_fetch_timestamp(last_fetch_file: str, cache_hours: int = 6):
    """Update fetch timestamp for caching system."""
    os.makedirs(os.path.dirname(last_fetch_file), exist_ok=True)
    timestamp_data = {
        'last_fetch': datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        'cache_duration_hours': cache_hours
    }
    with open(last_fetch_file, 'w') as f:
        json.dump(timestamp_data, f)
    logging.info("Updated fetch timestamp for caching")


# --- ALGORITHM 11: HEURISTIC CORNER/CARDS ESTIMATION ---

def estimate_corners_and_cards(xg_home: float, xg_away: float) -> Dict[str, float]:
    """
    Algorithm 11: Heuristic Secondary Market Estimation

    Estimates corners and cards from expected goals using industry averages.
    These are rule-of-thumb approximations for markets with limited data.

    Formulas:
    - Total_Corners = max(4.0, total_xG × 3.5)  # ~3.5 corners per goal
    - Corner_Share = xG_share (proportional to attacking threat)
    - Cards = min(6.0, 2.5 + total_xG × 0.8)    # Higher xG = more intensity

    Args:
        xg_home: Expected goals for home team
        xg_away: Expected goals for away team

    Returns:
        Dictionary with corner and card estimates
    """
    total_xg = xg_home + xg_away

    # Corner estimation (3.5 corners per goal, minimum 4)
    est_total_corners = max(4.0, total_xg * 3.5)

    # Split corners by attacking share
    home_corner_share = xg_home / total_xg if total_xg > 0 else 0.5
    away_corner_share = 1.0 - home_corner_share
    est_home_corners = max(1.0, est_total_corners * home_corner_share)
    est_away_corners = max(1.0, est_total_corners * away_corner_share)

    # Card estimation (baseline 2.5, increases with match intensity)
    est_cards = min(6.0, 2.5 + total_xg * 0.8)

    return {
        'TotalCorners': est_total_corners,
        'HomeCorners': est_home_corners,
        'AwayCorners': est_away_corners,
        'EstimatedCards': est_cards
    }


# --- RATING MODEL: GOAL-SUPREMACY DRIVEN PROBABILITIES ---

def compute_goal_supremacy_rating(history_df: pd.DataFrame, team: str, last_n: int = 6) -> float:
    """Compute recent goal-supremacy rating for a team over the last N matches.
    rating_team = sum(goals_for - goals_against) over last N matches the team played.
    Returns 0.0 if no history is available.
    """
    if history_df is None or history_df.empty or not team:
        return 0.0
    df = history_df.copy()
    # Normalize column names (robust to variants)
    rename_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in ('hometeam', 'home_team'):
            rename_map[c] = 'HomeTeam'
        elif lc in ('awayteam', 'away_team'):
            rename_map[c] = 'AwayTeam'
        elif lc in ('fthg', 'homegoals'):
            rename_map[c] = 'FTHG'
        elif lc in ('ftag', 'awaygoals'):
            rename_map[c] = 'FTAG'
        elif lc == 'date' and c != 'Date':
            rename_map[c] = 'Date'
    if rename_map:
        df = df.rename(columns=rename_map)
    if not set(['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']).issubset(df.columns):
        return 0.0

    df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
    df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df = df.sort_values('Date', na_position='last')

    played = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
    if played.empty:
        return 0.0

    last = played.tail(last_n)
    diffs = []
    for _, row in last.iterrows():
        if row['HomeTeam'] == team:
            gf, ga = row['FTHG'], row['FTAG']
        else:
            gf, ga = row['FTAG'], row['FTHG']
        try:
            diffs.append(float(gf) - float(ga))
        except Exception:
            continue
    return float(np.nansum(diffs)) if diffs else 0.0


def match_rating(home_team: str, away_team: str, history_df: pd.DataFrame, last_n: int = 6) -> float:
    """Match rating: r = rating_home − rating_away, where rating_* are goal-supremacy over last N."""
    rh = compute_goal_supremacy_rating(history_df, home_team, last_n=last_n)
    ra = compute_goal_supremacy_rating(history_df, away_team, last_n=last_n)
    return float(rh - ra)


def _build_training_ratings(history_df: pd.DataFrame, last_n: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Construct arrays (r, outcome) from historical matches.
    outcome: 0=Home win, 1=Draw, 2=Away win
    For each historical match, compute r using only matches prior to that match for each team.
    """
    if history_df is None or history_df.empty:
        return np.array([]), np.array([])

    df = history_df.copy()
    # Normalize columns
    rename_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in ('hometeam', 'home_team'):
            rename_map[c] = 'HomeTeam'
        elif lc in ('awayteam', 'away_team'):
            rename_map[c] = 'AwayTeam'
        elif lc in ('fthg', 'homegoals'):
            rename_map[c] = 'FTHG'
        elif lc in ('ftag', 'awaygoals'):
            rename_map[c] = 'FTAG'
        elif lc == 'date' and c != 'Date':
            rename_map[c] = 'Date'
    if rename_map:
        df = df.rename(columns=rename_map)
    if not set(['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']).issubset(df.columns):
        return np.array([]), np.array([])

    df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
    df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)
        df = df.sort_values('Date', na_position='last')
    else:
        df = df.reset_index(drop=True)

    # Build index of matches per team (chronological indexes)
    team_matches: Dict[str, List[int]] = {}
    for idx, row in df.iterrows():
        team_matches.setdefault(str(row['HomeTeam']), []).append(idx)
        team_matches.setdefault(str(row['AwayTeam']), []).append(idx)

    ratings: List[float] = []
    outcomes: List[int] = []

    for idx, row in df.iterrows():
        home = str(row['HomeTeam'])
        away = str(row['AwayTeam'])

        def last_n_diff(team: str) -> float:
            inds = [i for i in team_matches.get(team, []) if i < idx]
            if not inds:
                return 0.0
            recent = inds[-last_n:]
            diffs = []
            for j in recent:
                r = df.iloc[j]
                if r['HomeTeam'] == team:
                    gf, ga = r['FTHG'], r['FTAG']
                else:
                    gf, ga = r['FTAG'], r['FTHG']
                try:
                    diffs.append(float(gf) - float(ga))
                except Exception:
                    continue
            return float(np.nansum(diffs)) if diffs else 0.0

        rh = last_n_diff(home)
        ra = last_n_diff(away)
        r_val = rh - ra
        ratings.append(float(r_val))

        if row['FTHG'] > row['FTAG']:
            outcomes.append(0)
        elif row['FTHG'] == row['FTAG']:
            outcomes.append(1)
        else:
            outcomes.append(2)

    return np.array(ratings, dtype=float), np.array(outcomes, dtype=int)


def fit_rating_to_prob_models(history_df: pd.DataFrame, last_n: int = 6, min_sample_for_rating: int = 30) -> Dict:
    """Fit simple polynomial models mapping rating r -> (P_home, P_draw, P_away).
    - Home: linear
    - Draw: quadratic
    - Away: quadratic
    When insufficient samples, fall back to overall means; store sample size and bounds for diagnostics.
    """
    r, y = _build_training_ratings(history_df, last_n=last_n)
    if r.size == 0 or y.size == 0:
        return {
            'home_poly': None,
            'draw_poly': None,
            'away_poly': None,
            'fallback_probs': {'Home': 0.45, 'Draw': 0.25, 'Away': 0.30},
            'sample_size': 0,
            'min_sample_for_rating': int(min_sample_for_rating),
            'rating_min': 0.0,
            'rating_max': 0.0,
        }

    t_home = (y == 0).astype(float)
    t_draw = (y == 1).astype(float)
    t_away = (y == 2).astype(float)

    def fit_poly(x, t, deg):
        try:
            coefs = np.polyfit(x, t, deg)
            return np.poly1d(coefs)
        except Exception:
            return None

    home_poly = fit_poly(r, t_home, 1)
    draw_poly = fit_poly(r, t_draw, 2)
    away_poly = fit_poly(r, t_away, 2)

    fallback_probs = {
        'Home': float(np.mean(t_home)),
        'Draw': float(np.mean(t_draw)),
        'Away': float(np.mean(t_away)),
    }

    return {
        'home_poly': home_poly,
        'draw_poly': draw_poly,
        'away_poly': away_poly,
        'fallback_probs': fallback_probs,
        'sample_size': int(r.size),
        'min_sample_for_rating': int(min_sample_for_rating),
        'rating_min': float(np.min(r)),
        'rating_max': float(np.max(r)),
    }


def rating_probabilities_from_rating(r: float, models: Dict) -> Tuple[float, float, float]:
    """Evaluate fitted models at rating r; return normalized (p_home, p_draw, p_away).
    Uses fallback means if models are missing or too few samples.
    """
    if not models or models.get('sample_size', 0) < models.get('min_sample_for_rating', 30):
        fb = models.get('fallback_probs', {'Home': 0.45, 'Draw': 0.25, 'Away': 0.30}) if models else {'Home': 0.45, 'Draw': 0.25, 'Away': 0.30}
        s = fb['Home'] + fb['Draw'] + fb['Away']
        return fb['Home']/s, fb['Draw']/s, fb['Away']/s

    def eval_poly(poly, x, mean_val):
        if poly is None:
            return float(mean_val)
        try:
            return float(poly(x))
        except Exception:
            return float(mean_val)

    fb = models.get('fallback_probs', {'Home': 0.45, 'Draw': 0.25, 'Away': 0.30})
    pH = eval_poly(models.get('home_poly'), r, fb['Home'])
    pD = eval_poly(models.get('draw_poly'), r, fb['Draw'])
    pA = eval_poly(models.get('away_poly'), r, fb['Away'])

    # Clamp and normalize
    pH = min(max(pH, 0.001), 0.999)
    pD = min(max(pD, 0.001), 0.999)
    pA = min(max(pA, 0.001), 0.999)
    s = pH + pD + pA
    return pH / s, pD / s, pA / s

# Backwards-compatible alias
rating_probabilities = rating_probabilities_from_rating

# --- UTILITIES ---

def prob_to_decimal_odds(p: float) -> float:
    """Convert probability to decimal odds."""
    return round(1.0 / p, 2) if p > 0 else float('inf')


def format_bet_slip(parlay: Dict, bankroll: float = 100.0) -> Dict:
    """
    Format parlay into human-readable bet slip with Kelly staking.

    Args:
        parlay: Parlay dictionary from generate_parlays()
        bankroll: Total bankroll for stake calculation

    Returns:
        Formatted bet slip dictionary
    """
    p = parlay['probability']
    odds = parlay['decimal_odds']
    f = kelly_fraction(p, odds)
    stake = round(bankroll * f, 2)
    potential_return = round(stake * odds, 2)
    confidence = f"{p*100:.1f}%"

    return {
        'legs': parlay['legs'],
        'size': parlay['size'],
        'probability': p,
        'odds': odds,
        'stake_suggestion': stake,
        'potential_return': potential_return,
        'confidence': confidence
    }


ALGORITHM_INFO = {
    1: "Team Strength Calculation - Normalizes team performance vs league average",
    2: "Expected Goals (xG) Estimation - Multiplicative model with home advantage",
    3: "Poisson Probability Distribution - Mathematical foundation for goal modeling",
    4: "Score Probability Matrix - All possible match score probabilities",
    5: "Market Probability Extraction - Converts scores to betting markets",
    6: "Recent Form Analysis - Exponential decay weighting of recent matches",
    7: "Form-Season Blending - Weighted average of season stats and recent form",
    8: "Kelly Criterion Staking - Optimal bet sizing with conservative caps",
    9: "Parlay Generation - Value-ranked combination bet creation",
    10: "Smart Caching - Rate limiting with graceful degradation",
    11: "Heuristic Estimation - Rule-of-thumb secondary market predictions"
}

def print_algorithm_summary():
    """Print summary of all 11 algorithms."""
    print("🧮 Football Analytics - 11 Core Algorithms")
    print("=" * 60)
    for num, desc in ALGORITHM_INFO.items():
        print(f"{num:2d}. {desc}")
    print(f"\nTotal algorithms: {len(ALGORITHM_INFO)}")

if __name__ == "__main__":
    print_algorithm_summary()
