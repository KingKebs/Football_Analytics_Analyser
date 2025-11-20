"""
Football Analytics Analyser - Multi-league CLI

This script drives the end-to-end workflow:
- Load league table (from football-data/all-euro-football/<LEAGUE>.csv or fallback)
- Compute team strengths
- Optionally blend recent form from historical CSVs in data/old csv/
- Build a single-match suggestion and sample parlays
- Save results to data/

Requires: pandas, numpy
Optional: requests (for scraping fallback, currently disabled)
"""

import os
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import logging

# Helper to parse range filter
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

# Import algorithms module
from algorithms import (
    compute_basic_strengths,
    estimate_xg,
    score_probability_matrix,
    extract_markets_from_score_matrix,
    estimate_corners_and_cards,
    generate_parlays,
    kelly_fraction,
    prob_to_decimal_odds,
    format_bet_slip,
    # rating model helpers
    match_rating,
    fit_rating_to_prob_models,
    rating_probabilities_from_rating,
)

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')

# Paths
DATA_DIR = 'data'
OLD_CSV_SUBFOLDER = 'old csv'
EURO_FOOTBALL_DIR = os.path.join('football-data', 'all-euro-football')
LEAGUE_CACHE_DIR = os.path.join(DATA_DIR, 'cache')
LAST_TEAMS_JSON = os.path.join(DATA_DIR, 'last_teams.json')

# Compatibility helper for new layout
from data_file_utils import get_league_cache_path, get_team_strengths_path, ensure_dirs_for_writing

# Caching
CACHE_DURATION_HOURS = 6

# League mapping (subset; extend as needed)
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


def get_available_leagues() -> Dict[str, Dict[str, str]]:
    available = {}
    if not os.path.exists(EURO_FOOTBALL_DIR):
        return available
    for code, info in LEAGUE_MAPPINGS.items():
        csv_path = os.path.join(EURO_FOOTBALL_DIR, f"{code}.csv")
        if os.path.exists(csv_path):
            available[code] = {**info, 'csv_path': csv_path}
    return available


def display_available_leagues():
    leagues = get_available_leagues()
    if not leagues:
        print('No league data found in football-data/all-euro-football/')
        return
    # Flags
    flags = {
        'England': '🏴', 'Germany': '🇩🇪', 'Spain': '🇪🇸', 'Italy': '🇮🇹', 'France': '🇫🇷',
        'Portugal': '🇵🇹', 'Netherlands': '🇳🇱', 'Scotland': '🏴', 'Belgium': '🇧🇪',
        'Greece': '🇬🇷', 'Turkey': '🇹🇷'
    }
    by_country: Dict[str, List[Tuple[str, Dict[str, str]]]] = {}
    for code, info in leagues.items():
        by_country.setdefault(info['country'], []).append((code, info))
    print('\n🏆 Available European Leagues:')
    print('=' * 60)
    for country in sorted(by_country.keys()):
        flag = flags.get(country, '🏳️')
        print(f"\n{flag} {country.upper()}:")
        for code, info in sorted(by_country[country], key=lambda x: x[1]['tier']):
            tier_str = f"Tier {info['tier']}"
            print(f"  {code:<4} - {info['name']:<35} ({tier_str})")


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
    """Aggregate football-data match-level CSV into a league table (Team, P, F, A)."""
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
    # If it's already a league table, return as-is
    return df_raw


def load_league_table(league_code: str) -> pd.DataFrame:
    """Load league table for selected league from all-euro-football directory, aggregating if needed.
       Caches aggregated table to data/league_data_<code>.csv
    """
    # Ensure new dirs exist when writing cache
    ensure_dirs_for_writing()
    cache_path = get_league_cache_path(league_code)
    if os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path), tz=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600
        if age_hours < CACHE_DURATION_HOURS:
            logging.info(f"Loading cached league table: {cache_path} ({age_hours:.1f}h old)")
            return pd.read_csv(cache_path)
    # Aggregate from source CSV
    src = get_league_csv(league_code)
    info = get_league_info(league_code)
    logging.info(f"Loading {info['name']} from {src}")
    agg = aggregate_football_data(src)
    agg.to_csv(cache_path, index=False)
    logging.info(f"Saved aggregated league table to {cache_path}")
    return agg


def load_historical_matches(data_dir: str = DATA_DIR, subfolder: str = OLD_CSV_SUBFOLDER) -> pd.DataFrame:
    folder = os.path.join(data_dir, subfolder)
    if not os.path.isdir(folder):
        return pd.DataFrame()
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.csv')]
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


def build_single_match_suggestion(home_team: str, away_team: str, strengths_df: pd.DataFrame, min_confidence: float = 0.6, rating_models: Dict = None, history_df: pd.DataFrame = None, rating_model_config: Dict = None) -> Dict:
    # Compute xG and score matrix first (needed for Poisson probabilities and markets)
    xg_home, xg_away = estimate_xg(home_team, away_team, strengths_df)
    mat = score_probability_matrix(xg_home, xg_away, max_goals=6)

    external_probs = None
    if rating_models and rating_model_config and rating_model_config.get('model', 'none') != 'none':
        last_n = int(rating_model_config.get('last_n', 6))
        try:
            r = match_rating(home_team, away_team, history_df, last_n)
            # range filter guardrail
            rng = rating_model_config.get('range_filter')
            if rng and not (rng[0] < r <= rng[1]):
                r = None
            if r is not None:
                p_home_rating, p_draw_rating, p_away_rating = rating_probabilities_from_rating(r, rating_models)
            else:
                p_home_rating = p_draw_rating = p_away_rating = None
        except Exception:
            p_home_rating = p_draw_rating = p_away_rating = None

        if p_home_rating is not None:
            if rating_model_config.get('model') == 'goal_supremacy':
                # Use pure rating-based 1X2
                external_probs = {'1X2': {'Home': p_home_rating, 'Draw': p_draw_rating, 'Away': p_away_rating}}
            elif rating_model_config.get('model') == 'blended':
                # Blend Poisson and rating-based 1X2 using weight
                poisson_all = extract_markets_from_score_matrix(mat, min_confidence=0.0)
                poisson_1x2 = poisson_all.get('1X2', {})
                blend_weight = float(rating_model_config.get('blend_weight', 0.3))
                # defaults if poisson missing
                pH_pois = float(poisson_1x2.get('Home', 0.0))
                pD_pois = float(poisson_1x2.get('Draw', 0.0))
                pA_pois = float(poisson_1x2.get('Away', 0.0))
                blended_home = (1 - blend_weight) * pH_pois + blend_weight * p_home_rating
                blended_draw = (1 - blend_weight) * pD_pois + blend_weight * p_draw_rating
                blended_away = (1 - blend_weight) * pA_pois + blend_weight * p_away_rating
                total = blended_home + blended_draw + blended_away
                if total > 0:
                    external_probs = {'1X2': {'Home': blended_home / total, 'Draw': blended_draw / total, 'Away': blended_away / total}}

    markets = extract_markets_from_score_matrix(mat, min_confidence=min_confidence, external_probs=external_probs)
    corners = estimate_corners_and_cards(xg_home, xg_away)

    picks = []
    one_x_two = markets.get('1X2', {})
    if one_x_two:
        best_1x2 = max(one_x_two.items(), key=lambda x: x[1])
        picks.append({'market': '1X2', 'selection': best_1x2[0], 'prob': float(best_1x2[1]), 'odds': prob_to_decimal_odds(float(best_1x2[1]))})

    btts = markets.get('BTTS', {})
    btts_no = float(btts.get('No', 0))
    btts_yes = float(btts.get('Yes', 0))
    if btts_no > 0.6:
        picks.append({'market': 'BTTS', 'selection': 'No', 'prob': btts_no, 'odds': prob_to_decimal_odds(btts_no)})
    elif btts_yes > 0.6:
        picks.append({'market': 'BTTS', 'selection': 'Yes', 'prob': btts_yes, 'odds': prob_to_decimal_odds(btts_yes)})

    ou = markets.get('OU', {})
    ou_under = float(ou.get('Under2.5', 0))
    ou_over = float(ou.get('Over2.5', 0))
    if ou_under > 0.7:
        picks.append({'market': 'Over/Under 2.5', 'selection': 'Under2.5', 'prob': ou_under, 'odds': prob_to_decimal_odds(ou_under)})
    elif ou_over > 0.7:
        picks.append({'market': 'Over/Under 2.5', 'selection': 'Over2.5', 'prob': ou_over, 'odds': prob_to_decimal_odds(ou_over)})

    suggestion = {
        'home': home_team,
        'away': away_team,
        'xg_home': float(xg_home),
        'xg_away': float(xg_away),
        'markets': markets,
        'corners_cards': corners,
        'picks': picks,
        'score_matrix': mat.to_dict()
    }
    return suggestion


def main_interactive(bankroll: float = 100.0, league_code: str = None, rating_model: str = 'none', rating_last_n: int = 6, min_sample_for_rating: int = 30, rating_blend_weight: float = 0.3, min_confidence: float = 0.6):
    logging.info("Starting analytics workflow")
    logging.info(f"Bankroll set to {bankroll}")

    if not league_code:
        display_available_leagues()
        print("\n🎯 League Selection:")
        print("Enter a league code (e.g., E0 for EPL, D1 for Bundesliga, SP1 for La Liga)")
        print("Or press Enter for default English Premier League (E0)")
        user_input = input("\nSelect league: ").strip().upper()
        league_code = user_input if user_input else 'E0'
        if league_code not in LEAGUE_MAPPINGS:
            print(f"❌ Invalid league code: {league_code}")
            display_available_leagues()
            return
        info = get_league_info(league_code)
        print(f"\n✅ Selected: {info['name']} ({info['country']})")

    try:
        league_df = load_league_table(league_code=league_code)
    except Exception as e:
        logging.error(f"Failed to load league table for {league_code}: {e}")
        return

    strengths_df = compute_basic_strengths(league_df)

    history_df = load_historical_matches()
    if not history_df.empty:
        from algorithms import merge_form_into_strengths  # lazy import to avoid cycles
        strengths_df = merge_form_into_strengths(strengths_df, history_df, last_n=6, decay=0.6, alpha=0.4)
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            strengths_path = get_team_strengths_path(league_code)
            strengths_df.to_csv(strengths_path, index=False)
            logging.info(f"Updated strengths with recent form saved to {strengths_path}")
        except Exception:
            logging.warning("Could not save updated strengths CSV")
    else:
        logging.info("No historical match files found; using season strengths only.")

    # Fit rating models if requested and history available
    rating_models = None
    if rating_model and rating_model != 'none':
        if not history_df.empty:
            try:
                logging.info("Fitting rating-to-probability models from historical data...")
                rating_models = fit_rating_to_prob_models(history_df, last_n=rating_last_n, min_sample_for_rating=min_sample_for_rating)
                logging.info(f"Rating model fitted (samples={rating_models.get('sample_size', 0)})")
            except Exception as e:
                logging.warning(f"Failed to fit rating models: {e}")
        else:
            logging.warning("Rating model requested, but no historical match files found. Using season strengths only.")

    teams = list(strengths_df['Team'])
    logging.info(f"Teams available: {len(teams)}")
    print("Available teams (sample):")
    print('\n'.join(teams[:30]))
    home = input('Enter HOME_TEAM: ').strip()
    away = input('Enter AWAY_TEAM: ').strip()
    with open(LAST_TEAMS_JSON, 'w') as f:
        json.dump({'home_team': home, 'away_team': away, 'league': league_code}, f)

    rating_model_config = {
        'model': rating_model,
        'last_n': rating_last_n,
        'blend_weight': rating_blend_weight,
        'range_filter': rng,
    }

    suggestion = build_single_match_suggestion(home, away, strengths_df, min_confidence=min_confidence, rating_models=rating_models, history_df=history_df, rating_model_config=rating_model_config)

    print('\nMatch suggestion: {} v {}'.format(home, away))
    print('Estimated xG -> {}: {:.2f}, {}: {:.2f}'.format(home, suggestion['xg_home'], away, suggestion['xg_away']))
    print('\nTop market probabilities:')
    for mtype, vals in suggestion['markets'].items():
        print(f"  {mtype}: ")
        for sel, p in vals.items():
            print(f"    {sel}: {p*100:.1f}% (odds ~ {prob_to_decimal_odds(p)})")
    print('\nHeuristic corners/cards:')
    for k, v in suggestion['corners_cards'].items():
        print(f"  {k}: {v:.2f}")

    print('\nSuggested single-match picks:')
    predictions = []
    for pick in suggestion['picks']:
        print(f"  {pick['market']} - {pick['selection']}: {pick['prob']*100:.1f}% (odds {pick['odds']})")
        predictions.append((home, away, pick['prob'], pick['odds']))

    logging.info("Generating sample parlays")
    sample_preds = predictions.copy()
    for i in range(0, min(6, len(teams)-1), 2):
        extra_home = teams[i]
        extra_away = teams[i+1]
        s = build_single_match_suggestion(extra_home, extra_away, strengths_df, min_confidence=min_confidence, rating_models=rating_models, history_df=history_df, rating_model_config=rating_model_config)
        best = max(s['markets'].get('1X2', {}).items(), key=lambda x: x[1]) if s['markets'].get('1X2') else None
        if best:
            sample_preds.append((extra_home, extra_away, best[1], prob_to_decimal_odds(best[1])))

    parlays = generate_parlays(sample_preds, min_size=2, max_size=3, max_results=10)
    print('\nTop parlays:')
    for p in parlays:
        slip = format_bet_slip(p, bankroll=bankroll)
        print(f"- Legs ({p['size']}): {p['legs']}")
        print(f"  Prob: {p['probability']*100:.2f}%, Odds: {p['decimal_odds']:.2f}, Suggested stake: {slip['stake_suggestion']}, Potential return: {slip['potential_return']}")

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    os.makedirs(DATA_DIR, exist_ok=True)
    suggestion_file = os.path.join(DATA_DIR, f'suggestion_{home}_{away}_{timestamp}.json')
    with open(suggestion_file, 'w') as f:
        json.dump(suggestion, f, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    logging.info(f"Saved suggestion JSON to: {suggestion_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Football analytics assistant (multi-league)')
    parser.add_argument('--bankroll', type=float, default=100.0, help='Estimated bankroll for stake suggestions')
    parser.add_argument('--league', type=str, help='League code (e.g., E0, D1, SP1). If omitted, interactive select.')
    parser.add_argument('--rating-model', type=str, default='none', choices=['none', 'goal_supremacy', 'blended'], help='Rating model for 1X2 probabilities')
    parser.add_argument('--rating-last-n', type=int, default=6, help='Number of recent matches for goal supremacy rating')
    parser.add_argument('--min-sample-for-rating', type=int, default=30, help='Minimum sample size to fit rating models')
    parser.add_argument('--rating-blend-weight', type=float, default=0.3, help='Blend weight when using blended model (0..1)')
    parser.add_argument('--rating-range-filter', type=str, help='Comma-separated rating range (lo,hi); skip rating override when r is outside')
    parser.add_argument('--min-confidence', type=float, default=0.6, help='Minimum market probability to surface')
    args = parser.parse_args()
    rng = _parse_range_filter(args.rating_range_filter)
    main_interactive(bankroll=args.bankroll, league_code=args.league, rating_model=args.rating_model, rating_last_n=args.rating_last_n, min_sample_for_rating=args.min_sample_for_rating, rating_blend_weight=args.rating_blend_weight, min_confidence=args.min_confidence)
