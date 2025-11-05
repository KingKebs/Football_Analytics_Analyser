import pandas as pd
import numpy as np

from algorithms import compute_basic_strengths
from automate_football_analytics import build_single_match_suggestion

def test_blended_1x2_uses_external_probs():
    # Minimal strengths for two teams X and Y
    df = pd.DataFrame([
        {'Team': 'X', 'P': 10, 'F': 15, 'A': 10},
        {'Team': 'Y', 'P': 10, 'F': 8,  'A': 14},
    ])
    strengths = compute_basic_strengths(df)
    # Fake history to satisfy rating model call; not used in this blend test
    history = pd.DataFrame({'Date': [], 'HomeTeam': [], 'AwayTeam': [], 'FTHG': [], 'FTAG': []})

    # Inject rating_models with fallback so rating_probabilities_from_rating returns fallback probs
    rating_models = {'fallback_probs': {'Home': 0.55, 'Draw': 0.25, 'Away': 0.20}, 'sample_size': 0, 'min_sample_for_rating': 30}
    cfg = {'model': 'blended', 'last_n': 6, 'blend_weight': 1.0}  # weight 1 means pure rating

    s = build_single_match_suggestion('X', 'Y', strengths, min_confidence=0.0, rating_models=rating_models, history_df=history, rating_model_config=cfg)
    one = s['markets']['1X2']
    # With weight 1.0, expect near fallback ratios after normalization
    total = one['Home'] + one['Draw'] + one['Away']
    assert np.isclose(total, 1.0)
    assert one['Home'] > one['Away']

