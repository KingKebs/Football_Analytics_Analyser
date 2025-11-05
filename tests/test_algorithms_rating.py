import pandas as pd
import numpy as np

from algorithms import (
    compute_goal_supremacy_rating,
    match_rating,
    fit_rating_to_prob_models,
    rating_probabilities_from_rating,
)

def make_history():
    # Minimal synthetic history for two teams A and B
    data = [
        {"Date": "2024-01-01", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 2, "FTAG": 0},  # A +2
        {"Date": "2024-01-08", "HomeTeam": "B", "AwayTeam": "A", "FTHG": 1, "FTAG": 1},  # draw, A 0
        {"Date": "2024-01-15", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 0, "FTAG": 1},  # A -1
        {"Date": "2024-01-22", "HomeTeam": "B", "AwayTeam": "A", "FTHG": 0, "FTAG": 3},  # A +3
    ]
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def test_goal_supremacy_rating_basic():
    history = make_history()
    rA = compute_goal_supremacy_rating(history, 'A', last_n=3)
    rB = compute_goal_supremacy_rating(history, 'B', last_n=3)
    # Last 3 A matches (after first): draw(0), -1, +3 => total +2
    assert isinstance(rA, float)
    assert np.isfinite(rA)
    assert rA == 2.0
    # Symmetry: B should be -2
    assert rB == -2.0

def test_match_rating_diff():
    history = make_history()
    r = match_rating('A', 'B', history, last_n=3)
    assert r == 4.0  # A(2) - B(-2)

def test_fit_and_eval_models():
    history = make_history()
    models = fit_rating_to_prob_models(history, last_n=3, min_sample_for_rating=30)
    # With tiny history, models.sample_size < min_sample_for_rating, fallback is used
    assert 'sample_size' in models
    pH, pD, pA = rating_probabilities_from_rating(0.0, models)
    s = pH + pD + pA
    assert np.isclose(s, 1.0)
    assert 0 < pH < 1 and 0 < pD < 1 and 0 < pA < 1

