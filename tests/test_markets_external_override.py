import numpy as np
import pandas as pd

from algorithms import score_probability_matrix, extract_markets_from_score_matrix

def test_external_1x2_override_normalizes():
    mat = score_probability_matrix(1.2, 0.8, max_goals=4)
    ext = {'1X2': {'Home': 0.9, 'Draw': 0.2, 'Away': 0.2}}  # sums > 1, will be normalized
    markets = extract_markets_from_score_matrix(mat, min_confidence=0.0, external_probs=ext)
    one = markets['1X2']
    s = one['Home'] + one['Draw'] + one['Away']
    assert np.isclose(s, 1.0)
    assert one['Home'] > one['Away']

