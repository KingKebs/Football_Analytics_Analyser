import pandas as pd
import numpy as np
import pytest

from ml_features import engineer_features, TRAIN_FEATURE_COLUMNS, build_match_feature_row
from ml_utils import build_recency_weights, check_min_samples
from ml_training import train_models, predict_match
from ml_evaluation import evaluate_vs_poisson
from algorithms import score_probability_matrix, extract_markets_from_score_matrix


def synthetic_history(rows: int = 400):
    rng = np.random.default_rng(42)
    dates = pd.date_range('2024-08-01', periods=rows, freq='D')
    teams = ['TeamA','TeamB','TeamC','TeamD']
    data = []
    for i in range(rows):
        home = teams[rng.integers(0, len(teams))]
        away = teams[rng.integers(0, len(teams))]
        while away == home:
            away = teams[rng.integers(0, len(teams))]
        fthg = rng.integers(0,5)
        ftag = rng.integers(0,5)
        row = {
            'Date': dates[i],
            'HomeTeam': home,
            'AwayTeam': away,
            'FTHG': fthg,
            'FTAG': ftag,
            # basic shot / fouls / corners proxies
            'HS': rng.integers(1,20),
            'AS': rng.integers(1,20),
            'HST': rng.integers(0,10),
            'AST': rng.integers(0,10),
            'HF': rng.integers(5,20),
            'AF': rng.integers(5,20),
            'HC': rng.integers(0,10),
            'AC': rng.integers(0,10),
        }
        data.append(row)
    return pd.DataFrame(data)


def test_feature_engineering_shapes():
    hist = synthetic_history(200)
    feats = engineer_features(hist)
    assert not feats.empty
    # all training feature columns present
    for col in TRAIN_FEATURE_COLUMNS:
        assert col in feats.columns
    # targets present
    for col in ['TotalGoals','HomeWin','Draw','AwayWin','BTTS']:
        assert col in feats.columns


def test_recency_weights_sum_to_one():
    hist = synthetic_history(100)
    feats = engineer_features(hist)
    w = build_recency_weights(feats['Date'].values, decay=0.9)
    assert pytest.approx(w.sum(), 1e-6) == 1.0
    # Newest date gets largest weight
    assert w[-1] == max(w)


def test_train_models_rf_only():
    hist = synthetic_history(350)
    feats = engineer_features(hist)
    w = build_recency_weights(feats['Date'].values, decay=0.85)
    assert check_min_samples(feats, 300)
    models = train_models(feats, weights=w, algorithms=['rf'], cv_folds=3)
    assert 'regression' in models and 'classification' in models
    assert 'TotalGoals_RF' in models['regression']
    assert '1X2_RF' in models['classification']
    assert 'BTTS_RF' in models['classification']
    cv = models['cv_metrics']
    assert 'TotalGoals_RF' in cv
    assert cv['TotalGoals_RF']['MAE'] > 0


def test_predict_single_match_and_evaluate():
    hist = synthetic_history(320)
    feats = engineer_features(hist)
    w = build_recency_weights(feats['Date'].values, decay=0.85)
    models = train_models(feats, weights=w, algorithms=['rf'], cv_folds=3)
    # pick last two teams
    last_row = feats.iloc[-1]
    home, away = last_row['HomeTeam'], last_row['AwayTeam']
    feat_row_dict = build_match_feature_row(feats, home, away)
    feature_vector = np.array([feat_row_dict[c] for c in TRAIN_FEATURE_COLUMNS]).reshape(1,-1)
    pred = predict_match(models, feature_vector)
    assert 'pred_total_goals' in pred
    # baseline from arbitrary xG (use simple average goals for both teams)
    # Use mean lambda for demonstration
    xg_h = feats['FTHG'].mean()/feats['HomeTeam'].nunique()
    xg_a = feats['FTAG'].mean()/feats['AwayTeam'].nunique()
    mat = score_probability_matrix(max(xg_h,0.5), max(xg_a,0.5), max_goals=6)
    markets_full = extract_markets_from_score_matrix(mat, min_confidence=0.0)
    comp = evaluate_vs_poisson(markets_full, pred)
    assert '1X2' in comp and 'BTTS' in comp


def test_insufficient_samples_gate():
    hist = synthetic_history(50)  # too small
    feats = engineer_features(hist)
    assert not check_min_samples(feats, 300)

