"""Training routines for goal-market ML models.

Models:
- TotalGoals: regression (RandomForestRegressor, optional XGBRegressor)
- 1X2: classification (RandomForestClassifier, optional XGBClassifier)
- BTTS: classification (RandomForestClassifier, optional XGBClassifier)

Returned objects include fitted primary model(s) and cross-validation metrics.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, log_loss

from ml_utils import safe_import_xgboost
from ml_features import TRAIN_FEATURE_COLUMNS


def _prepare_arrays(df: pd.DataFrame):
    X = df[TRAIN_FEATURE_COLUMNS].values.astype(float)
    y_goals = df['TotalGoals'].values.astype(float)
    # For 1X2 classification choose label order: HomeWin=0, Draw=1, AwayWin=2
    y_1x2 = np.argmax(df[['HomeWin','Draw','AwayWin']].values, axis=1)
    y_btts = df['BTTS'].values.astype(int)
    return X, y_goals, y_1x2, y_btts


def train_models(df: pd.DataFrame, weights: np.ndarray, algorithms: List[str] = None, cv_folds: int = 5, random_state: int = 42) -> Dict:
    if algorithms is None:
        algorithms = ['rf','xgb']
    X, y_goals, y_1x2, y_btts = _prepare_arrays(df)
    results = {
        'regression': {},
        'classification': {},
        'cv_metrics': {},
        'meta': {'samples': len(df)}
    }

    use_xgb = 'xgb' in algorithms
    xgb_mod = safe_import_xgboost() if use_xgb else None
    if use_xgb and xgb_mod is None:
        use_xgb = False

    # Regression: Total Goals
    rf_reg = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=random_state, n_jobs=-1)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    mae_scores = []; rmse_scores = []
    for tr, te in kf.split(X):
        rf_reg.fit(X[tr], y_goals[tr], sample_weight=None if weights is None else weights[tr])
        pred = rf_reg.predict(X[te])
        mae_scores.append(mean_absolute_error(y_goals[te], pred))
        mse = mean_squared_error(y_goals[te], pred)
        rmse_scores.append(float(np.sqrt(mse)))
    results['cv_metrics']['TotalGoals_RF'] = {
        'MAE': float(np.mean(mae_scores)), 'RMSE': float(np.mean(rmse_scores))
    }
    results['regression']['TotalGoals_RF'] = rf_reg

    if use_xgb:
        try:
            xgb_reg = xgb_mod.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=random_state, n_jobs=-1)
            mae_scores = []; rmse_scores = []
            for tr, te in kf.split(X):
                xgb_reg.fit(X[tr], y_goals[tr], sample_weight=None if weights is None else weights[tr])
                pred = xgb_reg.predict(X[te])
                mae_scores.append(mean_absolute_error(y_goals[te], pred))
                mse = mean_squared_error(y_goals[te], pred)
                rmse_scores.append(float(np.sqrt(mse)))
            results['cv_metrics']['TotalGoals_XGB'] = {
                'MAE': float(np.mean(mae_scores)), 'RMSE': float(np.mean(rmse_scores))
            }
            results['regression']['TotalGoals_XGB'] = xgb_reg
        except Exception as e:
            logging.warning(f"XGB regression training failed: {e}")

    # Classification: 1X2
    rf_cls_1x2 = RandomForestClassifier(n_estimators=400, max_depth=12, random_state=random_state, n_jobs=-1)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    acc_scores = []; ll_scores = []
    for tr, te in skf.split(X, y_1x2):
        rf_cls_1x2.fit(X[tr], y_1x2[tr], sample_weight=None if weights is None else weights[tr])
        prob = rf_cls_1x2.predict_proba(X[te])
        pred = np.argmax(prob, axis=1)
        acc_scores.append(np.mean(pred == y_1x2[te]))
        # Log loss requires all classes present; guard
        try:
            ll_scores.append(log_loss(y_1x2[te], prob, labels=[0,1,2]))
        except ValueError:
            pass
    results['cv_metrics']['1X2_RF'] = {
        'Accuracy': float(np.mean(acc_scores)), 'LogLoss': float(np.mean(ll_scores)) if ll_scores else None
    }
    results['classification']['1X2_RF'] = rf_cls_1x2

    if use_xgb:
        try:
            xgb_cls_1x2 = xgb_mod.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=random_state, n_jobs=-1, eval_metric='mlogloss')
            acc_scores = []; ll_scores = []
            for tr, te in skf.split(X, y_1x2):
                xgb_cls_1x2.fit(X[tr], y_1x2[tr], sample_weight=None if weights is None else weights[tr])
                prob = xgb_cls_1x2.predict_proba(X[te])
                pred = np.argmax(prob, axis=1)
                acc_scores.append(np.mean(pred == y_1x2[te]))
                try:
                    ll_scores.append(log_loss(y_1x2[te], prob, labels=[0,1,2]))
                except ValueError:
                    pass
            results['cv_metrics']['1X2_XGB'] = {
                'Accuracy': float(np.mean(acc_scores)), 'LogLoss': float(np.mean(ll_scores)) if ll_scores else None
            }
            results['classification']['1X2_XGB'] = xgb_cls_1x2
        except Exception as e:
            logging.warning(f"XGB 1X2 training failed: {e}")

    # Classification: BTTS
    rf_cls_btts = RandomForestClassifier(n_estimators=400, max_depth=10, random_state=random_state, n_jobs=-1)
    skf2 = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    acc_scores = []; ll_scores = []
    for tr, te in skf2.split(X, y_btts):
        rf_cls_btts.fit(X[tr], y_btts[tr], sample_weight=None if weights is None else weights[tr])
        prob = rf_cls_btts.predict_proba(X[te])
        pred = np.argmax(prob, axis=1)
        acc_scores.append(np.mean(pred == y_btts[te]))
        try:
            ll_scores.append(log_loss(y_btts[te], prob, labels=[0,1]))
        except ValueError:
            pass
    results['cv_metrics']['BTTS_RF'] = {
        'Accuracy': float(np.mean(acc_scores)), 'LogLoss': float(np.mean(ll_scores)) if ll_scores else None
    }
    results['classification']['BTTS_RF'] = rf_cls_btts

    if use_xgb:
        try:
            xgb_cls_btts = xgb_mod.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=random_state, n_jobs=-1, eval_metric='logloss')
            acc_scores = []; ll_scores = []
            for tr, te in skf2.split(X, y_btts):
                xgb_cls_btts.fit(X[tr], y_btts[tr], sample_weight=None if weights is None else weights[tr])
                prob = xgb_cls_btts.predict_proba(X[te])
                pred = np.argmax(prob, axis=1)
                acc_scores.append(np.mean(pred == y_btts[te]))
                try:
                    ll_scores.append(log_loss(y_btts[te], prob, labels=[0,1]))
                except ValueError:
                    pass
            results['cv_metrics']['BTTS_XGB'] = {
                'Accuracy': float(np.mean(acc_scores)), 'LogLoss': float(np.mean(ll_scores)) if ll_scores else None
            }
            results['classification']['BTTS_XGB'] = xgb_cls_btts
        except Exception as e:
            logging.warning(f"XGB BTTS training failed: {e}")

    return results


def predict_match(models: Dict, feature_row: np.ndarray) -> Dict[str, float]:
    out = {}
    # Regression models
    reg_models = models.get('regression', {})
    if 'TotalGoals_XGB' in reg_models:
        out['pred_total_goals'] = float(reg_models['TotalGoals_XGB'].predict(feature_row)[0])
        out['pred_total_goals_model'] = 'XGB'
    elif 'TotalGoals_RF' in reg_models:
        out['pred_total_goals'] = float(reg_models['TotalGoals_RF'].predict(feature_row)[0])
        out['pred_total_goals_model'] = 'RF'
    # 1X2 classification
    cls_models = models.get('classification', {})
    if '1X2_XGB' in cls_models:
        probs = cls_models['1X2_XGB'].predict_proba(feature_row)[0]
        out['prob_1x2_home'] = float(probs[0]); out['prob_1x2_draw'] = float(probs[1]); out['prob_1x2_away'] = float(probs[2]); out['model_1x2'] = 'XGB'
    elif '1X2_RF' in cls_models:
        probs = cls_models['1X2_RF'].predict_proba(feature_row)[0]
        out['prob_1x2_home'] = float(probs[0]); out['prob_1x2_draw'] = float(probs[1]); out['prob_1x2_away'] = float(probs[2]); out['model_1x2'] = 'RF'
    # BTTS
    if 'BTTS_XGB' in cls_models:
        probs = cls_models['BTTS_XGB'].predict_proba(feature_row)[0]
        out['prob_btts_no'] = float(probs[0]); out['prob_btts_yes'] = float(probs[1]); out['model_btts'] = 'XGB'
    elif 'BTTS_RF' in cls_models:
        probs = cls_models['BTTS_RF'].predict_proba(feature_row)[0]
        out['prob_btts_no'] = float(probs[0]); out['prob_btts_yes'] = float(probs[1]); out['model_btts'] = 'RF'
    return out
