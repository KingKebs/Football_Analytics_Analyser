#!/usr/bin/env python3
"""Unified training pipeline implementing Section 2.5 Training Procedure.

Steps:
 1. Load curated / raw league CSVs per config (football-data/<LEAGUE>_<SEASON>.csv or fallback <LEAGUE>.csv)
 2. Concatenate, clean, feature engineer
 3. Time-based split (train slice / validation slice by dates)
 4. Train models (Total Goals regression, 1X2, BTTS) with optional XGB
 5. Evaluate metrics on validation slice (MAE, RMSE, log loss, accuracy)
 6. Probability calibration (BTTS & 1X2) via isotonic / Platt if miscalibrated (basic ECE approximation)
 7. Persist artifacts: model.pkl, metrics.json, calibration_curve.json, feature_manifest.json
 8. Embed data snapshot / git hash.

Usage:
  python3 train_pipeline.py --config config/train_config.yaml
"""
import argparse
import json
import os
import sys
import math
from pathlib import Path
from datetime import datetime
import hashlib
import logging
from typing import Dict, Any

import pandas as pd
import numpy as np

from ml_features import engineer_features, TRAIN_FEATURE_COLUMNS
from ml_training import train_models
from ml_utils import build_recency_weights, save_models

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
except Exception:
    IsotonicRegression = None
    LogisticRegression = None

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')


def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def read_league_csv(league: str, seasons: list) -> pd.DataFrame:
    rows = []
    base_dir = Path('football-data')
    for season in seasons:
        fname = f"{league}_{season}.csv"
        path = base_dir / fname
        if path.exists():
            try:
                df = pd.read_csv(path, low_memory=False)
                df['__season__'] = season
                rows.append(df)
                logging.info(f"Loaded {fname} rows={len(df)}")
            except Exception as e:
                logging.warning(f"Failed reading {fname}: {e}")
    # Fallback base league file if seasons list empty or no season files
    if not rows:
        base_path = base_dir / f"{league}.csv"
        if base_path.exists():
            df = pd.read_csv(base_path, low_memory=False)
            df['__season__'] = 'unknown'
            rows.append(df)
            logging.info(f"Loaded fallback {base_path.name} rows={len(df)}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def curate_raw(df: pd.DataFrame) -> pd.DataFrame:
    # Minimal cleaning: drop rows without essential columns, coerce numeric
    needed = ['Date','HomeTeam','AwayTeam','FTHG','FTAG']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date','HomeTeam','AwayTeam'])
    for col in ['FTHG','FTAG']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def approximate_ece(y_true_probs: np.ndarray, y_pred_probs: np.ndarray, bins: int = 10) -> float:
    # y_true_probs: binary outcome 0/1 or encoded correct class indicator
    # y_pred_probs: predicted probability of that outcome
    if len(y_true_probs) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins+1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        mask = (y_pred_probs >= lo) & (y_pred_probs < hi)
        if not np.any(mask):
            continue
        avg_conf = y_pred_probs[mask].mean()
        avg_acc = y_true_probs[mask].mean()
        ece += (abs(avg_conf - avg_acc) * (mask.sum() / len(y_true_probs)))
    return float(ece)


def calibrate_probs(probs: np.ndarray, labels: np.ndarray, method: str = 'auto'):
    # Returns calibrated probabilities and calibration curve data
    if method == 'platt' or (method == 'auto' and LogisticRegression is not None and len(np.unique(labels)) == 2):
        lr = LogisticRegression(max_iter=1000)
        lr.fit(probs.reshape(-1,1), labels)
        cal = lr.predict_proba(probs.reshape(-1,1))[:,1]
        return cal, {'method':'platt'}
    if method == 'isotonic' or (method == 'auto' and IsotonicRegression is not None):
        try:
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(probs, labels)
            cal = iso.transform(probs)
            return cal, {'method':'isotonic'}
        except Exception:
            pass
    # fallback: no change
    return probs, {'method':'none'}


def hash_dataframe(df: pd.DataFrame) -> str:
    # Stable hash of essential columns
    h = hashlib.sha256()
    subset = df[['Date','HomeTeam','AwayTeam','FTHG','FTAG']].copy()
    subset['Date'] = subset['Date'].astype(str)
    for row in subset.itertuples(index=False):
        h.update(('|'.join(map(str,row))+'\n').encode('utf-8'))
    return h.hexdigest()


def main(config_path: str):
    cfg = load_config(config_path)
    train_cfg = cfg.get('train', {})
    leagues = train_cfg.get('leagues', ['E0'])
    seasons = train_cfg.get('seasons', [])
    rolling_window = int(train_cfg.get('rolling_window', 6))
    min_samples = int(train_cfg.get('min_samples', 300))
    algorithms = train_cfg.get('algorithms', ['rf','xgb'])
    train_until = pd.to_datetime(train_cfg.get('train_until')) if train_cfg.get('train_until') else None
    val_from = pd.to_datetime(train_cfg.get('validation_from')) if train_cfg.get('validation_from') else None
    fast_mode = bool(train_cfg.get('fast_mode', False))

    calib_cfg = cfg.get('calibration', {})
    calib_enable = bool(calib_cfg.get('enable', True))
    ece_threshold = float(calib_cfg.get('ece_threshold', 0.03))
    bins = int(calib_cfg.get('bins', 10))
    method_pref = calib_cfg.get('method_preference', 'auto')

    out_cfg = cfg.get('output', {})
    artifacts_dir = Path(out_cfg.get('artifacts_dir','models'))
    tag = out_cfg.get('tag','run')
    save_calibrated = bool(out_cfg.get('save_calibrated', True))

    meta_cfg = cfg.get('meta', {})
    seed = int(meta_cfg.get('seed', 42))
    np.random.seed(seed)

    # Aggregate multi-league historical data
    hist_frames = []
    for lg in leagues:
        df = read_league_csv(lg, seasons)
        if not df.empty:
            df['__league__'] = lg
            hist_frames.append(df)
    if not hist_frames:
        logging.error("No historical data loaded; aborting")
        return 1
    raw_hist = pd.concat(hist_frames, ignore_index=True)
    curated = curate_raw(raw_hist)
    curated_hash = hash_dataframe(curated)
    logging.info(f"Curated rows: {len(curated)} hash={curated_hash[:12]}")

    features_df = engineer_features(curated, rolling_window=rolling_window)
    if len(features_df) < min_samples:
        logging.error(f"Not enough samples ({len(features_df)}) < min required {min_samples}")
        return 1

    # Time split
    if train_until is None or val_from is None:
        # simple 80/20 chronological
        cutoff_idx = int(len(features_df)*0.8)
        train_df = features_df.iloc[:cutoff_idx]
        val_df = features_df.iloc[cutoff_idx:]
        logging.info(f"Using 80/20 chronological split train={len(train_df)} val={len(val_df)}")
    else:
        train_df = features_df[features_df['Date'] <= train_until]
        val_df = features_df[features_df['Date'] >= val_from]
        logging.info(f"Time split train_until={train_until.date()} validation_from={val_from.date()} train={len(train_df)} val={len(val_df)}")

    # Recency weights (train slice only)
    weights = build_recency_weights(train_df['Date'].values, decay=0.85)

    # Optionally reduce estimators for fast_mode (we rely on ml_training defaults; could implement variant)
    models = train_models(train_df, weights=weights, algorithms=algorithms)

    # Evaluate on validation slice
    val_X = val_df[TRAIN_FEATURE_COLUMNS].values.astype(float)
    goals_true = val_df['TotalGoals'].values.astype(float)
    home_true = val_df['HomeWin'].values.astype(int)
    draw_true = val_df['Draw'].values.astype(int)
    away_true = val_df['AwayWin'].values.astype(int)
    btts_true = val_df['BTTS'].values.astype(int)

    # Predictions
    from ml_training import predict_match
    pred_rows = []
    for _, r in val_df.iterrows():
        feat_row_dict = {}
        for col in TRAIN_FEATURE_COLUMNS:
            feat_row_dict[col] = float(r[col])
        feature_row = np.array([feat_row_dict[c] for c in TRAIN_FEATURE_COLUMNS], dtype=float).reshape(1,-1)
        pm = predict_match(models, feature_row)
        pred_rows.append({**pm, 'TotalGoals_true': r['TotalGoals'], 'HomeWin_true': r['HomeWin'], 'Draw_true': r['Draw'], 'AwayWin_true': r['AwayWin'], 'BTTS_true': r['BTTS']})
    pred_df = pd.DataFrame(pred_rows)

    # Metrics
    mae = float(np.mean(np.abs(pred_df['pred_total_goals'] - pred_df['TotalGoals_true']))) if 'pred_total_goals' in pred_df else None
    rmse = float(np.sqrt(np.mean((pred_df['pred_total_goals'] - pred_df['TotalGoals_true'])**2))) if 'pred_total_goals' in pred_df else None

    # 1X2 log loss approximation using predicted probabilities (need mapping to true label index)
    def label_index(row):
        if row['HomeWin_true'] == 1: return 0
        if row['Draw_true'] == 1: return 1
        return 2
    y_labels = np.array([label_index(r) for _, r in pred_df.iterrows()])
    probs_1x2 = pred_df[['prob_1x2_home','prob_1x2_draw','prob_1x2_away']].values if {'prob_1x2_home','prob_1x2_draw','prob_1x2_away'}.issubset(pred_df.columns) else None
    logloss_1x2 = None
    acc_1x2 = None
    if probs_1x2 is not None and len(probs_1x2):
        # Guard against zero or missing probs
        eps = 1e-9
        probs_clipped = np.clip(probs_1x2, eps, 1.0 - eps)
        # Normalize row-wise just in case
        probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
        ll_terms = -np.log([probs_clipped[i, y_labels[i]] for i in range(len(y_labels))])
        logloss_1x2 = float(np.mean(ll_terms))
        preds_idx = np.argmax(probs_clipped, axis=1)
        acc_1x2 = float(np.mean(preds_idx == y_labels))

    # BTTS metrics
    prob_btts_yes = pred_df['prob_btts_yes'].values if 'prob_btts_yes' in pred_df else None
    btts_logloss = None
    btts_acc = None
    if prob_btts_yes is not None and len(prob_btts_yes):
        eps = 1e-9
        p_yes = np.clip(prob_btts_yes, eps, 1.0 - eps)
        btts_logloss = float(np.mean(- (btts_true * np.log(p_yes) + (1 - btts_true) * np.log(1 - p_yes))))
        btts_acc = float(np.mean(((p_yes >= 0.5).astype(int) == btts_true)))

    metrics = {
        'samples_train': len(train_df),
        'samples_val': len(val_df),
        'goals_MAE': mae,
        'goals_RMSE': rmse,
        '1X2_LogLoss': logloss_1x2,
        '1X2_Accuracy': acc_1x2,
        'BTTS_LogLoss': btts_logloss,
        'BTTS_Accuracy': btts_acc,
        'cv_metrics': models.get('cv_metrics', {})
    }

    # Calibration
    calibration_artifacts = {}
    if calib_enable and prob_btts_yes is not None and len(prob_btts_yes):
        ece_btts = approximate_ece(btts_true, prob_btts_yes, bins=bins)
        calibration_artifacts['BTTS_ECE_before'] = ece_btts
        if ece_btts > ece_threshold:
            cal_probs, cal_meta = calibrate_probs(prob_btts_yes, btts_true, method=method_pref)
            calibration_artifacts['BTTS_ECE_after'] = approximate_ece(btts_true, cal_probs, bins=bins)
            calibration_artifacts['BTTS_method'] = cal_meta['method']
            if save_calibrated:
                pred_df['prob_btts_yes_calibrated'] = cal_probs
        else:
            calibration_artifacts['BTTS_method'] = 'none_needed'

    # Prepare artifact directory
    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    out_dir = artifacts_dir / f"{tag}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    model_path = out_dir / 'model.pkl'
    save_models(models, str(model_path))

    # Feature manifest
    feat_manifest = {
        'feature_columns': TRAIN_FEATURE_COLUMNS,
        'rolling_window': rolling_window,
        'hash_curated': curated_hash,
        'leagues': leagues,
        'seasons': seasons,
    }
    (out_dir / 'feature_manifest.json').write_text(json.dumps(feat_manifest, indent=2))

    # Metrics
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    # Calibration artifacts
    if calibration_artifacts:
        (out_dir / 'calibration_curve.json').write_text(json.dumps(calibration_artifacts, indent=2))

    # Prediction sample (first 50 rows)
    (out_dir / 'validation_predictions_sample.json').write_text(pred_df.head(50).to_json(orient='records'))

    # Embed git commit hash if available
    try:
        import subprocess
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        (out_dir / 'git_commit.txt').write_text(commit)
    except Exception:
        pass

    logging.info(f"Artifacts saved to {out_dir}")
    logging.info(json.dumps(metrics, indent=2))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline (Section 2.5)')
    parser.add_argument('--config', required=True, help='Path to training config YAML')
    args = parser.parse_args()
    sys.exit(main(args.config))
