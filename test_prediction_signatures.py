#!/usr/bin/env python3
"""
Test script to verify ML prediction uniqueness and signature generation.
This script helps diagnose why all predictions might be identical.

Usage:
    python test_prediction_signatures.py [--league E0] [--ml-mode predict]
"""

import sys
import os
import json
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_feature_uniqueness():
    """Test that feature vectors are unique per match."""
    try:
        from ml_features import engineer_features, build_match_feature_row, TRAIN_FEATURE_COLUMNS

        logger.info("=" * 70)
        logger.info("TEST 1: Feature Vector Uniqueness")
        logger.info("=" * 70)

        # Try to load recent history data
        history_files = list(Path('football-data').glob('*.csv'))
        if not history_files:
            logger.warning("No history files found in football-data/")
            return False

        history_file = history_files[0]
        logger.info(f"Loading history from: {history_file}")

        history_df = pd.read_csv(history_file)
        logger.info(f"Loaded {len(history_df)} historical matches")

        # Engineer features
        feature_df = engineer_features(history_df, rolling_window=6)
        logger.info(f"Engineered {len(feature_df)} feature rows")

        if len(feature_df) < 10:
            logger.warning("Not enough matches to test uniqueness")
            return False

        # Sample some teams
        sample_rows = feature_df.tail(10)
        logger.info("\nFeature Vector Uniqueness Test (last 10 matches):")
        logger.info("-" * 70)

        feature_vectors = []
        for idx, row in sample_rows.iterrows():
            home = row['HomeTeam']
            away = row['AwayTeam']

            feat_dict = build_match_feature_row(feature_df, home, away)
            feat_vector = [feat_dict.get(c, 0.0) for c in TRAIN_FEATURE_COLUMNS]
            feature_vectors.append(feat_vector)

            logger.info(f"  {home:20s} vs {away:20s}: {feat_vector[:5]} ...")

        # Check uniqueness
        unique_vectors = []
        for fv in feature_vectors:
            if not any(np.allclose(fv, ufv) for ufv in unique_vectors):
                unique_vectors.append(fv)

        logger.info(f"\nTotal matches: {len(feature_vectors)}")
        logger.info(f"Unique feature vectors: {len(unique_vectors)}")

        if len(unique_vectors) < len(feature_vectors):
            logger.warning(f"⚠️  Only {len(unique_vectors)}/{len(feature_vectors)} vectors are unique!")
            logger.warning("This indicates feature vectors are being reused or falling back to league average.")
            return False
        else:
            logger.info(f"✅ All {len(feature_vectors)} feature vectors are unique!")
            return True

    except Exception as e:
        logger.error(f"Feature uniqueness test failed: {e}", exc_info=True)
        return False


def test_ml_predictions():
    """Test that ML predictions are unique for different inputs."""
    try:
        from ml_training import predict_match
        from ml_utils import load_models

        logger.info("\n" + "=" * 70)
        logger.info("TEST 2: ML Prediction Uniqueness")
        logger.info("=" * 70)

        # Find latest model directory
        model_dirs = list(Path('models').glob('baseline_v*'))
        if not model_dirs:
            logger.warning("No trained models found in models/")
            return False

        latest_model_dir = max(model_dirs, key=lambda p: p.stat().st_mtime)
        logger.info(f"Loading models from: {latest_model_dir}")

        models = load_models(str(latest_model_dir))
        if not models:
            logger.warning("Failed to load models")
            return False

        logger.info("Models loaded successfully")

        # Create different feature vectors
        np.random.seed(42)
        logger.info("\nPrediction Test with Different Feature Vectors:")
        logger.info("-" * 70)

        predictions = []
        for i in range(5):
            # Create slightly different feature vectors
            feature_row = np.random.rand(1, 20) * 5  # Random features in range [0, 5]
            match_id = f"team{i}|opponent{i}"

            pred = predict_match(models, feature_row, match_id=match_id)
            predictions.append(pred)

            logger.info(f"  Match {i+1} ({match_id}):")
            logger.info(f"    Total Goals: {pred.get('pred_total_goals', 'N/A'):.2f}")
            logger.info(f"    1X2: {pred.get('prob_1x2_home', 0):.2f}, {pred.get('prob_1x2_draw', 0):.2f}, {pred.get('prob_1x2_away', 0):.2f}")
            logger.info(f"    BTTS: {pred.get('prob_btts_yes', 0):.2f}, {pred.get('prob_btts_no', 0):.2f}")
            if '_feature_hash' in pred:
                logger.info(f"    Feature Hash: {pred['_feature_hash']}")

        # Check uniqueness
        unique_predictions = []
        for pred in predictions:
            # Compare key fields
            pred_tuple = (
                round(pred.get('pred_total_goals', 0), 2),
                round(pred.get('prob_1x2_home', 0), 2),
                round(pred.get('prob_1x2_draw', 0), 2),
                round(pred.get('prob_1x2_away', 0), 2),
            )
            if pred_tuple not in [up[:4] for up in unique_predictions]:
                unique_predictions.append(pred_tuple)

        logger.info(f"\nTotal predictions: {len(predictions)}")
        logger.info(f"Unique predictions: {len(unique_predictions)}")

        if len(unique_predictions) < len(predictions):
            logger.warning(f"⚠️  Only {len(unique_predictions)}/{len(predictions)} predictions are unique!")
            return False
        else:
            logger.info(f"✅ All {len(predictions)} predictions are unique!")
            return True

    except Exception as e:
        logger.error(f"ML prediction test failed: {e}", exc_info=True)
        return False


def test_signature_generation():
    """Test the signature generation logic."""
    try:
        import json
        import hashlib

        logger.info("\n" + "=" * 70)
        logger.info("TEST 3: Signature Generation")
        logger.info("=" * 70)

        # Simulate predictions
        test_cases = [
            {
                'home': 'Arsenal',
                'away': 'Liverpool',
                'total_goals': 2.61,
                'prob_home': 0.60,
                'prob_draw': 0.28,
                'prob_away': 0.12,
                'prob_btts_yes': 0.65,
                'prob_btts_no': 0.35,
                'feature_hash': 'abc12345',
            },
            {
                'home': 'Chelsea',
                'away': 'Manchester City',
                'total_goals': 2.61,  # Same probabilities
                'prob_home': 0.60,
                'prob_draw': 0.28,
                'prob_away': 0.12,
                'prob_btts_yes': 0.65,
                'prob_btts_no': 0.35,
                'feature_hash': 'def67890',  # Different feature hash
            },
        ]

        logger.info("\nSignature Generation Test:")
        logger.info("-" * 70)

        signatures = []
        for case in test_cases:
            home = case['home']
            away = case['away']

            sig_payload = {
                'match': f"{home}|{away}",
                'total_goals': round(case['total_goals'], 2),
                'prob_home': round(case['prob_home'], 2),
                'prob_draw': round(case['prob_draw'], 2),
                'prob_away': round(case['prob_away'], 2),
                'prob_btts_yes': round(case['prob_btts_yes'], 2),
                'prob_btts_no': round(case['prob_btts_no'], 2),
                'feature_hash': case['feature_hash'],
            }

            sig_json = json.dumps(sig_payload, sort_keys=True, separators=(',', ':'))
            sig_hash = hashlib.sha256(sig_json.encode('utf-8')).hexdigest()[:12]
            full_sig = f"{home[:3]}-{away[:3]}|TG={case['total_goals']:.2f}|1X2={case['prob_home']:.2f},{case['prob_draw']:.2f},{case['prob_away']:.2f}|BTTS={case['prob_btts_yes']:.2f},{case['prob_btts_no']:.2f}|{sig_hash}"

            logger.info(f"  {home} vs {away}:")
            logger.info(f"    Signature: {full_sig}")
            signatures.append(full_sig)

        # Check uniqueness
        unique_sigs = set(signatures)
        logger.info(f"\nTotal signatures: {len(signatures)}")
        logger.info(f"Unique signatures: {len(unique_sigs)}")

        if len(unique_sigs) == len(signatures):
            logger.info("✅ All signatures are unique (team names + feature hashes included)!")
            return True
        else:
            logger.warning("⚠️  Some signatures are identical (team names should make them unique)!")
            return False

    except Exception as e:
        logger.error(f"Signature generation test failed: {e}", exc_info=True)
        return False


def main():
    parser = argparse.ArgumentParser(description='Test ML prediction uniqueness and signatures')
    parser.add_argument('--league', default='E0', help='League code to test')
    parser.add_argument('--ml-mode', default='predict', help='ML mode')
    parser.add_argument('--test', choices=['features', 'predictions', 'signatures', 'all'], default='all',
                        help='Which test to run')

    args = parser.parse_args()

    logger.info(f"\nFootball Analytics - ML Prediction Signature Test Suite")
    logger.info(f"League: {args.league}, ML Mode: {args.ml-mode}")
    logger.info(f"Running test: {args.test}\n")

    results = {}

    if args.test in ['features', 'all']:
        results['features'] = test_feature_uniqueness()

    if args.test in ['predictions', 'all']:
        results['predictions'] = test_ml_predictions()

    if args.test in ['signatures', 'all']:
        results['signatures'] = test_signature_generation()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name.upper():20s}: {status}")

    all_passed = all(results.values())
    logger.info("\n" + ("✅ All tests passed!" if all_passed else "❌ Some tests failed - see details above"))

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

