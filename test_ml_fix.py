#!/usr/bin/env python3
"""Quick test to verify ML predictions are now unique."""
import sys
sys.path.insert(0, 'src')

from ml_features import engineer_features, build_match_feature_row, TRAIN_FEATURE_COLUMNS
from automate_football_analytics_fullLeague import load_historical_matches
import numpy as np

print("Loading historical data...")
history_df = load_historical_matches()
print(f"Loaded {len(history_df)} matches")

print("\nEngineering features...")
ml_feature_df = engineer_features(history_df)
print(f"Created {len(ml_feature_df)} feature rows")

# Test matches
matches = [
    ('Brighton', 'Crystal Palace'),
    ('Liverpool', 'Man City'),
    ('Arsenal', 'Chelsea'),
]

print("\n" + "="*60)
print("TESTING FEATURE EXTRACTION")
print("="*60)

features = []
for home, away in matches:
    print(f"\n{home} vs {away}:")
    feat = build_match_feature_row(ml_feature_df, home, away)
    features.append(feat)

    print(f"  Home shots: {feat['HS']:.2f}, Home roll GF: {feat['Home_roll_GF']:.2f}")
    print(f"  Away shots: {feat['AS']:.2f}, Away roll GF: {feat['Away_roll_GF']:.2f}")
    print(f"  Shots ratio: {feat['Shots_Ratio']:.2f}")

    # Convert to array for prediction
    feat_vector = np.array([feat[c] for c in TRAIN_FEATURE_COLUMNS], dtype=float)
    print(f"  Feature vector sum: {feat_vector.sum():.2f}")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)

# Check if any are identical
for i in range(len(features)):
    for j in range(i+1, len(features)):
        match1 = matches[i]
        match2 = matches[j]
        feat1 = features[i]
        feat2 = features[j]

        # Check if identical
        identical = feat1 == feat2

        # Compute similarity
        vec1 = np.array([feat1[c] for c in TRAIN_FEATURE_COLUMNS], dtype=float)
        vec2 = np.array([feat2[c] for c in TRAIN_FEATURE_COLUMNS], dtype=float)
        diff = np.abs(vec1 - vec2).sum()

        print(f"\n{match1[0]} v {match1[1]} vs {match2[0]} v {match2[1]}:")
        print(f"  Identical? {identical}")
        print(f"  Total difference: {diff:.4f}")

        if diff < 0.01:
            print(f"  ⚠️  PROBLEM: Features are nearly identical!")
        else:
            print(f"  ✅ Features are different")

print("\n" + "="*60)
print("RESULT: " + ("✅ FIX SUCCESSFUL" if diff > 0.01 else "❌ STILL BROKEN"))
print("="*60)

