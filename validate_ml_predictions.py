#!/usr/bin/env python3
"""
ML Predictions Validation Script
Run this to verify that ML predictions are unique across matches and leagues.
"""
import json
import sys
from pathlib import Path

def validate_ml_predictions(json_path):
    """Validate ML predictions from consolidated output."""

    print("=" * 80)
    print("ML PREDICTIONS VALIDATION")
    print("=" * 80)
    print(f"\nFile: {json_path}")

    # Load data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {json_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False

    # Extract matches
    matches = data.get('matches', {})
    if not matches:
        print("❌ No matches found in data")
        return False

    print(f"\n✅ Loaded data for {len(matches)} leagues")
    print(f"   Total matches: {data.get('summary', {}).get('total_matches', 0)}")
    print(f"   Total picks: {data.get('summary', {}).get('total_picks', 0)}")

    # Collect all predictions
    all_predictions = []
    for league, league_matches in matches.items():
        print(f"\n{league}: {len(league_matches)} matches")

        for match in league_matches:
            ml_pred = match.get('ml_predictions', {})
            if ml_pred:
                all_predictions.append({
                    'league': league,
                    'match': match.get('match', 'Unknown'),
                    'total_goals': ml_pred.get('total_goals', 'N/A'),
                    'h_d_a': ml_pred.get('h_d_a_probs', 'N/A'),
                    'btts': ml_pred.get('btts_probs', 'N/A'),
                    'dc': ml_pred.get('dc_probs', 'N/A')
                })

    if not all_predictions:
        print("\n❌ No ML predictions found in any match")
        return False

    print(f"\n✅ Found ML predictions for {len(all_predictions)} matches")

    # Check uniqueness
    total_goals_values = [p['total_goals'] for p in all_predictions]
    unique_count = len(set(total_goals_values))

    print("\n" + "=" * 80)
    print("UNIQUENESS CHECK")
    print("=" * 80)
    print(f"\nTotal matches with ML predictions: {len(all_predictions)}")
    print(f"Unique 'Total Goals' values: {unique_count}")

    if unique_count == len(all_predictions):
        print("\n✅ ✅ ✅ ALL PREDICTIONS ARE UNIQUE! ✅ ✅ ✅")
        success = True
    else:
        duplicates_count = len(all_predictions) - unique_count
        print(f"\n⚠️  Found {duplicates_count} duplicate prediction(s)")

        # Show duplicates
        from collections import Counter
        value_counts = Counter(total_goals_values)
        duplicates = {val: count for val, count in value_counts.items() if count > 1}

        if duplicates:
            print("\nDuplicate values:")
            for val, count in duplicates.items():
                print(f"  {val}: appears {count} times")
                matches_with_val = [p['match'] for p in all_predictions if p['total_goals'] == val]
                for m in matches_with_val:
                    print(f"    - {m}")

        success = False

    # Show sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    for i, pred in enumerate(all_predictions[:5]):
        print(f"\n{i+1}. {pred['league']}: {pred['match']}")
        print(f"   Total Goals: {pred['total_goals']}")
        print(f"   1X2: {pred['h_d_a']}")
        print(f"   BTTS: {pred['btts']}")
        print(f"   DC: {pred['dc']}")

    if len(all_predictions) > 5:
        print(f"\n... and {len(all_predictions) - 5} more matches")

    return success


if __name__ == '__main__':
    # Default to latest consolidated file
    default_path = "data/analysis/consolidated_full_league_20260208_20260208_164108.json"

    json_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    success = validate_ml_predictions(json_path)

    print("\n" + "=" * 80)
    if success:
        print("✅ VALIDATION PASSED")
        print("=" * 80)
        sys.exit(0)
    else:
        print("❌ VALIDATION FAILED")
        print("=" * 80)
        sys.exit(1)

