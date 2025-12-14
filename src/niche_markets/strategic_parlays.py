#!/usr/bin/env python3
"""
RECOMMENDED PARLAY STRATEGY - Lower Correlation Risk
"""

from parlay_optimizer import ParlayOptimizer

def get_strategic_recommendations():
    """Get strategic parlay recommendations with reduced correlation"""

    optimizer = ParlayOptimizer()
    matches = optimizer.get_all_priority_matches()

    # Separate matches by league and time
    league_one = [m for m in matches if m['league'] == 'League One']
    laliga2 = [m for m in matches if m['league'] == 'LaLiga2']
    greek = [m for m in matches if m['league'] == 'Super League']

    print("🎯 STRATEGIC PARLAY RECOMMENDATIONS")
    print("="*60)
    print("Focus: REDUCED CORRELATION RISK + OPTIMAL EDGE")
    print("="*60)

    print(f"\n💎 SLIP 1: ODD/EVEN PARLAY (RECOMMENDED)")
    print("-" * 50)

    # Best cross-league combination
    best_l1 = league_one[0]  # Barnsley vs Leyton Orient
    best_spanish = laliga2[0] if laliga2 else league_one[1]  # Gijon vs Granada CF or backup

    if laliga2:
        combined_prob = best_l1['odd_prob'] * best_spanish['odd_prob']
        print(f"Match 1: {best_l1['kick_off_time']} | {best_l1['home_team']} vs {best_l1['away_team']}")
        print(f"         📊 England League One | ODD probability: {best_l1['odd_prob']:.1%}")
        print(f"Match 2: {best_spanish['kick_off_time']} | {best_spanish['home_team']} vs {best_spanish['away_team']}")
        print(f"         📊 Spain LaLiga2 | ODD probability: {best_spanish['odd_prob']:.1%}")
        print(f"\n🎲 Combined Probability: {combined_prob:.1%}")
        print(f"💰 Expected Edge: +{(combined_prob-0.25)*100:.1f}% over fair parlay")
        print(f"⚠️  Correlation Risk: LOW (Different leagues + times)")
    else:
        # Fallback to best League One combination with time spacing
        best_l1_alt = league_one[2]  # Different match
        combined_prob = best_l1['odd_prob'] * best_l1_alt['odd_prob']
        print(f"Match 1: {best_l1['kick_off_time']} | {best_l1['home_team']} vs {best_l1['away_team']}")
        print(f"         📊 England League One | ODD probability: {best_l1['odd_prob']:.1%}")
        print(f"Match 2: {best_l1_alt['kick_off_time']} | {best_l1_alt['home_team']} vs {best_l1_alt['away_team']}")
        print(f"         📊 England League One | ODD probability: {best_l1_alt['odd_prob']:.1%}")
        print(f"\n🎲 Combined Probability: {combined_prob:.1%}")
        print(f"💰 Expected Edge: +{(combined_prob-0.25)*100:.1f}% over fair parlay")
        print(f"⚠️  Correlation Risk: MEDIUM (Same league, same time)")

    print(f"\n💎 SLIP 2: 2ND HALF PARLAY (RECOMMENDED)")
    print("-" * 50)

    # Best 2nd half combination
    if greek:
        # Mix League One + Greek for better diversification
        best_greek = greek[0]
        combined_prob_2nd = best_l1['second_half_prob'] * best_greek['second_half_prob']
        print(f"Match 1: {best_l1['kick_off_time']} | {best_l1['home_team']} vs {best_l1['away_team']}")
        print(f"         📊 England League One | 2nd Half probability: {best_l1['second_half_prob']:.1%}")
        print(f"Match 2: {best_greek['kick_off_time']} | {best_greek['home_team']} vs {best_greek['away_team']}")
        print(f"         📊 Greece Super League | 2nd Half probability: {best_greek['second_half_prob']:.1%}")
        print(f"\n🎲 Combined Probability: {combined_prob_2nd:.1%}")
        print(f"💰 Expected Edge: +{(combined_prob_2nd-0.25)*100:.1f}% over fair parlay")
        print(f"⚠️  Correlation Risk: LOW (Different leagues + countries)")
    else:
        # League One backup
        l1_second = league_one[1]
        l1_third = league_one[3]
        combined_prob_2nd = l1_second['second_half_prob'] * l1_third['second_half_prob']
        print(f"Match 1: {l1_second['kick_off_time']} | {l1_second['home_team']} vs {l1_second['away_team']}")
        print(f"         📊 England League One | 2nd Half probability: {l1_second['second_half_prob']:.1%}")
        print(f"Match 2: {l1_third['kick_off_time']} | {l1_third['home_team']} vs {l1_third['away_team']}")
        print(f"         📊 England League One | 2nd Half probability: {l1_third['second_half_prob']:.1%}")
        print(f"\n🎲 Combined Probability: {combined_prob_2nd:.1%}")
        print(f"💰 Expected Edge: +{(combined_prob_2nd-0.25)*100:.1f}% over fair parlay")
        print(f"⚠️  Correlation Risk: MEDIUM (Same league)")

    print(f"\n🎯 ALTERNATIVE SINGLE-BET STRATEGY")
    print("-" * 50)
    print(f"If parlays feel too risky, focus on single bets:")
    print(f"• League One ODD goals: 54.07% win rate (+4.1% edge)")
    print(f"• Spread across 3-4 different matches")
    print(f"• Lower variance, more consistent profits")

    print(f"\n📊 FINAL RECOMMENDATIONS")
    print("-" * 35)
    print(f"✅ BEST APPROACH: Cross-league parlays (if available)")
    print(f"✅ SAFE APPROACH: Single League One ODD bets")
    print(f"⚠️  AVOID: Multiple same-time League One parlays")
    print(f"⚠️  MONITOR: Team news before 17:00 kick-offs")

    print(f"\n💡 SUCCESS FACTORS")
    print("-" * 25)
    print(f"• Lower leagues = higher market inefficiency")
    print(f"• Cross-league = reduced correlation risk")
    print(f"• ODD goals = strongest statistical edge")
    print(f"• Diversification = more stable long-term results")

if __name__ == "__main__":
    get_strategic_recommendations()
