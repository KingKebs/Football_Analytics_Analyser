#!/usr/bin/env python3
"""
League One Odd/Even Analysis
Based on empirical data: League One has 54.07% odd rate (optimal for niche markets)
"""

import json
from parse_games import UpcomingGamesParser

def analyze_league_one_matches():
    """Quick analysis for today's League One matches"""

    parser = UpcomingGamesParser("/Users/admin/sites/Development/Football_Analytics_Analyser/src/niche_markets/data/upcomingGames-13122025.json")
    league_one_matches = parser.get_league_one_matches()

    print("🎯 LEAGUE ONE ODD/EVEN ANALYSIS")
    print("=" * 50)
    print(f"📊 League One Odd Rate: 54.07% (Empirical)")
    print(f"📊 2nd Half Goal Ratio: 1.260 (26% more goals)")
    print(f"📊 Total Matches Available: {len(league_one_matches)}")
    print()

    print("📅 TODAY'S TARGETS:")
    print("-" * 30)

    for i, match in enumerate(league_one_matches, 1):
        print(f"{i:2d}. {match['kick_off_time']} | {match['home_team']} vs {match['away_team']}")
        print(f"     🎯 Recommended: ODD total goals (54.07% hit rate)")
        print(f"     🎯 Secondary: 2nd Half highest scoring (1.26x ratio)")
        print()

    print("💡 STRATEGY NOTES:")
    print("• League One shows highest volatility for niche markets")
    print("• Focus on ODD total goals (primary market)")
    print("• 2nd half typically outscores 1st half by 26%")
    print("• Lower-division = higher variance = better odds")

    return league_one_matches

if __name__ == "__main__":
    analyze_league_one_matches()
