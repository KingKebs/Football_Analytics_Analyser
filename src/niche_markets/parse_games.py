#!/usr/bin/env python3
"""
Parse upcoming games for niche market analysis (Odd/Even, Highest Scoring Half)
Prioritizes lower leagues: League One, LaLiga2, Greek Super League, Dutch Eredivisie
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

class UpcomingGamesParser:

    # League priority for niche markets (higher = better for odd/even + 2nd half)
    LEAGUE_PRIORITY = {
        'League One': 10,      # English League One - ideal volatility
        'LaLiga2': 9,          # Spanish 2nd tier - high variance
        'Super League': 8,     # Greek league - unpredictable
        'Eredivisie': 6,       # Dutch league - moderate priority
        'Championship': 5,     # English 2nd tier
        'League Two': 8,       # English 4th tier - very volatile
    }

    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.matches = []

    def load_games(self) -> Dict:
        """Load games from JSON file"""
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def parse_matches(self) -> List[Dict]:
        """Extract and prioritize matches for niche markets"""
        data = self.load_games()
        prioritized_matches = []

        for country, leagues in data.items():
            for league_name, league_data in leagues.items():

                # Get priority score
                priority = self.LEAGUE_PRIORITY.get(league_name, 3)

                # Skip if very low priority
                if priority < 5:
                    continue

                matches = league_data.get('matches', [])

                for match in matches:
                    # Skip live matches
                    if match.get('time') == 'LIVE':
                        continue

                    match_info = {
                        'country': country,
                        'league': league_name,
                        'priority': priority,
                        'home_team': match['home'],
                        'away_team': match['away'],
                        'kick_off_time': match['time'],
                        'match_id': f"{country}_{league_name}_{match['home']}_vs_{match['away']}".replace(' ', '_')
                    }

                    prioritized_matches.append(match_info)

        # Sort by priority (highest first)
        return sorted(prioritized_matches, key=lambda x: x['priority'], reverse=True)

    def get_top_matches(self, limit: int = 10) -> List[Dict]:
        """Get top priority matches for analysis"""
        matches = self.parse_matches()
        return matches[:limit]

    def get_league_one_matches(self) -> List[Dict]:
        """Get only League One matches (optimal for odd/even)"""
        matches = self.parse_matches()
        return [m for m in matches if m['league'] == 'League One']

    def print_analysis_targets(self):
        """Print formatted matches ready for analysis"""
        matches = self.get_top_matches(15)

        print(f"=== NICHE MARKET ANALYSIS TARGETS ===")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Total High-Priority Matches: {len(matches)}\n")

        current_league = None
        for match in matches:
            if match['league'] != current_league:
                current_league = match['league']
                print(f"\n📊 {match['country']} - {match['league']} (Priority: {match['priority']})")
                print("-" * 50)

            print(f"{match['kick_off_time']} | {match['home_team']} vs {match['away_team']}")

        print(f"\n🎯 RECOMMENDED: Focus on League One matches for Odd/Even market")
        league_one = self.get_league_one_matches()
        print(f"League One matches available: {len(league_one)}")


def main():
    """Run the parser"""
    json_path = "/Users/admin/sites/Development/Football_Analytics_Analyser/src/niche_markets/data/upcomingGames-13122025.json"

    # Check if file exists
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return []

    try:
        parser = UpcomingGamesParser(json_path)

        # Print analysis targets
        parser.print_analysis_targets()

        # Export for further analysis
        matches = parser.get_top_matches()

        output_file = json_path.replace('.json', '_parsed.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'total_matches': len(matches),
                'matches': matches
            }, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Parsed data saved to: {output_file}")
        return matches

    except Exception as e:
        print(f"❌ Error parsing games: {str(e)}")
        return []

if __name__ == "__main__":
    main()
