#!/usr/bin/env python3
"""
Parlay Optimizer for Niche Markets
Creates optimal 2-slip combinations for Odd/Even and 2nd Half markets
"""

from parse_games import UpcomingGamesParser
import itertools
from typing import List, Dict, Tuple

class ParlayOptimizer:

    def __init__(self):
        self.parser = UpcomingGamesParser('data/upcomingGames-13122025.json')

        # Market probabilities based on empirical analysis
        self.market_probs = {
            'League One': {'odd': 0.5407, '2nd_half': 0.526},
            'LaLiga2': {'odd': 0.529, '2nd_half': 0.520},
            'Super League': {'odd': 0.529, '2nd_half': 0.518}
        }

    def get_all_priority_matches(self) -> List[Dict]:
        """Get all high-priority matches with market probabilities"""
        matches = self.parser.get_top_matches(20)

        enhanced_matches = []
        for match in matches:
            if match['priority'] >= 6:  # Only high-priority leagues
                league = match['league']

                # Add market probabilities
                match['odd_prob'] = self.market_probs.get(league, {}).get('odd', 0.50)
                match['second_half_prob'] = self.market_probs.get(league, {}).get('2nd_half', 0.50)
                match['combined_edge'] = (match['odd_prob'] - 0.5) + (match['second_half_prob'] - 0.5)

                enhanced_matches.append(match)

        return enhanced_matches

    def calculate_parlay_probability(self, matches: List[Dict], market_type: str) -> float:
        """Calculate combined parlay probability"""
        prob = 1.0
        for match in matches:
            if market_type == 'odd':
                prob *= match['odd_prob']
            elif market_type == '2nd_half':
                prob *= match['second_half_prob']
        return prob

    def assess_correlation_risk(self, matches: List[Dict]) -> str:
        """Assess correlation risk between matches"""
        same_time = len(set(m['kick_off_time'] for m in matches)) == 1
        same_league = len(set(m['league'] for m in matches)) == 1
        same_country = len(set(m['country'] for m in matches)) == 1

        if same_time and same_league:
            return "HIGH"
        elif same_league or (same_time and same_country):
            return "MEDIUM"
        else:
            return "LOW"

    def generate_optimal_parlays(self) -> Dict:
        """Generate optimal 2-slip parlay combinations"""
        matches = self.get_all_priority_matches()

        # Separate by leagues for better selection
        league_one = [m for m in matches if m['league'] == 'League One']
        other_leagues = [m for m in matches if m['league'] != 'League One']

        parlays = {
            'odd_even_parlays': [],
            'second_half_parlays': [],
            'mixed_parlays': []
        }

        # Generate Odd/Even parlays
        print("🎯 ANALYZING ODD/EVEN PARLAY COMBINATIONS...")

        # Best League One combinations (reduce correlation)
        for combo in itertools.combinations(league_one[:8], 2):  # Top 8 to reduce correlation
            parlay_prob = self.calculate_parlay_probability(combo, 'odd')
            correlation_risk = self.assess_correlation_risk(combo)

            if parlay_prob > 0.29:  # Only include if combined prob > 29%
                parlays['odd_even_parlays'].append({
                    'matches': combo,
                    'probability': parlay_prob,
                    'correlation_risk': correlation_risk,
                    'expected_edge': (parlay_prob - 0.25) * 100,  # vs 25% fair parlay odds
                    'type': 'League One Mix'
                })

        # Cross-league combinations (lower correlation)
        for l1_match in league_one[:5]:  # Top 5 League One
            for other_match in other_leagues[:3]:  # Top 3 other leagues
                combo = [l1_match, other_match]
                parlay_prob = self.calculate_parlay_probability(combo, 'odd')
                correlation_risk = self.assess_correlation_risk(combo)

                if parlay_prob > 0.28:
                    parlays['odd_even_parlays'].append({
                        'matches': combo,
                        'probability': parlay_prob,
                        'correlation_risk': correlation_risk,
                        'expected_edge': (parlay_prob - 0.25) * 100,
                        'type': 'Cross-League'
                    })

        # Generate 2nd Half parlays
        print("🎯 ANALYZING 2ND HALF PARLAY COMBINATIONS...")

        for combo in itertools.combinations(league_one[:6], 2):
            parlay_prob = self.calculate_parlay_probability(combo, '2nd_half')
            correlation_risk = self.assess_correlation_risk(combo)

            if parlay_prob > 0.27:  # Slightly lower threshold for 2nd half
                parlays['second_half_parlays'].append({
                    'matches': combo,
                    'probability': parlay_prob,
                    'correlation_risk': correlation_risk,
                    'expected_edge': (parlay_prob - 0.25) * 100,
                    'type': '2nd Half'
                })

        # Sort all parlays by expected edge
        for key in parlays:
            parlays[key].sort(key=lambda x: x['expected_edge'], reverse=True)
            parlays[key] = parlays[key][:5]  # Top 5 each

        return parlays

    def display_parlay_recommendations(self):
        """Display formatted parlay recommendations"""
        parlays = self.generate_optimal_parlays()

        print("\n" + "="*70)
        print("🎲 OPTIMAL PARLAY COMBINATIONS (2-SLIP STRATEGY)")
        print("="*70)

        print(f"\n🥇 TOP ODD/EVEN PARLAYS")
        print("-" * 50)

        for i, parlay in enumerate(parlays['odd_even_parlays'], 1):
            print(f"\n{i}. {parlay['type']} Parlay")
            print(f"   Combined Probability: {parlay['probability']:.1%}")
            print(f"   Expected Edge: +{parlay['expected_edge']:.1f}% over fair odds")
            print(f"   Correlation Risk: {parlay['correlation_risk']}")

            for j, match in enumerate(parlay['matches'], 1):
                print(f"   Match {j}: {match['kick_off_time']} | {match['home_team']} vs {match['away_team']}")
                print(f"           📊 {match['country']} {match['league']} | ODD: {match['odd_prob']:.1%}")

        print(f"\n🥈 TOP 2ND HALF PARLAYS")
        print("-" * 50)

        for i, parlay in enumerate(parlays['second_half_parlays'], 1):
            print(f"\n{i}. {parlay['type']} Parlay")
            print(f"   Combined Probability: {parlay['probability']:.1%}")
            print(f"   Expected Edge: +{parlay['expected_edge']:.1f}% over fair odds")
            print(f"   Correlation Risk: {parlay['correlation_risk']}")

            for j, match in enumerate(parlay['matches'], 1):
                print(f"   Match {j}: {match['kick_off_time']} | {match['home_team']} vs {match['away_team']}")
                print(f"           📊 {match['country']} {match['league']} | 2nd Half: {match['second_half_prob']:.1%}")

        print(f"\n📊 PARLAY STRATEGY GUIDE")
        print("-" * 40)
        print(f"• 2-match parlays multiply individual probabilities")
        print(f"• Target: >28% combined probability for positive edge")
        print(f"• Lower correlation risk = more stable results")
        print(f"• Cross-league parlays reduce simultaneous exposure")
        print(f"• League One matches offer highest individual edges")

        print(f"\n⚠️  RISK WARNINGS")
        print("-" * 25)
        print(f"• Parlays amplify both wins and losses")
        print(f"• HIGH correlation = all matches fail together")
        print(f"• 17:00 kick-offs = simultaneous risk exposure")
        print(f"• Consider single bets if unsure about parlays")

        return parlays

def main():
    optimizer = ParlayOptimizer()
    optimizer.display_parlay_recommendations()

if __name__ == "__main__":
    main()
