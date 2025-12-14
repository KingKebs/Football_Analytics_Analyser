#!/usr/bin/env python3
"""
Quick Parameter Generator
========================

Generates optimized CLI commands based on analysis of your system performance.
"""

def generate_optimized_commands():
    """Generate ready-to-use optimized commands"""

    leagues = ['E0', 'E1', 'D1', 'F1', 'I1', 'SP1', 'B1']

    configurations = {
        'balanced_recommended': {
            'name': 'Balanced Value Hunter (RECOMMENDED)',
            'params': '--min-confidence 0.68 --ml-mode predict --enable-double-chance --dc-min-prob 0.78 --dc-secondary-threshold 0.83 --dc-allow-multiple --verbose',
            'description': 'Optimized for 3-5 leg parlays, balanced risk/reward'
        },
        'conservative_safe': {
            'name': 'Conservative Multi-Parlay',
            'params': '--min-confidence 0.72 --ml-mode predict --enable-double-chance --dc-min-prob 0.82 --dc-secondary-threshold 0.87 --verbose',
            'description': 'High-probability selections for 5-8 leg parlays'
        },
        'ml_edge_hunter': {
            'name': 'ML Edge Exploiter',
            'params': '--min-confidence 0.62 --ml-mode predict --enable-double-chance --dc-min-prob 0.72 --dc-secondary-threshold 0.78 --dc-allow-multiple --verbose',
            'description': 'Maximum ML advantage for 2-4 leg parlays'
        },
        'your_current': {
            'name': 'Your Current Configuration',
            'params': '--min-confidence 0.6 --ml-mode predict --enable-double-chance --dc-min-prob 0.75 --dc-secondary-threshold 0.80 --dc-allow-multiple --verbose',
            'description': 'Your existing setup (working well)'
        }
    }

    print("OPTIMIZED CLI COMMANDS FOR FOOTBALL ANALYTICS")
    print("=" * 60)
    print()

    for config_key, config in configurations.items():
        print(f"{config['name']}")
        print(f"Description: {config['description']}")
        print("Commands:")
        for league in leagues:
            command = f"python cli.py --task full-league --league {league} {config['params']}"
            print(f"  {league}: {command}")
        print()
        print("-" * 60)
        print()

if __name__ == "__main__":
    generate_optimized_commands()
