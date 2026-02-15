#!/usr/bin/env python3
"""
Quick Parameter Generator
========================

Generates optimized CLI commands based on analysis of your system performance.
"""

def generate_optimized_commands():
    """Generate ready-to-use optimized commands"""

    leagues = ['E0', 'E1', 'D1', 'F1', 'I1', 'SP1', 'B1']

    print("=" * 80)
    print("DYNAMIC WORKFLOW (RECOMMENDED - Auto-detects leagues from upcoming matches)")
    print("=" * 80)
    print()
    print("# Interactive mode with prompts")
    print("python run_analysis_workflow.py --date 2026-02-11")
    print()
    print("# Auto mode (no prompts)")
    print("python run_analysis_workflow.py --date 2026-02-11 --auto --verbose")
    print()
    print("# Override detected leagues")
    print("python run_analysis_workflow.py --date 2026-02-11 --leagues E0,E2,E3")
    print()
    print("This workflow:")
    print("  1. Reads data/raw/upcomingMatches.json")
    print("  2. Auto-detects leagues and prompts for confirmation")
    print("  3. Converts to parsed fixtures (todays_fixtures_<DATE>.json)")
    print("  4. Runs full-league analysis with ML predictions (fixed feature engineering)")
    print("  5. Runs corners analysis with ML predictions")
    print("  6. Generates outputs for Streamlit UI")
    print()
    print("-" * 80)
    print()

    configurations = {
        'balanced_recommended': {
            'name': 'Balanced Value Hunter (RECOMMENDED)',
            'params': '--use-parsed-all --fixtures-date YYYYMMDD --ml-mode predict --enable-double-chance --dc-min-prob 0.78 --dc-secondary-threshold 0.83 --dc-allow-multiple --verbose',
            'description': 'Optimized for 3-5 leg parlays, balanced risk/reward, auto-detects leagues from parsed fixtures'
        },
        'conservative_safe': {
            'name': 'Conservative Multi-Parlay',
            'params': '--use-parsed-all --fixtures-date YYYYMMDD --ml-mode predict --enable-double-chance --dc-min-prob 0.82 --dc-secondary-threshold 0.87 --verbose',
            'description': 'High-probability selections for 5-8 leg parlays, auto-detects leagues'
        },
        'ml_edge_hunter': {
            'name': 'ML Edge Exploiter',
            'params': '--use-parsed-all --fixtures-date YYYYMMDD --ml-mode predict --enable-double-chance --dc-min-prob 0.72 --dc-secondary-threshold 0.78 --dc-allow-multiple --verbose',
            'description': 'Maximum ML advantage for 2-4 leg parlays, auto-detects leagues'
        },
    }

    print("MANUAL CLI COMMANDS (For individual league analysis)")
    print("=" * 80)
    print()

    for config_key, config in configurations.items():
        print(f"{config['name']}")
        print(f"Description: {config['description']}")
        print("Command:")
        print(f"  python cli.py --task full-league {config['params']}")
        print()
        print("  Note: Replace YYYYMMDD with your date (e.g., 20260211)")
        print("        Leagues are auto-detected from todays_fixtures_<DATE>.json")
        print()
        print("-" * 80)
        print()

    print()
    print("CORNERS ANALYSIS")
    print("=" * 80)
    print()
    print("# Auto-detect leagues from parsed fixtures")
    print("python cli.py --task corners --use-parsed-all --fixtures-date 20260211 --league ALL --min-team-matches 3 --corners-use-ml-prediction --verbose")
    print()
    print("-" * 80)
    print()



if __name__ == "__main__":
    generate_optimized_commands()
