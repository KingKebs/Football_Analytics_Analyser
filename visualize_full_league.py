#!/usr/bin/env python3

"""
Visualize Full League Analysis Results

This script reads analysis results and creates visualizations for:
- Full league suggestions: Expected Goals (xG) comparisons, pick probabilities, parlay overview
- Corner analysis: Corner correlations, team statistics, half-split estimates

Usage:
  python visualize_full_league.py [--input <path>] [--output_dir <path>] [--analysis_type <type>]
If --input is omitted, uses the most recent analysis file
"""

import argparse
import glob
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def find_latest_full_league(data_dir: str = 'data') -> Optional[str]:
    """Find the most recent full league suggestions JSON file."""
    paths = sorted(glob.glob(os.path.join(data_dir, 'full_league_suggestions_*.json')))
    return paths[-1] if paths else None


def load_full_league_data(path: str) -> dict:
    """Load the full league suggestions JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def plot_xg_comparison(suggestions: list, output_dir: str):
    """Plot expected goals comparison for each match."""
    if not suggestions:
        return

    matches = [f"{s['home']} vs {s['away']}" for s in suggestions]
    xg_home = [s['xg_home'] for s in suggestions]
    xg_away = [s['xg_away'] for s in suggestions]

    x = np.arange(len(matches))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, xg_home, width, label='Home xG', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, xg_away, width, label='Away xG', color='lightcoral', alpha=0.8)

    ax.set_xlabel('Matches')
    ax.set_ylabel('Expected Goals (xG)')
    ax.set_title('Expected Goals Comparison - Full League Round')
    ax.set_xticks(x)
    ax.set_xticklabels(matches, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'xg_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pick_probabilities(suggestions: list, output_dir: str):
    """Plot pick probabilities for each match."""
    if not suggestions:
        return

    fig, axes = plt.subplots(len(suggestions), 1, figsize=(10, 4*len(suggestions)))
    if len(suggestions) == 1:
        axes = [axes]

    for i, suggestion in enumerate(suggestions):
        ax = axes[i]
        match = f"{suggestion['home']} vs {suggestion['away']}"
        picks = suggestion.get('picks', [])

        if picks:
            markets = [p['market'] for p in picks]
            probs = [p['prob'] * 100 for p in picks]  # Convert to percentage
            odds = [p['odds'] for p in picks]

            bars = ax.bar(markets, probs, color='mediumseagreen', alpha=0.7)
            ax.set_ylabel('Probability (%)')
            ax.set_title(f'{match} - Pick Probabilities')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)

            # Add probability and odds labels
            for bar, prob, odd in zip(bars, probs, odds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{prob:.1f}%\n{odd:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No picks available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{match} - No Picks')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pick_probabilities.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_parlay_overview(parlays: list, output_dir: str):
    """Plot favorable parlay overview."""
    if not parlays:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Parlay probabilities
    sizes = [p['size'] for p in parlays]
    probs = [p['probability'] * 100 for p in parlays]  # Convert to percentage

    bars1 = ax1.bar(range(len(parlays)), probs, color='dodgerblue', alpha=0.7)
    ax1.set_ylabel('Probability (%)')
    ax1.set_title('Favorable Parlays - Probabilities')
    ax1.set_xticks(range(len(parlays)))
    ax1.set_xticklabels([f'Parlay {i+1}\n({sizes[i]} legs)' for i in range(len(parlays))])
    ax1.grid(True, alpha=0.3)

    # Add probability labels
    for bar, prob in zip(bars1, probs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{prob:.1f}%', ha='center', va='bottom', fontsize=8)

    # Parlay odds
    odds = [p['decimal_odds'] for p in parlays]

    bars2 = ax2.bar(range(len(parlays)), odds, color='orange', alpha=0.7)
    ax2.set_ylabel('Decimal Odds')
    ax2.set_title('Favorable Parlays - Decimal Odds')
    ax2.set_xticks(range(len(parlays)))
    ax2.set_xticklabels([f'Parlay {i+1}\n({sizes[i]} legs)' for i in range(len(parlays))])
    ax2.grid(True, alpha=0.3)

    # Add odds labels
    for bar, odd in zip(bars2, odds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{odd:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parlay_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_table(suggestions: list, parlays: list, output_dir: str):
    """Create a summary table of all data."""
    summary_data = []

    for i, suggestion in enumerate(suggestions):
        match = f"{suggestion['home']} vs {suggestion['away']}"
        xg_home = suggestion['xg_home']
        xg_away = suggestion['xg_away']
        picks = suggestion.get('picks', [])

        pick_summary = "; ".join([f"{p['market']} {p['selection']} ({p['prob']*100:.1f}%)" for p in picks])

        summary_data.append({
            'Match': match,
            'Home xG': xg_home,
            'Away xG': xg_away,
            'Picks': pick_summary
        })

    # Create DataFrame and save as CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(output_dir, 'full_league_summary.csv'), index=False)

    # Also save parlay summary
    if parlays:
        parlay_data = []
        for i, parlay in enumerate(parlays):
            parlay_data.append({
                'Parlay': f'Parlay {i+1}',
                'Size': parlay['size'],
                'Legs': "; ".join(parlay['legs']),
                'Probability': parlay['probability'],
                'Decimal Odds': parlay['decimal_odds'],
                'Stake Suggestion': parlay['stake_suggestion'],
                'Potential Return': parlay['potential_return']
            })

        parlay_df = pd.DataFrame(parlay_data)
        parlay_df.to_csv(os.path.join(output_dir, 'parlay_summary.csv'), index=False)


def find_latest_corners(data_dir: str = 'data/corners') -> Optional[str]:
    """Find the most recent corners analysis JSON file."""
    paths = sorted(glob.glob(os.path.join(data_dir, 'corners_correlations_*.json')))
    return paths[-1] if paths else None


def load_corners_data(corr_path: str) -> tuple:
    """Load corners analysis data (correlations and team stats)."""
    # Load correlations
    with open(corr_path, 'r') as f:
        correlations = json.load(f)

    # Find corresponding team stats file
    base_name = os.path.basename(corr_path).replace('corners_correlations', 'team_stats')
    team_stats_path = os.path.join(os.path.dirname(corr_path), base_name)

    team_stats = None
    if os.path.exists(team_stats_path):
        with open(team_stats_path, 'r') as f:
            team_stats = json.load(f)

    return correlations, team_stats


def plot_corner_correlations(correlations: dict, output_dir: str, league: str = ''):
    """Plot corner feature correlations."""
    if not correlations:
        return

    # Get top 15 correlations (excluding Total_Corners itself)
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [(k, v) for k, v in sorted_corr if k != 'Total_Corners'][:15]

    features = [f[0] for f in top_features]
    values = [f[1] for f in top_features]

    # Color based on positive/negative correlation
    colors = ['skyblue' if v >= 0 else 'lightcoral' for v in values]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(features)), values, color=colors, alpha=0.7)

    ax.set_xlabel('Correlation with Total Corners')
    ax.set_ylabel('Features')
    title = f'Corner Feature Correlations{f" - {league}" if league else ""}'
    ax.set_title(title)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.grid(True, alpha=0.3)

    # Add correlation values on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'corner_correlations{league.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_team_corner_stats(team_stats: dict, output_dir: str):
    """Plot team corner statistics."""
    if not team_stats:
        return

    league = team_stats.get('league', '')

    # Extract home and away stats
    home_stats = team_stats.get('home', {})
    away_stats = team_stats.get('away', {})

    if not home_stats or not away_stats:
        return

    # Convert to DataFrames
    home_df = pd.DataFrame(home_stats).T
    away_df = pd.DataFrame(away_stats).T

    # Get top teams by corners for
    top_home = home_df.nlargest(10, 'Avg_Corners_For')[['Avg_Corners_For', 'Avg_Corners_Against']]
    top_away = away_df.nlargest(10, 'Avg_Corners_For')[['Avg_Corners_For', 'Avg_Corners_Against']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Home teams
    x = np.arange(len(top_home))
    width = 0.35

    bars1 = ax1.bar(x - width/2, top_home['Avg_Corners_For'], width, label='Corners For', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, top_home['Avg_Corners_Against'], width, label='Corners Against', color='lightcoral', alpha=0.8)

    ax1.set_xlabel('Home Teams')
    ax1.set_ylabel('Average Corners')
    ax1.set_title(f'Top 10 Home Teams by Corners{f" - {league}" if league else ""}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_home.index, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Away teams
    bars3 = ax2.bar(x - width/2, top_away['Avg_Corners_For'], width, label='Corners For', color='skyblue', alpha=0.8)
    bars4 = ax2.bar(x + width/2, top_away['Avg_Corners_Against'], width, label='Corners Against', color='lightcoral', alpha=0.8)

    ax2.set_xlabel('Away Teams')
    ax2.set_ylabel('Average Corners')
    ax2.set_title(f'Top 10 Away Teams by Corners{f" - {league}" if league else ""}')
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_away.index, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'team_corner_stats{league.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_corner_distribution(df: pd.DataFrame, output_dir: str, league: str = ''):
    """Plot corner distribution and half-split analysis."""
    if df is None or 'Total_Corners' not in df.columns:
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Corner distribution histogram
    ax1.hist(df['Total_Corners'], bins=range(int(df['Total_Corners'].min()), int(df['Total_Corners'].max()) + 2),
             alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Total Corners')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Corner Distribution{f" - {league}" if league else ""}')
    ax1.grid(True, alpha=0.3)

    # Home vs Away corners
    if 'HC' in df.columns and 'AC' in df.columns:
        ax2.scatter(df['HC'], df['AC'], alpha=0.6, color='mediumseagreen')
        ax2.set_xlabel('Home Corners')
        ax2.set_ylabel('Away Corners')
        ax2.set_title('Home vs Away Corners')
        ax2.grid(True, alpha=0.3)

        # Add diagonal line
        max_corners = max(df['HC'].max(), df['AC'].max())
        ax2.plot([0, max_corners], [0, max_corners], 'r--', alpha=0.5, label='Equal corners')
        ax2.legend()

    # Half-split ratios
    if 'Est_1H_Corner_Ratio' in df.columns:
        ax3.hist(df['Est_1H_Corner_Ratio'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('1st Half Corner Ratio')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Half-Split Distribution')
        ax3.grid(True, alpha=0.3)

        # Add mean line
        mean_ratio = df['Est_1H_Corner_Ratio'].mean()
        ax3.axvline(mean_ratio, color='red', linestyle='--', alpha=0.8,
                   label=f'Mean: {mean_ratio:.3f}')
        ax3.legend()

    # Estimated vs Actual corners (if available)
    if 'Est_1H_Corners' in df.columns and 'Est_2H_Corners' in df.columns:
        ax4.scatter(df['Est_1H_Corners'], df['Est_2H_Corners'], alpha=0.6, color='purple')
        ax4.set_xlabel('Estimated 1st Half Corners')
        ax4.set_ylabel('Estimated 2nd Half Corners')
        ax4.set_title('Half-Split Estimates')
        ax4.grid(True, alpha=0.3)

        # Add diagonal line
        max_est = max(df['Est_1H_Corners'].max(), df['Est_2H_Corners'].max())
        ax4.plot([0, max_est], [0, max_est], 'r--', alpha=0.5, label='Equal halves')
        ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'corner_distribution{league.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_corners_summary(correlations: dict, team_stats: dict, output_dir: str, league: str = ''):
    """Create a summary table for corners analysis."""
    summary_data = []

    # Top correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for feature, corr in sorted_corr[:10]:
        if feature != 'Total_Corners':
            summary_data.append({
                'Type': 'Correlation',
                'Feature': feature,
                'Value': corr,
                'League': league
            })

    # Team stats summary
    if team_stats:
        home_stats = team_stats.get('home', {})
        if home_stats:
            # Get top team
            home_df = pd.DataFrame(home_stats).T
            if not home_df.empty and 'Avg_Corners_For' in home_df.columns:
                top_team = home_df['Avg_Corners_For'].idxmax()
                top_value = home_df['Avg_Corners_For'].max()
                summary_data.append({
                    'Type': 'Team Stat',
                    'Feature': f'Top Home Team: {top_team}',
                    'Value': top_value,
                    'League': league
                })

    # Create DataFrame and save
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(output_dir, f'corners_summary{league.lower()}.csv'), index=False)


def main():
    parser = argparse.ArgumentParser(description='Visualize full league analysis results')
    parser.add_argument('--input', type=str, help='Path to full league suggestions JSON file')
    parser.add_argument('--output_dir', type=str, default='visuals', help='Output directory for visualizations')

    args = parser.parse_args()

    # Find input file
    if args.input:
        input_path = args.input
    else:
        input_path = find_latest_full_league()
        if not input_path:
            print("No full league suggestions JSON file found in data directory")
            return

    print(f"Loading data from: {input_path}")

    # Load data
    data = load_full_league_data(input_path)
    suggestions = data.get('suggestions', [])
    parlays = data.get('favorable_parlays', [])

    print(f"Found {len(suggestions)} match suggestions and {len(parlays)} favorable parlays")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate visualizations
    print("Generating xG comparison plot...")
    plot_xg_comparison(suggestions, args.output_dir)

    print("Generating pick probabilities plot...")
    plot_pick_probabilities(suggestions, args.output_dir)

    print("Generating parlay overview plot...")
    plot_parlay_overview(parlays, args.output_dir)

    print("Creating summary tables...")
    create_summary_table(suggestions, parlays, args.output_dir)

    # Corner analysis (if corner data exists)
    corner_input_path = find_latest_corners()
    if corner_input_path:
        print(f"\nLoading corner data from: {corner_input_path}")
        correlations, team_stats = load_corners_data(corner_input_path)

        # Extract league name from file path
        league_name = ''
        if team_stats and 'league' in team_stats:
            league_name = team_stats['league']

        print("Generating corner correlations plot...")
        plot_corner_correlations(correlations, args.output_dir, league_name)

        print("Generating team corner stats plot...")
        plot_team_corner_stats(team_stats, args.output_dir)

        # Try to load the corresponding CSV data for distribution plot
        csv_path = corner_input_path.replace('corners_correlations', 'corners_analysis').replace('.json', '.csv')
        if os.path.exists(csv_path):
            print("Generating corner distribution plot...")
            df = pd.read_csv(csv_path)
            plot_corner_distribution(df, args.output_dir, league_name)
        else:
            print("CSV data not found, skipping distribution plot")

        print("Creating corners summary table...")
        create_corners_summary(correlations, team_stats, args.output_dir, league_name)

    print(f"\nVisualizations saved to: {args.output_dir}")
    print("Files created:")
    print("  - xg_comparison.png")
    print("  - pick_probabilities.png")
    print("  - parlay_overview.png")
    print("  - full_league_summary.csv")
    print("  - parlay_summary.csv")
    if corner_input_path:
        league_name = os.path.basename(corner_input_path).split('_')[2]  # Extract league name from file name
        print("  - corner_correlations.png")
        print("  - team_corner_stats.png")
        print("  - corner_distribution.png")
        print("  - corners_summary.csv")


if __name__ == '__main__':
    main()
