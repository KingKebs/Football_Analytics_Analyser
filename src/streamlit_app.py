#!/usr/bin/env python3
"""
Streamlit App for Football Analytics Visualization

This app provides interactive visualizations for:
- Full league analysis results (xG, probabilities, parlays)
- Corner analysis results (correlations, team stats, distributions)

Usage:
  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import math

# Streamlit performance caches
@st.cache_data(ttl=120)
def _load_json(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

@st.cache_data(ttl=120)
def load_latest_parsed_corners():
    corners_dir = 'data/corners'
    paths = sorted(glob.glob(os.path.join(corners_dir, 'parsed_corners_predictions_*.json')))
    if not paths:
        return None, None
    latest = paths[-1]
    return _load_json(latest), latest

# Helper to display dataframes with a stretch/content toggle
def show_dataframe(df, stretch: bool = True, fallback_width: int = 800):
    """Render a dataframe using use_container_width when available, else fallback to fixed width."""
    if stretch:
        try:
            st.dataframe(df, use_container_width=True)
            return
        except TypeError:
            # older streamlit versions may not accept use_container_width on dataframe
            pass
    st.dataframe(df, width=fallback_width)

# Inject light CSS for card-like visuals
SPORTS_CSS = """
<style>
/***** Card look *****/
.block-container {padding-top: 1rem;}
.s-card {border: 1px solid #e6e6e6; border-radius: 10px; padding: 12px 14px; background: #ffffff; box-shadow: 0 1px 3px rgba(0,0,0,0.04);}
.s-subtle {color:#6b7280; font-size: 0.9rem;}
.s-badge {display:inline-block; padding:2px 8px; border-radius:12px; background:#eef2ff; color:#3730a3; font-weight:600; margin-left:6px;}
.s-metric {font-weight:700;}
.s-grid {gap: 10px;}
hr {margin: 1rem 0;}
</style>
"""

st.markdown(SPORTS_CSS, unsafe_allow_html=True)

# Set page config
st.set_page_config(
    page_title="Football Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_latest_full_league_data():
    """Load all recent full league suggestions data from the same analysis run."""
    data_dir = 'data/analysis'
    candidates = glob.glob(os.path.join(data_dir, 'full_league_suggestions_*.json'))
    if not candidates:
        return None, None

    # Get the most recent file to determine the base timestamp (date + hour-minute)
    latest_file = max(candidates, key=os.path.getmtime)
    full_timestamp = os.path.basename(latest_file).split('_')[-1].replace('.json', '')

    # Extract date and hour-minute from timestamp
    # Example: 065856 -> use 0658 to match 065840, 065843, 065847, 065850, 065853, 065856
    if len(full_timestamp) >= 4:  # Format: HHMMSS
        base_time = full_timestamp[:4]  # HHMM (first 4 chars)
    else:
        base_time = full_timestamp

    # Get the date part from the filename (YYYYMMDD)
    filename_parts = os.path.basename(latest_file).split('_')
    if len(filename_parts) >= 4:
        date_part = filename_parts[-2]  # Should be 20251206
        base_pattern = f"{date_part}_{base_time}"  # 20251206_0658
    else:
        base_pattern = base_time

    # Find all files from the same analysis run (same date and hour-minute)
    same_run_files = [f for f in candidates if base_pattern in os.path.basename(f)]

    # Load all files from the same run
    all_data = []
    all_paths = []
    for file_path in sorted(same_run_files):
        data = _load_json(file_path)
        if data:
            # Add league info to the data
            league_code = os.path.basename(file_path).split('_')[3]  # Extract league from filename
            if isinstance(data, dict):
                data['league_code'] = league_code
                data['file_path'] = file_path
            all_data.append(data)
            all_paths.append(file_path)

    return all_data, all_paths


def load_latest_corner_predictions_data():
    """Load the most recent corner batch predictions data."""
    corners_dir = 'data/corners'
    paths = sorted(glob.glob(os.path.join(corners_dir, 'batch_predictions_*.json')))
    if not paths:
        return None
    return _load_json(paths[-1])


def load_latest_corners_data():
    """Load the most recent corners analysis data."""
    corners_dir = 'data/corners'

    # Find latest correlations file
    corr_paths = sorted(glob.glob(os.path.join(corners_dir, 'corners_correlations_*.json')))
    if not corr_paths:
        return None, None, None

    # Load correlations
    with open(corr_paths[-1], 'r') as f:
        correlations = json.load(f)

    # Load team stats
    team_stats_path = corr_paths[-1].replace('corners_correlations', 'team_stats')
    team_stats = None
    if os.path.exists(team_stats_path):
        with open(team_stats_path, 'r') as f:
            team_stats = json.load(f)

    # Load CSV data
    csv_path = corr_paths[-1].replace('corners_correlations', 'corners_analysis').replace('.json', '.csv')
    df = None
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

    return correlations, team_stats, df


def create_correlation_plot(correlations, league_name=""):
    """Create correlation bar chart."""
    if not correlations:
        return None

    # Get top 15 correlations (excluding Total_Corners itself)
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = [(k, v) for k, v in sorted_corr if k != 'Total_Corners'][:15]

    features = [f[0] for f in top_features]
    values = [f[1] for f in top_features]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['skyblue' if v >= 0 else 'lightcoral' for v in values]
    bars = ax.barh(range(len(features)), values, color=colors, alpha=0.7)

    ax.set_xlabel('Correlation with Total Corners')
    title = f'Corner Feature Correlations{f" - {league_name}" if league_name else ""}'
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)

    # Add correlation values on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=8)

    plt.tight_layout()
    return fig


def create_team_stats_plot(team_stats):
    """Create team statistics comparison plot."""
    if not team_stats:
        return None

    league = team_stats.get('league', '')

    # Extract home and away stats
    home_stats = team_stats.get('home', {})
    away_stats = team_stats.get('away', {})

    if not home_stats or not away_stats:
        return None

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

    ax1.set_xlabel('Home Teams', fontsize=12)
    ax1.set_ylabel('Average Corners', fontsize=12)
    ax1.set_title(f'Top 10 Home Teams by Corners{f" - {league}" if league else ""}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(top_home.index, rotation=45, ha='right', fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Away teams
    bars3 = ax2.bar(x - width/2, top_away['Avg_Corners_For'], width, label='Corners For', color='skyblue', alpha=0.8)
    bars4 = ax2.bar(x + width/2, top_away['Avg_Corners_Against'], width, label='Corners Against', color='lightcoral', alpha=0.8)

    ax2.set_xlabel('Away Teams', fontsize=12)
    ax2.set_ylabel('Average Corners', fontsize=12)
    ax2.set_title(f'Top 10 Away Teams by Corners{f" - {league}" if league else ""}', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_away.index, rotation=45, ha='right', fontsize=9)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_distribution_plots(df, league_name=""):
    """Create distribution plots for corner analysis."""
    if df is None or 'Total_Corners' not in df.columns:
        return None

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Corner distribution histogram
    ax1.hist(df['Total_Corners'], bins=range(int(df['Total_Corners'].min()), int(df['Total_Corners'].max()) + 2),
             alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Total Corners', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Corner Distribution{f" - {league_name}" if league_name else ""}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Home vs Away corners
    if 'HC' in df.columns and 'AC' in df.columns:
        ax2.scatter(df['HC'], df['AC'], alpha=0.6, color='mediumseagreen')
        ax2.set_xlabel('Home Corners', fontsize=12)
        ax2.set_ylabel('Away Corners', fontsize=12)
        ax2.set_title('Home vs Away Corners', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add diagonal line
        max_corners = max(df['HC'].max(), df['AC'].max())
        ax2.plot([0, max_corners], [0, max_corners], 'r--', alpha=0.5, label='Equal corners')
        ax2.legend()

    # Half-split ratios
    if 'Est_1H_Corner_Ratio' in df.columns:
        ax3.hist(df['Est_1H_Corner_Ratio'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('1st Half Corner Ratio', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Half-Split Distribution', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Add mean line
        mean_ratio = df['Est_1H_Corner_Ratio'].mean()
        ax3.axvline(mean_ratio, color='red', linestyle='--', alpha=0.8,
                   label=f'Mean: {mean_ratio:.3f}')
        ax3.legend()

    # Estimated vs Actual corners (if available)
    if 'Est_1H_Corners' in df.columns and 'Est_2H_Corners' in df.columns:
        ax4.scatter(df['Est_1H_Corners'], df['Est_2H_Corners'], alpha=0.6, color='purple')
        ax4.set_xlabel('Estimated 1st Half Corners', fontsize=12)
        ax4.set_ylabel('Estimated 2nd Half Corners', fontsize=12)
        ax4.set_title('Half-Split Estimates', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Add diagonal line
        max_est = max(df['Est_1H_Corners'].max(), df['Est_2H_Corners'].max())
        ax4.plot([0, max_est], [0, max_est], 'r--', alpha=0.5, label='Equal halves')
        ax4.legend()

    plt.tight_layout()
    return fig


def create_xg_comparison_plot(suggestions):
    """Create expected goals comparison plot."""
    if not suggestions:
        return None

    matches = [f"{s['home']} vs {s['away']}" for s in suggestions]
    xg_home = [s['xg_home'] for s in suggestions]
    xg_away = [s['xg_away'] for s in suggestions]

    x = np.arange(len(matches))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, xg_home, width, label='Home xG', color='skyblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, xg_away, width, label='Away xG', color='lightcoral', alpha=0.8)

    ax.set_xlabel('Matches', fontsize=12)
    ax.set_ylabel('Expected Goals (xG)', fontsize=12)
    ax.set_title('Expected Goals Comparison - Full League Round', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(matches, rotation=45, ha='right', fontsize=9)
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
    return fig


def main():
    st.title("H⚽VA  Football Analytics Dashboard")

    # Load data
    full_league_data_list, full_league_paths = load_latest_full_league_data()
    correlations, team_stats, corners_df = load_latest_corners_data()
    corner_predictions = load_latest_corner_predictions_data()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a view",
        ["Full League Suggestions", "Corner Analysis", "Corner Predictions", "Live Corner Predictor"]
    )
    # Toggle to control whether charts/dataframes stretch to the container width (width='stretch')
    stretch_charts = st.sidebar.checkbox("Stretch charts/tables to container width (width='stretch')", value=True)

    if app_mode == "Full League Suggestions":
        st.header("Full League Match & Parlay Suggestions")
        if not full_league_data_list:
            st.warning("No full league suggestion files found in `data/analysis/`")
            st.info("Run the league analysis pipeline to generate suggestions.")
        else:
            # Display source info for all files from the same run
            if full_league_paths:
                latest_path = full_league_paths[0]
                ts = os.path.basename(latest_path).split('_')[-1].replace('.json','')
                league_codes = [data.get('league_code', 'Unknown') for data in full_league_data_list if isinstance(data, dict)]

                st.caption(f"Analysis run: {ts} | Leagues: {', '.join(league_codes)} | Files: {len(full_league_paths)}")
                with st.expander("📁 View all files"):
                    for path in full_league_paths:
                        st.text(f"• {path}")

            # Combine all suggestions from all leagues
            all_suggestions = []
            all_parlays = []
            for data in full_league_data_list:
                if isinstance(data, dict):
                    league_code = data.get('league_code', 'Unknown')
                    suggestions = data.get('suggestions', [])
                    parlays = data.get('favorable_parlays', [])

                    # Add league info to each suggestion
                    for suggestion in suggestions:
                        suggestion['league'] = league_code
                    all_suggestions.extend(suggestions)

                    # Add league info to each parlay
                    for parlay in parlays:
                        parlay['league'] = league_code
                    all_parlays.extend(parlays)

            # Quick filters
            fcol1, fcol2, fcol3 = st.columns([2,1,1])
            with fcol1:
                team_filter = st.text_input("Filter by team name", "").strip().lower()
            with fcol2:
                prob_cut = st.slider("Min pick probability % (for display)", 0, 100, 55)
            with fcol3:
                # League filter
                available_leagues = list(set([s.get('league', 'Unknown') for s in all_suggestions]))
                league_filter = st.selectbox("Filter by league", ['All'] + available_leagues)

            st.subheader(f"📋 Match Suggestions ({len(all_suggestions)} matches from {len(league_codes)} leagues)")

            # Build dataframe for line-by-line display
            if all_suggestions:
                rows = []
                for s in all_suggestions:
                    home = s.get('home', '')
                    away = s.get('away', '')
                    league = s.get('league', 'Unknown')

                    # Apply team filter
                    if team_filter and (team_filter not in home.lower() and team_filter not in away.lower()):
                        continue

                    # Apply league filter
                    if league_filter != 'All' and league != league_filter:
                        continue

                    picks = s.get('picks', [])
                    top_pick = None
                    if picks:
                        top_pick = max(picks, key=lambda p: p.get('prob', 0))

                    # Get ML prediction if available
                    mp = s.get('ml_prediction') or {}

                    # Get corner info
                    cc = s.get('corners_cards', {})

                    row = {
                        'League': league,
                        'Match': f"{home} vs {away}",
                        'Home xG': f"{s.get('xg_home', 0):.2f}",
                        'Away xG': f"{s.get('xg_away', 0):.2f}",
                        'Top Pick': f"{top_pick['selection']} ({top_pick.get('prob', 0)*100:.1f}%)" if top_pick and (top_pick.get('prob', 0)*100) >= prob_cut else "—",
                        'Market': top_pick.get('market', '—') if top_pick else "—",
                        'Total Corners': f"{cc.get('TotalCorners', 0):.1f}",
                        'Home Corners': f"{cc.get('HomeCorners', 0):.1f}",
                        'Away Corners': f"{cc.get('AwayCorners', 0):.1f}",
                        'Est. Cards': f"{cc.get('EstimatedCards', 0):.1f}",
                    }
                    rows.append(row)

                if rows:
                    df_suggestions = pd.DataFrame(rows)
                    show_dataframe(df_suggestions, stretch=stretch_charts)
                else:
                    st.info("No suggestions match your filters.")
            else:
                st.info("No suggestions available.")

            # Display Parlay Suggestions
            st.subheader(f"🎰 Favorable Parlays ({len(all_parlays)} parlays from all leagues)")
            if all_parlays:
                # Filter parlays by league if selected
                filtered_parlays = all_parlays
                if league_filter != 'All':
                    filtered_parlays = [p for p in all_parlays if p.get('league') == league_filter]

                if filtered_parlays:
                    for i, parlay in enumerate(filtered_parlays):
                        with st.expander(f"Parlay {i+1} ({parlay.get('league', 'Unknown')}) - {parlay.get('legs', 0)} legs, {parlay.get('combined_prob', 0)*100:.1f}% prob"):
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                legs = parlay.get('selections', [])
                                for j, leg in enumerate(legs):
                                    match = leg.get('match', 'Unknown match')
                                    selection = leg.get('selection', 'Unknown')
                                    prob = leg.get('prob', 0) * 100
                                    st.write(f"**Leg {j+1}:** {match} - {selection} ({prob:.1f}%)")
                            with col2:
                                st.metric("Combined Probability", f"{parlay.get('combined_prob', 0)*100:.1f}%")
                                if 'combined_odds' in parlay:
                                    st.metric("Combined Odds", f"{parlay.get('combined_odds', 0):.2f}")
                                if 'expected_return' in parlay:
                                    st.metric("Expected Return", f"${parlay.get('expected_return', 0):.2f}")
                else:
                    st.info("No parlays match your league filter.")
            else:
                st.info("No favorable parlays found.")

    elif app_mode == "Corner Predictions":
        st.header("Corner Predictions")

        # Check for parsed fixture corner predictions first (newest format)
        parsed_data, parsed_path = load_latest_parsed_corners()

        if parsed_data:
            st.subheader("📊 Parsed Fixture Corner Predictions (Dynamic League Detection)")
            st.caption(f"Latest file: {parsed_path}")

            preds = parsed_data.get('predictions', [])
            skipped = parsed_data.get('skipped', [])

            st.info(f"✅ {len(preds)} predictions generated, {len(skipped)} matches skipped")

            if preds:
                # League selector
                leagues_in_preds = list(set([p.get('league_code') for p in preds]))
                selected_leagues = st.multiselect("Filter by leagues", leagues_in_preds, default=leagues_in_preds)

                # Filter predictions
                filtered_preds = [p for p in preds if p.get('league_code') in selected_leagues]

                # Build dataframe of key metrics
                rows = []
                for p in filtered_preds:
                    rng = p.get('pred_total_corners_range') or [None, None]
                    rows.append({
                        'League': p.get('league_code'),
                        'Match': f"{p.get('home_team')} vs {p.get('away_team')}",
                        'ExpHome': f"{p.get('expected_home_corners', 0):.1f}",
                        'ExpAway': f"{p.get('expected_away_corners', 0):.1f}",
                        'TotalMean': f"{p.get('pred_total_corners_mean', 0):.1f}",
                        'Range': f"{rng[0]:.1f}-{rng[1]:.1f}" if rng[0] and rng[1] else "N/A",
                        '1H%': f"{p.get('pred_1h_ratio_mean', 0)*100:.0f}%",
                        '1H Corners': f"{p.get('pred_1h_corners_mean', 0):.1f}",
                        '2H Corners': f"{p.get('pred_2h_corners_mean', 0):.1f}",
                        'ML Used': "✅" if p.get('ml_used', False) else "❌"
                    })

                if rows:
                    df_parsed = pd.DataFrame(rows)
                    # Sort by league then total mean desc
                    df_parsed = df_parsed.sort_values(['League','TotalMean'], ascending=[True, False])
                    show_dataframe(df_parsed, stretch=stretch_charts)

                    # Show skipped matches summary
                    if skipped:
                        with st.expander(f"⚠️ {len(skipped)} matches skipped"):
                            skip_reasons = {}
                            for s in skipped:
                                reason = s.get('reason', 'unknown')
                                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

                            for reason, count in skip_reasons.items():
                                st.write(f"• {reason}: {count} matches")
                else:
                    st.info("No predictions match your league filter.")
            else:
                st.warning("No predictions found in the parsed file.")

        # Fallback to old batch predictions if no parsed data
        elif corner_predictions:
            st.subheader("📋 Legacy Corner Predictions")
            st.info("Showing older batch prediction format")

            # Display source info
            c_paths = sorted(glob.glob(os.path.join('data/corners','batch_predictions_*.json')))
            if c_paths:
                st.caption(f"Latest file: {c_paths[-1]}")

            # Support both legacy list and new dict structure
            if isinstance(corner_predictions, list):
                leagues_map = {'ALL': corner_predictions}
            elif isinstance(corner_predictions, dict) and 'predictions_by_league' in corner_predictions:
                leagues_map = corner_predictions['predictions_by_league'] or {}
                # Show summary
                totals = corner_predictions.get('totals', {})
                if totals:
                    st.info(f"Summary: {totals}")
            else:
                leagues_map = {}

            if not leagues_map:
                st.warning("No predictions available in the latest file.")
            else:
                for lg in sorted(leagues_map.keys()):
                    preds = leagues_map.get(lg, [])
                    st.subheader(f"League {lg} – {len(preds)} predictions")
                    for pred in preds:
                        home = pred.get('home_team') or pred.get('home') or 'N/A'
                        away = pred.get('away_team') or pred.get('away') or 'N/A'
                        with st.expander(f"{home} vs {away}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                total_mean = pred.get('pred_total_corners_mean', 0.0)
                                rng = pred.get('pred_total_corners_range') or [pred.get('pred_total_corners_min', 0.0), pred.get('pred_total_corners_max', 0.0)]
                                st.metric("Predicted Total Corners", f"{total_mean:.2f}")
                                if isinstance(rng, (list, tuple)) and len(rng) == 2:
                                    st.write(f"Range: {rng[0]:.2f} - {rng[1]:.2f}")
                                st.metric("Predicted 1H Corners", f"{pred.get('pred_1h_corners_mean', 0):.2f}")
                                st.metric("Predicted 2H Corners", f"{pred.get('pred_2h_corners_mean', 0):.2f}")
                            with col2:
                                st.subheader("Market Recommendations")
                                lines = pred.get('total_corner_lines', [])
                                if not lines:
                                    st.write("No specific line recommendations available.")
                                else:
                                    for line in lines:
                                        rec = line.get('recommendation')
                                        if rec:
                                            st.success(f"**Over/Under {line.get('line','?')}**: {rec} ({line.get('p_over', 0)*100:.1f}% Over)")
                                fh_lines = pred.get('first_half_lines', [])
                                if fh_lines:
                                    st.write("First Half Lines:")
                                    for line in fh_lines:
                                        rec = line.get('recommendation')
                                        if rec:
                                            st.info(f"1H {line.get('line','?')}: {rec} (Over {line.get('p_over', 0)*100:.1f}%)")
        else:
            st.warning("No corner prediction files found in `data/corners/`.")
            st.info("Run corner analysis with `--use-parsed-all` to generate predictions.")

    elif app_mode == "Corner Analysis":
        st.header("Corner Analysis Deep Dive")
        if not correlations:
            st.warning("No corner analysis files found.")
        else:
            league_name = correlations.get('league_name', '')
            st.subheader(f"Analysis for: {league_name}")

            # Correlation plot
            corr_fig = create_correlation_plot(correlations.get('correlations', {}), league_name)
            if corr_fig:
                st.pyplot(corr_fig, use_container_width=stretch_charts)

            # Team stats plot
            team_fig = create_team_stats_plot(team_stats)
            if team_fig:
                st.pyplot(team_fig, use_container_width=stretch_charts)

            # Distribution plots
            dist_fig = create_distribution_plots(corners_df, league_name)
            if dist_fig:
                st.pyplot(dist_fig, use_container_width=stretch_charts)

    elif app_mode == "Live Corner Predictor":
        st.header("Live Corner Predictor")
        st.info("Select a league and enter two teams to get a corner prediction.")

        league_stats_map = list_league_team_stats()
        if not league_stats_map:
            st.error("No team stats files found in `data/corners/`. Cannot make predictions.")
            st.stop()

        leagues = sorted(league_stats_map.keys())
        selected_league = st.selectbox("Select League", leagues)

        if selected_league:
            team_stats_path = league_stats_map[selected_league]
            team_stats_data = load_team_stats(team_stats_path)
            enriched_df = load_enriched_csv_for_team_stats(team_stats_path)

            home_teams = sorted(team_stats_data.get('home', {}).keys())
            away_teams = sorted(team_stats_data.get('away', {}).keys())

            col1, col2 = st.columns(2)
            with col1:
                home_team = st.selectbox("Home Team", home_teams)
            with col2:
                away_team = st.selectbox("Away Team", away_teams)

            if st.button("Predict Corners"):
                if home_team and away_team:
                    prediction, error = compute_match_prediction(team_stats_data, enriched_df, home_team, away_team)
                    if error:
                        st.error(error)
                    else:
                        st.success("Prediction successful!")
                        st.metric("Predicted Total Corners", f"{prediction['pred_total_mean']:.2f}")
                        st.write(f"Range: {prediction['pred_total_range'][0]:.2f} - {prediction['pred_total_range'][1]:.2f}")

                        c1, c2 = st.columns(2)
                        c1.metric("1H Corners", f"{prediction['pred_1h_mean']:.2f}")
                        c2.metric("2H Corners", f"{prediction['pred_2h_mean']:.2f}")

                        st.subheader("Market Lines (Total Corners)")
                        lines_df = pd.DataFrame(prediction['total_lines'])
                        show_dataframe(lines_df, stretch_charts)

                        st.subheader("Market Lines (1H Corners)")
                        half_lines_df = pd.DataFrame(prediction['half_lines'])
                        show_dataframe(half_lines_df, stretch_charts)

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit | Data from football-data.co.uk")


if __name__ == '__main__':
    main()
