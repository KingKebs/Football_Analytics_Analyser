import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime
import sys

DATA_DIR = "data"
DATA_ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")

st.set_page_config(page_title="Football Analytics Suggestions Viewer", layout="wide")
st.title("📊 Football Analytics Suggestions Viewer")

# Optional command-line arg (when using: streamlit run view_suggestions.py -- --dir data/analysis)
custom_dir = None
if "--dir" in sys.argv:
    try:
        di = sys.argv.index("--dir")
        custom_dir = sys.argv[di+1]
    except Exception:
        custom_dir = None

# List all suggestion JSON files in analysis (preferred) then root data fallback
def list_suggestion_files():
    search_dirs = []
    if custom_dir:
        search_dirs.append(custom_dir)
        # If searching analysis dir, also include corners dir
        if custom_dir == DATA_ANALYSIS_DIR or custom_dir.endswith("/analysis"):
            corners_dir = os.path.join(DATA_DIR, "corners")
            if os.path.isdir(corners_dir):
                search_dirs.append(corners_dir)
    else:
        # prefer analysis dir
        if os.path.isdir(DATA_ANALYSIS_DIR):
            search_dirs.append(DATA_ANALYSIS_DIR)
        # also search corners dir
        corners_dir = os.path.join(DATA_DIR, "corners")
        if os.path.isdir(corners_dir):
            search_dirs.append(corners_dir)
        # legacy root
        if os.path.isdir(DATA_DIR):
            search_dirs.append(DATA_DIR)
    files = []
    for d in search_dirs:
        try:
            for f in os.listdir(d):
                if f.endswith(".json"):
                    # Match both full_league_suggestions_ and parsed_corners_predictions_ files
                    if f.startswith("full_league_suggestions_") or f.startswith("parsed_corners_predictions_"):
                        files.append(os.path.join(d, f))
        except Exception:
            continue
    # Deduplicate and sort newest first (based on filename timestamp or mtime fallback)
    uniq = list(dict.fromkeys(files))
    def sort_key(path):
        base = os.path.basename(path)
        # Handle both filename patterns
        ts_str = base.replace("full_league_suggestions_", "").replace("parsed_corners_predictions_", "").replace(".json", "")
        parts = ts_str.split("_")
        if len(parts) >= 1:
            ts = "_".join(parts[-2:]) if len(parts) >= 2 else parts[0]
            try:
                return datetime.strptime(ts, "%Y%m%d_%H%M%S")
            except ValueError:
                return datetime.fromtimestamp(os.path.getmtime(path))
        return datetime.fromtimestamp(os.path.getmtime(path))
    uniq.sort(key=sort_key, reverse=True)
    return uniq

files = list_suggestion_files()
if not files:
    st.warning("No suggestion files found in data/analysis or data/ directory.")
    st.stop()

# Load all suggestions from all files
all_suggestions = []
for filepath in files:
    file = os.path.basename(filepath)
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        suggestions = data.get("suggestions", []) or data.get("predictions", [])

        # Parse filename to extract league and timestamp
        # Handles both: full_league_suggestions_E0_20251120_232056.json
        #         and: parsed_corners_predictions_20251121.json
        league = "Unknown"
        timestamp = datetime.fromtimestamp(os.path.getmtime(filepath))
        file_type = "unknown"

        if file.startswith("full_league_suggestions_"):
            file_type = "full_league"
            ts_str = file.replace("full_league_suggestions_", "").replace(".json", "")
            parts = ts_str.split("_")
            if len(parts) >= 2:
                league = parts[0]
                timestamp_str = "_".join(parts[1:])
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                except ValueError:
                    pass
        elif file.startswith("parsed_corners_predictions_"):
            file_type = "corners"
            ts_str = file.replace("parsed_corners_predictions_", "").replace(".json", "")
            try:
                timestamp = datetime.strptime(ts_str, "%Y%m%d")
            except ValueError:
                pass
            league = "Corners"

        for s in suggestions:
            s_copy = s.copy()
            s_copy["file"] = file
            s_copy["full_path"] = filepath
            s_copy["league_code"] = s.get("league_code", league)
            s_copy["timestamp"] = timestamp
            s_copy["file_type"] = file_type
            all_suggestions.append(s_copy)
    except Exception as e:
        st.warning(f"Failed to load {file}: {e}")

if not all_suggestions:
    st.warning("No suggestions found in any file.")
    st.stop()

def flatten_suggestion(s):
    picks = s.get("picks", [])
    picks_str = ", ".join([f"{p['market']} {p['selection']} ({p['prob']*100:.1f}%, odds {p['odds']:.2f})" for p in picks]) if picks else "-"
    file_type = s.get("file_type", "unknown")
    file_type_icon = "📊" if file_type == "full_league" else "📍" if file_type == "corners" else "❓"

    # Handle different field names for corners vs league suggestions
    home = s.get("home", s.get("home_team", "-"))
    away = s.get("away", s.get("away_team", "-"))
    xg_home = s.get("xg_home", s.get("expected_home_corners", 0))
    xg_away = s.get("xg_away", s.get("expected_away_corners", 0))

    # For corners, show corner predictions instead of picks
    if file_type == "corners":
        total_corners = s.get("expected_total_corners", 0)
        picks_str = f"Total Corners: {total_corners:.1f} (H:{xg_home:.1f}, A:{xg_away:.1f})"

    return {
        "Type": file_type_icon,
        "File": s.get("file", "-"),
        "Timestamp": s.get("timestamp", datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
        "League": s.get("league_code", "-"),
        "Home": home,
        "Away": away,
        "xG Home": f"{xg_home:.2f}",
        "xG Away": f"{xg_away:.2f}",
        "Picks": picks_str,
        "Num Picks": len(picks) if picks else 1,  # Count corner predictions as 1
    }

df = pd.DataFrame([flatten_suggestion(s) for s in all_suggestions])

# Sidebar controls
st.sidebar.header("Filters")

# File type filter
file_types = sorted(df["Type"].unique())
file_type_labels = {
    "📊": "Full League Suggestions",
    "📍": "Corner Predictions",
    "❓": "Unknown"
}
selected_file_types_display = st.sidebar.multiselect(
    "File Type",
    [file_type_labels.get(ft, ft) for ft in file_types],
    default=[file_type_labels.get(ft, ft) for ft in file_types]
)
selected_file_types = [k for k, v in file_type_labels.items() if v in selected_file_types_display]
df_filtered = df[df["Type"].isin(selected_file_types)]

# League filter
leagues = sorted(df_filtered["League"].unique())
selected_leagues = st.sidebar.multiselect("League", leagues, default=leagues)
df_filtered = df_filtered[df_filtered["League"].isin(selected_leagues)]

# Date range
timestamps = pd.to_datetime(df_filtered["Timestamp"]) if not df_filtered.empty else pd.to_datetime(df["Timestamp"])
if not timestamps.empty:
    min_date = timestamps.min().date()
    max_date = timestamps.max().date()
    date_range = st.sidebar.date_input("Date range", [min_date, max_date])
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df_filtered[(pd.to_datetime(df_filtered["Timestamp"]).dt.date >= start_date) & (pd.to_datetime(df_filtered["Timestamp"]).dt.date <= end_date)]

# Probability threshold filter (applies to any pick's prob)
prob_threshold = st.sidebar.slider("Min pick probability (%)", 0.0, 100.0, 0.0, 1.0)
if prob_threshold > 0:
    keep_rows = []
    for s in all_suggestions:
        picks = s.get("picks", [])
        if any(p.get("prob", 0)*100 >= prob_threshold for p in picks):
            keep_rows.append(s)
    df_filtered = pd.DataFrame([flatten_suggestion(s) for s in keep_rows])

# Latest file only toggle
latest_only = st.sidebar.checkbox("Show latest file only", value=False)
if latest_only:
    latest_timestamp = df_filtered["Timestamp"].max()
    df_filtered = df_filtered[df_filtered["Timestamp"] == latest_timestamp]

# Sort options
sort_col = st.sidebar.selectbox("Sort by", ["Timestamp", "League", "Home", "Num Picks"], index=0)
ascending = st.sidebar.checkbox("Ascending", value=False)
df_filtered = df_filtered.sort_values(sort_col, ascending=ascending)

st.subheader("Suggestion Matches")
st.caption(f"Showing {len(df_filtered)} of {len(df)} total suggestions | Source dirs: {', '.join({os.path.dirname(f) for f in files})}")
st.dataframe(df_filtered, use_container_width=True)

st.markdown("---")
st.subheader("Match Details")
if df_filtered.empty:
    st.info("No rows match the current filters.")
else:
    row_idx = st.number_input("Select row (0-based)", min_value=0, max_value=len(df_filtered)-1, value=0)
    match = df_filtered.iloc[row_idx]
    st.write(f"**{match['Home']} vs {match['Away']}** (League: {match['League']}, File: {match['File']})")
    st.write(f"xG: {match['xG Home']} - {match['xG Away']}")
    st.write(f"Picks: {match['Picks']}")
    # Underlying JSON path
    full_path = next((s['full_path'] for s in all_suggestions if s.get('file') == match['File'] and (s.get('home') == match['Home'] or s.get('home_team') == match['Home'])), None)
    if full_path:
        with open(full_path, 'r') as f:
            raw_json = json.load(f)
        if st.checkbox("Show raw JSON for this file"):
            st.json(raw_json)
    # Export filtered view
    if st.button("Export filtered table to CSV"):
        export_name = f"filtered_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_filtered.to_csv(export_name, index=False)
        st.success(f"Exported to {export_name}")
