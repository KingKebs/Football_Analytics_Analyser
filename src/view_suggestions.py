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
    else:
        # prefer analysis dir
        if os.path.isdir(DATA_ANALYSIS_DIR):
            search_dirs.append(DATA_ANALYSIS_DIR)
        # legacy root
        if os.path.isdir(DATA_DIR):
            search_dirs.append(DATA_DIR)
    files = []
    for d in search_dirs:
        try:
            for f in os.listdir(d):
                if f.startswith("full_league_suggestions_") and f.endswith(".json"):
                    files.append(os.path.join(d, f))
        except Exception:
            continue
    # Deduplicate and sort newest first (based on filename timestamp or mtime fallback)
    uniq = list(dict.fromkeys(files))
    def sort_key(path):
        base = os.path.basename(path)
        parts = base.replace("full_league_suggestions_", "").replace(".json", "").split("_")
        if len(parts) >= 2:
            ts = "_".join(parts[1:])
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
        suggestions = data.get("suggestions", [])
        parts = file.replace("full_league_suggestions_", "").replace(".json", "").split("_")
        if len(parts) >= 2:
            league = parts[0]
            timestamp_str = "_".join(parts[1:])
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except ValueError:
                timestamp = datetime.fromtimestamp(os.path.getmtime(filepath))
        else:
            league = "Unknown"
            timestamp = datetime.fromtimestamp(os.path.getmtime(filepath))
        for s in suggestions:
            s_copy = s.copy()
            s_copy["file"] = file
            s_copy["full_path"] = filepath
            s_copy["league_code"] = s.get("league_code", league)
            s_copy["timestamp"] = timestamp
            all_suggestions.append(s_copy)
    except Exception as e:
        st.warning(f"Failed to load {file}: {e}")

if not all_suggestions:
    st.warning("No suggestions found in any file.")
    st.stop()

def flatten_suggestion(s):
    picks = s.get("picks", [])
    picks_str = ", ".join([f"{p['market']} {p['selection']} ({p['prob']*100:.1f}%, odds {p['odds']:.2f})" for p in picks]) if picks else "-"
    return {
        "File": s.get("file", "-"),
        "Timestamp": s.get("timestamp", datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
        "League": s.get("league_code", "-"),
        "Home": s.get("home", "-"),
        "Away": s.get("away", "-"),
        "xG Home": f"{s.get('xg_home', 0):.2f}",
        "xG Away": f"{s.get('xg_away', 0):.2f}",
        "Picks": picks_str,
        "Num Picks": len(picks),
    }

df = pd.DataFrame([flatten_suggestion(s) for s in all_suggestions])

# Sidebar controls
st.sidebar.header("Filters")
leagues = sorted(df["League"].unique())
selected_leagues = st.sidebar.multiselect("League", leagues, default=leagues)
df_filtered = df[df["League"].isin(selected_leagues)]

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
    full_path = next((s['full_path'] for s in all_suggestions if s.get('file') == match['File'] and s.get('home') == match['Home']), None)
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
