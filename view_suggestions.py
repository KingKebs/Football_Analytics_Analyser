import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime

DATA_DIR = "data"

st.set_page_config(page_title="Football Analytics Suggestions Viewer", layout="wide")
st.title("📊 Football Analytics Suggestions Viewer")

# List all suggestion JSON files in data/
def list_suggestion_files():
    files = [f for f in os.listdir(DATA_DIR) if f.startswith("full_league_suggestions_") and f.endswith(".json")]
    files.sort(reverse=True)
    return files

files = list_suggestion_files()
if not files:
    st.warning("No suggestion files found in data/ directory.")
    st.stop()

# Load all suggestions from all files
all_suggestions = []
for file in files:
    try:
        with open(os.path.join(DATA_DIR, file), "r") as f:
            data = json.load(f)
        suggestions = data.get("suggestions", [])
        # Parse file name to extract league and timestamp
        parts = file.replace("full_league_suggestions_", "").replace(".json", "").split("_")
        if len(parts) >= 2:
            league = parts[0]
            timestamp_str = "_".join(parts[1:])
            try:
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except ValueError:
                timestamp = datetime.now()  # fallback
        else:
            league = "Unknown"
            timestamp = datetime.now()
        for s in suggestions:
            s_copy = s.copy()
            s_copy["file"] = file
            s_copy["league_code"] = league
            s_copy["timestamp"] = timestamp
            all_suggestions.append(s_copy)
    except Exception as e:
        st.warning(f"Failed to load {file}: {e}")

if not all_suggestions:
    st.warning("No suggestions found in any file.")
    st.stop()

# Convert to DataFrame for easier filtering
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
    }

df = pd.DataFrame([flatten_suggestion(s) for s in all_suggestions])

# Sidebar filters
leagues = sorted(df["League"].unique())
selected_leagues = st.sidebar.multiselect("Filter by league:", leagues, default=leagues)
df_filtered = df[df["League"].isin(selected_leagues)]

# Filter by date range
timestamps = pd.to_datetime(df["Timestamp"])
min_date = timestamps.min().date()
max_date = timestamps.max().date()
date_range = st.sidebar.date_input("Filter by date range:", [min_date, max_date])
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df_filtered[(pd.to_datetime(df_filtered["Timestamp"]).dt.date >= start_date) & (pd.to_datetime(df_filtered["Timestamp"]).dt.date <= end_date)]

st.dataframe(df_filtered, use_container_width=True)

# Show details for a selected match
st.markdown("---")
st.subheader("Match Details")
row_idx = st.number_input("Select match row for details (0-based):", min_value=0, max_value=len(df_filtered)-1, value=0)
match = df_filtered.iloc[row_idx]
st.write(f"**{match['Home']} vs {match['Away']}** (League: {match['League']}, File: {match['File']})")
st.write(f"xG: {match['xG Home']} - {match['xG Away']}")
st.write(f"Picks: {match['Picks']}")
