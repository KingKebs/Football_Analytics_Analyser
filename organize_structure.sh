#!/bin/bash

# Script to organize files and create directory structure for Football_Analytics_Analyser
# This script creates the required directories and moves files to match the project structure

set -e  # Exit on any error

echo "Organizing Football Analytics Analyser project structure..."

# Create main directories
mkdir -p data
mkdir -p football-data
mkdir -p tmp
mkdir -p __pycache__
mkdir -p venv
mkdir -p .git
mkdir -p .idea
mkdir -p visuals

# Create subdirectories
mkdir -p data/"old csv"
mkdir -p football-data/all-euro-football

# Move files to their correct locations (if they exist and are not already there)

# Root level files (most should already be in place)
# README.md, README_Downloader_Script.md, ALGORITHMS.md, requirements.txt
# algorithms.py, automate_football_analytics.py, automate_football_analytics_fullLeague.py
# download_all_tabs.py, visualize_score_table.py
# .gitignore, .DS_Store (if exists)

# Data directory files
# Move any home_away_team_strengths_*.csv files to data/
if ls home_away_team_strengths_*.csv 1> /dev/null 2>&1; then
    mv home_away_team_strengths_*.csv data/ 2>/dev/null || true
fi

# Move any league_data_*.csv files to data/
if ls league_data_*.csv 1> /dev/null 2>&1; then
    mv league_data_*.csv data/ 2>/dev/null || true
fi

# Move any suggestion_*.json files to data/
if ls suggestion_*.json 1> /dev/null 2>&1; then
    mv suggestion_*.json data/ 2>/dev/null || true
fi

# Football-data directory
# Move data.zip to football-data/ if it exists in root
if [ -f data.zip ]; then
    mv data.zip football-data/
fi

# Move any *.csv files in root that might belong to all-euro-football (though they should already be there)
# Assuming all-euro-football CSVs are already in place

# Tmp directory
# Move any *.log files in root to tmp/
if ls *.log 1> /dev/null 2>&1; then
    mv *.log tmp/ 2>/dev/null || true
fi

# __pycache__ directory
# Move any *.pyc files in root to __pycache__/
if ls *.pyc 1> /dev/null 2>&1; then
    mv *.pyc __pycache__/ 2>/dev/null || true
fi

# Move any __pycache__ subdirs content if needed
if [ -d __pycache__ ] && ls __pycache__/*.pyc 1> /dev/null 2>&1; then
    # Already in place
    true
fi

# Venv directory (usually already exists)
# .git, .idea should already exist

# Visuals directory
# Move any *.png files in root to visuals/
if ls *.png 1> /dev/null 2>&1; then
    mv *.png visuals/ 2>/dev/null || true
fi

echo "Project structure organization complete!"
echo ""
echo "Current structure:"
find . -type f -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.csv" -o -name "*.json" -o -name "*.log" -o -name "*.pyc" | head -20
echo "... (truncated)"
echo ""
echo "To verify the full structure, run: tree -a -I '.git|venv' or ls -la"
