# Football Analytics Analyser - Project Structure Analysis

**Generated:** November 13, 2025

---

## Current State Overview

### ✅ What's Working Well

1. **Core Analysis Workflow**
   - `automate_football_analytics_fullLeague.py` - Main entry point for league analysis
   - `algorithms.py` - Contains rating models, Poisson/xG calculations, odds utilities
   - Data pipeline: CSV → Analysis → JSON suggestions

2. **Data Management**
   - Downloads: `download_all_tabs.py` fetches from football-data.co.uk
   - Organization: `organize_structure.py` structures raw data
   - Storage: `/data/` directory contains 40+ analysis files
   - CSV outputs: League tables, team strengths, fixture data

3. **Output Generation**
   - Full league suggestions (JSON): `full_league_suggestions_*.json`
   - Team strengths: `home_away_team_strengths*.csv`
   - Analysis results: `analysis_results_*.json`
   - Parlay generation & betting strategies

4. **Visualization & Reporting**
   - `visualize_full_league.py` - League-wide visualizations
   - `visualize_score_table.py` - Score matrix plots
   - `web_output_examples.py` - Dashboard examples

---

## 🚨 Current Issues & Pain Points

### 1. **Scattered Analysis Scripts** (No Unified CLI)
```
Root level: 8+ independent Python scripts
├── automate_football_analytics.py (single match)
├── automate_football_analytics_fullLeague.py (league)
├── analyze_suggestions_results.py (backtest harness)
├── view_suggestions.py (results viewer)
├── parse_match_log.py (log parsing)
├── check_downloaded_data.py (data validation)
├── organize_structure.py (data organization)
└── download_all_tabs.py (data downloader)
```

**Problem:** No central entry point. Users need to know which script to run for each task.

### 2. **Data Organization Issues**
- `/data/` contains 40+ mixed files (JSON, CSV, old CSVs in subdirectories)
- Naming convention inconsistent (full_league_suggestions_* vs league_data_*)
- No clear separation between:
  - Raw input data
  - Intermediate processing outputs
  - Final analysis results
  - Temporary/deprecated files

### 3. **Outdated/Deprecated Files**
- `data/old csv/` - unclear what these are
- Multiple duplicate league_data files (B1, D1, D2, E0, E1, etc.)
- Old suggestion files from October (suggestion_Benfica_*.json)
- Unclear file retention policy

### 4. **Configuration Management**
- CLI flags scattered across different scripts
- No centralized config file or environment setup
- Hard to reproduce analyses with different parameters

### 5. **Documentation Gaps**
- README covers basic usage but not:
  - Full workflow description
  - Available CLI options
  - How to chain tasks together
  - Troubleshooting guide
  - File output structure

### 6. **Testing & Validation**
- `/tests/` directory exists but likely empty or minimal
- No validation pipelines for:
  - Downloaded data integrity
  - Output consistency
  - Regression testing

### 7. **Logging & Observability**
- `/tmp/` contains various `.log` files but no centralized logging
- Hard to track execution history across multiple script runs

---

## 📋 Recommended Actions

### Priority 1: Create Unified CLI Tool
Build a single entry point that delegates to all analysis workflows:

```bash
python cli.py --task full-league --league E0 --model blended
python cli.py --task analyze-single --home Team1 --away Team2
python cli.py --task backtest --results-file data/full_league_suggestions_*.json
python cli.py --task download --leagues E0,SP1 --seasons AUTO
python cli.py --task organize --source football-data --target data/processed
python cli.py --task validate --check-data
```

**Deliverable:** `cli.py` - unified command dispatcher with help, logging, error handling

---

### Priority 2: Reorganize Data Directory

```
data/
├── raw/                          # Original CSVs from football-data.co.uk
│   ├── E0_2324.csv
│   ├── E0_2425.csv
│   └── ...
├── processed/                    # Cleaned & enriched data
│   ├── league_tables/
│   └── team_strengths/
├── analysis/                     # Analysis outputs (JSON)
│   ├── 20251109_E0_suggestions.json
│   ├── 20251109_SP1_suggestions.json
│   └── ...
├── archive/                      # Old results (>7 days)
│   └── 2025-10-??/
├── temp/                         # Working files (auto-cleanup)
└── cache/                        # Model coefficients, lookup tables
```

**Deliverable:** `organize_data_structure.py` - automated restructuring + migration script

---

### Priority 3: Enhance Documentation

Create these docs:
1. **WORKFLOW.md** - Step-by-step guide for each analysis type
2. **FILE_MANIFEST.md** - Purpose of every file/directory
3. **TROUBLESHOOTING.md** - Common errors and solutions
4. **QUICK_REFERENCE.md** - 1-page CLI cheat sheet

---

### Priority 4: Consolidate Logging

Replace scattered logs in `/tmp/` with:
- Centralized logging to `logs/` directory
- Timestamped per-execution logs
- Log levels: DEBUG, INFO, WARNING, ERROR
- Searchable log aggregator

---

### Priority 5: Build Configuration Module

Create `config.py`:
```python
class AnalysisConfig:
    LEAGUE = 'E0'
    SEASON = '2024-25'
    RATING_MODEL = 'blended'
    BLEND_WEIGHT = 0.3
    LAST_N_MATCHES = 6
    ...
```

Allow override via:
- CLI flags
- Environment variables
- Config file (config.yaml)
- Interactive prompts

---

## 🎯 Implementation Roadmap

| Phase | Task | Est. Time | Deliverable |
|-------|------|-----------|------------|
| 1 | Create unified CLI | 2-3 hrs | `cli.py` |
| 2 | Add CLI help & validation | 1 hr | Help text + error handling |
| 3 | Reorganize data structure | 1.5 hrs | `organize_data_structure.py` + migration |
| 4 | Centralize logging | 1 hr | `logging.yaml` + logger setup |
| 5 | Write documentation | 2 hrs | WORKFLOW.md + QUICK_REFERENCE.md |
| 6 | Refactor config | 1.5 hrs | `config.py` + config.yaml |
| **Total** | | **~9 hours** | Production-ready CLI tool suite |

---

## 💡 Quick Wins (Implement Now)

1. ✅ **Corners Analysis** - Already scaffolded! (`corners_analysis.py`)
   - Integrates with data pipeline
   - Outputs model-ready features

2. **Add script** - `validate_data.py`
   ```bash
   python validate_data.py --check-all
   # Validates: CSV integrity, corner data, team coverage, date ranges
   ```

3. **Add script** - `cleanup_data.py`
   ```bash
   python cleanup_data.py --archive-old --days 7
   # Moves data >7 days old to archive/
   ```

4. **Add to README**
   ```
   ## Quick Tasks
   - Full league analysis: python cli.py --task full-league --league E0
   - Analyze corners: python cli.py --task analyze-corners
   - View results: python cli.py --task view --file latest
   ```

---

## 📊 File Inventory

**Scripts (Root):** 12 files
- Core analysis (4): automate_*, analyze_*, algorithms.py
- Utilities (5): download_*, organize_*, parse_*, check_*, view_*
- Visualization (3): visualize_*, web_output_*

**Data Output (data/):** 44 files
- Analysis results: 18 JSON files
- Team data: 24 CSV files
- Supporting: 2 misc files

**Configuration:** 1 file (requirements.txt)

**Documentation:** 3 files (README.md, ENHANCEMENTS_TO_SCRIPT.md, ALGORITHMS.md)

---

## ✨ Next Steps

1. **Immediate (Today):** Create `cli.py` unified interface
2. **Short-term (This week):** Reorganize `/data/` directory
3. **Medium-term:** Write comprehensive documentation
4. **Long-term:** Build web dashboard (Streamlit app already scaffolded)

---

*This analysis helps prioritize development efforts and reduce cognitive load for future maintenance.*

