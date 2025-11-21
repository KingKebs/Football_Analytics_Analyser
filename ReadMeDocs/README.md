# Football Analytics Analyser

A comprehensive Python-based football match analysis and prediction system with rating models, ML pipelines, corner analysis, and backtesting capabilities.

**Status:** ✅ Production Ready | **Last Updated:** November 2025

---

## 📖 Quick Navigation

### 🚀 Getting Started
- **[START_HERE.md](ReadMeDocs/START_HERE.md)** - First-time user guide (10 min read)
- **[QUICK_START.md](ReadMeDocs/QUICK_START.md)** - Quick setup and basic commands
- **[QUICK_REFERENCE.md](ReadMeDocs/QUICK_REFERENCE.md)** - CLI command reference

### 📊 Core Features
- **[SYSTEM_OVERVIEW.md](ReadMeDocs/SYSTEM_OVERVIEW.md)** - Project capabilities and architecture
- **[PROJECT_STRUCTURE_ANALYSIS.md](ReadMeDocs/PROJECT_STRUCTURE_ANALYSIS.md)** - Detailed architecture audit

### ⚙️ Usage Guides
- **[WORKFLOW_GUIDE.md](ReadMeDocs/WORKFLOW_GUIDE.md)** - Complete analysis workflows
- **[FILE_MANAGEMENT_GUIDE.md](ReadMeDocs/FILE_MANAGEMENT_GUIDE.md)** - Data organization and cleanup
- **[ML_MODE_GUIDE.md](ReadMeDocs/ML_MODE_GUIDE.md)** - Machine learning integration

### 🎯 Corner Analysis (NEW)
- **[SUMMARY_CORNERS_ANALYSIS.md](ReadMeDocs/SUMMARY_CORNERS_ANALYSIS.md)** - Executive summary (2 min read)
- **[CORNERS_ANALYSIS_QUICK_EXPLANATION.md](ReadMeDocs/CORNERS_ANALYSIS_QUICK_EXPLANATION.md)** - High-level overview
- **[CORNERS_ANALYSIS_FLOW.md](ReadMeDocs/CORNERS_ANALYSIS_FLOW.md)** - Detailed execution flow with code
- **[CORNERS_ANALYSIS_ARCHITECTURE.md](ReadMeDocs/CORNERS_ANALYSIS_ARCHITECTURE.md)** - Visual data flow diagrams

### 📚 Reference & Analysis
- **[CORNER_PREDICTIONS_GUIDE.md](ReadMeDocs/CORNER_PREDICTIONS_GUIDE.md)** - Corner prediction methodology
- **[CORNER_VS_FULLLEAGUE_COMPARISON.md](ReadMeDocs/CORNER_VS_FULLLEAGUE_COMPARISON.md)** - Compare corner and full league analysis
- **[AUTOMATION_COMPARISON.md](ReadMeDocs/AUTOMATION_COMPARISON.md)** - Automation strategies comparison
- **[SCRIPT_FLOW_DIAGRAM.md](ReadMeDocs/SCRIPT_FLOW_DIAGRAM.md)** - Script execution flows
- **[SCRIPT_RELATIONSHIPS.md](ReadMeDocs/SCRIPT_RELATIONSHIPS.md)** - How scripts interact
- **[MODELING_PROGRESS.md](ReadMeDocs/MODELING_PROGRESS.md)** - Model development history
- **[IMPLEMENTATION_SUMMARY.md](ReadMeDocs/IMPLEMENTATION_SUMMARY.md)** - Implementation notes
- **[INDEX.md](ReadMeDocs/INDEX.md)** - Comprehensive file index
- **[ThisIsNotYetAmodel.md](ReadMeDocs/ThisIsNotYetAmodel.md)** - Predictive modeling guide

---

## 🎬 Quick Start

### Installation
```bash
pip install -r requirements.txt
python setup.py  # Validate environment
```

### Basic Usage

**Analyze a league:**
```bash
python cli.py --task full-league --league E0 --rating-model blended
```

**Download data:**
```bash
python cli.py --task download --leagues E0,SP1 --season AUTO
```

**Corner analysis:**
```bash
python cli.py --task corners --league ALL --use-parsed-all --fixtures-date 20251121
```

**View results:**
```bash
python cli.py --task view
```

---

## 📁 Project Structure

```
Football_Analytics_Analyser/
├── Core Analysis Scripts
│   ├── cli.py                              # Unified CLI dispatcher
│   ├── algorithms.py                       # Core algorithms
│   ├── automate_football_analytics.py      # Single match analysis
│   ├── automate_football_analytics_fullLeague.py  # League analysis
│   └── corners_analysis.py                 # Corner pattern analysis
│
├── Machine Learning
│   ├── ml_training.py                      # Model training pipeline
│   ├── ml_evaluation.py                    # Cross-validation metrics
│   ├── ml_features.py                      # Feature engineering
│   ├── ml_utils.py                         # ML utilities
│   └── train_pipeline.py                   # End-to-end training
│
├── Data Management
│   ├── data_manager.py                     # Data organization tool
│   ├── download_all_tabs.py                # Download from football-data.co.uk
│   ├── organize_structure.py               # Data structuring
│   ├── check_downloaded_data.py            # Data validation
│   └── parse_match_log.py                  # Log parsing
│
├── Utilities & Visualization
│   ├── view_suggestions.py                 # Results viewer (Streamlit)
│   ├── analyze_suggestions_results.py      # Backtest analyzer
│   ├── visualize_full_league.py            # League visualization
│   ├── visualize_score_table.py            # Score table plots
│   └── web_output_examples.py              # Web dashboard examples
│
├── Configuration & Setup
│   ├── setup.py                            # Environment validator
│   ├── config.example.yaml                 # Config template
│   └── requirements.txt                    # Dependencies
│
├── Documentation (THIS DIRECTORY ONLY!)
│   └── ReadMeDocs/                         # All markdown documentation
│       ├── START_HERE.md                   # Entry point for new users
│       ├── QUICK_START.md
│       ├── SYSTEM_OVERVIEW.md
│       ├── CORNERS_ANALYSIS_*.md           # Corner analysis guides
│       └── ...15 more guides...
│
├── Data Directories
│   ├── data/                               # Analysis results & outputs
│   ├── football-data/                      # Downloaded league CSVs
│   ├── models/                             # Trained ML models
│   ├── logs/                               # Execution logs
│   └── ReadMeDocs/                         # **Documentation only**
│
└── Supporting
    ├── tests/                              # Test files
    ├── tools/                              # Utility scripts
    └── .gitignore, requirements.txt, etc.
```

---

## 🔧 Key Tasks

| Task | Command | Purpose |
|------|---------|---------|
| Download Data | `python cli.py --task download --leagues E0,SP1` | Fetch from football-data.co.uk |
| Organize Data | `python cli.py --task organize` | Structure raw data |
| Validate Data | `python cli.py --task validate --check-all` | Verify data integrity |
| Analyze League | `python cli.py --task full-league --league E0` | Generate match predictions |
| Single Match | `python cli.py --task single-match --home Team1 --away Team2` | Analyze one match |
| Corner Analysis | `python cli.py --task corners --league ALL --use-parsed-all` | Predict corners |
| View Results | `python cli.py --task view` | View suggestions (Streamlit) |
| Backtest | `python cli.py --task backtest` | Evaluate ROI/yield |

---

## 🎯 Core Capabilities

### ✅ Rating Models
- **Blended**: Recent form + baseline ratings (recommended)
- **Poisson**: Shot-based probability distribution
- **xG**: Expected goals model

### ✅ ML Integration
- Linear Regression, RandomForest, XGBoost
- 5-fold cross-validation with recency weighting
- Feature engineering (19+ engineered features per match)

### ✅ Corner Analysis
- Team corner statistics and correlations
- 1H/2H split estimation
- Multi-league fixture prediction
- Batch processing with skip tracking

### ✅ Data Management
- Automatic file archival (>7 days old)
- Data manifest generation
- JSON validation and cleanup
- League-based organization

### ✅ Visualization
- Streamlit-based results viewer
- Score table plots
- Web dashboard examples
- Backtest ROI analysis

---

## 📊 Supported Leagues

**England:** E0 (EPL), E1 (Championship), E2 (League One), E3 (League Two)  
**Germany:** D1 (Bundesliga), D2 (2. Bundesliga)  
**Spain:** SP1 (La Liga), SP2 (Segunda)  
**Italy:** I1 (Serie A), I2 (Serie B)  
**France:** F1 (Ligue 1), F2 (Ligue 2)  
**Other:** N1 (Netherlands), P1 (Portugal), SC0-SC3 (Scotland), B1 (Belgium), G1 (Greece), T1 (Turkey), EC (Europe)

---

## 🚀 Example Workflows

### Workflow 1: Full Analysis
```bash
# Download latest data
python cli.py --task download --leagues E0,SP1 --season AUTO

# Validate
python cli.py --task validate --check-all

# Analyze both leagues
python cli.py --task full-league --league E0,SP1 --ml-mode predict

# View results
python cli.py --task view

# Backtest predictions
python cli.py --task backtest
```

### Workflow 2: Corner Predictions
```bash
# Analyze all leagues and predict corners for upcoming fixtures
python cli.py --task corners --league ALL --use-parsed-all --fixtures-date 20251121 --top-n 5

# Output: data/corners/parsed_corners_predictions_20251121.json
```

### Workflow 3: Single Match Analysis
```bash
python cli.py --task single-match --home "Arsenal" --away "Chelsea" --rating-model blended
```

---

## 📈 Performance Characteristics

| Operation | Duration | Notes |
|-----------|----------|-------|
| Setup validation | ~20 sec | One-time only |
| Data download (1 league) | 2-5 min | Network dependent |
| League analysis (E0) | 10-30 sec | Historical data size |
| Corner analysis (1 league) | 3-8 sec | Pandas operations |
| All 22 leagues + fixture predictions | 2-3 min | Cached stats reuse |

---

## 🐛 Troubleshooting

**"ModuleNotFoundError"**
```bash
pip install -r requirements.txt
```

**"File not found"**
```bash
python cli.py --task download --leagues E0
```

**"JSON decode error"**
```bash
python data_manager.py --validate
python cli.py --task full-league --league E0
```

**Check logs**
```bash
tail -50 logs/cli_*.log
```

---

## 📚 Documentation Organization

All documentation is consolidated in **ReadMeDocs/** directory:
- Entry point: `ReadMeDocs/START_HERE.md`
- Quick reference: `ReadMeDocs/QUICK_REFERENCE.md`
- Corner analysis: `ReadMeDocs/CORNERS_ANALYSIS_*.md`
- Complete index: `ReadMeDocs/INDEX.md`

**Root directory contains only code and configuration, not documentation.**

---

## ✨ Key Features Summary

| Feature | Status | Location |
|---------|--------|----------|
| Unified CLI | ✅ | cli.py |
| Rating Models | ✅ | algorithms.py |
| ML Integration | ✅ | ml_*.py |
| Corner Analysis | ✅ | corners_analysis.py |
| Data Management | ✅ | data_manager.py |
| Streamlit Viewer | ✅ | view_suggestions.py |
| Backtesting | ✅ | analyze_suggestions_results.py |
| Comprehensive Docs | ✅ | ReadMeDocs/ |

---

## 🎓 Learning Path

**Beginner (30 min):**
1. Read `ReadMeDocs/START_HERE.md`
2. Run `python setup.py`
3. Run first analysis: `python cli.py --task full-league --league E0`

**Intermediate (2 hours):**
1. Read `ReadMeDocs/SYSTEM_OVERVIEW.md`
2. Try all CLI tasks
3. Review `ReadMeDocs/WORKFLOW_GUIDE.md`

**Advanced (4 hours):**
1. Study `algorithms.py` (core rating models)
2. Review `ml_training.py` (ML pipeline)
3. Explore `corners_analysis.py` (corner prediction)
4. Review `ReadMeDocs/PROJECT_STRUCTURE_ANALYSIS.md`

---

## 📝 License & Attribution

This project analyzes publicly available football data from football-data.co.uk.

---

## 🤝 Support & Questions

```bash
# Show help
python cli.py --task help

# Check environment
python setup.py

# View logs
tail logs/cli_*.log

# Full documentation
cat ReadMeDocs/START_HERE.md
```

---

**Status:** ✅ Production Ready  
**Documentation:** Complete (18 guides in ReadMeDocs/)  
**Code Quality:** Tested and validated  
**Last Updated:** November 21, 2025


