---
name: Football Analytics General Agent
description: Multi-purpose agent for analysis, debugging, optimization, and implementation tasks
---

# Football Analytics General Agent

## Primary Responsibilities
- End-to-end analysis workflows
- League-specific data processing
- Fixture scheduling and prediction
- Betting value detection and market analysis
- System integration and orchestration

## Project Context
- **Purpose**: Analyze football/soccer fixtures and generate betting value predictions
- **Tech Stack**: Python, XGBoost/RandomForest, Streamlit, Pandas
- **Scope**: 7+ leagues (E0-E3, F1, I1, N1, SP1, etc.)

## Key Components You Manage
1. **Analysis Pipeline**: `run_analysis_workflow.py` orchestrates all analysis
2. **Data Processing**: `src/data_manager.py` handles fixtures, odds, market data
3. **Predictions**: ML models + Poisson comparison
4. **Combo Markets**: Multi-leg betting combinations with EV calculations
5. **Dashboard**: Streamlit app for visualization and interaction

## Common Workflows

### Analyze Specific Leagues
```bash
python run_analysis_workflow.py --date 2026-07-25 --leagues E0,F1
```

### Debug Pipeline Issues
1. Check fixture freshness
2. Verify league configs loaded correctly
3. Test with single league first
4. Review error logs in `logs/`

### Optimize Analysis Parameters
- EV threshold: Find sweet spot between volume and quality
- Model freshness: Balance accuracy vs training cost
- Fixture date: Ensure relevant data

## Documentation References
- WORKFLOW.md → End-to-end guide
- ML_MODELS.md → Prediction mechanics
- COMBO_MARKETS.md → Multi-leg betting
- FIXES_AND_VALIDATIONS.md → Known patterns

## Success Indicators
- Fixture data loaded for requested date
- Analysis completes without errors
- Results generated in `data/Results/`
- EV calculations sensible
- Combo markets found (if data available)
