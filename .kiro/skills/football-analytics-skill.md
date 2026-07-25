---
name: Football Analytics Skill
description: Debug, implement, and optimize football prediction pipelines, ML models, and betting analysis workflows
applies_to: ["*.py", "docs/**/*.md", "config/**/*.json"]
---

# Football Analytics Analysis Skill

## When to Use
- **Debugging** ML predictions or combo market calculations
- **Implementing** new leagues or market types
- **Optimizing** fixture analysis or EV calculations
- **Validating** data pipelines and model freshness
- **Fixing** workflow orchestration or API integration issues

## ML Model System
- **Models**: XGBoost, RandomForest in `models/baseline_v1_*/`
- **Signature**: Deterministic hash validation (prevent duplicate predictions)
- **Freshness**: Max age configurable (default 7 days)
- **Output**: Predictions with team, odds, model, confidence

## Combo Markets
- **Definition**: Multi-leg betting combinations (Goal + OU)
- **EV Calculation**: (1/odds - 1) × 100
- **Threshold**: Configurable (default 5%)
- **Data Source**: Odds + OU market data required
- **Issue**: "No opportunities" → check OU market data presence

## Fixture Pipeline
```
API → data/raw/upcomingMatches.json 
   → data/fixtures/ (cached)
   → league_analysis/ (processed)
   → Results/ (output)
```

## League Codes
```
E0=England Prem, E1-E3=English divisions
F1=French Ligue 1
I1=Italian Serie A
N1=Dutch Eredivisie
SP1=Spanish La Liga
```

## Common Debugging Tasks

### Debug Prediction Signature Failures
1. Check `ML_MODELS.md` Section 2 (Predictions Fix)
2. Verify deterministic hashing in `src/prediction_validators.py`
3. Test with `python validate_ml_predictions.py`
4. Review recent model updates in `models/baseline_v1_*/metadata.json`

### Add New League or Market Type
1. Update `config/leagues.json` (verify 2-letter code)
2. Add test case in `tests/`
3. Run: `python run_analysis_workflow.py --leagues [CODE]`
4. Verify results in `data/league_analysis/`

### Optimize EV Threshold
1. Review current settings in workflow args
2. Test with range (3%, 5%, 8%)
3. Compare recommendation volume vs quality in output
4. Document in `docs/PARAMETER_SUMMARY.md`

### Validate Data Pipeline
1. Check fixture freshness: `data/todays_fixtures_*.json`
2. Verify league_data_*.csv in `data/` (one per league code)
3. Run: `python test_fixtures_download.py`
4. Check `logs/` for pipeline errors

## Debugging Checklist
- [ ] Is model fresh? Check `model_max_age` in workflow args
- [ ] Does fixture data exist? Verify `data/todays_fixtures_*.json`
- [ ] League codes correct? Confirm in `config/leagues.json`
- [ ] Combo market data present? Check for OU odds in output
- [ ] Signature collision? Search `FIXES_AND_VALIDATIONS.md` Section 2
- [ ] Workflow completed? Review `data/Results/` output files

## File Organization
- **Config**: `config/` (leagues.json, api_keys.env, train_config.yaml)
- **Data**: `data/` (raw, fixtures, league_analysis, Results)
- **Models**: `models/baseline_v1_*` (trained XGBoost/RandomForest)
- **Source**: `src/` (core logic, validators, enrichment)
- **Tests**: `test_*.py` (validation suite)
- **Docs**: `docs/` (architecture, troubleshooting, guides)
