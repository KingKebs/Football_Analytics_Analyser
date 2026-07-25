---
name: Testing & CI Skill
description: Run tests, validate pipelines, and ensure code quality
applies_to: ["test_*.py", "src/**/*.py"]
---

# Testing & CI Skill

## Test Suites

### Integration Tests (`test_phase1_integration.py`)
- Full workflow execution
- End-to-end data pipeline
- Fixture loading and processing
- Analysis generation
- Result validation

### Prediction Signature Tests (`test_prediction_signatures.py`)
- Deterministic hash validation
- Duplicate prediction detection
- Signature consistency
- Model prediction consistency

### ML Validation (`test_ml_fix.py`)
- Model loading and inference
- Feature engineering correctness
- Prediction format validation
- Signature verification

### Fixtures Download (`test_fixtures_download.py`)
- API connectivity
- Fixture data parsing
- Date filtering
- League filtering

## Running Tests
```bash
# Full integration test
python test_phase1_integration.py

# Prediction signatures
python test_prediction_signatures.py

# ML validation
python test_ml_fix.py

# Fixtures download
python test_fixtures_download.py

# All tests
bash test_system.sh
```

## Validation Checks
- Data integrity (CSV/JSON format)
- Pipeline completion (all stages)
- Output generation (Results files)
- Prediction quality (EV, confidence)
- Signature consistency (no duplicates)

## CI Pipeline
- Pre-commit: Syntax check, formatting
- Pull request: Full test suite
- Merge: Integration test validation
- Deployment: End-to-end workflow test

## Common Issues & Fixes
- **Signature Conflicts**: Run signature validation test
- **Pipeline Failures**: Check integration test logs
- **ML Prediction Issues**: Run ML validation test
- **Fixture Data Problems**: Run fixtures download test

## Success Criteria
- All tests pass ✅
- Zero signature conflicts ✅
- All output files generated ✅
- Predictions have valid EV ✅
- No regressions vs baseline ✅
