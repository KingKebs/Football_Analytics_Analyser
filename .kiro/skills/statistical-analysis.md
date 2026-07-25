---
name: Statistical Analysis Skill
description: Implement Poisson models, probability calculations, and statistical validation
applies_to: ["src/**/*.py", "docs/**/*.md"]
---

# Statistical Analysis Skill

## Poisson Model
- **Base**: Poisson probability distribution for goal counts
- **Usage**: Generate goal probabilities, predict match outcomes
- **Input**: Team attack/defense ratings
- **Output**: Home/Draw/Away probabilities

## Key Calculations
- **Team Strength**: Attack and defense ratings (logs)
- **Expected Goals**: λ parameter from historical data
- **Goal Probability**: P(X=k) = (e^-λ × λ^k) / k!
- **Match Outcomes**: Combine both teams' probabilities

## Probability Models
- 1X2 (Home/Draw/Away)
- Over/Under Goals
- Both Teams Score
- Asian Handicap
- Combo Markets (multi-leg combinations)

## Validation Methods
- Cross-validation (k-fold)
- Accuracy metrics (precision, recall, ROC-AUC)
- Calibration analysis
- Comparison vs ML models
- Backtesting on historical data

## EV (Expected Value) Calculation
```
EV = (Probability × Odds) - 1
```
- Positive EV = Value bet (expected profit)
- Threshold typically 5% for recommendations

## Statistical Tests
- Chi-square tests (model fit)
- Z-tests (probability differences)
- Correlation analysis (team ratings)
- Confidence intervals (prediction bands)

## Common Tasks
- **Calculate Team Ratings**: From historical data
- **Generate Predictions**: Poisson probability model
- **Compare vs ML**: Statistical significance tests
- **Validate Thresholds**: EV analysis, sensitivity
- **Calibration Check**: Predicted vs actual outcomes

## Integration with ML
- Poisson as baseline/fallback
- ML model comparison analysis
- Ensemble predictions (combine both)
- Statistical validation of ML improvements
