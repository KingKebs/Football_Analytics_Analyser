# Niche Markets Guide

**Last Updated:** December 2025  
**Covers:** Odd/Even goals, Highest Scoring Half, Cross-League Parlays, parameter tuning

---

## Contents

1. [Market Overview & Statistical Drivers](#1-market-overview--statistical-drivers)
2. [Priority Leagues for Niche Markets](#2-priority-leagues-for-niche-markets)
3. [Running Niche Market Predictions](#3-running-niche-market-predictions)
4. [Parameter Tuning](#4-parameter-tuning)
5. [Cross-League Parlay Generation](#5-cross-league-parlay-generation)
6. [Architecture & Module Map](#6-architecture--module-map)
7. [Backtested Results](#7-backtested-results)

---

## 1. Market Overview & Statistical Drivers

### Odd/Even Goals

The natural scoreline distribution skews slightly toward odd totals (~52–54%):
- Most common scorelines (1-0, 2-1, 3-1, 3-0) are odd-total
- Late consolation goals and trailing-team surges tend to convert even totals to odd ones
- Lower leagues and youth leagues amplify this bias due to higher goal variance

**Best leagues:** E2 (54.07%), G1 (52.93%), SC1 (52.70%)

### Highest Scoring Half (2nd Half)

Second halves produce more goals than first halves (league ratio ~1.25–1.45) because:
- Physical fatigue reduces defensive organisation after 60 minutes
- Attacking substitutions (typically 60–75') inject fresh pace
- Trailing teams push forward, opening space for counters
- Stoppage time effectively extends the half

**Best leagues:** SP2 (ratio 1.368), N1 (1.332), E2 (1.260)

**Youth leagues** (Jong AZ, Jong PSV, Jong Ajax, etc.) carry the highest ratio (~1.45) due to severe fitness disparities.

### Combined parlays

Mixing Odd/Even legs (E2, G1, SC1 at ~1.85 odds) with 2nd Half legs (SP2, N1 at ~2.10 odds) creates diverse, low-correlation parlays.

---

## 2. Priority Leagues for Niche Markets

Empirical analysis across 21 leagues, 3 seasons (2023–26), 15,958 matches:

| Rank | League | Code | Odd Rate | Half Ratio | Recommendation |
|------|--------|------|----------|------------|----------------|
| 1 | England League One | E2 | **54.07%** | 1.260 | ✅ Highest priority |
| 2 | Greece Super League | G1 | 52.93% | 1.253 | ✅ Very high |
| 3 | Spain La Liga 2 | SP2 | 49.23% | **1.368** | ✅ Very high (2nd half) |
| 4 | Netherlands Eredivisie | N1 | 46.72% | 1.332 | ✅ High |
| 5 | Scotland Championship | SC1 | 52.70% | 1.164 | ✅ High |

Lower divisions consistently outperform elite leagues:

| Metric | Elite (E0, D1) | Lower Divisions | Edge |
|--------|---------------|-----------------|------|
| Odd Rate | 48.7% | **49.7%** | +2.05% |
| Half Ratio | 1.249 | **1.260** | +0.88% |
| Goal Variance | 2.09 | **2.18** | +4.31% |

---

## 3. Running Niche Market Predictions

The niche market modules live in `src/niche_markets/`:

```
src/niche_markets/
├── __init__.py
├── odd_even_predictor.py
├── half_comparison_predictor.py
├── league_priors.py
├── lower_league_analysis.py
└── parlay_optimizer.py
```

### CLI — single league

```bash
# Full-league analysis including niche markets
python cli.py --task full-league --leagues E2 \
  --ml-mode predict --enable-double-chance \
  --dc-min-prob 0.75 --dc-secondary-threshold 0.80 --dc-allow-multiple \
  --min-confidence 0.6 --verbose
```

### Python API

```python
from src.niche_markets import OddEvenPredictor, HalfComparisonPredictor

# Odd/Even — English League One
oe = OddEvenPredictor(league_code='E2')
result = oe.predict_with_full_context(
    xg_home=1.6, xg_away=1.3,
    home_odd_rate=0.58, away_odd_rate=0.51,
)
# result: {'Odd': 0.511, 'Even': 0.489, 'confidence': 0.02, 'recommended': False}

# Highest Scoring Half — Spain La Liga 2
hc = HalfComparisonPredictor(league_code='SP2')
result = hc.predict_half_scoring(
    xg_home=1.7, xg_away=1.4,
    home_half_ratio=1.40, away_half_ratio=1.35,
)
# result: {'2nd_Half': 0.493, '1st_Half': 0.278, 'confidence': 0.24, 'recommended': True}
```

### Daily batch

```python
from src.niche_markets import OddEvenPredictor

matches = [
    {'league': 'E2', 'home_team': 'Portsmouth', 'away_team': 'Oxford',
     'xg_home': 1.6, 'xg_away': 1.3, 'home_odd_rate': 0.58, 'away_odd_rate': 0.51},
    # … more fixtures
]
predictor = OddEvenPredictor(league_code='E2')
top_picks = [p for p in predictor.bulk_predict(matches) if p['recommended']]
```

---

## 4. Parameter Tuning

Current baseline: `--min-confidence 0.6 --dc-min-prob 0.75 --dc-secondary-threshold 0.80 --dc-allow-multiple`

**Observed performance on that baseline:**
- DC selection rate: 26.1% (well-calibrated)
- Average DC probability: 0.786 (above threshold — good)
- ML home edge: +0.085; ML away edge: −0.087
- Most selected market: Double Chance "12"

### Three ready-to-use configurations

| Strategy | Confidence | DC min | DC secondary | Use when |
|----------|-----------|--------|--------------|----------|
| **Conservative (5–8 legs)** | 0.72 | 0.82 | 0.87 | Low bankroll / long parlays |
| **Balanced — RECOMMENDED** | 0.68 | 0.78 | 0.83 | Daily use |
| **Edge Exploiter (2–4 legs)** | 0.62 | 0.72 | 0.78 | Short, aggressive parlays |

```bash
# Balanced (recommended)
python cli.py --task full-league --use-parsed-all \
  --min-confidence 0.68 --ml-mode predict \
  --enable-double-chance --dc-min-prob 0.78 \
  --dc-secondary-threshold 0.83 --dc-allow-multiple --verbose
```

### League-specific parameter adjustments

| Parameter | Elite (E0, D1) | Lower (E2, E3) | Reason |
|-----------|---------------|----------------|--------|
| `--min-confidence` | 0.60 | 0.52 | More volatility in lower leagues |
| Form decay | 0.70 | 0.55 | Recent form matters more in lower leagues |
| Bayesian weight (odd/even) | 0.20 | 0.30 | Trust team history more |

### Parlay market mix

1. **Primary:** Double Chance "12" (your strongest market)
2. **Secondary:** BTTS Yes (~0.692 average probability)
3. **Tertiary:** Over 2.5 Goals (~0.764 average probability)
4. **Safety leg:** Under 3.5 Goals

Mix leagues to reduce correlation. ML-enhanced picks where home probability delta > 0.085.

---

## 5. Cross-League Parlay Generation

When using `--use-parsed-all`, the system automatically generates cross-league parlays. Previously `favorable_parlays` was always empty; that bug was fixed by adding `_generate_cross_league_parlays()` to `main_full_league_multiple()`.

### How it works

1. All league suggestions are aggregated after each league run.
2. `_generate_enhanced_parlays()` scores combinations using:
   - League diversity (cross-league combos score higher)
   - Market bonuses (DC and BTTS preferred)
   - Value metric: `prob × (odds − 1) − (1 − prob)`
3. Individual league files are updated with their relevant parlays via `_update_league_file_with_parlays()`.
4. A `cross_league_summary_<DATE>.json` is written to `data/analysis/`.

### Output structure

```json
// data/analysis/full_league_suggestions_E0_<TS>.json
{
  "suggestions": [...],
  "favorable_parlays": [
    {
      "legs": ["Team A v Team B (E0): DC 12", "Team C v Team D (F1): BTTS Yes"],
      "size": 2,
      "probability": 0.547,
      "decimal_odds": 1.83,
      "leagues_involved": ["E0", "F1"],
      "market_types": ["Double Chance", "BTTS"]
    }
  ],
  "cross_league_info": {
    "total_cross_league_parlays": 20,
    "relevant_to_this_league": 8
  }
}
```

### Real-data test result

- 35 predictions from 21 suggestions across 6 leagues
- 20 total parlays generated, 10 favorable
- Average favorable parlay: 35.7% probability, 2.84 odds

**Sample output:**
```
4-Leg Parlay — Probability: 37.7% | Odds: 2.68
Leagues: F1, I1, E0
  Verona v Atalanta (I1): Double Chance X2
  Leeds v Liverpool  (E0): Double Chance 12
  Man City v Sunderland (E0): Double Chance 12
  Toulouse v Strasbourg (F1): BTTS Yes
```

### Recommended parlay slip compositions

| Slip | Legs | Leagues | Min confidence | Target odds |
|------|------|---------|---------------|-------------|
| Pure Odd/Even | 10–14 | E2, G1, SC1 | 0.08 | 20–50 |
| Pure 2nd Half | 5–6 | SP2, N1, E0 | 0.15 (≥ 0.48 P2H) | 30–100 |
| Mixed | 9–14 OE + 5 2H | Combined | — | 40–200 |

---

## 6. Architecture & Module Map

### Hybrid model layers

```
Layer 1: Base Goal Expectation (Poisson/Dixon-Coles, xG/90)
    ↓
Layer 2: Time-Segmented Models (xG by half, fatigue-adjusted Poisson)
    ↓
Layer 3: Context Adjustments (Markov game state, league Bayesian priors)
    ↓
Layer 4: Market Outputs (P(2nd > 1st), P(Odd), parlay probabilities)
```

### Key league priors (in `src/niche_markets/league_priors.py`)

| Code | Odd Rate | Half Ratio | Style |
|------|----------|------------|-------|
| E2 | 0.541 | 1.26 | Volatile |
| G1 | 0.529 | 1.25 | Open |
| SP2 | 0.492 | 1.37 | Low fitness |
| N1 | 0.467 | 1.33 | Open play |
| D1 | 0.515 | 1.35 | High intensity |
| YOUTH | 0.550 | 1.45 | Max variance |
| E0 | 0.505 | 1.25 | Balanced (baseline) |

### Data requirements

Required CSV columns: `FTHG`, `FTAG`, `HTHG`, `HTAG`, `HomeTeam`, `AwayTeam`, `Date`, `Div`  
Optional (improves xG): `HS`, `AS`, `HST`, `AST`  
Source: [football-data.co.uk](https://www.football-data.co.uk)

---

## 7. Backtested Results

Simulated on 2024–25 season, E2 + G1 + SP2 (1,587 matches):

| Market | Accuracy | Baseline | Edge |
|--------|----------|----------|------|
| Odd/Even — E2 | 56.2% | 50% | **+12.4%** |
| Odd/Even — G1 | 54.8% | 50% | **+9.6%** |
| 2nd Half — SP2 | 50.1% | 33.3% | **+50.4%** |
| 2nd Half — N1 | 48.9% | 33.3% | **+46.8%** |

### Target metrics (6-month evaluation)

- Odd/Even accuracy > 54%
- 2nd Half accuracy > 47%
- Combined parlay hit rate > 8% for 10-leg slips
- Simulated ROI > 10% with Kelly staking

### Known limitations

1. Youth league data is not on football-data.co.uk — must source separately.
2. When teams are missing from historical data, predictions fall back to league averages (logged as warnings).
3. Rolling stats will be 0 at season start; accuracy improves after 5+ matches.

---

## See Also

- `ReadMeDocs/QUICK_REFERENCE.md` — CLI command reference
- `ReadMeDocs/ML_MODE_GUIDE.md` — ML pipeline details
- `ReadMeDocs/WORKFLOW_GUIDE.md` — Daily workflow including `run_analysis_workflow.py`
- `src/niche_markets/` — Source code
