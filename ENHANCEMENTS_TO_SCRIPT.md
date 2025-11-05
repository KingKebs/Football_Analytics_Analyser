# Rating-driven match model — actionable enhancements

## Short actionable summary
- Build a ratings-driven single-number "match rating" (goal‑supremacy over recent N matches).
- Fit empirical mappings from match rating → P(home), P(draw), P(away).
- Convert probabilities to fair decimal odds and compare to market/bookmaker odds to find value bets.
- Backtest and report yield/ROI by rating bins; offer ensemble with existing Poisson/xG outputs.

## Table of contents
1. Practical recipe (what the book recommends)
2. Where to integrate this in the repo (file / function map)
3. Concrete algorithm & implementation details (step-by-step)
4. Backtesting and diagnostics
5. CLI + runtime wiring
6. Guardrails, edge-cases and recommended defaults
7. Next steps / prioritized TODOs

---

## 1) Practical recipe (from the book)
- Compute a recent-form rating for each team using goal difference (goals_for − goals_against) over the last N matches (commonly N=4..6).
- Define the match rating as: `match_rating = home_rating − away_rating`.
- Empirically fit mappings from `match_rating` → outcome probabilities. The book suggests:
  - Linear regression works well for home-win probability.
  - Quadratic (or low-degree polynomial) fits are often better for draw and away probabilities.
- Where sample sizes are small (extreme ratings), apply smoothing (binning / fallback to wider bins / overall mean).
- Convert fitted probabilities to fair odds: `fair_decimal_odds = 1.0 / prob` and compute EV vs market odds.
- Backtest across seasons and report yield/profit by rating ranges; the book reports best returns typically for central rating windows.

## 2) Where to integrate in this repo (file + spot)
- `algorithms.py`
  - Add rating functions and model fits.
  - Keep existing Poisson/xG extraction but allow externally supplied probability vectors.
- `automate_football_analytics.py` and `automate_football_analytics_fullLeague.py`
  - Add CLI flags to enable the rating model and tune parameters.
- `analyze_suggestions_results.py`
  - Add a backtest harness to evaluate yield/ROI, strike-rate, profit by rating-bin.
- Optional new module: `odds_utils.py` (or add to `algorithms.py`)
  - Helpers: overround computation, best price selection, EV calculation.

Suggested function signatures (to implement)
- `compute_goal_supremacy_rating(history_df, team, last_n=6) -> float`
- `match_rating(home_team, away_team, history_df, last_n=6) -> float`
- `fit_rating_to_prob_models(history_df, last_n=6) -> dict` (returns callables/models for P(home|r), P(draw|r), P(away|r))
- `rating_probabilities(home, away, models) -> (p_home, p_draw, p_away)`
- `extract_markets_from_score_matrix(..., external_probs=None)` — accept external probability vectors when provided.

## 3) Concrete algorithm & implementation details
A minimal workable implementation path (low friction):

1. Compute goal‑supremacy ratings
   - For each team, compute `rating_team = sum(goals_for − goals_against)` over the last `last_n` matches. Optionally normalize by `last_n`.
   - Default `last_n = 6`.
   - The match rating is `r = rating_home − rating_away`.

2. Build training set for fitting
   - For historical matches, compute `r` and label the observed outcome (one-hot for home/draw/away).
   - Optionally bin `r` (e.g., integer bins) to produce stable frequency estimates and counts per bin for diagnostics.

3. Fit models
   - Fit three simple models for each outcome (linear for home, quadratic or polynomial for draw/away), or try monotonic/isotonic regression.
   - After prediction, clamp probabilities to `[0.001, 0.999]` and renormalize so they sum to 1: `p = p / (p_home + p_draw + p_away)`.
   - Compute and store diagnostics: sample counts per bin, R² or log-loss, and residual plots for debugging.

4. Smoothing & low-sample handling
   - For bins with sample counts < `min_sample_for_rating` (default 30), fallback to a wider bin estimate or overall mean probabilities.
   - Provide a `rating_range_filter` option to focus suggestions to a pragmatic window (e.g., `[-5, +5]` or narrower like `[-2, +2]`).

5. From probabilities → fair odds → EV
   - `fair_decimal_odds = round(1.0 / prob, 2)`
   - Expected value (per unit stake): `EV = prob * (market_decimal_odds - 1) - (1 - prob)`
     - Equivalently check `prob * market_decimal_odds - 1 > 0` for positive EV.
   - Suggest only bets with `EV > ev_threshold` and where supporting sample size is sufficient.

6. Blending with Poisson/xG
   - Allow an ensemble weight `w` such that final probabilities = `w * rating_probs + (1-w) * poisson_probs`.
   - Expose `w` as a tunable parameter per run.

## 4) Backtesting and diagnostics (what to produce)
- A backtest harness should:
  - Replay historical matches and generate suggestions using the rating model (and optionally blended with xG).
  - Simulate betting with unit stake and/or common staking methods (flat stake, Kelly, fraction of bankroll).
  - Output: overall ROI/yield, strike rate, profit by rating-bin, sample counts, and cumulative P&L time-series.
- Produce tables similar to the book: rating-bin | matches | strike% | ROI | mean EV.
- Plot yield vs rating-bin; surface where sample sizes are low.

## 5) CLI + runtime wiring
- Add CLI flags to the automation scripts:
  - `--rating-model {none,goal_supremacy,blended}` (default: `none`)
  - `--rating-last-n INT` (default: `6`)
  - `--rating-range-filter STR` (e.g., `-10,10`)
  - `--min-sample-for-rating INT` (default: `30`)
  - `--rating-ev-threshold FLOAT` (default: `0.0`)
  - `--rating-blend-weight FLOAT` (when `blended`, weight for rating model, default e.g. `0.3`)
- When enabled, call the rating functions during match suggestion generation and optionally blend with Poisson output.

## 6) Guardrails, edge-cases and recommended defaults
- Defaults: `last_n=6`, `min_sample_for_rating=30`, `ev_threshold=0.0`, `rating_blend_weight=0.3`.
- Avoid suggesting bets for extreme ratings with tiny counts (fallback or skip).
- Clamp probabilities to avoid zero/one outcomes and renormalize.
- Log model diagnostics (R² / log-loss / sample counts).
- Use shopping for best available market odds across bookies for EV computation.

## 7) Next steps / prioritized TODOs (Priority 1)
1. Implement the minimal rating functions in `algorithms.py`:
   - `compute_goal_supremacy_rating`, `match_rating`, `fit_rating_to_prob_models`, `rating_probabilities`.
2. Add CLI flags and wiring to `automate_football_analytics.py` and `automate_football_analytics_fullLeague.py` to enable the rating model.
3. Add a backtest runner in `analyze_suggestions_results.py` to evaluate yield by rating-bin.
4. Add unit tests (happy path + low-sample bin behavior) and a small example notebook or script that fits the model on a small dataset and prints diagnostics.

---

## Appendix — quick reference: formulas & code hints
- Match rating (per match): `r = (sum_{i=1..N} (gf_home_i - ga_home_i)) - (sum_{i=1..N} (gf_away_i - ga_away_i))`.
- Fair odds: `fair_decimal_odds = round(1.0 / p, 2)`.
- EV (unit stake): `EV = p * (market_odds - 1) - (1 - p)` — suggest when EV > 0 (or > ev_threshold).


<!-- End of enhanced, markdown-friendly enhancements document -->

