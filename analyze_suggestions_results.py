#!/usr/bin/env python3
"""
Compare suggested picks (from full_league_suggestions JSON) to actual match results.
Usage:
  python analyze_suggestions_results.py --suggestions data/full_league_suggestions_E0_20251102_095227.json \
    --results data/sample_results_20251102.csv

Results file format (CSV): date,home,away,home_score,away_score
Or JSON: list of {date, home, away, home_score, away_score}

The script prints per-pick correctness, per-market accuracy and parlay diagnostics.
Optional: backtest rating model by rating bins with --backtest-rating.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

# Lazy imports for rating tools (to keep base usage light)
# Will import when --backtest-rating is requested


def load_suggestions(path: Path):
    with path.open() as f:
        data = json.load(f)
    return data.get("suggestions", [])


def load_results(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        with path.open() as f:
            return json.load(f)
    else:
        rows = []
        with path.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                # normalize keys
                rows.append({
                    "date": r.get("date") or r.get("match_date") or r.get("Date"),
                    "home": r.get("home") or r.get("Home"),
                    "away": r.get("away") or r.get("Away"),
                    "home_score": int(r.get("home_score") or r.get("HomeScore") or r.get("home_goals") or 0),
                    "away_score": int(r.get("away_score") or r.get("AwayScore") or r.get("away_goals") or 0),
                })
        return rows


def find_result_for(match, results):
    # match: dict with home/away, optional date
    h = match.get("home", "").lower()
    a = match.get("away", "").lower()
    d = match.get("date")
    for r in results:
        if r.get("home") and r.get("away"):
            if r["home"].lower() == h and r["away"].lower() == a:
                return r
            # sometimes reversed naming
            if r["home"].lower() == a and r["away"].lower() == h:
                # return reversed but mark home/away swapped
                return {**r, "swapped": True}
    return None


def evaluate_pick(pick, result):
    # pick: {'market':..., 'selection':...}
    sel = pick.get("selection")
    if result is None:
        return None
    hs = int(result.get("home_score", 0))
    as_ = int(result.get("away_score", 0))

    # 1X2 market
    if pick.get("market") == "1X2":
        if sel.lower() in ("home", "h", "1"):
            return hs > as_
        if sel.lower() in ("away", "a", "2"):
            return as_ > hs
        if sel.lower() in ("draw", "d", "x"):
            return hs == as_
        return False

    # BTTS
    if pick.get("market", "").upper().startswith("BTTS"):
        if sel.lower() in ("yes", "y"):
            return hs > 0 and as_ > 0
        if sel.lower() in ("no", "n"):
            return not (hs > 0 and as_ > 0)
        return False

    # Over/Under
    if pick.get("market", "").lower().startswith("over") or "over" in str(sel).lower() or "under" in str(sel).lower() or pick.get("market") .lower().startswith("over/under"):
        # selection examples: Over2.5, Under2.5, Over1.5
        s = str(sel).lower()
        import re
        m = re.search(r"(over|under)\s*([0-9]+\.?[0-9]*)", s)
        if not m:
            # sometimes market stored as "Over2.5"
            m = re.search(r"(over|under)([0-9]+\.?[0-9]*)", s)
        if not m:
            return False
        typ = m.group(1)
        val = float(m.group(2))
        total = hs + as_
        if typ == "over":
            return total > val
        else:
            return total < val

    # default: unknown market -> return None
    return None


def summarize(suggestions, results):
    per_market = defaultdict(lambda: {"correct": 0, "total": 0})
    per_match = []
    overall = {"checked": 0, "correct": 0}
    failed = []

    for s in suggestions:
        match = {"home": s.get("home"), "away": s.get("away"), "date": s.get("date")}
        result = find_result_for(match, results)
        picks = s.get("picks", [])
        match_res = {"home": match["home"], "away": match["away"], "result": result, "picks": []}
        for p in picks:
            res = evaluate_pick(p, result)
            match_res["picks"].append({"pick": p, "won": res})
            if res is not None:
                per_market[p.get("market")]["total"] += 1
                if res:
                    per_market[p.get("market")]["correct"] += 1
                    overall["correct"] += 1
                overall["checked"] += 1
                if not res:
                    failed.append({"match": match, "pick": p, "result": result})
        per_match.append(match_res)

    # compute accuracies
    market_acc = {m: (v["correct"], v["total"], (v["correct"]/v["total"] if v["total"] else None)) for m, v in per_market.items()}
    overall_acc = (overall["correct"], overall["checked"], (overall["correct"]/overall["checked"] if overall["checked"] else None))
    return {"per_match": per_match, "market_acc": market_acc, "overall_acc": overall_acc, "failed": failed}


def parlay_diagnostics(suggestions, min_prob=0.6):
    # Build a parlay from picks that meet min_prob threshold (use pick['prob'])
    picks = []
    for s in suggestions:
        for p in s.get("picks", []):
            prob = p.get("prob")
            if prob and prob >= min_prob:
                picks.append(p)
                break  # one pick per match
    # compute parlay success probability as product of probs
    if not picks:
        return {"picks_in_parlay": 0}
    from math import prod
    probs = [p.get("prob") for p in picks]
    parlay_prob = prod(probs)
    expected_odds = 1
    for p in picks:
        if p.get("odds"):
            expected_odds *= float(p.get("odds"))
    return {"picks_in_parlay": len(picks), "parlay_prob": parlay_prob, "expected_odds": expected_odds, "picks": picks}


# ---------------- Rating-model backtest additions ----------------

def _load_history(data_dir: str = 'data', subfolder: str = 'old csv') -> pd.DataFrame:
    folder = Path(data_dir) / subfolder
    if not folder.is_dir():
        return pd.DataFrame()
    files = [p for p in folder.iterdir() if p.suffix.lower() == '.csv']
    if not files:
        return pd.DataFrame()
    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp, low_memory=False)
            rename_map = {}
            for c in df.columns:
                lc = str(c).lower()
                if lc in ('hometeam', 'home_team'):
                    rename_map[c] = 'HomeTeam'
                if lc in ('awayteam', 'away_team'):
                    rename_map[c] = 'AwayTeam'
                if lc in ('fthg', 'homegoals'):
                    rename_map[c] = 'FTHG'
                if lc in ('ftag', 'awaygoals'):
                    rename_map[c] = 'FTAG'
                if lc == 'date' and c != 'Date':
                    rename_map[c] = 'Date'
            df = df.rename(columns=rename_map)
            if set(['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']).issubset(df.columns):
                dfs.append(df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']])
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    all_df['Date'] = pd.to_datetime(all_df['Date'], errors='coerce', dayfirst=True)
    all_df = all_df.sort_values('Date', na_position='last')
    return all_df


def _compute_match_rating(home: str, away: str, history_df: pd.DataFrame, last_n: int) -> float:
    # Local import to avoid hard dependency unless requested
    from algorithms import match_rating
    try:
        return float(match_rating(home, away, history_df, last_n=last_n))
    except Exception:
        return 0.0


def backtest_by_rating_bins(suggestions: List[Dict], results: List[Dict], history_df: pd.DataFrame, last_n: int = 6, bins: List[float] = None) -> Dict:
    """Compute strike-rate and ROI by rating bins using 1X2 picks in suggestions.
    ROI computed with provided pick odds if present; otherwise uses fair odds 1/p.
    """
    from algorithms import fit_rating_to_prob_models, rating_probabilities_from_rating

    if bins is None:
        bins = [-10, -6, -4, -2, -1, 0, 1, 2, 4, 6, 10]

    # Fit models (optional, for diagnostics; not required to bin)
    models = None
    try:
        models = fit_rating_to_prob_models(history_df, last_n=last_n, min_sample_for_rating=30) if not history_df.empty else None
    except Exception:
        models = None

    # Prepare bins
    bins_sorted = sorted(bins)
    bin_labels = [f"({bins_sorted[i]},{bins_sorted[i+1]}]" for i in range(len(bins_sorted)-1)]
    stats = {lab: {"matches": 0, "wins": 0, "losses": 0, "stake": 0.0, "profit": 0.0} for lab in bin_labels}

    for s in suggestions:
        home = s.get('home'); away = s.get('away')
        if not home or not away:
            continue
        # find 1X2 pick (use the first if multiple)
        pick = None
        for p in s.get('picks', []):
            if p.get('market') == '1X2':
                pick = p; break
        if not pick:
            continue
        # resolve result
        result = find_result_for({"home": home, "away": away}, results) if results else None
        if not result:
            continue
        won = evaluate_pick(pick, result)
        if won is None:
            continue
        # rating and bin
        r = _compute_match_rating(home, away, history_df, last_n)
        # find bin index
        bi = None
        for i in range(len(bins_sorted)-1):
            if bins_sorted[i] < r <= bins_sorted[i+1]:
                bi = i; break
        if bi is None:
            continue
        lab = bin_labels[bi]
        # stake and profit (unit stake)
        odds = float(pick.get('odds')) if pick.get('odds') else (1.0 / max(1e-6, float(pick.get('prob', 0.0))))
        stake = 1.0
        profit = (odds - 1.0) if won else -1.0
        # update
        st = stats[lab]
        st["matches"] += 1
        st["wins"] += 1 if won else 0
        st["losses"] += 0 if won else 1
        st["stake"] += stake
        st["profit"] += profit

    # finalize ratios
    out_rows = []
    for lab in bin_labels:
        st = stats[lab]
        if st["matches"] == 0:
            sr = None; roi = None
        else:
            sr = st["wins"] / st["matches"]
            roi = st["profit"] / max(1e-9, st["stake"])
        out_rows.append({
            "bin": lab,
            "matches": st["matches"],
            "strike_rate": sr,
            "roi": roi,
            "wins": st["wins"],
            "losses": st["losses"],
        })
    return {"bins": out_rows, "models_fitted": bool(models), "sample_size": models.get('sample_size', 0) if models else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suggestions", required=True)
    parser.add_argument("--results", required=False)
    # Rating backtest options
    parser.add_argument("--backtest-rating", action="store_true", help="Run rating-bin backtest on suggestions (1X2 picks)")
    parser.add_argument("--history-dir", type=str, default="data/old csv", help="Directory with historical CSVs")
    parser.add_argument("--rating-last-n", type=int, default=6, help="Matches to use for goal-supremacy rating")
    parser.add_argument("--rating-bins", type=str, default="-10,-6,-4,-2,-1,0,1,2,4,6,10", help="Comma-separated rating bin edges")

    args = parser.parse_args()

    sug_path = Path(args.suggestions)
    res_path = Path(args.results) if args.results else None

    suggestions = load_suggestions(sug_path)
    results = []
    if res_path:
        results = load_results(res_path)
    else:
        print("No results file provided. To evaluate accuracy provide --results <path>")

    analysis = summarize(suggestions, results)

    print("Overall accuracy (correct/checked/ratio):", analysis["overall_acc"])
    print("Per-market accuracy:")
    for m, (c, t, r) in analysis["market_acc"].items():
        print(f"  {m}: {c}/{t} -> {r}")

    print("Failed picks (sample):")
    for f in analysis["failed"][:20]:
        print(json.dumps(f, default=str))

    diag = parlay_diagnostics(suggestions, min_prob=0.65)
    print("\nParlay diagnostics for picks >=65%:")
    print(json.dumps(diag, indent=2, default=str))

    # Optional backtest
    if args.backtest_rating:
        history_df = _load_history(data_dir=str(Path(args.history_dir).parent), subfolder=str(Path(args.history_dir).name))
        if history_df.empty:
            print("\n[Backtest] No historical CSVs found; skipping rating-bin backtest.")
        else:
            bins = [float(x.strip()) for x in args.rating_bins.split(',') if x.strip()]
            bt = backtest_by_rating_bins(suggestions, results, history_df, last_n=args.rating_last_n, bins=bins)
            print("\n[Backtest] Rating-bin strike rate and ROI (unit stake):")
            for row in bt["bins"]:
                print(f"  {row['bin']}: matches={row['matches']}, strike={None if row['strike_rate'] is None else round(row['strike_rate']*100,1)}%, ROI={None if row['roi'] is None else round(row['roi']*100,1)}% (W{row['wins']}/L{row['losses']})")

    # save output
    out = Path("data/analysis_results_20251102.json")
    out.write_text(json.dumps(analysis, default=str, indent=2))
    print("Analysis saved to", out)

if __name__ == '__main__':
    main()
