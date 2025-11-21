#!/usr/bin/env python3
"""
parse_match_log.py

Parse the plain-text match log (e.g. sunday_02112025_matches.log) into a structured CSV/JSON
without external dependencies.

Usage:
  python3 parse_match_log.py --input sunday_02112025_matches.log

Outputs:
  data/todays_fixtures_<YYYYMMDD>.csv and .json
"""
import argparse
import csv
import json
import os
import re
from datetime import datetime
import difflib

TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")
LEAGUE_HEADER_RE = re.compile(r"^[A-Za-z][A-Za-z\s]+:\s*$")
DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})")
NOISE_WORDS = {"standings", "live standings", "live", "all", "odds", "finished", "scheduled"}


def _is_header_line(s: str) -> bool:
    return bool(LEAGUE_HEADER_RE.match(s))


def _is_time_line(s: str) -> bool:
    return bool(TIME_RE.match(s))


def _is_noise_line(s: str) -> bool:
    if not s:
        return True
    sl = s.strip().lower()
    if sl in NOISE_WORDS:
        return True
    if sl in {"-", "--"}:
        return True
    if sl.isdigit():
        return True
    return False


def _norm(name: str) -> str:
    return (name or '').strip().lower()


def find_league_files(league_code: str):
    candidates = []
    roots = [
        os.path.join('football-data', 'all-euro-football'),
        'football-data'
    ]
    patterns = [
        f"{league_code}_",
        f"{league_code}.csv"
    ]
    for root in roots:
        if not os.path.isdir(root):
            continue
        try:
            for fname in os.listdir(root):
                if fname.startswith(league_code + '_') and fname.endswith('.csv'):
                    candidates.append(os.path.join(root, fname))
                elif fname == f"{league_code}.csv":
                    candidates.append(os.path.join(root, fname))
        except Exception:
            continue
    return sorted(candidates)


def load_league_teams(league_codes):
    """Return dict league_code -> set of normalized team names found in CSVs."""
    leagues = {}
    for code in league_codes:
        teams = set()
        for path in find_league_files(code):
            try:
                with open(path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    # normalize header keys to lower
                    field_map = {k: k for k in reader.fieldnames or []}
                    # Accept a variety of headers
                    home_key = None
                    away_key = None
                    for k in field_map:
                        lk = k.lower()
                        if lk in ('hometeam', 'home team'):
                            home_key = k
                        if lk in ('awayteam', 'away team'):
                            away_key = k
                    if not home_key or not away_key:
                        continue
                    for row in reader:
                        h = _norm(row.get(home_key, ''))
                        a = _norm(row.get(away_key, ''))
                        if h:
                            teams.add(h)
                        if a:
                            teams.add(a)
            except Exception:
                continue
        leagues[code] = teams
    return leagues


def resolve_fixture_league(home: str, away: str, league_codes, league_teams):
    """Pick the most likely league for a fixture based on team presence in provided leagues."""
    if not league_codes:
        return ''
    h = _norm(home)
    a = _norm(away)
    scores = {code: 0 for code in league_codes}
    # Exact presence
    for code in league_codes:
        teams = league_teams.get(code, set())
        if h in teams:
            scores[code] += 2
        if a in teams:
            scores[code] += 2
    # Fuzzy presence if still zero
    if max(scores.values()) == 0:
        for code in league_codes:
            teams = list(league_teams.get(code, set()))
            if not teams:
                continue
            if h:
                cand = difflib.get_close_matches(h, teams, n=1, cutoff=0.88)
                if cand:
                    scores[code] += 1
            if a:
                cand = difflib.get_close_matches(a, teams, n=1, cutoff=0.88)
                if cand:
                    scores[code] += 1
    # Choose best, tie -> first in provided order
    best = max(scores.values())
    if best <= 0:
        return league_codes[0]  # fallback
    top = [c for c, s in scores.items() if s == best]
    for c in league_codes:
        if c in top:
            return c
    return league_codes[0]


def parse_log_lines(lines, default_league: str = "", include_postponed: bool = False, multi_leagues=None):
    """Return list of match dicts with fields: date, league, competition, time, home, away, status"""
    matches = []
    league = (multi_leagues[0] if multi_leagues else (default_league or None))
    competition = None
    parsed_date = None

    # preload team sets if multi-leagues provided
    league_codes = [c.strip().upper() for c in (multi_leagues or []) if c.strip()]
    league_teams = load_league_teams(league_codes) if league_codes else {}
    infer_competition = not bool(league_codes)  # don't infer competition when leagues are explicitly provided

    i = 0
    n = len(lines)
    # Try to find a date near the top
    for j in range(0, min(20, n)):
        m = DATE_RE.search(lines[j])
        if m:
            day = int(m.group(1)); month = int(m.group(2))
            now = datetime.now(); year = now.year
            try:
                candidate = datetime(year, month, day)
            except Exception:
                candidate = now
            if (candidate - now).days > 60:
                year = year - 1; candidate = datetime(year, month, day)
            parsed_date = candidate.date().isoformat()
            break

    while i < n:
        raw = lines[i].strip(); i += 1
        if not raw:
            continue
        if _is_header_line(raw):
            league = raw.rstrip(":").strip().upper()
            competition = None
            j = i
            while j < n and competition is None:
                cand = lines[j].strip(); j += 1
                if not cand:
                    continue
                if _is_header_line(cand) or _is_time_line(cand) or _is_noise_line(cand):
                    continue
                competition = cand
            continue
        if raw.lower() == 'postponed':
            status = 'Postponed'
            teams = []
            while i < n and len(teams) < 2:
                token = lines[i].strip(); i += 1
                if not token or _is_noise_line(token) or _is_header_line(token) or _is_time_line(token):
                    if _is_header_line(token) or _is_time_line(token):
                        i -= 1; break
                    continue
                teams.append(token)
            home = teams[0] if len(teams) >= 1 else ""
            away = teams[1] if len(teams) >= 2 else ""
            if include_postponed:
                # Resolve league if multi-league supplied
                resolved_league = resolve_fixture_league(home, away, league_codes, league_teams) if league_codes else (league or default_league or "")
                comp_val = "" if league_codes else (competition or "")
                matches.append({
                    "date": parsed_date or "",
                    "league": resolved_league,
                    "competition": comp_val,
                    "time": "",
                    "home": home,
                    "away": away,
                    "status": status
                })
            continue
        if raw and not _is_time_line(raw) and not _is_header_line(raw) and not _is_noise_line(raw):
            if infer_competition and competition is None and (league or default_league or league_codes):
                competition = raw
                continue
        if _is_time_line(raw):
            time = raw
            collected = []
            k = 0
            while i < n and k < 8:
                token = lines[i].strip(); i += 1; k += 1
                if not token or _is_noise_line(token):
                    continue
                if _is_header_line(token) or _is_time_line(token):
                    i -= 1; break
                collected.append(token)
            teams = []
            for t in collected:
                if not teams or (t != teams[-1]):
                    teams.append(t)
                if len(teams) >= 2:
                    break
            home = teams[0] if len(teams) >= 1 else ""
            away = teams[1] if len(teams) >= 2 else ""
            # Resolve league for this fixture
            resolved_league = resolve_fixture_league(home, away, league_codes, league_teams) if league_codes else (league or default_league or "")
            # When explicit leagues provided, suppress competition label
            comp_val = "" if league_codes else (competition or "")
            matches.append({
                "date": parsed_date or "",
                "league": resolved_league,
                "competition": comp_val,
                "time": time,
                "home": home,
                "away": away,
                "status": "Scheduled"
            })
            continue
    return matches


def write_outputs(matches, out_base):
    csv_path = out_base + ".csv"
    json_path = out_base + ".json"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "league", "competition", "time", "home", "away", "status"])
        writer.writeheader()
        for m in matches:
            writer.writerow(m)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)
    return csv_path, json_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="sunday_02112025_matches.log", help="input log file")
    p.add_argument("--output-dir", "-o", default="data", help="output directory")
    p.add_argument("--league-code", "-l", default="", help="override league code for all entries (deprecated in favor of --leagues)")
    p.add_argument("--leagues", default="", help="comma-separated league codes to resolve fixtures (e.g. E2,E3)")
    p.add_argument("--include-postponed", action='store_true', help="include postponed fixtures in outputs")
    args = p.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    leagues = [c.strip().upper() for c in args.leagues.split(',') if c.strip()] if args.leagues else []
    try:
        matches = parse_log_lines(lines, default_league=args.league_code, include_postponed=args.include_postponed, multi_leagues=leagues)
    except Exception as e:
        print(f"Parse error: {e}")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    date_suffix = datetime.now().strftime("%Y%m%d")
    out_base = os.path.join(args.output_dir, f"todays_fixtures_{date_suffix}")
    csv_path, json_path = write_outputs(matches, out_base)

    print(f"Parsed {len(matches)} matches from {args.input}")
    for m in matches:
        label = m.get('time') or m.get('status') or ''
        prefix = f"{label} - " if label else ""
        lg = m.get('league') or ''
        lgp = f"[{lg}] " if lg else ''
        print(f"{lgp}{prefix}{m['home']} vs {m['away']}")

    by_league = {}
    for m in matches:
        lg = m.get('league') or 'UNKNOWN'
        by_league[lg] = by_league.get(lg, 0) + 1
    for league, cnt in by_league.items():
        print(f"  {league}: {cnt}")
    print(f"Wrote CSV -> {csv_path}")
    print(f"Wrote JSON -> {json_path}")

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for _ in range(30):
                line = f.readline()
                if not line:
                    break
                print(line.rstrip())
    except Exception:
        pass


if __name__ == '__main__':
    main()
