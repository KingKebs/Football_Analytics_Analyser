#!/usr/bin/env python3
"""Convert hierarchical upcomingMatches.json into standard todays_fixtures_<DATE>.csv/json.

Input format example (data/raw/upcomingMatches.json):
{
  "ENGLAND": {
    "Premier League": { "Standings": {}, "Fixtures": [ {"time":"14:30","home":"Burnley","away":"Chelsea"}, ... ] }
  },
  "FRANCE": { ... }
}

Output schema (aligned with parse_match_log.py output):
[{
  "date": "YYYY-MM-DD",
  "league": "E0",              # mapped league code if known else country code fallback
  "competition": "Premier League",  # original competition name
  "time": "14:30",
  "home": "Burnley",
  "away": "Chelsea",
  "status": "Scheduled"
}]

Usage:
  python3 convert_upcoming_matches.py --input data/raw/upcomingMatches.json --output-dir data
  python3 convert_upcoming_matches.py --date 2025-11-21

Mapping rules:
 - Premier League -> E0
 - Championship -> E1
 - Ligue 1 -> F1
 - Ligue 2 -> F2
 - Bundesliga -> D1
 - 2. Bundesliga -> D2
 - Serie A -> I1
 - Serie B -> I2
 - Eredivisie -> N1
 - La Liga -> SP1
 - Segunda División -> SP2
 - If no direct map, fallback to country code abbreviation (first letter(s)) or blank.

After conversion you can run:
  python3 cli.py --task full-league --use-parsed-all --min-confidence 0.6
which will pick up the newly created todays_fixtures_<DATE>.json from data/.
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import csv
import re

LEAGUE_MAP = {
    'Premier League': 'E0',
    'Championship': 'E1',
    'Ligue 1': 'F1',
    'Ligue 2': 'F2',
    'Bundesliga': 'D1',
    '2. Bundesliga': 'D2',
    'Serie A': 'I1',
    'Serie B': 'I2',
    'Eredivisie': 'N1',
    'La Liga': 'SP1',
    'Segunda División': 'SP2',
}

COUNTRY_FALLBACK = {
    'ENGLAND': 'E0',
    'FRANCE': 'F1',
    'GERMANY': 'D1',
    'ITALY': 'I1',
    'NETHERLANDS': 'N1',
    'SPAIN': 'SP1'
}

TIME_CLEAN_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def normalize_time(t: str) -> str:
    if not t:
        return ''
    m = TIME_CLEAN_RE.match(t.strip())
    if not m:
        return t.strip()[:5]
    h = int(m.group(1)); mnt = m.group(2)
    return f"{h:02d}:{mnt}"


def convert(data: dict, out_date: str) -> list:
    fixtures_out = []
    for country, comps in data.items():
        if not isinstance(comps, dict):
            continue
        for competition, payload in comps.items():
            if not isinstance(payload, dict):
                continue
            fx_list = payload.get('Fixtures') or []
            if not isinstance(fx_list, list):
                continue
            league_code = LEAGUE_MAP.get(competition) or COUNTRY_FALLBACK.get(country.upper(), '')
            for fx in fx_list:
                home = fx.get('home') or fx.get('Home') or ''
                away = fx.get('away') or fx.get('Away') or ''
                time = normalize_time(fx.get('time') or fx.get('Time') or '')
                if not home or not away:
                    continue
                fixtures_out.append({
                    'date': out_date,
                    'league': league_code,
                    'competition': competition,
                    'time': time,
                    'home': home,
                    'away': away,
                    'status': 'Scheduled'
                })
    return fixtures_out


def write_outputs(fixtures: list, output_dir: str, out_date: str):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f"todays_fixtures_{out_date.replace('-', '')}")
    csv_path = base + '.csv'
    json_path = base + '.json'
    # CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['date','league','competition','time','home','away','status'])
        w.writeheader()
        for r in fixtures:
            w.writerow(r)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(fixtures, f, ensure_ascii=False, indent=2)
    return csv_path, json_path


def main():
    ap = argparse.ArgumentParser(description='Convert upcomingMatches.json to todays_fixtures format')
    ap.add_argument('--input', default='data/raw/upcomingMatches.json', help='Path to upcoming matches JSON')
    ap.add_argument('--output-dir', default='data', help='Output directory for todays_fixtures files')
    ap.add_argument('--date', help='Override date (YYYY-MM-DD); default today UTC')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}")
        return 1
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON: {e}")
        return 1

    out_date = args.date or datetime.utcnow().date().isoformat()
    fixtures = convert(raw, out_date)
    if not fixtures:
        print("No fixtures extracted.")
        return 1

    csv_path, json_path = write_outputs(fixtures, args.output_dir, out_date)
    print(f"✓ Wrote {len(fixtures)} fixtures")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    # sample output
    for r in fixtures[:10]:
        print(f"{r['time']} {r['league'] or '-'} {r['home']} vs {r['away']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

