#!/usr/bin/env python3
"""
Helper to run corner analysis or full-league processing for leagues discovered in a fixtures JSON.

This script dynamically reads unique `league` codes from the fixtures JSON (e.g., "E0", "E1") and invokes your main CLI module
for each league found. It translates helper flags into the CLI flags so we don't need to modify `cli.py`.

Key behavior changes (per your request):
- Do NOT hard-code competition names; read league codes from the fixtures file.
- If `--competitions` is provided, filter fixtures by competition names first, then detect leagues.
- Translate helper flags:
    --ml  -> --ml-mode predict
    --dc  -> --enable-double-chance (and optionally pass dc thresholds)
- Default is dry-run (prints commands). Use --call-cli to actually execute.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def load_leagues_from_fixtures(fixtures_path: str, competitions: str | None = None) -> Set[str]:
    p = Path(fixtures_path)
    if not p.exists():
        raise FileNotFoundError(f"Fixtures file not found: {fixtures_path}")
    data = json.loads(p.read_text(encoding='utf-8'))
    comps = None
    if competitions:
        comps = {c.strip() for c in competitions.split(',') if c.strip()}
    leagues: Set[str] = set()
    for m in data:
        if comps and m.get('competition') not in comps:
            continue
        league = m.get('league')
        if league:
            leagues.add(league)
    return leagues


def build_command_for_league(cli_module: str, cli_task: str, league_code: str, data_file: str, ml: bool, dc: bool, dc_min_prob: float | None, dc_secondary: float | None, dc_allow_multiple: bool, extra_args: List[str]) -> List[str]:
    cmd = [sys.executable, '-m', cli_module, '--task', cli_task, '--leagues', league_code]
    # Pass the fixtures file so the CLI can optionally read it if it supports --file or --input; prefer --file
    # We'll pass --file if the CLI supports it; cli.py has --file argument.
    if data_file:
        cmd.extend(['--file', data_file])

    # Map helper --ml to CLI --ml-mode predict
    if ml:
        cmd.extend(['--ml-mode', 'predict'])

    # Map helper --dc to CLI's enable double chance flag and thresholds
    if dc:
        cmd.append('--enable-double-chance')
        if dc_min_prob is not None:
            cmd.extend(['--dc-min-prob', str(dc_min_prob)])
        if dc_secondary is not None:
            cmd.extend(['--dc-secondary-threshold', str(dc_secondary)])
        if dc_allow_multiple:
            cmd.append('--dc-allow-multiple')

    # Forward any extra raw args
    if extra_args:
        cmd.extend(extra_args)

    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['corner', 'full'], required=True)
    p.add_argument('--data-file', default='data/todays_fixtures_20251129.json')
    p.add_argument('--competitions', help='Optional comma-separated competition names to filter fixtures first')
    p.add_argument('--ml', action='store_true', help='Enable ML mode (maps to --ml-mode predict)')
    p.add_argument('--dc', action='store_true', help='Enable Double Chance (maps to --enable-double-chance)')
    p.add_argument('--dc-min-prob', type=float, help='DC min prob threshold to forward to CLI')
    p.add_argument('--dc-secondary-threshold', type=float, help='DC secondary threshold to forward to CLI')
    p.add_argument('--dc-allow-multiple', action='store_true', help='Forward --dc-allow-multiple to CLI')
    p.add_argument('--call-cli', action='store_true', help='Actually invoke the CLI module instead of only printing commands')
    p.add_argument('--cli-module', default='cli', help='Module to run with -m when --call-cli is used (default: cli)')
    p.add_argument('--dry-run', action='store_true', help='Only print prepared commands; do not execute (default)')
    p.add_argument('--extra', nargs=argparse.REMAINDER, help='Extra args to forward to the CLI')
    args = p.parse_args()

    try:
        leagues = load_leagues_from_fixtures(args.data_file, args.competitions)
    except Exception as e:
        print(f'Error loading fixtures: {e}')
        raise

    if not leagues:
        print('No leagues found in fixtures (after optional competition filtering). Exiting.')
        return

    print('Detected leagues in fixtures:', ', '.join(sorted(leagues)))

    cli_task = 'full-league' if args.mode == 'full' else 'corners'
    extra_args = args.extra or []

    for league in sorted(leagues):
        cmd = build_command_for_league(
            cli_module=args.cli_module,
            cli_task=cli_task,
            league_code=league,
            data_file=args.data_file,
            ml=args.ml,
            dc=args.dc,
            dc_min_prob=args.dc_min_prob,
            dc_secondary=args.dc_secondary_threshold,
            dc_allow_multiple=args.dc_allow_multiple,
            extra_args=extra_args,
        )
        print('Prepared command:', ' '.join(cmd))
        if args.call_cli and not args.dry_run:
            print(f'Invoking CLI for league {league}...')
            proc = subprocess.run(cmd, capture_output=True, text=True)
            print('--- STDOUT ---')
            print(proc.stdout)
            print('--- STDERR ---')
            print(proc.stderr)
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)

    if args.dry_run:
        print('\nDry-run: no commands were executed. Use --call-cli to actually run the CLI per league.')


if __name__ == '__main__':
    main()
