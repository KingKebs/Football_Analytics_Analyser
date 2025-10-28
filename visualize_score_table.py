"""
Visualize score probability matrix from the latest suggestion JSON.

Usage:
  python visualize_score_table.py [--input <path>] [--output <path>]
If --input is omitted, the script will pick the most recent data/suggestion_*.json
Saves a heatmap PNG (default: data/score_heatmap.png)
"""
import argparse
import glob
import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def find_latest_suggestion(data_dir: str = 'data') -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(data_dir, 'suggestion_*.json')))
    return paths[-1] if paths else None


def load_score_matrix(path: str) -> pd.DataFrame:
    with open(path, 'r') as f:
        data = json.load(f)
    mat = data.get('score_matrix')
    if isinstance(mat, dict):
        # Likely dict of dicts
        # Keys are strings representing integers; ensure proper ordering
        rows = sorted((int(k), v) for k, v in mat.items())
        max_col = 0
        for _, cols in rows:
            max_col = max(max_col, *[int(c) for c in cols.keys()])
        idx = list(range(max_col + 1))
        df = pd.DataFrame(index=idx, columns=idx, dtype=float)
        for r, cols in rows:
            for c, p in cols.items():
                df.loc[r, int(c)] = float(p)
        df = df.fillna(0.0)
        return df
    elif isinstance(mat, list):
        return pd.DataFrame(mat)
    else:
        raise ValueError('Unrecognized score_matrix format in suggestion JSON')


def plot_heatmap(df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(df, annot=False, cmap='viridis')
    ax.invert_yaxis()
    plt.title('Score Probability Heatmap (Home goals = rows, Away goals = cols)')
    plt.xlabel('Away Goals')
    plt.ylabel('Home Goals')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize score probability matrix from suggestion JSON')
    parser.add_argument('--input', type=str, help='Path to suggestion JSON')
    parser.add_argument('--output', type=str, default='data/score_heatmap.png', help='Output PNG path')
    args = parser.parse_args()

    in_path = args.input or find_latest_suggestion()
    if not in_path or not os.path.exists(in_path):
        print('No suggestion JSON found. Generate one with automate_football_analytics.py first.')
        return

    df = load_score_matrix(in_path)
    plot_heatmap(df, args.output)
    print(f'Heatmap saved to {args.output}')


if __name__ == '__main__':
    main()

