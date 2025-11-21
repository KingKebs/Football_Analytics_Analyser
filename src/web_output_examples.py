#!/usr/bin/env python3

"""
Web Output Examples for Football Analytics Full League Script

This script demonstrates how to format the output from automate_football_analytics_fullLeague.py
for web applications, including JSON API responses and HTML dashboards.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

# Sample data structure (simulating the output from the full league script)
SAMPLE_DATA = {
    "suggestions": [
        {
            "home": "Manchester City",
            "away": "Liverpool",
            "xg_home": 2.45,
            "xg_away": 1.23,
            "picks": [
                {"market": "1X2", "selection": "Home", "prob": 0.65, "odds": 1.54},
                {"market": "BTTS", "selection": "Yes", "prob": 0.72, "odds": 1.39}
            ]
        },
        {
            "home": "Arsenal",
            "away": "Chelsea",
            "xg_home": 1.89,
            "xg_away": 1.67,
            "picks": [
                {"market": "Over/Under 2.5", "selection": "Over2.5", "prob": 0.78, "odds": 1.28}
            ]
        }
    ],
    "favorable_parlays": [
        {
            "size": 2,
            "legs": ["Manchester City v Liverpool (1X2 Home)", "Arsenal v Chelsea (Over2.5)"],
            "probability": 0.51,
            "decimal_odds": 1.97,
            "stake_suggestion": 25.0,
            "potential_return": 49.25
        },
        {
            "size": 3,
            "legs": ["Manchester City v Liverpool (BTTS Yes)", "Arsenal v Chelsea (Over2.5)", "Another Match"],
            "probability": 0.56,
            "decimal_odds": 2.34,
            "stake_suggestion": 20.0,
            "potential_return": 46.80
        }
    ]
}

def generate_json_api_response(data: Dict[str, Any], league_code: str = "E0") -> str:
    """Generate a JSON API response for web consumption."""
    response = {
        "status": "success",
        "league": league_code,
        "timestamp": datetime.now().isoformat(),
        "data": data,
        "metadata": {
            "total_suggestions": len(data.get("suggestions", [])),
            "total_parlays": len(data.get("favorable_parlays", [])),
            "generated_by": "Football Analytics Analyser v2.0"
        }
    }
    return json.dumps(response, indent=2)

def generate_html_dashboard(data: Dict[str, Any], league_code: str = "E0") -> str:
    """Generate an HTML dashboard for displaying the results."""
    suggestions = data.get("suggestions", [])
    parlays = data.get("favorable_parlays", [])

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Football Analytics - {league_code} Full League Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        .match-card {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; background: #f9f9f9; }}
        .match-header {{ font-weight: bold; font-size: 1.2em; color: #2c3e50; }}
        .xg-display {{ margin: 10px 0; }}
        .pick {{ display: inline-block; margin: 5px; padding: 5px 10px; background: #3498db; color: white; border-radius: 3px; }}
        .parlay-card {{ border: 1px solid #27ae60; margin: 10px 0; padding: 15px; border-radius: 5px; background: #d5f4e6; }}
        .probability {{ color: #e74c3c; font-weight: bold; }}
        .odds {{ color: #9b59b6; font-weight: bold; }}
        .stake {{ color: #f39c12; font-weight: bold; }}
        .return {{ color: #27ae60; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>⚽ Football Analytics Dashboard</h1>
        <h2>League: {league_code} - Full Analysis</h2>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>📊 Match Suggestions ({len(suggestions)} matches)</h2>
"""

    for suggestion in suggestions:
        html += f"""
        <div class="match-card">
            <div class="match-header">{suggestion['home']} vs {suggestion['away']}</div>
            <div class="xg-display">
                Expected Goals: <strong>{suggestion['home']}</strong> {suggestion['xg_home']:.2f} - {suggestion['xg_away']:.2f} <strong>{suggestion['away']}</strong>
            </div>
            <div>
                <strong>Recommended Picks:</strong>
                {"".join([f'<span class="pick">{pick["market"]} {pick["selection"]} ({pick["prob"]*100:.1f}% / {pick["odds"]:.2f})</span>' for pick in suggestion['picks']])}
            </div>
        </div>
"""

    html += f"""
        <h2>🎲 Favorable Parlays ({len(parlays)} combinations)</h2>
"""

    if parlays:
        html += """
        <table>
            <thead>
                <tr>
                    <th>Size</th>
                    <th>Legs</th>
                    <th>Probability</th>
                    <th>Decimal Odds</th>
                    <th>Suggested Stake</th>
                    <th>Potential Return</th>
                </tr>
            </thead>
            <tbody>
"""
        for parlay in parlays:
            html += f"""
                <tr>
                    <td>{parlay['size']}</td>
                    <td>{"<br>".join(parlay['legs'])}</td>
                    <td class="probability">{parlay['probability']*100:.1f}%</td>
                    <td class="odds">{parlay['decimal_odds']:.2f}</td>
                    <td class="stake">${parlay['stake_suggestion']:.2f}</td>
                    <td class="return">${parlay['potential_return']:.2f}</td>
                </tr>
"""
        html += """
            </tbody>
        </table>
"""
    else:
        html += "<p>No favorable parlays found meeting the criteria (prob > 50%, odds > 2.0).</p>"

    html += """
    </div>
</body>
</html>
"""
    return html

def generate_csv_export(data: Dict[str, Any], league_code: str = "E0") -> str:
    """Generate CSV export for spreadsheet analysis."""
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)

    # Suggestions
    writer.writerow(["Match Suggestions"])
    writer.writerow(["Home", "Away", "XG Home", "XG Away", "Pick Market", "Pick Selection", "Probability", "Odds"])

    for suggestion in data.get("suggestions", []):
        for pick in suggestion.get("picks", []):
            writer.writerow([
                suggestion["home"],
                suggestion["away"],
                suggestion["xg_home"],
                suggestion["xg_away"],
                pick["market"],
                pick["selection"],
                pick["prob"],
                pick["odds"]
            ])

    writer.writerow([])
    writer.writerow(["Favorable Parlays"])
    writer.writerow(["Size", "Legs", "Probability", "Decimal Odds", "Stake Suggestion", "Potential Return"])

    for parlay in data.get("favorable_parlays", []):
        writer.writerow([
            parlay["size"],
            "; ".join(parlay["legs"]),
            parlay["probability"],
            parlay["decimal_odds"],
            parlay["stake_suggestion"],
            parlay["potential_return"]
        ])

    return output.getvalue()

def main():
    """Demonstrate different output formats."""

    print("=== JSON API Response Example ===")
    json_response = generate_json_api_response(SAMPLE_DATA, "E0")
    print(json_response[:500] + "...\n")

    print("=== HTML Dashboard Example (first 300 chars) ===")
    html_output = generate_html_dashboard(SAMPLE_DATA, "E0")
    print(html_output[:300] + "...\n")

    print("=== CSV Export Example ===")
    csv_output = generate_csv_export(SAMPLE_DATA, "E0")
    print(csv_output[:200] + "...\n")

    # Save examples to files
    with open("sample_api_response.json", "w") as f:
        f.write(json_response)

    with open("sample_dashboard.html", "w") as f:
        f.write(html_output)

    with open("sample_export.csv", "w") as f:
        f.write(csv_output)

    print("Sample files saved: sample_api_response.json, sample_dashboard.html, sample_export.csv")

if __name__ == "__main__":
    main()
