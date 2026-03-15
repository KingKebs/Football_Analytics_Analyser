"""
digest.py  (Integration H)
----------------------------
Generates a daily digest after the full workflow completes.
Produces either an HTML email or a standalone HTML file
(WeasyPrint PDF optional, not required).

Called after run_corners_analysis() in the workflow:
    from src.ai_enrichment.digest import generate_digest
    generate_digest(consolidated_path, corners_path, alerts_path, date_str)

Zero external API calls at generation time.
The LLM explainer (Integration D) must have run first --
ai_rationale fields are included in the digest if present.

Output:
    data/digests/digest_<DATE>.html   (always written)
    data/digests/digest_<DATE>.pdf    (only if WeasyPrint installed)

Email sending:
    Set DIGEST_EMAIL_TO in environment to auto-send after generation.
    Uses smtplib with SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS env vars.
    Email sending is optional -- digest is useful as a standalone HTML file.
"""

import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# -- Colour palette (matches your project's aesthetic) ------------------------
BLUE    = "#1A56A0"
GREEN   = "#1A7A4A"
AMBER   = "#B45309"
RED     = "#B91C1C"
BGLIGHT = "#F8FAFF"
BORDER  = "#DDE3EE"


def _load_json(path: str) -> dict:
    if not path or not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _fmt_pct(val) -> str:
    try:
        return f"{float(val):.0%}"
    except (TypeError, ValueError):
        return str(val) if val else "--"


def _fmt_f2(val) -> str:
    try:
        return f"{float(val):.2f}"
    except (TypeError, ValueError):
        return str(val) if val else "--"


def _edge_colour(edge) -> str:
    try:
        e = float(edge)
        if e >= 0.10: return GREEN
        if e >= 0.05: return AMBER
        return RED
    except (TypeError, ValueError):
        return "#666"


def _build_html(
    consolidated: dict,
    corners: dict,
    alerts: dict,
    date_str: str,
) -> str:
    """Build the complete HTML digest string."""

    date_human = datetime.strptime(date_str, "%Y%m%d").strftime("%A, %d %B %Y") \
        if len(date_str) == 8 else date_str

    predictions = consolidated.get("predictions", [])
    alert_list  = alerts.get("alerts", [])
    corners_list = corners.get("predictions", corners.get("matches", []))

    total_scanned = len(predictions)
    alert_count   = len(alert_list)
    corners_count = len(corners_list)

    # -- Value alert rows -----------------------------------------------------
    alert_rows = ""
    for a in alert_list[:10]:   # cap at 10 in email
        home = a.get("home_team") or a.get("homeTeam") or a.get("home", "?")
        away = a.get("away_team") or a.get("awayTeam") or a.get("away", "?")
        prediction  = a.get("prediction") or a.get("suggested_bet", "")
        probability = _fmt_pct(a.get("probability") or a.get("confidence"))
        edge        = _fmt_pct(a.get("edge", 0))
        kelly       = _fmt_pct(a.get("kelly_fraction") or a.get("kelly", 0))
        rationale   = a.get("ai_rationale", "")
        league      = a.get("league") or a.get("competition", "")
        ec          = _edge_colour(a.get("edge", 0))

        alert_rows += f"""
        <tr>
          <td style="padding:10px 8px;border-bottom:1px solid {BORDER};">
            <strong style="color:#1a1a1a">{home} vs {away}</strong>
            <br><span style="font-size:11px;color:#666">{league}</span>
          </td>
          <td style="padding:10px 8px;border-bottom:1px solid {BORDER};text-align:center">
            <span style="background:{BLUE};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px">{prediction}</span>
          </td>
          <td style="padding:10px 8px;border-bottom:1px solid {BORDER};text-align:center;color:{ec};font-weight:600">{edge}</td>
          <td style="padding:10px 8px;border-bottom:1px solid {BORDER};text-align:center">{probability}</td>
          <td style="padding:10px 8px;border-bottom:1px solid {BORDER};text-align:center;color:{BLUE}">{kelly}</td>
          <td style="padding:10px 8px;border-bottom:1px solid {BORDER};font-size:12px;color:#444;max-width:260px">{rationale}</td>
        </tr>"""

    if not alert_rows:
        alert_rows = f"""
        <tr>
          <td colspan="6" style="padding:20px;text-align:center;color:#888">
            No high-edge alerts found for {date_human}
          </td>
        </tr>"""

    # -- Corners rows ---------------------------------------------------------
    corners_rows = ""
    for c in corners_list[:8]:
        home = c.get("home_team") or c.get("homeTeam") or c.get("home", "?")
        away = c.get("away_team") or c.get("awayTeam") or c.get("away", "?")
        pred = c.get("prediction") or c.get("corners_prediction", "")
        conf = _fmt_pct(c.get("probability") or c.get("confidence"))
        avg  = _fmt_f2(c.get("expected_corners") or c.get("avg_corners"))

        corners_rows += f"""
        <tr>
          <td style="padding:8px;border-bottom:1px solid {BORDER}">{home} vs {away}</td>
          <td style="padding:8px;border-bottom:1px solid {BORDER};text-align:center">{pred}</td>
          <td style="padding:8px;border-bottom:1px solid {BORDER};text-align:center">{avg}</td>
          <td style="padding:8px;border-bottom:1px solid {BORDER};text-align:center">{conf}</td>
        </tr>"""

    if not corners_rows:
        corners_rows = f'<tr><td colspan="4" style="padding:16px;text-align:center;color:#888">No corners predictions available</td></tr>'

    # -- HTML shell ------------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Football Analytics Digest -- {date_human}</title>
<style>
  body {{font-family:Arial,sans-serif;background:#f0f2f5;margin:0;padding:24px;color:#1a1a1a}}
  .card {{background:#fff;border-radius:8px;border:1px solid {BORDER};margin-bottom:24px;overflow:hidden}}
  .card-header {{background:{BLUE};color:#fff;padding:14px 20px;font-size:15px;font-weight:600}}
  .card-body {{padding:20px}}
  table {{width:100%;border-collapse:collapse;font-size:13px}}
  th {{background:{BGLIGHT};color:#555;font-weight:600;padding:8px;text-align:left;border-bottom:2px solid {BORDER}}}
  .stat-grid {{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:0}}
  .stat {{background:{BGLIGHT};border:1px solid {BORDER};border-radius:6px;padding:14px 18px;flex:1;min-width:100px;text-align:center}}
  .stat-val {{font-size:26px;font-weight:700;color:{BLUE}}}
  .stat-lbl {{font-size:11px;color:#888;margin-top:4px}}
  .footer {{text-align:center;font-size:11px;color:#aaa;padding:16px}}
</style>
</head>
<body>
<div style="max-width:860px;margin:0 auto">

  <!-- Header -->
  <div style="background:{BLUE};color:#fff;padding:20px 24px;border-radius:8px 8px 0 0;margin-bottom:0">
    <div style="font-size:11px;opacity:.7;margin-bottom:4px">FOOTBALL ANALYTICS ANALYSER</div>
    <div style="font-size:22px;font-weight:700">Daily Digest</div>
    <div style="font-size:14px;opacity:.85;margin-top:4px">{date_human}</div>
  </div>

  <!-- Summary stats -->
  <div class="card" style="border-radius:0 0 8px 8px;margin-top:0">
    <div class="card-body">
      <div class="stat-grid">
        <div class="stat">
          <div class="stat-val">{total_scanned}</div>
          <div class="stat-lbl">Predictions analysed</div>
        </div>
        <div class="stat">
          <div class="stat-val" style="color:{'#1A7A4A' if alert_count > 0 else '#888'}">{alert_count}</div>
          <div class="stat-lbl">Value alerts</div>
        </div>
        <div class="stat">
          <div class="stat-val">{corners_count}</div>
          <div class="stat-lbl">Corners predictions</div>
        </div>
        <div class="stat">
          <div class="stat-val" style="font-size:14px">{datetime.utcnow().strftime('%H:%M')} UTC</div>
          <div class="stat-lbl">Generated</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Value Alerts -->
  <div class="card">
    <div class="card-header">Value Alerts -- High Edge Bets</div>
    <div class="card-body" style="padding:0">
      <table>
        <thead>
          <tr>
            <th>Match</th><th style="text-align:center">Prediction</th>
            <th style="text-align:center">Edge</th><th style="text-align:center">Prob</th>
            <th style="text-align:center">Kelly</th><th>AI Rationale</th>
          </tr>
        </thead>
        <tbody>{alert_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Corners -->
  <div class="card">
    <div class="card-header">Corners Predictions</div>
    <div class="card-body" style="padding:0">
      <table>
        <thead>
          <tr>
            <th>Match</th><th style="text-align:center">Prediction</th>
            <th style="text-align:center">Exp. Corners</th><th style="text-align:center">Confidence</th>
          </tr>
        </thead>
        <tbody>{corners_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="footer">
    Generated by Football Analytics Analyser &nbsp;.&nbsp;
    KingKebs/Football_Analytics_Analyser &nbsp;.&nbsp;
    {date_human}
    <br>This digest is for informational purposes only.
  </div>

</div>
</body>
</html>"""
    return html


def _try_pdf(html: str, pdf_path: str) -> bool:
    """Attempt PDF generation via WeasyPrint (optional dependency)."""
    try:
        from weasyprint import HTML as WPhtml
        WPhtml(string=html).write_pdf(pdf_path)
        logger.info("PDF digest written to %s", pdf_path)
        return True
    except ImportError:
        logger.debug("WeasyPrint not installed -- skipping PDF generation.")
        return False
    except Exception as exc:
        logger.warning("PDF generation failed: %s", exc)
        return False


def _try_send_email(html: str, date_human: str) -> bool:
    """Send HTML digest via SMTP if env vars are set."""
    to_addr   = os.environ.get("DIGEST_EMAIL_TO", "")
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pass = os.environ.get("SMTP_PASS", "")

    if not all([to_addr, smtp_host, smtp_user, smtp_pass]):
        logger.debug("DIGEST_EMAIL_TO / SMTP vars not set -- skipping email send.")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Football Analytics Digest -- {date_human}"
        msg["From"]    = smtp_user
        msg["To"]      = to_addr
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_addr, msg.as_string())

        logger.info("Digest emailed to %s", to_addr)
        return True

    except Exception as exc:
        logger.warning("Email send failed: %s", exc)
        return False


def generate_digest(
    consolidated_path: str,
    corners_path: str,
    alerts_path: str,
    date_str: str,
    output_dir: str = "data/digests",
) -> str:
    """
    Main entry point for Integration H.

    Call from run_analysis_workflow.py after run_corners_analysis():

        from src.ai_enrichment.digest import generate_digest
        digest_path = generate_digest(
            consolidated_path,
            corners_path,
            alerts_path,    # from post_scan_value_bets()
            self.date,
        )

    Args:
        consolidated_path: Path to consolidated_full_league_<DATE>.json
        corners_path:      Path to parsed_corners_predictions_<DATE>.json
        alerts_path:       Path to value_alerts_<DATE>.json
        date_str:          Date in YYYYMMDD format
        output_dir:        Output directory for digest files

    Returns:
        Path to the written HTML digest file.
    """
    logger.info("=== Integration H: generate_digest() ===")

    consolidated = _load_json(consolidated_path)
    corners      = _load_json(corners_path)
    alerts       = _load_json(alerts_path)

    date_human = datetime.strptime(date_str, "%Y%m%d").strftime("%A, %d %B %Y") \
        if len(date_str) == 8 else date_str

    html = _build_html(consolidated, corners, alerts, date_str)

    # -- Write HTML -----------------------------------------------------------
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    html_path = str(Path(output_dir) / f"digest_{date_str}.html")
    with open(html_path, "w") as f:
        f.write(html)
    logger.info("HTML digest written to %s", html_path)

    # -- Optional PDF ---------------------------------------------------------
    pdf_path = str(Path(output_dir) / f"digest_{date_str}.pdf")
    _try_pdf(html, pdf_path)

    # -- Optional email -------------------------------------------------------
    _try_send_email(html, date_human)

    return html_path
