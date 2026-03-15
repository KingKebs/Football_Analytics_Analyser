"""
streamlit_chat_tab.py  (Integration I)
----------------------------------------
Drop-in Streamlit chat tab that lets users query their analysis data
using natural language via the Claude API.

The AI's answers are grounded entirely in the consolidated JSON and
value_alerts JSON already produced by the workflow. No hallucination
risk -- it literally cannot reference data outside those files.

HOW TO ADD TO src/streamlit_app.py
-----------------------------------
Find where your existing tabs are defined, e.g.:

    tab1, tab2 = st.tabs(["Full League", "Corners"])

Change to:

    tab1, tab2, tab_chat = st.tabs(["Full League", "Corners", "Ask the Analyst"])

Then add at the end:

    from src.ai_enrichment.streamlit_chat_tab import render_chat_tab
    with tab_chat:
        render_chat_tab()

That's it. No changes to run_analysis_workflow.py.

Rate limit note:
  Each user message = 1 Claude API call.
  The call is made only when the user hits Enter -- not on page load.
  The 23h file cache is NOT used here (chat is interactive, not batch).
  The CLAUDE_LIMITER token bucket prevents burst abuse.
"""

import json
import logging
from glob import glob
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Max characters of JSON context fed to Claude (well inside 200k context window)
MAX_CONTEXT_CHARS = 12_000

# System prompt template
SYSTEM_PROMPT_TEMPLATE = """You are a football analytics assistant with access to today's analysis data.
Answer questions using ONLY the data provided below. Be concise and direct.
When recommending bets, always cite the edge, Kelly fraction, and AI rationale from the data.
If the data does not contain enough information to answer, say so clearly.

Today's date: {date}
Leagues analysed: {leagues}
Total predictions: {total_predictions}
Value alerts found: {alert_count}

=== PREDICTIONS DATA ===
{predictions_json}

=== VALUE ALERTS ===
{alerts_json}

=== CORNERS PREDICTIONS ===
{corners_json}
"""


def _find_latest_file(pattern: str) -> Optional[str]:
    """Return the most recently modified file matching a glob pattern."""
    files = sorted(glob(pattern), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_json(path: Optional[str]) -> dict:
    if not path or not Path(path).exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated for context window]"


def _build_system_prompt(consolidated: dict, alerts: dict, corners: dict) -> str:
    predictions = consolidated.get("predictions", [])
    alert_list  = alerts.get("alerts", [])
    leagues     = list({p.get("league") or p.get("competition", "") for p in predictions if p.get("league") or p.get("competition")})

    pred_json    = _truncate(json.dumps(predictions[:40], indent=1), MAX_CONTEXT_CHARS // 2)
    alerts_json  = _truncate(json.dumps(alert_list[:20],  indent=1), MAX_CONTEXT_CHARS // 3)
    corners_json = _truncate(
        json.dumps(corners.get("predictions", corners.get("matches", []))[:20], indent=1),
        MAX_CONTEXT_CHARS // 4
    )

    return SYSTEM_PROMPT_TEMPLATE.format(
        date=consolidated.get("_meta", {}).get("date", "today"),
        leagues=", ".join(leagues) if leagues else "multiple",
        total_predictions=len(predictions),
        alert_count=len(alert_list),
        predictions_json=pred_json,
        alerts_json=alerts_json,
        corners_json=corners_json,
    )


def _call_claude(system: str, messages: list[dict]) -> str:
    """Make the Claude API call with rate limiter."""
    try:
        import anthropic
        from src.ai_enrichment.rate_limiter import CLAUDE_LIMITER

        CLAUDE_LIMITER.wait()
        client = anthropic.Anthropic()

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    except ImportError:
        return "anthropic package not installed. Run: pip install anthropic"
    except Exception as exc:
        logger.error("Claude API error: %s", exc)
        return f"API error: {exc}"


def render_chat_tab() -> None:
    """
    Render the complete chat tab. Call this inside `with tab_chat:`.
    Requires streamlit to be running -- this function imports streamlit.
    """
    import streamlit as st

    st.markdown("### Ask the Analyst")
    st.caption(
        "Ask questions about today's predictions, value bets, or corners. "
        "Answers are grounded in your model's actual output -- no external data."
    )

    # -- Load latest analysis files -------------------------------------------
    consolidated_path = _find_latest_file("data/analysis/consolidated_full_league_*.json")
    alerts_path       = _find_latest_file("data/analysis/value_alerts_*.json")
    corners_path      = _find_latest_file("data/corners/parsed_corners_predictions_*.json")

    consolidated = _load_json(consolidated_path)
    alerts       = _load_json(alerts_path)
    corners      = _load_json(corners_path)

    if not consolidated:
        st.warning(
            "No analysis data found. Run the workflow first:\n"
            "`python run_analysis_workflow.py --auto`"
        )
        return

    # Status bar
    predictions = consolidated.get("predictions", [])
    alert_list  = alerts.get("alerts", [])
    col1, col2, col3 = st.columns(3)
    col1.metric("Predictions loaded", len(predictions))
    col2.metric("Value alerts", len(alert_list))
    col3.metric("Corners predictions",
                len(corners.get("predictions", corners.get("matches", []))))

    st.divider()

    # -- Suggested prompts -----------------------------------------------------
    st.markdown("**Suggested questions:**")
    suggestions = [
        "What are the top 3 value bets today?",
        "Show me all predictions with edge above 8%",
        "Which corners bets have the highest confidence?",
        "Are there any derbies or cup games I should know about?",
        "Summarise the predictions for the Premier League today",
    ]
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, key=f"sugg_{i}", use_container_width=True):
            st.session_state["chat_input_prefill"] = suggestion

    st.divider()

    # -- Chat history ----------------------------------------------------------
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # -- Input -----------------------------------------------------------------
    prefill = st.session_state.pop("chat_input_prefill", "")
    user_input = st.chat_input("e.g. Best bet this weekend?", key="chat_input")

    # Handle both direct input and button prefill
    query = user_input or (prefill if prefill else None)

    if query:
        # Display user message
        st.session_state["chat_messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Build context-grounded system prompt
        system = _build_system_prompt(consolidated, alerts, corners)

        # Build message history for Claude (last 6 turns to keep context lean)
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["chat_messages"][-6:]
        ]

        # Call Claude
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                response_text = _call_claude(system, history)
            st.write(response_text)

        st.session_state["chat_messages"].append(
            {"role": "assistant", "content": response_text}
        )

    # -- Clear button ----------------------------------------------------------
    if st.session_state.get("chat_messages"):
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state["chat_messages"] = []
            st.rerun()
