"""
WORKFLOW_PATCH_GUIDE.py
Shows exactly where to add all integrations in run_analysis_workflow.py.
Run: python WORKFLOW_PATCH_GUIDE.py
"""

PATCH = """
+======================================================================+
|  run_analysis_workflow.py  PATCH -- feature/ai-enrichment-layer      |
+======================================================================+

STEP 0  New imports (top of file)
----------------------------------
    from src.ai_enrichment import (
        pre_run_fetch,
        analyse_context_warnings, print_context_warnings,
        post_process_explanations,
        post_scan_value_bets,
        generate_digest,
    )

STEP 0b  New CLI flags
-----------------------
    parser.add_argument("--skip-fetch", action="store_true",
        help="Skip Integration A: do not auto-fetch fixtures")
    parser.add_argument("--skip-ai", action="store_true",
        help="Skip D+E+H: no LLM explainer, no value scan, no digest")

CHANGE 1 -- Integration A  [BEFORE Step 1: read_upcoming_matches()]
--------------------------------------------------------------------
    if not getattr(self.args, 'skip_fetch', False):
        try:
            fetched = pre_run_fetch(self.date_str)
            if fetched:
                self.logger.info("[AI-A] Fixtures fetched -> %s", fetched)
        except Exception as exc:
            self.logger.warning("[AI-A] Fetch failed (%s). Using existing file.", exc)

CHANGE 2 -- Integration C  [INSIDE detect_leagues(), before Step 3 prompt]
---------------------------------------------------------------------------
    ctx = analyse_context_warnings(self.upcoming_fixtures, detected_leagues)
    if ctx['summary']:
        print_context_warnings(ctx)

CHANGE 3 -- Integration D  [AFTER run_full_league_analysis()]
--------------------------------------------------------------
    if consolidated_path and not getattr(self.args, 'skip_ai', False):
        try:
            n = post_process_explanations(consolidated_path, self.date)
            self.logger.info("[AI-D] Rationales added to %d predictions", n)
        except Exception as exc:
            self.logger.warning("[AI-D] Explainer failed (%s). Continuing.", exc)

CHANGE 4 -- Integration E  [AFTER Integration D]
-------------------------------------------------
    alerts_path = ""
    if consolidated_path and not getattr(self.args, 'skip_ai', False):
        try:
            alerts_path = post_scan_value_bets(consolidated_path, self.date)
            self.logger.info("[AI-E] Value alerts -> %s", alerts_path)
        except Exception as exc:
            self.logger.warning("[AI-E] Value scan failed (%s). Continuing.", exc)

CHANGE 5 -- Integration H  [AFTER run_corners_analysis()]
----------------------------------------------------------
    if not getattr(self.args, 'skip_ai', False):
        try:
            digest_path = generate_digest(
                consolidated_path, corners_path, alerts_path, self.date)
            self.logger.info("[AI-H] Digest -> %s", digest_path)
        except Exception as exc:
            self.logger.warning("[AI-H] Digest failed (%s). Continuing.", exc)

CHANGE 6 -- Integration I  [src/streamlit_app.py only]
-------------------------------------------------------
    tab1, tab2, tab_chat = st.tabs(["Full League", "Corners", "Ask the Analyst"])
    ...
    from src.ai_enrichment.streamlit_chat_tab import render_chat_tab
    with tab_chat:
        render_chat_tab()

========================================================================
FULL EXECUTION ORDER AFTER PATCHING
========================================================================
  [AI-A]  pre_run_fetch()                  1 API call, 23h cached
  Step 1  read_upcoming_matches()
  Step 2  detect_leagues()
  [AI-C]    analyse_context_warnings()     zero API calls
  Step 3  confirm_leagues()
  Step 4  convert_fixtures()
  Step 5  run_full_league_analysis()
  [AI-D]    post_process_explanations()    1 Claude call, batched+cached
  [AI-E]    post_scan_value_bets()         zero API calls
  Step 6  run_corners_analysis()
  [AI-H]  generate_digest()               zero API calls
  [AI-I]  render_chat_tab()               1 Claude call per user message

  Net calls per day: 2 first run / 0 re-runs (cache)

========================================================================
USAGE AFTER PATCHING
========================================================================
  python run_analysis_workflow.py --auto            # full AI pipeline
  python run_analysis_workflow.py --auto --skip-ai  # analysis only
  python run_analysis_workflow.py --auto --skip-fetch  # no fixture fetch

  # Cron (09:00 daily)
  0 9 * * * cd /path/to/repo && \\
    python run_analysis_workflow.py --auto --verbose \\
    >> logs/workflow_$(date +%%Y%%m%%d).log 2>&1
"""

if __name__ == "__main__":
    print(PATCH)
