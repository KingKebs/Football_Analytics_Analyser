---
name: AI Enrichment Skill
description: Manage AI integration points, explanation generation, value scanning, and context enrichment
applies_to: ["src/ai_enrichment/**/*.py", "run_analysis_workflow.py"]
---

# AI Enrichment Skill

## Purpose
Integrate AI-powered explanations, analysis scanning, and value enrichment throughout the football analytics pipeline.

## Core Components
- **Explainer**: Generate natural language explanations for predictions
- **Value Scanner**: Post-process predictions to detect value opportunities
- **Context Analyzer**: Examine warnings and anomalies
- **Digest Generator**: Create summary reports

## Integration Points
1. **Pre-run Fetch** (`pre_run_fetch`): Gather context before analysis
2. **Context Analysis** (`analyse_context_warnings`): Review warnings
3. **Explanation Printing** (`print_context_warnings`): Format output
4. **Post-processing** (`post_process_explanations`): Enrich predictions
5. **Value Scanning** (`post_scan_value_bets`): Find high-value opportunities
6. **Digest Generation** (`generate_digest`): Create reports

## Typical Workflow
```python
# Pre-analysis
context = pre_run_fetch(fixture_data, league_configs)

# During analysis
# ... normal prediction generation ...

# Post-analysis
explained = post_process_explanations(predictions)
value_bets = post_scan_value_bets(explained)
digest = generate_digest(value_bets, context)
```

## Common Customizations
- Adjust explanation templates (confidence thresholds, detail level)
- Modify value scanning criteria (EV ranges, market preferences)
- Enhance context analysis (warning types, severity)
- Customize digest output (format, sorting)

## Documentation
- See `src/ai_enrichment/` for implementation
- Review `run_analysis_workflow.py` for integration examples
- Check docs for AI integration architecture
