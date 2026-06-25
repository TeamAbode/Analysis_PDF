# Jury Analyst Pipeline

Three-phase local tool for turning Alchemer mock-juror exports into branded PDF reports.

## Quick start

**Easiest (one click):** Double-click `start.command` in Finder. The first run will set up a virtual environment and install dependencies automatically (about 30 seconds), then open the app in your browser. After that, it's instant — close the Terminal window to stop the server.

First-time only: when prompted, paste your Anthropic API key into `.env` and re-run.

**Manual (terminal):**

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set your Anthropic API key (one-time)
cp .env.example .env
# then edit .env and paste your key

# 3. Run
python app.py

# 4. Open http://localhost:8765 in your browser
```

## Workflow

**Phase 1 — Clean.** Upload Alchemer CSV + Alchemer survey PDF. Configure filters (status, speeders, straightliners, duplicates). Review flagged respondents in a table with per-person override checkboxes. Download exclusions audit CSV. Commit clean dataset.

**Phase 2 — Analyze.** Survey PDF is parsed for case metadata (parties, fact text, county). Edit if needed. The pipeline also auto-discovers extra survey questions outside the standard schema and presents them with Include/Predictor checkboxes — uncheck to drop, check "Predictor?" to add a numeric/scale question to the predictive-indicators model. Run analysis to produce every chart and stat deterministically. No AI here.

**Phase 3 — Report.** Claude writes the strategic insight prose, juror profile summaries, voir dire questions, key-takeaway callouts at the top of each section, and one-line interpretive notes under each chart. All numbers come from Phase 2; the AI only handles synthesis. Regenerate any section individually if you don't like the output. Download the PDF.

## What lives where

```
app.py                  FastAPI server, routes for each phase
pipeline/
  __init__.py
  clean.py              Filter logic + exclusion audit
  analyze.py            Charts, stats, predictive indicators, open-ended extraction
  auto_discover.py      Scans for non-canonical survey questions, classifies & charts each
  report.py             Prompt assembly, AI calls (with retry on overload), PDF rendering
  survey_parse.py       Extract case metadata from Alchemer survey PDF
  schema.py             Column-name mappings (Alchemer variable conventions)
prompts/                Per-section prompt templates (edit freely)
templates/report.html   Jinja2 template — drives the PDF layout
static/                 CSS + frontend JS
frontend/               HTML UI for each phase
workspace/<case_id>/    Per-case outputs (raw, clean, charts, analysis, report)
```

## Filter defaults (Phase 1)

- **Status**: keep only `Complete` — anyone who didn't finish, attention-check failures, etc. are caught here.
- **Speeders**: remove if total time < 1/3 of median panel duration.
- **Straightliners**: remove if any Likert grid (TIPI, Just World, Litigation Attitudes) shows 100% identical responses — and only test grids with ≥ 5 items, so short scales don't trigger false positives.
- **Duplicates**: keep first occurrence per RID/email.

All defaults are tunable in the UI. After preview, you see every flagged respondent and can manually uncheck individuals to keep them.

## Auto-discovery of extra questions (Phase 2)

After loading the clean dataset, the pipeline scans for any column that isn't part of the standard schema, isn't a system field, and isn't already covered by a known scale. Each surviving column is classified:

- **Grouped scale**: items sharing a numeric-suffix prefix (e.g. `Authoritarian1`–`Authoritarian4`) become a composite scale with a per-item chart.
- **Binary**: 0/1 or Yes/No → small two-bar chart.
- **Numeric / Likert**: ordinal/continuous → distribution chart.
- **Categorical**: ≤ 15 unique values → frequency table + bar chart.
- **Free text**: many unique values or longer responses → verbatim collected and summarized into themes by Claude.

Each discovered question appears in the Phase 2 UI with two checkboxes:

- **Include?** (default on): include this question in Section 9 of the PDF report.
- **Predictor?** (default off, only available for numeric/binary/scale): add this question to the predictive-indicators model alongside the standard predictors. Useful if you're testing whether a case-specific attitudinal item predicts plaintiff support.

Columns are auto-skipped if 100% of respondents picked the same value (no signal) or if they're system metadata (IP, region, session ID, validity check, etc.).

## Voice & framing

The report is plaintiff-counsel-facing. No statistics jargon — "regression," "coefficient," "p-value," etc. are stripped from all user-visible text. The predictive-indicators section talks about "predictive strength" and "directional pull" rather than coefficients, and reports model accuracy as "we can correctly anticipate X% of jurors' verdict direction."

Each major section opens with a short **Key Takeaway** callout (auto-generated by Claude from the section's stats). Each chart caption ends with a one-line interpretive note ("what jumps out").

## Notes

- Files persist per `case_id` in `workspace/`. Re-running a phase overwrites that phase's outputs but doesn't touch earlier phases.
- The `clean.csv` from Phase 1 is what Phase 2 reads. The `analysis_bundle.json` from Phase 2 is what Phase 3 reads. Each phase is independent — you can re-run Phase 3 with different prompt tweaks without re-cleaning.
- AI never sees individual juror rows. It receives aggregated stats + verbatim juror quotes (already selected by Phase 2).
- The model used by Phase 3 is set in `pipeline/report.py` (`MODEL = "claude-opus-4-7"`). Change there to swap models.
- API calls retry up to 3 times with exponential backoff (2s, 5s, 12s) on 529 overloaded errors. If all retries fail, the affected section renders as a clean placeholder in the PDF with a note to use the "Regenerate" button.
- Open-ended prompts instruct Claude to ignore non-substantive responses ("N/A", "nothing", single-word answers) and never theme from them.
- Prompts live in `prompts/*.txt` and are plain text — edit them to change the AI's voice or scope without touching code.

## Surveys with extra/missing questions

If a case has a few extra questions, auto-discovery picks them up automatically and adds them to Section 9. If you want a specific case-custom item added to the predictive-indicators model, just tick the "Predictor?" box in Phase 2.

If a case has more or fewer facts, the pipeline auto-detects them by scanning for `Importance_Outcome_F<N>:Fact N` columns — no code changes needed.

If a case lacks some standard predictors (e.g. no damage-cap question), the model gracefully drops that predictor from the equation.
