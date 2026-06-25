"""
Phase 3: report generation.

Reads analysis_bundle.json + chart images, calls the Anthropic API for the
prose sections (executive summary, per-fact strategic insights, juror profiles,
voir dire, open-ended themes), then renders the final PDF with WeasyPrint.

The AI never sees raw juror data — only aggregated stats + curated verbatim
quotes that Phase 2 already extracted.

The deterministic numbers come from the analysis bundle. The AI's job is
prose synthesis, not calculation.
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Optional

import anthropic
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML, CSS


load_dotenv()


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-7"   # most recent capable model as of May 2026
_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Create a .env file in the project "
                "root with: ANTHROPIC_API_KEY=sk-ant-..."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def call_ai(prompt: str, max_tokens: int = 1500, system: Optional[str] = None,
            max_retries: int = 3) -> str:
    """Single call to Claude with retry on transient overload (529) errors.
    Returns the text content. Raises if all retries fail."""
    import time

    client = get_client()
    kwargs = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system

    # Backoff schedule for transient errors (529 overloaded, 529 rate-limited,
    # connection timeouts). After max_retries the last error is raised.
    backoff = [2, 5, 12]
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(**kwargs)
            out = "".join(b.text for b in resp.content if hasattr(b, "text"))
            return out.strip()
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # Only retry on transient errors
            transient = (
                "overload" in err_str
                or "529" in err_str
                or "rate_limit" in err_str
                or "timeout" in err_str
                or "connection" in err_str
            )
            if not transient or attempt == max_retries - 1:
                raise
            wait = backoff[min(attempt, len(backoff) - 1)]
            time.sleep(wait)
    raise last_err   # safety; should be unreachable


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt_template(name: str) -> str:
    path = PROMPTS_DIR / name
    return path.read_text(encoding="utf-8")


def render_prompt(template_name: str, ctx: dict) -> str:
    """Simple {{var}} substitution into prompt templates."""
    tpl = load_prompt_template(template_name)
    env = Environment()
    return env.from_string(tpl).render(**ctx)


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def _fmt_money(v) -> str:
    if v is None:
        return "—"
    return f"{int(v):,}"


def generate_executive_summary(bundle: dict) -> str:
    comp = bundle["compensation"]
    verdict = bundle["verdict"]
    meta = bundle["case_metadata"]

    # Top bracket
    top_bracket = "—"; top_bracket_pct = 0
    if comp.get("bracket_selection", {}).get("pct"):
        bp = comp["bracket_selection"]["pct"]
        if bp:
            top_key = max(bp, key=bp.get)
            top_bracket = top_key
            top_bracket_pct = bp[top_key]

    ctx = {
        "case_caption": meta.get("case_caption", ""),
        "county_state": meta.get("county_state", ""),
        "plaintiff_name": meta.get("plaintiff_name", ""),
        "defendant_names_joined": ", ".join(meta.get("defendant_names", [])),
        "n_respondents": bundle["n_respondents"],
        "pct_plaintiff_leaning": verdict.get("pct_plaintiff_leaning"),
        "pct_defense_leaning": verdict.get("pct_defense_leaning"),
        "pct_liable_yes": verdict.get("pct_liable_yes"),
        "mean_responsibility_plaintiff": verdict.get("mean_responsibility_plaintiff"),
        "mean_responsibility_defendant": verdict.get("mean_responsibility_defendant"),
        "mean_award_all_formatted": _fmt_money(comp.get("mean_award_all")),
        "median_award_all_formatted": _fmt_money(comp.get("median_award_all")),
        "mean_award_liable_formatted": _fmt_money(comp.get("mean_award_liable")),
        "top_bracket": top_bracket,
        "top_bracket_pct": top_bracket_pct,
    }
    prompt = render_prompt("exec_summary.txt", ctx)
    return call_ai(prompt, max_tokens=800)


def generate_fact_insight(bundle: dict, fact: dict) -> str:
    meta = bundle["case_metadata"]
    quotes_block = "\n".join(f"- \"{q}\"" for q in fact["quotes"][:5]) or "(no responses)"

    ctx = {
        "case_caption": meta.get("case_caption", ""),
        "plaintiff_name": meta.get("plaintiff_name", ""),
        "fact_num": fact["num"],
        "fact_text": fact["text"],
        "importance_mean": fact["importance_mean"],
        "influence_mean": fact["influence_mean"],
        "pct_plaintiff": fact["pct_plaintiff"],
        "pct_defense": fact["pct_defense"],
        "direction": fact["direction"],
        "quotes_block": quotes_block,
    }
    prompt = render_prompt("fact_insight.txt", ctx)
    return call_ai(prompt, max_tokens=600)


def generate_section_callouts(bundle: dict) -> dict:
    """One batched API call to produce all section takeaways + chart captions.
    Returns a dict of short interpretive strings keyed by section.
    On parse failure, returns an empty dict (template will skip the callouts).
    """
    meta = bundle["case_metadata"]
    v = bundle["verdict"]
    c = bundle["compensation"]
    p = bundle["predictors"]

    # Compact stats block
    def fm(v):
        return "—" if v is None else (f"${v:,}" if isinstance(v, int) and v > 1000 else str(v))

    stats_lines = [
        f"- Plaintiff-leaning: {v.get('pct_plaintiff_leaning')}%",
        f"- Defense-leaning: {v.get('pct_defense_leaning')}%",
        f"- Found Liable: {v.get('pct_liable_yes')}%",
        f"- Mean Responsibility on Plaintiff: {v.get('mean_responsibility_plaintiff')}%",
        f"- Mean Responsibility on Defendant(s): {v.get('mean_responsibility_defendant')}%",
        f"- Mean Award (all): {fm(c.get('mean_award_all'))}",
        f"- Median Award (all): {fm(c.get('median_award_all'))}",
        f"- Mean Award (liable only): {fm(c.get('mean_award_liable'))}",
        f"- Deserving Mean (1-5): {c.get('deserving_mean')}",
        f"- Eggshell Mean (1-7): {c.get('eggshell_mean')}",
    ]

    # A/B sampling-design results (if present)
    ab = c.get("ab_analysis") or {}
    if ab.get("ab_present") and ab.get("ttest"):
        t = ab["ttest"]
        stats_lines.append(
            f"- A/B SAMPLING TEST: {t.get('interpretation','(no interpretation)')}"
        )
    elif ab.get("per_variant"):
        # Only one variant filled — note that briefly
        which = list(ab.get("per_variant", {}).keys())
        if which:
            v0 = ab["per_variant"][which[0]]
            stats_lines.append(
                f"- A/B SAMPLING: only the '{v0.get('raw_label')}' variant has data "
                f"(n={v0.get('n')}); A/B comparison not possible."
            )

    # Expected case value (per defendant)
    for block in c.get("case_value", []) or []:
        if not block.get("original"): continue
        sc = block["original"]["scenarios"]
        st = block["original"]["stats"]
        def_label = block.get("defendant_label", "Defendant")
        stats_lines.append(
            f"- EXPECTED CASE VALUE — {def_label}: "
            f"Most Likely {fm(sc.get('most_likely',{}).get('value'))}, "
            f"Worst Day (defense) {fm(sc.get('worst_day_defense',{}).get('value'))}, "
            f"Plaintiff Verdict in Isolation {fm(sc.get('plaintiff_verdict_isolated',{}).get('value'))}; "
            f"P(plaintiff verdict)={st.get('p_plaintiff_verdict',0)*100:.1f}%"
        )

    defense_facts = [f for f in bundle.get("facts", []) if f.get("direction") == "DEFENSE"]
    plaintiff_facts = [f for f in bundle.get("facts", []) if f.get("direction") == "PLAINTIFF"]
    defense_fact_str = ", ".join(f"F{f['num']} ({f['pct_defense']}% defense)" for f in defense_facts[:4]) or "(none)"
    plaintiff_fact_str = ", ".join(f"F{f['num']} ({f['pct_plaintiff']}% plaintiff)" for f in plaintiff_facts[:4]) or "(none)"

    indicators = []
    for ind in p.get("coefficients", [])[:6]:
        direction = ind.get("direction", "")
        indicators.append(f"- {ind['predictor']} ({direction.lower()} pull)")
    indicators_block = "\n".join(indicators) or "(no model)"

    ctx = {
        "case_caption": meta.get("case_caption", ""),
        "plaintiff_name": meta.get("plaintiff_name", ""),
        "n_respondents": bundle["n_respondents"],
        "stats_block": "\n".join(stats_lines),
        "defense_facts_list": defense_fact_str,
        "plaintiff_facts_list": plaintiff_fact_str,
        "indicators_list": indicators_block,
    }
    prompt = render_prompt("section_callouts.txt", ctx)
    raw = call_ai(prompt, max_tokens=1500)
    # Strip code fences if AI added them despite the instruction
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        out = json.loads(raw)
        if isinstance(out, dict):
            return out
    except Exception:
        pass
    return {}


def generate_juror_profiles(bundle: dict) -> str:
    preds = bundle["predictors"]
    coefs = preds.get("coefficients", [])
    if not coefs:
        return ""

    coef_lines = []
    for c in coefs:
        coef_lines.append(f"- {c['predictor']}: {c['coefficient']:+.3f} ({c['direction']})")
    ctx = {
        "case_caption": bundle["case_metadata"].get("case_caption", ""),
        "coefficients_block": "\n".join(coef_lines),
        "accuracy_pct": round((preds.get("accuracy") or 0) * 100, 1),
        "n_in_model": preds.get("n", 0),
    }
    prompt = render_prompt("juror_profiles.txt", ctx)
    return call_ai(prompt, max_tokens=900)


def generate_voir_dire(bundle: dict) -> str:
    facts = bundle["facts"]
    plaintiff_facts = sorted(
        [f for f in facts if f["direction"] == "PLAINTIFF"],
        key=lambda f: -f["pct_plaintiff"],
    )[:4]
    defense_facts = sorted(
        [f for f in facts if f["direction"] == "DEFENSE"],
        key=lambda f: -f["pct_defense"],
    )[:3]

    def _fact_line(f):
        return (f"F{f['num']} — \"{f['text'][:120]}{'...' if len(f['text'])>120 else ''}\" "
                f"(influence={f['influence_mean']}/6, %plt={f['pct_plaintiff']}%)")

    plaintiff_block = "\n".join(_fact_line(f) for f in plaintiff_facts) or "(none)"
    defense_block = "\n".join(_fact_line(f) for f in defense_facts) or "(none)"

    # Top defense predictors
    defense_preds = [c for c in bundle["predictors"].get("coefficients", []) if c["direction"] == "Defense"][:5]
    defense_pred_block = "\n".join(
        f"- {c['predictor']} (coef {c['coefficient']:+.3f})" for c in defense_preds
    ) or "(none)"

    ctx = {
        "case_caption": bundle["case_metadata"].get("case_caption", ""),
        "plaintiff_name": bundle["case_metadata"].get("plaintiff_name", ""),
        "plaintiff_facts_block": plaintiff_block,
        "defense_facts_block": defense_block,
        "defense_predictors_block": defense_pred_block,
    }
    prompt = render_prompt("voir_dire.txt", ctx)
    return call_ai(prompt, max_tokens=700)


def generate_open_ended_themes(bundle: dict, question_key: str) -> str:
    """question_key in: 'narrative', 'evidence_gap', 'unanswered'"""
    responses = bundle["open_ended"].get(question_key, [])
    if not responses:
        return ""
    topic_map = {
        "narrative": ("Juror narratives — what jurors think really happened",
                      "In your own words, what do you think really happened in this case?"),
        "evidence_gap": ("Evidence gaps — what would have changed verdicts",
                         "What specific additional evidence, testimony, or information would have changed your verdict?"),
        "unanswered": ("Biggest unanswered questions",
                       "What was the biggest unanswered question you still had after reading the case facts?"),
    }
    topic, question_text = topic_map[question_key]
    # Truncate huge response sets to keep prompt size reasonable
    if len(responses) > 60:
        responses = responses[:60]
    block = "\n".join(f"- {r}" for r in responses)
    ctx = {
        "question_topic": topic,
        "question_text": question_text,
        "responses_block": block,
    }
    prompt = render_prompt("open_ended_themes.txt", ctx)
    return call_ai(prompt, max_tokens=1500)


def generate_discovery_summary(question_label: str, responses: list[str]) -> str:
    """AI summary for a free-text auto-discovered question.
    Uses the same themes prompt as standard open-endeds, treating the question
    label as the topic. Returns a single short paragraph or themed list."""
    if not responses:
        return ""
    if len(responses) > 60:
        responses = responses[:60]
    block = "\n".join(f"- {r}" for r in responses)
    ctx = {
        "question_topic": f"Auto-discovered survey question: {question_label}",
        "question_text": question_label,
        "responses_block": block,
    }
    prompt = render_prompt("open_ended_themes.txt", ctx)
    return call_ai(prompt, max_tokens=1000)


# ---------------------------------------------------------------------------
# Markdown to safe HTML for inline insertion into Jinja template
# ---------------------------------------------------------------------------

FAILED_PREFIX = "__FAILED__:"


def section_html(text: str, section_label: str = "this section") -> str:
    """Wrap md_to_html with failure-sentinel handling.
    If the AI generation failed for this section, render a clean placeholder
    callout instead of stamping the raw error message into the report."""
    if not text:
        return ""
    if text.startswith(FAILED_PREFIX):
        return (
            '<div class="alert">'
            f'<strong>{section_label.capitalize()} could not be generated.</strong> '
            "The AI service was temporarily unavailable. "
            "Open Phase 3 in the app and click <em>Regenerate</em> on this section, "
            "then re-download the PDF."
            "</div>"
        )
    return md_to_html(text)


def md_to_html(text: str) -> str:
    """Convert AI markdown output to HTML for the PDF template.

    Supports:
      - paragraphs (blank-line separated)
      - bold (**text** or __text__)
      - italics (*text* or _text_) — only when not part of a bold marker
      - bullet lists: lines starting with '-', '*', or '•'
      - numbered lists: lines starting with '1.', '2.', etc.
      - Q-prefixed items: 'Q1', 'Q2:' etc. become numbered items
      - section headers: short uppercase lines or **Bolded titles** on their own line

    HTML special chars are escaped first.
    """
    if not text:
        return ""

    # 1. HTML-escape special chars BEFORE any markdown parsing
    text = text.strip()
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def inline_markdown(s: str) -> str:
        """Convert **bold** and *italic* in a string. No backrefs."""
        # Bold first (longer marker wins). Use a callable to avoid backref issues.
        s = re.sub(r"\*\*([^*\n]+?)\*\*", lambda m: f"<strong>{m.group(1)}</strong>", s)
        s = re.sub(r"__([^_\n]+?)__",     lambda m: f"<strong>{m.group(1)}</strong>", s)
        # Italics — careful not to swallow remaining asterisks accidentally
        s = re.sub(r"(?<!\*)\*([^*\n]+?)\*(?!\*)", lambda m: f"<em>{m.group(1)}</em>", s)
        return s

    lines = text.split("\n")
    out_parts: list[str] = []
    para_buf: list[str] = []
    list_buf: list[str] = []
    list_type: Optional[str] = None   # "ul" or "ol"

    def flush_paragraph():
        if not para_buf:
            return
        joined = " ".join(para_buf).strip()
        if joined:
            out_parts.append(f"<p>{inline_markdown(joined)}</p>")
        para_buf.clear()

    def flush_list():
        nonlocal list_type
        if not list_buf:
            return
        tag = list_type or "ul"
        items = "".join(f"<li>{inline_markdown(it)}</li>" for it in list_buf)
        out_parts.append(f"<{tag}>{items}</{tag}>")
        list_buf.clear()
        list_type = None

    bullet_re   = re.compile(r"^\s*[-*•]\s+(.*)$")
    numbered_re = re.compile(r"^\s*\d+[.)]\s+(.*)$")
    q_prefix_re = re.compile(r"^\s*(Q\d+)\s*[:.]?\s*(.*)$")

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        # Bullet list?
        m = bullet_re.match(line)
        if m:
            flush_paragraph()
            if list_type == "ol":
                flush_list()
            list_type = "ul"
            list_buf.append(m.group(1))
            continue

        # Numbered list?
        m = numbered_re.match(line)
        if m:
            flush_paragraph()
            if list_type == "ul":
                flush_list()
            list_type = "ol"
            list_buf.append(m.group(1))
            continue

        # Q-prefixed item (voir dire format)?
        m = q_prefix_re.match(line)
        if m:
            flush_paragraph()
            if list_type == "ul":
                flush_list()
            list_type = "ol"
            qtag, rest = m.group(1), m.group(2)
            list_buf.append(f"<strong>{qtag}</strong> {rest}")
            continue

        # Regular paragraph line
        flush_list()
        para_buf.append(stripped)

    flush_paragraph()
    flush_list()
    return "\n".join(out_parts)


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def build_report_context(bundle: dict, ai_sections: dict) -> dict:
    """Combine the analysis bundle with the AI-generated prose into a single
    context dict for Jinja rendering."""
    from datetime import datetime

    # Format compensation values for display
    comp = bundle["compensation"]
    comp_fmt = dict(comp)
    for k in ["mean_award_all", "median_award_all", "mean_award_liable", "median_award_liable"]:
        v = comp.get(k)
        comp_fmt[f"{k}_fmt"] = f"${int(v):,}" if v else "—"

    # Eggshell mean display
    comp_fmt["eggshell_mean_fmt"] = f"{comp.get('eggshell_mean'):.2f}" if comp.get("eggshell_mean") else "—"

    # --- A/B analysis (sampling-design framework) ---
    ab = comp.get("ab_analysis") or {}
    ab_fmt = None
    if ab.get("ab_present") or ab.get("per_variant"):
        ab_fmt = {
            "present":      ab.get("ab_present", False),
            "variant_col":  ab.get("variant_col"),
            "interpretation": (ab.get("ttest") or {}).get("interpretation"),
            "significant": (ab.get("ttest") or {}).get("significant", False),
            "p_value":     (ab.get("ttest") or {}).get("p_value"),
            "rows": [],
        }
        # Add an "Overall" row first, then each variant
        ov = ab.get("overall", {})
        if ov.get("n"):
            ab_fmt["rows"].append({
                "label":    "Overall",
                "n":        ov.get("n"),
                "mean":     f"${ov.get('mean'):,.0f}" if ov.get("mean") is not None else "—",
                "median":   f"${ov.get('median'):,.0f}" if ov.get("median") is not None else "—",
                "winsor":   f"${ov.get('winsor_mean_5_95'):,.0f}" if ov.get("winsor_mean_5_95") is not None else "—",
            })
        for v_id in ("A", "B"):
            stats = ab.get("per_variant", {}).get(v_id)
            if not stats: continue
            ab_fmt["rows"].append({
                "label":    f"{stats.get('raw_label') or v_id}",
                "n":        stats.get("n"),
                "mean":     f"${stats.get('mean'):,.0f}" if stats.get("mean") is not None else "—",
                "median":   f"${stats.get('median'):,.0f}" if stats.get("median") is not None else "—",
                "winsor":   f"${stats.get('winsor_mean_5_95'):,.0f}" if stats.get("winsor_mean_5_95") is not None else "—",
            })

    # --- Expected Case Value (per defendant) ---
    cv_blocks = []
    for block in comp.get("case_value", []) or []:
        formatted = {
            "defendant":      block.get("defendant_label", ""),
            "resp_mean":      block.get("responsibility", {}).get("mean"),
            "resp_median":    block.get("responsibility", {}).get("median"),
            "n_original":     block.get("n_with_original_responsibility", 0),
            "n_pivot":        block.get("n_with_pivot_responsibility", 0),
            "original":       _format_cv_scenarios(block.get("original")),
            "pivot":          _format_cv_scenarios(block.get("pivot")),
        }
        cv_blocks.append(formatted)

    # Bracket as ordered list (sorted descending by n)
    bracket = comp.get("bracket_selection", {})
    bracket_rows = []
    if bracket.get("counts"):
        for label, n in sorted(bracket["counts"].items(), key=lambda kv: -kv[1]):
            bracket_rows.append({
                "label": label, "n": n,
                "pct": bracket["pct"].get(label, 0),
            })

    # Demand reaction
    demand = comp.get("demand_reaction", {})
    demand_rows = []
    if demand.get("counts"):
        for label, n in sorted(demand["counts"].items(), key=lambda kv: -kv[1]):
            demand_rows.append({
                "label": label, "n": n,
                "pct": demand["pct"].get(label, 0),
            })

    # Deserving distribution
    deserving = comp.get("deserving_distribution", {})
    deserving_rows = []
    if deserving.get("counts"):
        order = ["Not at all (0%)", "Somewhat (25%)", "Moderate (50%)",
                 "A great deal (75%)", "Completely (100%)"]
        for label in order:
            if label in deserving["counts"]:
                deserving_rows.append({
                    "label": label, "n": deserving["counts"][label],
                    "pct": deserving["pct"].get(label, 0),
                })

    # Defense / plaintiff facts split
    facts_with_insights = []
    for f in bundle["facts"]:
        f_copy = dict(f)
        f_copy["insight_html"] = section_html(
            ai_sections.get(f"fact_insight_{f['num']}", ""),
            section_label=f"strategic insight for Fact {f['num']}",
        )
        facts_with_insights.append(f_copy)

    callouts = ai_sections.get("section_callouts") or {}
    if not isinstance(callouts, dict):
        callouts = {}

    # Discovered questions — attach AI summary HTML for free-text items;
    # add per-item display annotations for grouped scales so the template can
    # render (R) markers next to reverse-scored items.
    discovered_with_html = []
    for rec in bundle.get("discovered_questions", []):
        rec_copy = dict(rec)
        if rec.get("kind") == "free_text":
            key = f"discovery_summary_{rec['id']}"
            rec_copy["summary_html"] = section_html(
                ai_sections.get(key, ""),
                f"summary for {rec['label'][:40]}",
            )
        if rec.get("kind") == "grouped_scale":
            # Build a per-item display list: { item_label, mean, reversed, corr }
            per_item = (rec.get("stats") or {}).get("per_item") or []
            items_display = []
            for it in per_item:
                col = str(it.get("col", ""))
                # Trim long Alchemer column to the question text (before colon)
                short = col.split(":", 1)[0].strip()
                if len(short) > 100: short = short[:97] + "…"
                items_display.append({
                    "label":    short,
                    "raw_col":  col,
                    "mean":     it.get("mean"),
                    "reverse":  bool(it.get("reverse_scored")),
                    "manual":   bool(it.get("reverse_scored_manual")),
                    "corr":     it.get("corr_with_rest"),
                })
            rec_copy["items_display"] = items_display
        discovered_with_html.append(rec_copy)

    return {
        "case": bundle["case_metadata"],
        "n_respondents": bundle["n_respondents"],
        "date_generated": datetime.now().strftime("%B %d, %Y"),
        "demographics": bundle["demographics"],
        "facts": facts_with_insights,
        "verdict": bundle["verdict"],
        "defendant_attribution": bundle["defendant_attribution"],
        "plaintiff_by_demo": bundle["plaintiff_support_by_demo"],
        "compensation": comp_fmt,
        "bracket_rows": bracket_rows,
        "demand_rows": demand_rows,
        "deserving_rows": deserving_rows,
        # NEW: A/B sampling-design and per-defendant case-value
        "ab_analysis": ab_fmt,
        "case_value_blocks": cv_blocks,
        "predictors": bundle["predictors"],
        "open_ended": bundle["open_ended"],
        "charts": bundle["charts"],
        # AI sections (already rendered to HTML; sentinel-aware)
        "exec_summary_html": section_html(ai_sections.get("exec_summary", ""), "executive summary"),
        "juror_profiles_html": section_html(ai_sections.get("juror_profiles", ""), "juror profiles"),
        "voir_dire_html": section_html(ai_sections.get("voir_dire", ""), "voir dire strategy"),
        "narrative_themes_html": section_html(ai_sections.get("narrative_themes", ""), "open-ended narrative themes"),
        "evidence_gap_themes_html": section_html(ai_sections.get("evidence_gap_themes", ""), "evidence-gap themes"),
        "unanswered_themes_html": section_html(ai_sections.get("unanswered_themes", ""), "unanswered-question themes"),
        "section_callouts": callouts,
        "discovered_questions": discovered_with_html,
    }


def _format_cv_scenarios(cv: Optional[dict]) -> Optional[dict]:
    """Convert a compute_case_value() output into display-ready strings.
    'most_likely' is omitted from the display (per Yasmine's request) — the
    Most Likely outcome on this case-value model is often $0 (defense verdict)
    which buries the more useful scenarios."""
    if not cv:
        return None
    scenarios = cv.get("scenarios", {})
    out = {"scenarios": []}
    # Order: removed "most_likely"
    order = ["best_day_defense", "worst_day_defense", "plaintiff_verdict_isolated"]
    for k in order:
        s = scenarios.get(k)
        if not s: continue
        out["scenarios"].append({
            "key": k,
            "label": s.get("label", k),
            "description": s.get("description", ""),
            "value_fmt": f"${int(s.get('value', 0)):,}" if s.get("value") is not None else "—",
        })
    st = cv.get("stats", {})
    out["p_plaintiff_pct"] = (
        f"{st.get('p_plaintiff_verdict', 0) * 100:.1f}%"
        if st.get("p_plaintiff_verdict") is not None else "—"
    )
    out["p_defense_pct"] = (
        f"{st.get('p_defense_verdict', 0) * 100:.1f}%"
        if st.get("p_defense_verdict") is not None else "—"
    )
    out["winsor_low"]  = f"${int(st.get('winsor_low', 0)):,}" if st.get("winsor_low") else "—"
    out["winsor_high"] = f"${int(st.get('winsor_high', 0)):,}" if st.get("winsor_high") else "—"
    out["n_pool"] = st.get("n_respondents_in_pool", 0)
    return out


def render_pdf(context: dict, workspace_dir: str, output_path: str):
    """Render the report.html template + chart images to PDF via WeasyPrint."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template("report.html")
    html_str = template.render(**context)

    # base_url so weasyprint can find /charts/foo.png and the static/logo
    base_url = workspace_dir
    HTML(string=html_str, base_url=base_url).write_pdf(output_path)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_report(workspace_dir: str, regenerate_sections: Optional[list[str]] = None) -> dict:
    """
    Generate (or regenerate) AI prose sections + render the final PDF.

    Args:
        workspace_dir: where analysis_bundle.json lives.
        regenerate_sections:
            - None: regenerate ALL sections (fresh AI for every prose block)
            - []: regenerate NOTHING; just re-render the PDF from existing AI
                  output and existing overrides. Used by the "Save edits"
                  flow on Phase 3.
            - list of keys: regenerate just those sections

    Returns the dict of ai_sections (also written to ai_sections.json).
    """
    workspace_dir = Path(workspace_dir)
    bundle_path = workspace_dir / "analysis_bundle.json"
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    # Load existing AI sections if any (for surgical regeneration)
    ai_path = workspace_dir / "ai_sections.json"
    ai_sections = {}
    if ai_path.exists() and regenerate_sections is not None:
        with open(ai_path, "r", encoding="utf-8") as f:
            ai_sections = json.load(f)

    # All section keys
    fact_keys = [f"fact_insight_{f['num']}" for f in bundle["facts"]]
    # Discovery summaries — one per free-text discovered question
    discovery_keys = [
        f"discovery_summary_{rec['id']}"
        for rec in bundle.get("discovered_questions", [])
        if rec.get("kind") == "free_text" and rec.get("include", True)
    ]
    all_keys = [
        "exec_summary",
        *fact_keys,
        "juror_profiles",
        "voir_dire",
        "narrative_themes",
        "evidence_gap_themes",
        "unanswered_themes",
        *discovery_keys,
        "section_callouts",
    ]

    to_generate = regenerate_sections if regenerate_sections is not None else all_keys

    # Run generators. On failure, store a structured sentinel (FAILED_PREFIX,
    # defined at module level) that section_html() converts to a placeholder
    # callout in the PDF. Errors are also logged to stdout.
    for key in to_generate:
        try:
            if key == "exec_summary":
                ai_sections[key] = generate_executive_summary(bundle)
            elif key == "section_callouts":
                # Stored as JSON-serializable dict, not as text
                ai_sections[key] = generate_section_callouts(bundle)
            elif key == "juror_profiles":
                ai_sections[key] = generate_juror_profiles(bundle)
            elif key == "voir_dire":
                ai_sections[key] = generate_voir_dire(bundle)
            elif key == "narrative_themes":
                ai_sections[key] = generate_open_ended_themes(bundle, "narrative")
            elif key == "evidence_gap_themes":
                ai_sections[key] = generate_open_ended_themes(bundle, "evidence_gap")
            elif key == "unanswered_themes":
                ai_sections[key] = generate_open_ended_themes(bundle, "unanswered")
            elif key.startswith("fact_insight_"):
                num = int(key.split("_")[-1])
                fact = next((f for f in bundle["facts"] if f["num"] == num), None)
                if fact:
                    ai_sections[key] = generate_fact_insight(bundle, fact)
            elif key.startswith("discovery_summary_"):
                rec_id = key[len("discovery_summary_"):]
                rec = next((r for r in bundle.get("discovered_questions", []) if r["id"] == rec_id), None)
                if rec and rec.get("kind") == "free_text":
                    ai_sections[key] = generate_discovery_summary(rec["label"], rec.get("responses", []))
        except Exception as e:
            err_msg = str(e)
            short_err = err_msg[:200]
            print(f"[report] Section '{key}' failed after retries: {short_err}")
            # For section_callouts, store {} on failure instead of sentinel
            if key == "section_callouts":
                ai_sections[key] = {}
            else:
                ai_sections[key] = f"{FAILED_PREFIX}{short_err}"

    # Save AI output
    with open(ai_path, "w", encoding="utf-8") as f:
        json.dump(ai_sections, f, indent=2, ensure_ascii=False)

    # Apply user overrides (edits + skip flags) if a report_overrides.json
    # exists. Edits replace AI text; skipped sections become empty strings,
    # which the template's `{% if %}` guards convert to omission.
    overrides_path = workspace_dir / "report_overrides.json"
    sections_for_render = dict(ai_sections)
    if overrides_path.exists():
        try:
            ov = json.loads(overrides_path.read_text(encoding="utf-8"))
            edits = ov.get("edits", {}) or {}
            skip = set(ov.get("skip", []) or [])
            for key, text in edits.items():
                if key in sections_for_render or key in ("exec_summary", "juror_profiles",
                                                          "voir_dire", "narrative_themes",
                                                          "evidence_gap_themes", "unanswered_themes") \
                   or key.startswith("fact_insight_") or key.startswith("discovery_summary_"):
                    sections_for_render[key] = text
            for key in skip:
                sections_for_render[key] = ""
        except (OSError, json.JSONDecodeError) as e:
            print(f"[report] Failed to apply overrides: {e}")

    # Render PDF (uses overrides-applied sections, not the raw ai_sections)
    context = build_report_context(bundle, sections_for_render)
    pdf_path = workspace_dir / "report.pdf"
    render_pdf(context, str(workspace_dir), str(pdf_path))

    return ai_sections
