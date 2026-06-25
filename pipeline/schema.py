"""
Column-name mappings for Alchemer exports.

Alchemer assigns column headers from question text + variable name. Surveys
often diverge case-to-case, so this module resolves columns to canonical
*roles* using a two-layer system:

  1. Literal/regex patterns — used for stable Alchemer variable names that
     don't change across cases (TIPI items, Eggshell_plaintiff, etc.).
  2. Content-based scoring resolver — used for roles whose column header
     varies by case (open-ended prose questions, comp-reasoning, narrative,
     etc.). Each role has a scorer that examines column name AND column
     content. The best-scoring column wins, subject to a confidence floor.

The Phase 2 mapping-review UI is mandatory: every case writes a
`column_mapping.json` that overrides any of the auto-resolved choices.
That file is the single source of truth — once written, it is what the
rest of the pipeline reads.
"""
from __future__ import annotations
import re
import json
from pathlib import Path
import pandas as pd
from typing import Optional, Callable


# ---------------------------------------------------------------------------
# Stable canonical columns (literal/regex patterns)
# ---------------------------------------------------------------------------
# These are Alchemer variable names that, in practice, don't vary across the
# cases we run. If one of these ever drifts, fix the pattern here.

CANONICAL_COLUMNS = {
    # Meta / system
    "response_id":      [r"^Response ID$"],
    "status":           [r"^Status$"],
    "time_started":     [r"^Time Started$"],
    "date_submitted":   [r"^Date Submitted$"],
    "email":            [r"Enter your email address"],
    "rid":              [r"^RID$"],

    # Demographics
    "age":              [r"^Age$"],
    "sex":              [r"^Sex$"],
    "race":             [r"^Race$"],
    "education":        [r"^Education$"],
    "income":           [r"^Income$"],
    "political_party":  [r"^Political_Party$"],
    "political_view":   [r"^Political_View$"],
    "marital_status":   [r"^Marital_Status$"],
    "housing":          [r"^Housing$"],
    "employment":       [r"^Employment$"],
    "children":         [r"^Children$"],
    "military":         [r"^Military$"],

    # Pre-info bias slider
    "pre_info_bias":    [r"^pre_info_bias$"],

    # Outcome variables (mostly stable Alchemer names)
    "favor_support":    [r"^Favor_Support$"],
    "liable":           [r"^liable$"],
    "responsibility_plaintiff": [r"^Plaintiff[^:]*:Responsibility$"],
    "responsibility_defendant": [r"^Defendant/s:Responsibility$", r"^Defendant\(s\):Responsibility$"],
    "total_compensation": [r"^Total_Compensation$"],
    "deserving":        [r"^Deserving_Total_Compensation$"],
    "comp_bracket":     [r"reasonable amount of compensation"],
    "demand_reaction":  [r"^Damages$"],

    # Defendant attribution (conditional on Favor_Support)
    "defendant_supported": [r"Which defendant are you MOST in support of"],
    "defendant_most_liable": [r"Which defendant do you think is MOST liable"],

    # Litigation/psychographic attitudes (predictors — stable variable names)
    "eggshell":         [r"^Eggshell_plaintiff$"],
    "would_sue":        [r"^Would_You_Sue$"],
    "ever_sued":        [r"^Ever_Been_Sued$"],
    "family_owned":     [r"^FamilyOwned$"],
    "businesses_getaway": [r"^institutional_skepticism"],
    "damage_limits_ps": [r"^Damage_Limits_Pain_Suffering$"],
    "damage_limits_punitive": [r"^Damage_Limits_Punitive$"],
    "lawsuits_general": [r"^Lawsuits_General$"],
    "lawsuits_frivolous": [r"^Lawsuits_Frivolous$"],
    "comfort_damages":  [r"comfortable.*awarding a substantial"],
    "qanon":            [r"^qanon$"],
    "validity_check":   [r"^Validity_Check2$"],

    # Prior jury
    "prior_jury":       [r"^Prior_Jury$"],
    "jury_foreperson":  [r"^Jury_Foreperson$"],
}

# ---------------------------------------------------------------------------
# Variable-text roles (resolved by scoring, not literal patterns)
# ---------------------------------------------------------------------------
# Each entry: role_id -> dict with:
#   "label":      human-readable label for the mapping UI
#   "kind":       "free_text" | "categorical" | "numeric"  (expected content shape)
#   "multi":      True if multiple columns can fill this role (e.g. reasoning_extra)
#   "scorer":     (col_name, series) -> float in [0,1]   higher = better fit

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _avg_len(series: pd.Series) -> float:
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    return float(s.str.len().mean()) if len(s) else 0.0


def _is_free_text(series: pd.Series) -> bool:
    """Heuristic: column reads like prose, not a code or short label."""
    s = series.dropna().astype(str).str.strip()
    s = s[s != ""]
    if len(s) == 0:
        return False
    return _avg_len(series) >= 30 and s.nunique() >= max(10, int(0.5 * len(s)))


def _keyword_score(col: str, must: list[str], nice: list[str] = ()) -> float:
    """
    Score a column header against keyword sets.
    Returns ~1.0 if all MUST terms appear and most NICE terms also appear.
    Returns 0.0 if any MUST term is missing.
    """
    name = _normalize(col)
    for term in must:
        if term not in name:
            return 0.0
    score = 0.6
    if nice:
        hits = sum(1 for term in nice if term in name)
        score += 0.4 * (hits / len(nice))
    return min(score, 1.0)


def _looks_like_money(series: pd.Series) -> bool:
    """Heuristic: does this column contain dollar-value-shaped numbers?
    Used to disambiguate a damages-amount column from a 1-5 Likert column
    whose header happens to include the word 'compensation'."""
    # Clean and try numeric
    s = series.dropna().astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
    s = s[s != ""]
    if len(s) < 10:
        return False
    n = pd.to_numeric(s, errors="coerce").dropna()
    if len(n) < 10:
        return False
    # All Likert-shaped? (1-5 or 1-7 integers, narrow range)
    if n.max() <= 7 and n.min() >= 1 and (n == n.astype(int)).all():
        return False
    # Money typically has a wide range and reaches into the thousands or more
    if n.max() < 100:
        return False
    return True


def _make_scorer(must_any: list[list[str]], nice: list[str] = (),
                 require_free_text: bool = True,
                 require_money_content: bool = False,
                 penalize_terms: list[str] = ()) -> Callable:
    """
    Build a scorer that fires if ANY of the `must_any` keyword sets all match,
    optionally boosted by `nice` terms, with optional `penalize_terms` that
    reduce the score (used to disambiguate similar prompts).

    If `require_free_text`, the column must also look like prose.
    If `require_money_content`, the column's values must look like dollar
      amounts (excludes 1-5 Likert scales whose headers happen to mention
      'compensation' or 'damages').
    """
    def score(col: str, series: pd.Series) -> float:
        best = 0.0
        for must in must_any:
            s = _keyword_score(col, must, nice)
            if s > best:
                best = s
        if best == 0.0:
            return 0.0
        if require_free_text and not _is_free_text(series):
            return 0.0
        if require_money_content and not _looks_like_money(series):
            return 0.0
        name = _normalize(col)
        for bad in penalize_terms:
            if bad in name:
                best -= 0.25
        return max(0.0, min(best, 1.0))
    return score


# --- Role definitions ------------------------------------------------------
# These are case-variable roles. The scorer decides which (if any) column
# fills each role. The mapping-review UI lets the user override.

ROLE_DEFINITIONS: dict[str, dict] = {
    "narrative": {
        "label": "Narrative — juror retells the case in their own words",
        "kind": "free_text",
        "multi": False,
        "scorer": _make_scorer(
            must_any=[
                ["own words", "happened"],
                ["in your own words"],
                ["tell the story"],
                ["describe what happened"],
            ],
            nice=["case", "story", "events"],
        ),
    },
    "evidence_gap": {
        "label": "Evidence gap — what additional evidence would have helped",
        "kind": "free_text",
        "multi": False,
        "scorer": _make_scorer(
            must_any=[
                ["additional evidence"],
                ["more evidence"],
                ["what evidence", "missing"],
                ["what evidence", "needed"],
                ["what other information", "wanted"],
            ],
        ),
    },
    "unanswered": {
        "label": "Unanswered questions — what jurors still want to know",
        "kind": "free_text",
        "multi": False,
        "scorer": _make_scorer(
            must_any=[
                ["unanswered question"],
                ["unanswered"],
                ["still want", "know"],
                ["liked", "concrete information"],
                ["wished", "known"],
            ],
        ),
    },
    "comp_reasoning": {
        "label": "Compensation reasoning — why jurors chose their damages amount",
        "kind": "free_text",
        "multi": False,
        "scorer": _make_scorer(
            must_any=[
                ["reasoning", "damage award"],
                ["reasoning", "total damage"],
                ["reasoning", "damages"],
                ["explain", "damage award"],
                ["explain", "total"],
                ["why", "amount", "award"],
            ],
            nice=["compensation", "damages", "amount"],
        ),
    },
    "decision_reasoning": {
        "label": "Decision reasoning — why jurors picked the side they did",
        "kind": "free_text",
        "multi": False,
        "scorer": _make_scorer(
            must_any=[
                ["reasoning", "decision", "support"],
                ["explain", "decision", "support"],
                ["specific fact", "influenced", "decision"],
                ["evidence", "most influenced", "decision"],
                ["why", "support", "party"],
            ],
            nice=["fact", "evidence", "influenced"],
        ),
    },
    "reasoning_extra": {
        # Multi-fill: any *other* prose "explain your reasoning" prompt
        # (e.g. after fact reveals, pivot points) flows in here.
        "label": "Additional reasoning prompts — other 'explain your thinking' questions",
        "kind": "free_text",
        "multi": True,
        "scorer": _make_scorer(
            must_any=[
                ["reasoning", "additional"],
                ["reasoning", "after"],
                ["reasoning", "facts"],
                ["explain", "changed"],
                ["why", "changed", "mind"],
                ["explain", "reasoning"],
            ],
            penalize_terms=["damage award", "total damage"],   # those belong to comp_reasoning
        ),
    },
    "sponsor_belief": {
        "label": "Sponsor belief — who jurors think funded the study",
        "kind": "categorical",
        "multi": False,
        "scorer": _make_scorer(
            must_any=[["sponsored", "study"], ["who", "sponsored"], ["who paid"]],
            require_free_text=False,
        ),
    },
    "ab_variant": {
        "label": "A/B variant column — which damages-question format each respondent saw",
        "kind": "categorical",
        "multi": False,
        "scorer": _make_scorer(
            must_any=[
                ["a/b", "split"],
                ["split", "test"],
                ["variant"],
                ["branch"],
            ],
            require_free_text=False,
        ),
    },
    "total_compensation_combined": {
        "label": "Combined-total damages — the single 'all damages combined' amount each juror picks",
        "kind": "numeric",
        "multi": False,
        "scorer": _make_scorer(
            must_any=[
                ["considering all", "damages"],
                ["total amount you would award"],
                ["combined", "amount", "award"],
            ],
            require_free_text=False,
            require_money_content=True,
            penalize_terms=["deserving", "rate the level"],
        ),
    },
    "total_compensation_slider": {
        "label": "Damages — SLIDER variant column (filled by Branch A respondents only)",
        "kind": "numeric",
        "multi": False,
        "scorer": _make_scorer(
            must_any=[
                ["total_compensation"],
                ["please use the sliding scale"],
                ["sliding scale"],
            ],
            require_free_text=False,
            require_money_content=True,
            penalize_terms=["deserving", "deserving_total_compensation"],
        ),
    },
    "total_compensation_open": {
        "label": "Damages — OPEN-FORM variant column (filled by Branch B respondents only)",
        "kind": "numeric",
        "multi": False,
        "scorer": _make_scorer(
            must_any=[
                ["please indicate", "amount", "below"],
                ["write the full dollar amount"],
                ["complete number using all digits"],
            ],
            require_free_text=False,
            require_money_content=True,
            penalize_terms=["deserving"],
        ),
    },
}


# ---------------------------------------------------------------------------
# Scale groups (for straightlining detection + summary stats)
# ---------------------------------------------------------------------------

SCALE_GROUPS = {
    "tipi": r":TIPI$",
    "just_world": r":Global_Belief_in_a_Just_World$",
    "litigation_attitudes": r":Litigation_attitudes$|^Litigation_attitudes(\.\d+)?$",
    "single_item_personality": r":Single_Item$",
}


# ---------------------------------------------------------------------------
# Case fact columns
# ---------------------------------------------------------------------------
# Anchor on the `:Fact <N>` SUFFIX, not the prefix. Surveys vary the prefix
# ("Importance_Outcome_F1" vs "How important is this information..."), but
# the trailing ":Fact N" is consistent because Alchemer derives it from the
# row label.

FACT_SUFFIX = re.compile(r":Fact\s*(\d+)\s*$", re.IGNORECASE)

# Keyword sets identifying which of the two paired questions per fact this is.
FACT_IMPORTANCE_KEYS = ["importance_outcome", "how important"]
FACT_INFLUENCE_KEYS  = ["influence_verdict", "how would this information influence",
                       "influence your verdict"]

FACT_REASONING_PATTERN = re.compile(r"^Reasoning_F(\d+)$", re.IGNORECASE)


def _fact_kind(col: str) -> Optional[str]:
    """Return 'importance', 'influence', or None for a `*:Fact N` column."""
    name = _normalize(col)
    for k in FACT_IMPORTANCE_KEYS:
        if k in name:
            return "importance"
    for k in FACT_INFLUENCE_KEYS:
        if k in name:
            return "influence"
    return None


# ---------------------------------------------------------------------------
# Likert encodings (unchanged from prior version)
# ---------------------------------------------------------------------------

SIX_POINT_FACT_INFLUENCE = {
    "Strong positive influence in favor of the Defendant/s": 1,
    "Positive influence in favor of the Defendant/s": 2,
    "Somewhat positive influence in favor of the Defendant/s": 3,
    "Somewhat positive influence in favor of the Plaintiff": 4,
    "Positive influence in favor of the Plaintiff": 5,
    "Strong positive influence in favor of the Plaintiff": 6,
}

SIX_POINT_FACT_IMPORTANCE = {
    "Not at all important": 1,
    "Unimportant": 2,
    "Somewhat unimportant": 3,
    "Somewhat important": 4,
    "Important": 5,
    "Very important": 6,
}

SIX_POINT_FAVOR_SUPPORT = {
    "Strongly in favor of the Defendant/s": 1,
    "In favor of the Defendant/s": 2,
    "Somewhat in favor of the Defendant/s": 3,
    "Somewhat in favor of the Plaintiff": 4,
    "In favor of the Plaintiff": 5,
    "Strongly in favor of the Plaintiff": 6,
}

FIVE_POINT_DESERVING = {
    "Not at all (0%)": 1, "Somewhat (25%)": 2, "Moderate (50%)": 3,
    "A great deal (75%)": 4, "Completely (100%)": 5,
}

FIVE_POINT_LIKERT = {
    "Strongly Disagree": 1, "Disagree": 2, "Neutral": 3,
    "Agree": 4, "Strongly Agree": 5,
}

SEVEN_POINT_LIKERT = {
    "Very Strongly Disagree": 1, "Very strongly disagree": 1,
    "Strongly Disagree": 2, "Strongly disagree": 2,
    "Disagree": 3, "Slightly Disagree": 3, "Slightly disagree": 3,
    "Neutral": 4,
    "Slightly Agree": 5, "Slightly agree": 5,
    "Agree": 5,
    "Strongly Agree": 6, "Strongly agree": 6,
    "Very Strongly Agree": 7, "Very strongly agree": 7,
}


# ---------------------------------------------------------------------------
# Mapping override: column_mapping.json
# ---------------------------------------------------------------------------
# Structure on disk:
# {
#   "roles": {
#     "narrative":          {"columns": ["..."], "confidence": 0.92, "manual": true|false},
#     "evidence_gap":       {"columns": [],      "confidence": 0.0,  "manual": true},   # skipped
#     "reasoning_extra":    {"columns": ["A","B","C"], "confidence": 0.88, "manual": false},
#     ...
#   },
#   "auto_resolved": true|false   # false once the user clicks "Confirm" in the UI
# }

def load_column_mapping(case_dir: str | Path) -> Optional[dict]:
    p = Path(case_dir) / "column_mapping.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def save_column_mapping(case_dir: str | Path, mapping: dict) -> None:
    p = Path(case_dir) / "column_mapping.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def find_column(df: pd.DataFrame, canonical_or_role: str,
                mapping: Optional[dict] = None) -> Optional[str]:
    """
    Return the actual df column matching the canonical name OR role, or None.

    Resolution priority:
      1. If `mapping` is provided and contains this name as a role, use it.
      2. Otherwise, fall back to the literal CANONICAL_COLUMNS regex table.
      3. (For roles), call resolve_role to score-match against df columns.

    Empty / whitespace / unknown column entries from a mapping payload are
    treated as "no match" — they can occur when the Phase 2 UI saves a role
    row that was added but never given a column.
    """
    # 1. Override mapping wins
    if mapping and "roles" in mapping:
        role_entry = mapping["roles"].get(canonical_or_role)
        if role_entry is not None:
            cols = [c for c in role_entry.get("columns", [])
                    if c and str(c).strip() and c in df.columns]
            return cols[0] if cols else None

    # 2. Literal canonical
    if canonical_or_role in CANONICAL_COLUMNS:
        patterns = CANONICAL_COLUMNS[canonical_or_role]
        for col in df.columns:
            for pat in patterns:
                if re.search(pat, col, re.IGNORECASE):
                    return col
        return None

    # 3. Scoring role
    if canonical_or_role in ROLE_DEFINITIONS:
        matches = resolve_role(df, canonical_or_role)
        return matches[0]["column"] if matches else None

    return None


def find_columns_multi(df: pd.DataFrame, role: str,
                       mapping: Optional[dict] = None) -> list[str]:
    """Like find_column but returns ALL columns assigned to a multi-fill role.
    Empty / whitespace / unknown column entries are filtered out."""
    if mapping and "roles" in mapping:
        role_entry = mapping["roles"].get(role)
        if role_entry is not None:
            return [c for c in role_entry.get("columns", [])
                    if c and str(c).strip() and c in df.columns]

    if role in ROLE_DEFINITIONS:
        matches = resolve_role(df, role)
        if ROLE_DEFINITIONS[role].get("multi"):
            return [m["column"] for m in matches]
        return [matches[0]["column"]] if matches else []
    return []


def resolve_role(df: pd.DataFrame, role: str,
                 threshold: float = 0.55) -> list[dict]:
    """
    Score every df column against the role's scorer; return matches above
    threshold, sorted high → low. Each match: {"column", "score"}.
    """
    if role not in ROLE_DEFINITIONS:
        return []
    scorer = ROLE_DEFINITIONS[role]["scorer"]
    multi = ROLE_DEFINITIONS[role].get("multi", False)
    scored = []
    for col in df.columns:
        try:
            s = scorer(col, df[col])
        except Exception:
            s = 0.0
        if s >= threshold:
            scored.append({"column": col, "score": round(s, 2)})
    scored.sort(key=lambda x: -x["score"])
    if not multi:
        if scored:
            top = scored[0]["score"]
            scored = [m for m in scored if m["score"] >= top - 0.001 or len(scored) == 1]
            scored = scored[:3]
    return scored


def auto_resolve_all(df: pd.DataFrame) -> dict:
    """
    Run all role scorers against df. Returns a `column_mapping.json`-shaped
    dict with `auto_resolved=True`. Used to pre-fill the mapping-review UI.

    A column claimed by a higher-priority (single-fill) role can't also be
    claimed by a lower-priority (multi-fill) role like reasoning_extra.
    """
    roles_out = {}
    claimed = set()
    # Single-fill first so multi-fill doesn't steal a column that's a better
    # unique fit somewhere else.
    role_ids = sorted(ROLE_DEFINITIONS.keys(),
                      key=lambda r: ROLE_DEFINITIONS[r].get("multi", False))
    for role_id in role_ids:
        matches = resolve_role(df, role_id)
        matches = [m for m in matches if m["column"] not in claimed]
        if not matches:
            roles_out[role_id] = {"columns": [], "confidence": 0.0, "manual": False,
                                  "candidates": []}
            continue
        multi = ROLE_DEFINITIONS[role_id].get("multi", False)
        if multi:
            cols = [m["column"] for m in matches]
        else:
            cols = [matches[0]["column"]]
        for c in cols:
            claimed.add(c)
        roles_out[role_id] = {
            "columns": cols,
            "confidence": matches[0]["score"],
            "manual": False,
            "candidates": [m for m in matches[:5]],
        }
    return {"roles": roles_out, "auto_resolved": True}


# ---------------------------------------------------------------------------
# Helpers — unchanged callers (find_scale_columns, detect_facts, etc.)
# ---------------------------------------------------------------------------

def find_scale_columns(df: pd.DataFrame, scale_name: str) -> list[str]:
    if scale_name not in SCALE_GROUPS:
        return []
    pat = SCALE_GROUPS[scale_name]
    return [c for c in df.columns if re.search(pat, c)]


def detect_facts(df: pd.DataFrame) -> list[int]:
    """Return sorted list of fact numbers present in the dataset.

    Uses the `:Fact <N>` suffix so prefix variations don't matter.
    Only counts a fact if it has at least one column matching that suffix
    AND that column looks like a fact question (importance OR influence).
    """
    nums = set()
    for col in df.columns:
        m = FACT_SUFFIX.search(col)
        if m and _fact_kind(col):
            nums.add(int(m.group(1)))
    return sorted(nums)


def _fact_col_of_kind(df: pd.DataFrame, fact_num: int, kind: str) -> Optional[str]:
    """Find the column for (fact_num, kind) where kind in {'importance','influence'}."""
    for col in df.columns:
        m = FACT_SUFFIX.search(col)
        if not m or int(m.group(1)) != fact_num:
            continue
        if _fact_kind(col) == kind:
            return col
    return None


def fact_importance_col(df: pd.DataFrame, fact_num: int) -> Optional[str]:
    return _fact_col_of_kind(df, fact_num, "importance")


def fact_influence_col(df: pd.DataFrame, fact_num: int) -> Optional[str]:
    return _fact_col_of_kind(df, fact_num, "influence")


def fact_reasoning_col(df: pd.DataFrame, fact_num: int) -> Optional[str]:
    for col in df.columns:
        m = FACT_REASONING_PATTERN.match(col)
        if m and int(m.group(1)) == fact_num:
            return col
    return None


def coerce_likert(series: pd.Series, mapping: dict) -> pd.Series:
    """Convert a text Likert series to numeric using the mapping.
    Numeric values pass through unchanged."""
    def _convert(v):
        if pd.isna(v):
            return None
        if isinstance(v, (int, float)):
            return v
        s = str(v).strip()
        if s in mapping:
            return mapping[s]
        for k, num in mapping.items():
            if k.lower().strip() == s.lower():
                return num
        try:
            return float(s)
        except ValueError:
            return None
    return series.apply(_convert)
