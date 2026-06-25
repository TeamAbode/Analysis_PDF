"""
Auto-discovery of non-canonical survey questions.

After Phase 2's main analysis runs, this scans the cleaned dataframe for any
column that ISN'T in the canonical schema, ISN'T an Alchemer system column,
and ISN'T already covered by a known scale or fact item. Each surviving
column is classified and summarized:

  - numeric_likert    → continuous/ordinal; produce a bar chart of distribution
  - categorical       → ≤ 15 unique values; produce a frequency table
  - binary            → 0/1 or Yes/No; produce a small two-bar chart
  - free_text         → many unique values, longer responses; collect verbatims
  - grouped_scale     → items sharing a numeric-suffix prefix (Authoritarian1-4)

Each discovered item gets:
  {
    "id": "stable-key",
    "label": "question text or column header",
    "kind": "numeric_likert" | "categorical" | "binary" | "free_text" | "grouped_scale",
    "n": int,
    "stats": {...kind-specific...},
    "chart_path": "auto/discoveryN.png" (when applicable),
    "responses": [verbatim, ...] (only for free_text),
    "include_default": bool,
    "predictor_default": False,
  }

The Phase 2 UI receives this list with checkboxes ("Include?" and "Predictor?")
and the user can toggle each before re-running.
"""
from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import schema


# Brand colors — kept in sync with analyze.py
JA_PRIMARY = "#0000B7"
JA_SECONDARY = "#0073DF"
JA_MID = "#4D4DCC"
JA_TEXT = "#231F20"


# ---------------------------------------------------------------------------
# What counts as "system" or "already known"
# ---------------------------------------------------------------------------

SYSTEM_COL_PATTERNS = [
    # Alchemer system / response metadata
    r"^Response ID$", r"^Status$", r"^Time Started$", r"^Time Spent$", r"^Date Submitted$",
    r"^Contact ID$", r"^SessionID$", r"^Session ID$",
    r"^IP Address$", r"^Country$", r"^Region$", r"^State/Region$", r"^City$", r"^Postal$",
    r"^Latitude$", r"^Longitude$", r"^User Agent$", r"^Browser$", r"^Device$",
    r"^Referer$", r"^Tags$", r"^Legacy Comments$", r"^Comments$",
    r"^Language$", r"^Survey Done$",
    # Common screener / system data — sample composition vars, not survey content
    r"^Group$",                    # experimental group assignment
    r"^PID$",                      # respondent platform ID
    r"^RID$",                      # respondent ID
    # Validity / attention checks (irrelevant in report since survey filter already handled)
    r"^Validity_Check\d?$",
    # Trailing leftover responsibility columns from other case versions
    r":Responsibility$",
    # Screeners that confirm jury eligibility (everyone in clean dataset passed by definition)
    r"^18OrOlder$",
    r"^consent$", r"^Consent$",
    r"^County$",                   # numeric "1=lives in county" flag (different from county_state)
    r"jury summons",               # "Do you have a current jury summons"
    # Free-text demographics that shouldn't be auto-discovered as questions
    r"^What is your occupation\??\s*$",
    r"Other not stated above:\s*:Sex",
    r"Other not stated above::?Sex",
    # Spelled-out money validation columns — these accompany the digit-entry
    # damages questions and are noise once the digit column is captured.
    r"spelled out in full words",
    r"this time spelled out",
    # The big damages-options preamble columns (the one with all the
    # "Past Medical Expenses: $164,642.86..." text) — these are display-only
    # blocks Alchemer attaches to the rating question, not standalone data.
    r"Based on the information provided, please rate the level of compensation",
]

# Stable, friendly label for binary-numeric columns whose header reads like a
# question. Used when we don't have a richer label to show.
def _is_system(col: str) -> bool:
    for p in SYSTEM_COL_PATTERNS:
        if re.search(p, col, re.IGNORECASE):
            return True
    return False


def _known_canonical_cols(df: pd.DataFrame, mapping: Optional[dict] = None) -> set:
    """Union of all columns the main pipeline already analyzes.

    If a `column_mapping.json` is provided, columns assigned to any of its
    roles (including multi-fill roles like reasoning_extra) are added to
    the 'known' set so auto-discovery doesn't double-list them.
    """
    known = set()
    # Canonical (literal-pattern) columns
    for canon in schema.CANONICAL_COLUMNS:
        c = schema.find_column(df, canon)
        if c:
            known.add(c)
    # Likert scale items (TIPI, Just_World, Litigation_attitudes, Single_Item)
    for scale_name in schema.SCALE_GROUPS:
        for c in schema.find_scale_columns(df, scale_name):
            known.add(c)
    # Fact items — importance, influence, reasoning, AND "how convincing"
    # ("How convincing" rolls up to fact stats; not a standalone discovery item)
    for n in schema.detect_facts(df):
        for c in [
            schema.fact_importance_col(df, n),
            schema.fact_influence_col(df, n),
            schema.fact_reasoning_col(df, n),
        ]:
            if c:
                known.add(c)
    # Capture How-convincing columns explicitly
    for col in df.columns:
        m = schema.FACT_SUFFIX.search(col)
        if m and "how convincing" in str(col).lower():
            known.add(col)
    # Columns claimed by the role mapping (open-ends, comp_reasoning, etc.)
    if mapping and "roles" in mapping:
        for role_entry in mapping["roles"].values():
            for c in role_entry.get("columns", []):
                known.add(c)
    return known


# ---------------------------------------------------------------------------
# Per-column classification
# ---------------------------------------------------------------------------

def _try_numeric(series: pd.Series) -> tuple[pd.Series, float]:
    """Coerce to numeric (stripping $/%/commas). Return (numeric_series, parse_rate)."""
    s = series.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "N/A": pd.NA, "n/a": pd.NA})
    s = s.str.replace("%", "", regex=False).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
    numeric = pd.to_numeric(s, errors="coerce")
    non_null_orig = series.notna().sum()
    if non_null_orig == 0:
        return numeric, 0.0
    parse_rate = numeric.notna().sum() / non_null_orig
    return numeric, parse_rate


def _clean_label(s: str) -> str:
    """Strip stray latin-1 encoding artifacts and trailing whitespace."""
    if s is None:
        return ""
    # Common mojibake patterns when latin-1 bytes get UTF-8 decoded
    s = s.replace("\xa0", " ")           # non-breaking space
    s = s.replace("Â", "")                # stray latin-1 NBSP encoding
    s = s.replace("\u00a0", " ")
    # Collapse repeated whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _classify_column(col: str, series: pd.Series) -> Optional[dict]:
    """Return a discovery record for the column, or None to skip it."""
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return None   # always-empty column — skip

    n_unique = non_null.astype(str).str.strip().replace({"": pd.NA}).dropna().nunique()
    if n_unique == 0:
        return None

    # Skip uniform-answer columns (everyone picked the same value — no signal).
    # Free-text columns with long unique strings often have high n_unique
    # naturally; uniform check only meaningful when n_unique is small.
    if n_unique == 1:
        return None

    # Try numeric coercion
    numeric, parse_rate = _try_numeric(series)
    is_numeric = parse_rate >= 0.80

    # Average response length (for free-text detection)
    str_vals = non_null.astype(str).str.strip()
    avg_len = float(str_vals.str.len().mean()) if len(str_vals) else 0
    max_len = int(str_vals.str.len().max()) if len(str_vals) else 0

    # ---- Binary ----
    # Numeric 0/1 OR text Yes/No
    text_lower = str_vals.str.lower()
    is_yes_no = text_lower.isin(["yes", "no", "y", "n"]).all() and n_unique == 2
    is_zero_one = is_numeric and set(numeric.dropna().unique()).issubset({0, 1, 0.0, 1.0}) and n_unique == 2
    if is_yes_no or is_zero_one:
        if is_yes_no:
            yes_count = int((text_lower == "yes").sum() + (text_lower == "y").sum())
            no_count = int((text_lower == "no").sum() + (text_lower == "n").sum())
        else:
            yes_count = int((numeric == 1).sum())
            no_count = int((numeric == 0).sum())
        return {
            "label": _clean_label(col), "kind": "binary",
            "n": n,
            "stats": {
                "yes_count": yes_count, "no_count": no_count,
                "pct_yes": round(100 * yes_count / max(yes_count + no_count, 1), 1),
            },
            "chart_path": None,   # filled in by caller
        }

    # ---- Numeric (Likert / scale / continuous) ----
    if is_numeric and n_unique > 2:
        vals = numeric.dropna()
        return {
            "label": _clean_label(col), "kind": "numeric_likert",
            "n": n,
            "stats": {
                "mean": round(float(vals.mean()), 2),
                "median": round(float(vals.median()), 2),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "distribution": vals.value_counts().sort_index().to_dict(),
            },
            "chart_path": None,
        }

    # ---- Categorical ----
    # Short string values, fewer than 16 unique levels
    if n_unique <= 15 and avg_len < 50:
        counts = str_vals.value_counts()
        return {
            "label": _clean_label(col), "kind": "categorical",
            "n": n,
            "stats": {
                "counts": counts.to_dict(),
                "pct": {k: round(100 * v / counts.sum(), 1) for k, v in counts.to_dict().items()},
            },
            "chart_path": None,
        }

    # ---- Free text ----
    # Many unique values OR long average response length
    if n_unique > 15 or avg_len >= 50:
        # Keep verbatim responses (longer than 10 chars) for AI summarization
        responses = [v for v in str_vals.tolist() if len(v) > 10]
        return {
            "label": _clean_label(col), "kind": "free_text",
            "n": n,
            "stats": {"n_responses": len(responses), "avg_length": int(avg_len), "max_length": max_len},
            "chart_path": None,
            "responses": responses[:80],   # cap for prompt size
        }

    return None   # uncategorizable


# ---------------------------------------------------------------------------
# Grouped-scale detection
# ---------------------------------------------------------------------------
# Three detection passes, chained. Earlier passes consume columns so later
# passes don't re-claim them.
#
# Pass A: number-suffix groups       (Authoritarian1..Authoritarian4)  threshold ≥ 3
# Pass B: shared `:alias` suffix     (...:plaintiff_perception)        threshold ≥ 2
# Pass C: NAMED_SCALES registry      (case-specific named groupings)   threshold ≥ 2
#
# Items are coerced to numeric (handling Likert text via schema mappings) before
# being averaged. Items whose item-total correlation is strongly negative are
# flagged as reverse-scored and flipped (1↔max) before averaging.

GROUP_NUMERIC_SUFFIX = re.compile(r"^(.+?)(\d+)$")
COLON_ALIAS_SUFFIX = re.compile(r":([A-Za-z][A-Za-z0-9_]+)\s*$")

# NAMED_SCALES: hand-defined groupings for items that conceptually belong
# together but share neither a number suffix nor a colon alias. Match by
# substring against the column header (lowercased). All listed phrases must
# appear in the column (in any order) for it to count as a member.
NAMED_SCALES = {
    "lawsuit_attitudes": {
        "label": "Lawsuit attitudes (tort reform)",
        "patterns": [
            ["claiming injuries", "taking advantage"],
            ["large damage awards", "hurt the economy"],
            ["lawsuit abuse"],
            ["large jury damage awards", "excessive"],
        ],
    },
}

REVERSE_SCORE_CORR_THRESHOLD = -0.10   # item-total correlation below this -> flag

# Maximum value used for flipping (Likert max). Detected per scale from the
# observed value range; falls back to 5 if undetermined.
def _scale_max(values: list[pd.Series]) -> int:
    flat = pd.concat(values, axis=0)
    flat = pd.to_numeric(flat, errors="coerce").dropna()
    if flat.empty:
        return 5
    m = int(flat.max())
    return m if m in (5, 7) else max(5, m)


def _coerce_likert_to_numeric(series: pd.Series) -> tuple[pd.Series, float]:
    """
    Coerce a Likert series to numeric, trying:
      1. direct numeric parse
      2. schema.FIVE_POINT_LIKERT  (text labels)
      3. schema.SEVEN_POINT_LIKERT (text labels)
    Returns (numeric_series, parse_rate).
    """
    # Try plain numeric first
    numeric, rate = _try_numeric(series)
    if rate >= 0.75:
        return numeric, rate

    # Try text Likert mappings
    from . import schema   # local import to avoid circulars
    for mapping in (schema.FIVE_POINT_LIKERT, schema.SEVEN_POINT_LIKERT):
        coerced = schema.coerce_likert(series, mapping)
        coerced = pd.to_numeric(coerced, errors="coerce")
        non_null = series.notna().sum()
        if non_null == 0:
            continue
        new_rate = coerced.notna().sum() / non_null
        if new_rate >= 0.75:
            return coerced, new_rate
    return numeric, rate   # fall back to plain numeric, even if low


def _auto_detect_reverse_scored(
    items_numeric: list[pd.Series], item_cols: list[str], scale_max: int,
) -> tuple[pd.Series, list[dict]]:
    """
    Compute item-total correlations. Any item whose correlation with the
    sum of the OTHERS is negative beyond REVERSE_SCORE_CORR_THRESHOLD is
    flagged as reverse-scored, and its values are flipped via
    `flipped = (scale_max + 1) - original` before composition.

    Returns (composite_series, per_item_meta).
        per_item_meta: list of {"col", "mean", "corr_with_rest", "reverse_scored"}
    """
    n_items = len(items_numeric)
    if n_items < 2:
        composite = items_numeric[0] if items_numeric else pd.Series(dtype=float)
        meta = [{"col": item_cols[0], "mean": round(float(composite.mean()), 2)
                 if composite.notna().any() else None,
                 "corr_with_rest": None, "reverse_scored": False}] if items_numeric else []
        return composite, meta

    # Stack into a frame for correlation work
    frame = pd.concat(items_numeric, axis=1)
    frame.columns = item_cols
    per_item = []
    flipped_items = []
    for col, ser in zip(item_cols, items_numeric):
        rest = frame.drop(columns=[col]).mean(axis=1, skipna=True)
        # Pearson on the overlapping non-null rows
        joined = pd.concat([ser, rest], axis=1).dropna()
        if len(joined) >= 10:
            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        else:
            corr = None
        reverse = (corr is not None and corr < REVERSE_SCORE_CORR_THRESHOLD)
        if reverse:
            flipped = (scale_max + 1) - ser
            flipped_items.append(flipped)
        else:
            flipped_items.append(ser)
        per_item.append({
            "col": col,
            "mean": round(float(ser.mean()), 2) if ser.notna().any() else None,
            "corr_with_rest": round(corr, 3) if corr is not None else None,
            "reverse_scored": bool(reverse),
        })

    composite = pd.concat(flipped_items, axis=1).mean(axis=1, skipna=True)
    return composite, per_item


def _build_scale_record(
    df: pd.DataFrame,
    prefix: str,
    label: str,
    cols: list[str],
    coerce_text_likert: bool = True,
) -> Optional[dict]:
    """
    Build a grouped_scale record from a list of column names.
    Returns None if items can't be coerced to numeric reliably.
    """
    items_numeric = []
    parse_rates = []
    for c in cols:
        if coerce_text_likert:
            n_series, rate = _coerce_likert_to_numeric(df[c])
        else:
            n_series, rate = _try_numeric(df[c])
        if rate < 0.75:
            return None   # bail — can't trust the numeric coercion
        items_numeric.append(n_series)
        parse_rates.append(rate)

    scale_max = _scale_max(items_numeric)
    composite, per_item = _auto_detect_reverse_scored(items_numeric, cols, scale_max)

    n_reverse = sum(1 for x in per_item if x["reverse_scored"])
    rec = {
        "label": f"{label} (composite)",
        "kind": "grouped_scale",
        "prefix": prefix,
        "item_cols": cols,
        "n": int(composite.notna().sum()),
        "stats": {
            "composite_mean": round(float(composite.mean()), 2) if composite.notna().any() else None,
            "composite_median": round(float(composite.median()), 2) if composite.notna().any() else None,
            "scale_max": scale_max,
            "n_items": len(cols),
            "n_reverse_scored": n_reverse,
            "per_item": per_item,
        },
        "chart_path": None,
        "composite_series": composite,
    }
    return rec


def _detect_grouped_scales(df: pd.DataFrame, candidate_cols: list[str]) -> tuple[list[dict], set]:
    """
    Chain three detection passes. Returns (records, consumed_column_names).
    """
    out: list[dict] = []
    consumed: set = set()

    # Pass A: number-suffix groups (e.g. Authoritarian1..Authoritarian4)
    groups: dict[str, list[str]] = defaultdict(list)
    for col in candidate_cols:
        if col in consumed:
            continue
        # Skip fact item columns — handled by the dedicated fact detector
        if schema.FACT_SUFFIX.search(col):
            continue
        m = GROUP_NUMERIC_SUFFIX.match(col)
        if m:
            prefix = m.group(1).strip("_-:. ")
            if prefix and not prefix.isdigit():
                groups[prefix].append(col)
    for prefix, cols in groups.items():
        if len(cols) < 3:
            continue
        cols_sorted = sorted(cols, key=lambda c: int(GROUP_NUMERIC_SUFFIX.match(c).group(2)))
        rec = _build_scale_record(df, prefix, prefix, cols_sorted)
        if rec:
            out.append(rec)
            consumed.update(cols_sorted)

    # Pass B: shared `:alias` suffix (e.g. ...:plaintiff_perception)
    alias_groups: dict[str, list[str]] = defaultdict(list)
    for col in candidate_cols:
        if col in consumed:
            continue
        # Skip fact item columns — they're handled by the fact-suffix detector
        if schema.FACT_SUFFIX.search(col):
            continue
        m = COLON_ALIAS_SUFFIX.search(col)
        if m:
            alias_groups[m.group(1)].append(col)
    for alias, cols in alias_groups.items():
        if len(cols) < 2:
            continue
        # Skip aliases we don't want to treat as scales (handled elsewhere)
        if alias.lower() in {"responsibility", "tipi",
                              "global_belief_in_a_just_world",
                              "litigation_attitudes", "single_item"}:
            continue
        # Keep original column order
        cols_sorted = [c for c in candidate_cols if c in cols]
        rec = _build_scale_record(df, alias, alias.replace("_", " ").title(), cols_sorted)
        if rec:
            out.append(rec)
            consumed.update(cols_sorted)

    # Pass C: NAMED_SCALES (case-specific named groupings)
    for scale_id, spec in NAMED_SCALES.items():
        members: list[str] = []
        for col in candidate_cols:
            if col in consumed:
                continue
            low = col.lower()
            for must_all in spec["patterns"]:
                if all(term.lower() in low for term in must_all):
                    members.append(col)
                    break
        if len(members) < 2:
            continue
        rec = _build_scale_record(df, scale_id, spec["label"], members)
        if rec:
            out.append(rec)
            consumed.update(members)

    return out, consumed


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans", "DejaVu Sans", "Arial"],
        "axes.edgecolor": JA_TEXT, "axes.labelcolor": JA_TEXT,
        "axes.titlecolor": JA_PRIMARY, "axes.titleweight": "bold",
        "xtick.color": JA_TEXT, "ytick.color": JA_TEXT,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": "#E5E5E5", "grid.linewidth": 0.6,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _chart_binary(rec: dict, out_path: str):
    _set_style()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    yes_n, no_n = rec["stats"]["yes_count"], rec["stats"]["no_count"]
    bars = ax.bar(["Yes", "No"], [yes_n, no_n], color=[JA_PRIMARY, "#C0392B"])
    for b, v in zip(bars, [yes_n, no_n]):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, str(v),
                ha="center", fontweight="bold", color=JA_TEXT)
    ax.set_title(rec["label"][:80], fontsize=11, color=JA_PRIMARY)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()


def _chart_categorical(rec: dict, out_path: str):
    _set_style()
    counts = rec["stats"]["counts"]
    items = sorted(counts.items(), key=lambda kv: kv[1])
    labels = [k[:40] for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.32 * len(items) + 1.2)))
    ax.barh(labels, values, color=JA_SECONDARY)
    for i, v in enumerate(values):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=9, color=JA_TEXT)
    ax.set_title(rec["label"][:80], fontsize=11, color=JA_PRIMARY)
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.3); ax.grid(axis="y", visible=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()


def _chart_numeric(rec: dict, out_path: str, series: pd.Series):
    _set_style()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    vals = series.dropna()
    # Choose bin strategy: if 6 or fewer unique → bar; else histogram
    if vals.nunique() <= 8:
        counts = vals.value_counts().sort_index()
        ax.bar([str(k) for k in counts.index], counts.values, color=JA_PRIMARY)
        for i, (k, v) in enumerate(zip(counts.index, counts.values)):
            ax.text(i, v + 0.3, str(v), ha="center", fontsize=9, color=JA_TEXT)
        ax.set_xlabel("Value")
    else:
        ax.hist(vals, bins=12, color=JA_SECONDARY, edgecolor="white")
        ax.axvline(vals.mean(), color="#C0392B", linestyle="--",
                   label=f"Mean: {vals.mean():.2f}")
        ax.axvline(vals.median(), color=JA_TEXT, linestyle=":",
                   label=f"Median: {vals.median():.2f}")
        ax.legend(frameon=False)
    ax.set_title(rec["label"][:80], fontsize=11, color=JA_PRIMARY)
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()


def _chart_grouped_scale(rec: dict, out_path: str):
    _set_style()
    items = rec["stats"]["per_item"]
    labels = [it["col"][:30] for it in items]
    means = [it["mean"] or 0 for it in items]
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.4 * len(items) + 1.2)))
    bars = ax.barh(labels, means, color=JA_MID)
    for i, v in enumerate(means):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9, color=JA_TEXT)
    composite = rec["stats"].get("composite_mean")
    if composite is not None:
        ax.axvline(composite, color="#C0392B", linestyle="--",
                   label=f"Composite mean: {composite:.2f}")
        ax.legend(frameon=False)
    ax.set_title(f"{rec['prefix']} scale — per-item means",
                 fontsize=11, color=JA_PRIMARY)
    ax.set_xlabel("Mean")
    ax.grid(axis="x", alpha=0.3); ax.grid(axis="y", visible=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def discover_extra_questions(df: pd.DataFrame, output_dir: str,
                             mapping: Optional[dict] = None) -> list[dict]:
    """
    Discover and summarize non-canonical survey questions.

    Args:
        df: cleaned dataframe (from Phase 1)
        output_dir: workspace directory; charts written to <output_dir>/charts/auto/
        mapping: optional column_mapping.json contents. If provided, columns
                 claimed by any role in the mapping are excluded from discovery.

    Returns: list of discovery records (one per discovered question), each with
        an "id", "label", "kind", and kind-specific stats + chart_path.
    """
    output_dir = Path(output_dir)
    charts_dir = output_dir / "charts" / "auto"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-load mapping from disk if not passed in (for direct callers)
    if mapping is None:
        mapping = schema.load_column_mapping(str(output_dir))

    known = _known_canonical_cols(df, mapping=mapping)
    candidates = [c for c in df.columns if c not in known and not _is_system(c)]

    # 1. Detect grouped scales first (consume their items)
    grouped, consumed = _detect_grouped_scales(df, candidates)
    candidates = [c for c in candidates if c not in consumed]

    # 2. Classify the rest
    records = []
    next_id = 1
    for rec in grouped:
        rec_id = f"auto_{next_id:03d}"
        next_id += 1
        rec["id"] = rec_id
        chart_path = f"charts/auto/{rec_id}.png"
        _chart_grouped_scale(rec, str(output_dir / chart_path))
        rec["chart_path"] = chart_path
        rec["include_default"] = True
        rec["predictor_default"] = False
        # Strip series before serialization
        composite = rec.pop("composite_series", None)
        records.append(rec)

    for col in candidates:
        rec = _classify_column(col, df[col])
        if rec is None:
            continue
        rec_id = f"auto_{next_id:03d}"
        next_id += 1
        rec["id"] = rec_id
        rec["column"] = col   # original DataFrame column name (label is the cleaned display text)
        rec["include_default"] = True
        rec["predictor_default"] = False
        # Generate chart per kind
        chart_path = f"charts/auto/{rec_id}.png"
        full = str(output_dir / chart_path)
        try:
            if rec["kind"] == "binary":
                _chart_binary(rec, full)
            elif rec["kind"] == "categorical":
                _chart_categorical(rec, full)
            elif rec["kind"] == "numeric_likert":
                numeric, _ = _try_numeric(df[col])
                _chart_numeric(rec, full, numeric)
            else:
                chart_path = None    # free_text: no chart
        except Exception as e:
            print(f"[auto_discover] chart for {col} failed: {e}")
            chart_path = None
        rec["chart_path"] = chart_path
        records.append(rec)

    return records


def composite_series_for(df: pd.DataFrame, rec: dict) -> Optional[pd.Series]:
    """For grouped_scale, recompute the composite series on demand (e.g. for
    use as a logistic-regression predictor). Honors reverse-scoring flags
    stored in rec['stats']['per_item']. Returns None for non-scale kinds."""
    if rec.get("kind") != "grouped_scale":
        return None
    cols = rec.get("item_cols", [])
    if not cols:
        return None
    per_item = rec.get("stats", {}).get("per_item", [])
    reverse_map = {x["col"]: x.get("reverse_scored", False) for x in per_item}
    scale_max = rec.get("stats", {}).get("scale_max", 5)

    pieces = []
    for c in cols:
        n, _ = _coerce_likert_to_numeric(df[c])
        if reverse_map.get(c):
            n = (scale_max + 1) - n
        pieces.append(n)
    if not pieces:
        return None
    return pd.concat(pieces, axis=1).mean(axis=1, skipna=True)
