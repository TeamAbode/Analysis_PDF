"""
A/B damages-question handling — as a SAMPLING DESIGN, not a damages structure.

The A/B variant column (e.g. "Branch A" / "Branch B", or "Slider" / "open ended")
tells us which input mode each respondent saw for damages questions. We don't
try to reconcile sub-category columns into a single value — instead we treat
each variant as its own sample of jurors and:

  1. Report combined-total stats overall AND per variant.
  2. Run Welch's t-test on combined-total awards by variant. If significant,
     surface that — it's a methodological finding ("the slider produced
     systematically different awards than the open-form question").
  3. Report sub-category stats per variant (since they're filled per-variant).

Single-variant cases (no A/B column or only one variant value present) skip
the t-test and report sub-categories using whatever column is populated.
"""
from __future__ import annotations
import re
from typing import Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Variant detection
# ---------------------------------------------------------------------------

# Aliases for variant values. Maps lowercase trimmed value -> canonical
# variant label ("A" or "B"). "A" = slider variant by convention; "B" = open.
VARIANT_VALUES = {
    "branch a":   "A",
    "branch b":   "B",
    "a":          "A",
    "b":          "B",
    "slider":     "A",
    "sliders":    "A",
    "open":       "B",
    "open ended": "B",
    "open-ended": "B",
    "open form":  "B",
    "open-form":  "B",
}


def find_variant_column(df: pd.DataFrame, mapping: Optional[dict] = None) -> Optional[str]:
    """
    Resolve the A/B variant column. Priority:
      1. mapping["roles"]["ab_variant"]["columns"][0]  (user-confirmed)
      2. Auto-detect by column name + content

    Returns the column name or None if no A/B sampling design is present.
    """
    if mapping and "roles" in mapping:
        entry = mapping["roles"].get("ab_variant")
        if entry and entry.get("columns"):
            return entry["columns"][0]

    # Auto-detect: look for column names mentioning "a/b" or "split testing"
    name_patterns = [r"a\s*/\s*b\s*split", r"a/b\s*test", r"split\s*test", r"variant",
                     r"^Group$", r"^Branch$"]
    for col in df.columns:
        for pat in name_patterns:
            if re.search(pat, col, re.IGNORECASE):
                # Verify content looks like an A/B variant column
                vals = df[col].dropna().astype(str).str.strip().str.lower().unique()
                if 2 <= len(vals) <= 4:
                    if any(v in VARIANT_VALUES for v in vals):
                        return col
    return None


def normalize_variant_series(series: pd.Series) -> pd.Series:
    """Return a series of canonical 'A'/'B' values (or NaN where unmappable)."""
    norm = series.astype(str).str.strip().str.lower().map(VARIANT_VALUES)
    return norm


# ---------------------------------------------------------------------------
# A/B analysis
# ---------------------------------------------------------------------------

def analyze_ab_split(
    df: pd.DataFrame,
    combined_total_col: Optional[str],
    variant_col: Optional[str],
    slider_col: Optional[str] = None,
    open_col: Optional[str] = None,
) -> dict:
    """
    Produce A/B analysis for the combined-total damages question.

    Two patterns are supported:

      Pattern A — SINGLE COLUMN: every juror filled the same combined-total
                   column regardless of variant. `combined_total_col` is set;
                   we split by variant.

      Pattern B — TWO-COLUMN A/B: Branch A filled `slider_col`, Branch B
                   filled `open_col`. We synthesize the per-variant award
                   series from those columns and compare across the two
                   variants (Welch's t-test).

    Returns:
      {
        "ab_present":          True/False (only true if both variants have data),
        "variant_col":         column name (or None),
        "variant_labels":      {"A": "Slider", "B": "open ended"},
        "source_columns":      {"A": "<slider_col or combined_col>",
                                "B": "<open_col or combined_col>"},
        "overall":             { n, mean, median, winsor_mean_5_95 },
        "per_variant": {
            "A": { n, mean, median, winsor_mean_5_95, raw_label, source_col },
            "B": { ... },
        },
        "ttest": { t_statistic, p_value, diff_means, interpretation, significant } | None,
      }
    """
    # --- Build a per-juror "award" series, picking the right column for each ---
    if slider_col or open_col:
        # Pattern B (two-column A/B). Build a per-row award series by reading
        # whichever column matches each respondent's variant.
        slider_vals = (pd.to_numeric(_clean_money_series(df[slider_col]), errors="coerce")
                       if slider_col and slider_col in df.columns
                       else pd.Series([float("nan")] * len(df), index=df.index))
        open_vals = (pd.to_numeric(_clean_money_series(df[open_col]), errors="coerce")
                     if open_col and open_col in df.columns
                     else pd.Series([float("nan")] * len(df), index=df.index))
        # Combine: prefer the column matching the variant, else fall back to whichever has data
        if variant_col and variant_col in df.columns:
            norm = normalize_variant_series(df[variant_col])
            awards = pd.Series([float("nan")] * len(df), index=df.index, dtype=float)
            awards.loc[norm == "A"] = slider_vals.loc[norm == "A"]
            awards.loc[norm == "B"] = open_vals.loc[norm == "B"]
            # For rows where the variant column is NaN, fall back to whichever has a value
            unknown = norm.isna()
            awards.loc[unknown] = slider_vals.where(slider_vals.notna(), open_vals).loc[unknown]
        else:
            awards = slider_vals.where(slider_vals.notna(), open_vals)
    else:
        # Pattern A: single column for everyone
        if not combined_total_col or combined_total_col not in df.columns:
            return _empty_ab_result(variant_col)
        awards = pd.to_numeric(_clean_money_series(df[combined_total_col]), errors="coerce")

    out: dict = {
        "ab_present": False,
        "variant_col": variant_col,
        "variant_labels": {},
        "source_columns": {},
        "overall": _award_stats(awards),
        "per_variant": {},
        "ttest": None,
    }

    if not variant_col or variant_col not in df.columns:
        return out

    norm = normalize_variant_series(df[variant_col])
    if not norm.notna().any():
        return out

    # Build raw-label lookup so we can show "Slider" / "open ended" in the report
    raw_by_canon: dict[str, str] = {}
    for raw_val, canon in zip(df[variant_col].astype(str), norm):
        if pd.notna(canon) and canon not in raw_by_canon:
            raw_by_canon[canon] = str(raw_val).strip()
    out["variant_labels"] = raw_by_canon
    out["source_columns"] = {"A": slider_col or combined_total_col,
                              "B": open_col or combined_total_col}

    # Per-variant stats — KEY CHANGE: read each variant FROM ITS OWN COLUMN
    # so the slider-only and open-only column patterns are reflected correctly.
    for v in ("A", "B"):
        if slider_col or open_col:
            src_col = slider_col if v == "A" else open_col
            if not src_col or src_col not in df.columns: continue
            v_series = pd.to_numeric(_clean_money_series(df[src_col]), errors="coerce")
            v_mask = (norm == v)
            sub = v_series[v_mask]
        else:
            sub = awards[norm == v]
        if sub.notna().sum() == 0:
            continue
        stats = _award_stats(sub)
        stats["raw_label"] = raw_by_canon.get(v, v)
        stats["source_col"] = (slider_col if v == "A" else open_col) or combined_total_col
        out["per_variant"][v] = stats

    # T-test (requires both variants ≥ 10)
    if "A" in out["per_variant"] and "B" in out["per_variant"]:
        if slider_col or open_col:
            a_vals = pd.to_numeric(_clean_money_series(df[slider_col]), errors="coerce") \
                       [norm == "A"].dropna() if slider_col and slider_col in df.columns else pd.Series([])
            b_vals = pd.to_numeric(_clean_money_series(df[open_col]), errors="coerce") \
                       [norm == "B"].dropna() if open_col and open_col in df.columns else pd.Series([])
        else:
            a_vals = awards[norm == "A"].dropna()
            b_vals = awards[norm == "B"].dropna()
        if len(a_vals) >= 10 and len(b_vals) >= 10:
            out["ab_present"] = True
            t_stat, p_val = _welch_ttest(a_vals.to_numpy(), b_vals.to_numpy())
            diff = float(b_vals.mean() - a_vals.mean())
            out["ttest"] = {
                "t_statistic": round(float(t_stat), 3),
                "p_value":     round(float(p_val), 4),
                "diff_means":  round(diff, 0),
                "significant": bool(p_val < 0.05),
                "interpretation": _interpret_ab(
                    a_label=raw_by_canon.get("A", "A"),
                    b_label=raw_by_canon.get("B", "B"),
                    a_mean=float(a_vals.mean()),
                    b_mean=float(b_vals.mean()),
                    p=float(p_val),
                ),
            }

    return out


def _clean_money_series(s: pd.Series) -> pd.Series:
    """Strip $ , and whitespace from a money column so to_numeric works."""
    if s.dtype != "object":
        return s
    return s.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()


def _empty_ab_result(variant_col):
    return {
        "ab_present": False,
        "variant_col": variant_col,
        "variant_labels": {},
        "source_columns": {},
        "overall": {"n": 0, "mean": None, "median": None, "winsor_mean_5_95": None},
        "per_variant": {},
        "ttest": None,
    }


def _award_stats(series: pd.Series) -> dict:
    clean = series.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return {"n": 0, "mean": None, "median": None, "winsor_mean_5_95": None}
    if len(clean) >= 10 and (clean > 0).sum() >= 5:
        pos = clean[clean > 0]
        lo, hi = np.percentile(pos, [5, 95])
        wins = np.clip(clean, lo, hi)
        wins_mean = float(wins.mean())
    else:
        wins_mean = None
    return {
        "n":     int(len(clean)),
        "mean":  round(float(clean.mean()), 0),
        "median": round(float(np.median(clean)), 0),
        "winsor_mean_5_95": round(wins_mean, 0) if wins_mean is not None else None,
    }


def _welch_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test for unequal variances. Returns (t, p)."""
    a_mean, b_mean = a.mean(), b.mean()
    a_var, b_var = a.var(ddof=1), b.var(ddof=1)
    n_a, n_b = len(a), len(b)
    se = np.sqrt(a_var / n_a + b_var / n_b)
    if se == 0:
        return 0.0, 1.0
    t_stat = (a_mean - b_mean) / se
    # Welch–Satterthwaite degrees of freedom
    num = (a_var / n_a + b_var / n_b) ** 2
    den = (a_var ** 2) / (n_a ** 2 * (n_a - 1)) + (b_var ** 2) / (n_b ** 2 * (n_b - 1))
    df_ws = num / den if den > 0 else (n_a + n_b - 2)
    # Two-sided p-value via the survival function of Student's t.
    # We avoid scipy dependency by approximating via the symmetric CDF:
    # For practical use, use the t-distribution CDF if scipy is available,
    # else fall back to a normal approximation (fine at large df).
    p_value = _t_two_sided_p(t_stat, df_ws)
    return float(t_stat), float(p_value)


def _t_two_sided_p(t_stat: float, df: float) -> float:
    """Two-sided p-value for Student's t. Uses scipy if available, else falls
    back to a normal approximation (good when df > ~30)."""
    try:
        from scipy import stats as _stats
        return float(2.0 * (1.0 - _stats.t.cdf(abs(t_stat), df)))
    except Exception:
        # Normal approximation
        from math import erf, sqrt
        z = abs(t_stat)
        # Two-sided: P(|Z| > z) = 2 * (1 - Phi(z))
        return float(2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0)))))


def _interpret_ab(a_label: str, b_label: str, a_mean: float, b_mean: float, p: float) -> str:
    """One-sentence plain-English interpretation of the A/B finding.
    No statistics jargon — written for trial-team / mediation audience."""
    diff = b_mean - a_mean
    higher_label = b_label if diff > 0 else a_label
    lower_label = a_label if diff > 0 else b_label
    higher_val = max(a_mean, b_mean)
    lower_val = min(a_mean, b_mean)
    if higher_val > 0:
        pct = (higher_val - lower_val) / max(lower_val, 1) * 100
    else:
        pct = 0
    if p < 0.001:
        certainty = "with very strong evidence"
    elif p < 0.01:
        certainty = "with strong evidence"
    elif p < 0.05:
        certainty = "with moderate evidence"
    else:
        certainty = "but the difference is within normal sample variation"
    return (f"Jurors shown the '{higher_label}' format awarded on average "
            f"${higher_val:,.0f}, about {pct:.0f}% higher than those shown "
            f"the '{lower_label}' format (${lower_val:,.0f}), {certainty}.")


# ---------------------------------------------------------------------------
# Sub-category stats per variant
# ---------------------------------------------------------------------------

def per_variant_subcategory_stats(
    df: pd.DataFrame,
    subcategory_cols: list[str],
    variant_col: Optional[str],
) -> list[dict]:
    """
    For each sub-category column, return descriptive stats per variant.
    Format:
      [
        {
          "column": "...",
          "label":  human-readable label (truncated),
          "overall": {n, mean, median, winsor_mean_5_95},
          "per_variant": {"A": {...}, "B": {...}}  or {}  if no A/B,
        },
        ...
      ]
    """
    out = []
    norm = None
    if variant_col and variant_col in df.columns:
        norm = normalize_variant_series(df[variant_col])

    for col in subcategory_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        record = {
            "column": col,
            "label":  col[:100],
            "overall": _award_stats(series),
            "per_variant": {},
        }
        if norm is not None and norm.notna().any():
            for v in ("A", "B"):
                sub = series[norm == v]
                if sub.notna().sum() == 0:
                    continue
                record["per_variant"][v] = _award_stats(sub)
        # Only include sub-categories that actually have data somewhere
        if record["overall"]["n"] > 0:
            out.append(record)
    return out
