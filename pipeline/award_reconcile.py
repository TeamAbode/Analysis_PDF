"""
Damages award reconciliation across an A/B split.

Some surveys capture the award amount twice per respondent: once as a numeric /
free-text figure (`compensation_amount`) and once spelled out in words
(`compensation_in_words`). When the survey also A/B-tests the elicitation format,
each branch has its own pair of columns (pandas de-dupes the second branch as
`...amount.1` / `...in_words.1`).

This module:
  1. finds the per-branch amount + words columns and which branch each serves,
  2. parses both answers to numbers and keeps only respondents whose two answers
     AGREE (drops disagreements and unparseable answers),
  3. reports per-branch stats (mean/median/…) on the cleaned awards and compares
     Branch A vs Branch B.
"""
from __future__ import annotations
import re
from typing import Optional

import numpy as np
import pandas as pd

from . import damages_ab
from .money_parse import parse_money

_AMOUNT_RE = re.compile(r"^\s*compensation_amount(\.\d+)?\s*$", re.IGNORECASE)
_WORDS_RE = re.compile(r"^\s*compensation_in_words(\.\d+)?\s*$", re.IGNORECASE)


def _suffix(col: str) -> str:
    """Return the pandas dedup suffix ('' or '.1', '.2', …) of a column name."""
    m = re.search(r"(\.\d+)\s*$", col)
    return m.group(1) if m else ""


def find_award_ab_columns(df: pd.DataFrame, mapping: Optional[dict] = None) -> Optional[dict]:
    """Locate the per-branch amount/words columns and the variant column.

    Returns a dict {variant_col, branches: {label: {amount, words}}} or None if
    the paired-column pattern isn't present.
    """
    amount_cols = [c for c in df.columns if _AMOUNT_RE.match(str(c))]
    words_cols = {_suffix(c): c for c in df.columns if _WORDS_RE.match(str(c))}
    if not amount_cols:
        return None

    variant_col = damages_ab.find_variant_column(df, mapping=mapping)
    norm = (damages_ab.normalize_variant_series(df[variant_col])
            if variant_col else pd.Series([None] * len(df), index=df.index))

    branches: dict[str, dict] = {}
    for amt in amount_cols:
        words = words_cols.get(_suffix(amt))
        # Decide which branch this amount column serves: the variant label under
        # which it's most often answered.
        filled = df[amt].notna()
        if variant_col and filled.any():
            label = norm[filled].dropna().mode()
            branch = label.iloc[0] if len(label) else _suffix(amt) or "A"
        else:
            branch = _suffix(amt) or "A"
        branches[branch] = {"amount": amt, "words": words}

    if not branches:
        return None
    return {"variant_col": variant_col, "branches": branches}


def _clean_stats(vals: np.ndarray) -> dict:
    if len(vals) == 0:
        return {"n": 0, "mean": None, "median": None, "min": None,
                "max": None, "std": None}
    return {
        "n": int(len(vals)),
        "mean": int(round(float(np.mean(vals)))),
        "median": int(round(float(np.median(vals)))),
        "min": int(round(float(np.min(vals)))),
        "max": int(round(float(np.max(vals)))),
        "std": int(round(float(np.std(vals, ddof=1)))) if len(vals) > 1 else 0,
    }


def reconcile_awards(df: pd.DataFrame, mapping: Optional[dict] = None,
                     rel_tol: float = 0.01) -> Optional[dict]:
    """Clean and compare per-branch awards. Returns a report dict + a combined
    cleaned award Series (matched respondents only), or None if not applicable.
    """
    found = find_award_ab_columns(df, mapping=mapping)
    if not found:
        return None
    variant_col = found["variant_col"]
    norm = (damages_ab.normalize_variant_series(df[variant_col])
            if variant_col else pd.Series([None] * len(df), index=df.index))

    combined = pd.Series(np.nan, index=df.index, dtype=float)
    branches_out: dict[str, dict] = {}
    branch_clean_vals: dict[str, np.ndarray] = {}

    for branch in sorted(found["branches"].keys()):
        cols = found["branches"][branch]
        amt_col, words_col = cols["amount"], cols["words"]
        n_assigned = int((norm == branch).sum()) if variant_col else None

        both = 0
        matched_idx = []
        mismatch_examples = []
        n_mismatch = n_unverifiable = 0
        for idx, row in df[[amt_col] + ([words_col] if words_col else [])].iterrows():
            a_raw = row[amt_col]
            w_raw = row[words_col] if words_col else None
            if pd.isna(a_raw) and (words_col is None or pd.isna(w_raw)):
                continue
            a = parse_money(a_raw)
            w = parse_money(w_raw) if words_col else None
            if words_col is None:
                # No words column to check against — keep the parsed number.
                if a is not None:
                    combined.at[idx] = a
                    matched_idx.append(idx)
                continue
            both += 1
            if a is None or w is None:
                n_unverifiable += 1
                continue
            agree = (a == 0 and w == 0) or abs(a - w) <= rel_tol * max(abs(a), abs(w), 1.0)
            if agree:
                combined.at[idx] = a
                matched_idx.append(idx)
            else:
                n_mismatch += 1
                if len(mismatch_examples) < 8:
                    mismatch_examples.append({
                        "numeric": str(a_raw), "words": str(w_raw),
                        "parsed_numeric": a, "parsed_words": w,
                    })

        vals = combined.loc[matched_idx].dropna().to_numpy(dtype=float)
        branch_clean_vals[branch] = vals
        branches_out[branch] = {
            "label": f"Branch {branch}",
            "amount_col": amt_col,
            "words_col": words_col,
            "n_assigned": n_assigned,
            "n_both_answered": both,
            "n_matched": int(len(vals)),
            "n_mismatch": n_mismatch,
            "n_unverifiable": n_unverifiable,
            "pct_matched": (round(100 * len(vals) / both, 1) if both else None),
            "clean": _clean_stats(vals),
            "mismatch_examples": mismatch_examples,
        }

    comparison = _compare_branches(branch_clean_vals)

    return {
        "present": True,
        "variant_col": variant_col,
        "rel_tol": rel_tol,
        "branches": branches_out,
        "comparison": comparison,
        "combined_clean": combined,   # Series; stripped before JSON serialization
    }


def _compare_branches(branch_vals: dict[str, np.ndarray]) -> Optional[dict]:
    labels = [b for b in sorted(branch_vals) if len(branch_vals[b]) > 0]
    if len(labels) != 2:
        return None
    a, b = labels
    va, vb = branch_vals[a], branch_vals[b]
    out = {
        "a": a, "b": b,
        "mean_a": int(round(float(np.mean(va)))), "mean_b": int(round(float(np.mean(vb)))),
        "median_a": int(round(float(np.median(va)))), "median_b": int(round(float(np.median(vb)))),
        "mean_diff": int(round(float(np.mean(vb) - np.mean(va)))),
        "median_diff": int(round(float(np.median(vb) - np.median(va)))),
        "p_value": None,
        "test": None,
    }
    try:
        from scipy import stats
        # Mann-Whitney U: robust to the heavy right-skew of damages awards.
        _, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
        out["p_value"] = round(float(p), 4)
        out["test"] = "Mann-Whitney U"
    except Exception:
        pass
    return out
