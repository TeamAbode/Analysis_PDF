"""
Expected Case Value model.

A trial-award-prediction model that goes beyond mean / median by:
  1. Reconciling A/B split damages questions (slider vs. open-form digits+words)
  2. Applying juror-level comparative fault (award x % defendant responsibility,
     conditional on Liable=Yes per juror)
  3. Winsorizing at the 5th/95th percentile to dampen single-respondent outliers
  4. Monte Carlo simulating 10,000 12-juror panels with a 9/12 liability
     supermajority gate, producing a verdict distribution
  5. Reporting four scenarios from the verdict distribution:
        Most Likely          (median verdict)
        Best Day (defense)   (10th percentile verdict)
        Worst Day (defense)  (90th percentile verdict)
        Plaintiff Verdict In Isolation (median of >$0 verdicts only)

Methodology references (general — not exact replicas):
  - Diamond & Casper (1992) on jury damage decisions
  - Greene & Bornstein (2003) "Determining Damages"
  - Vidmar & Hans (2007) "American Juries"
  - Hastie, Schkade & Payne (1998) on group polarization in damage awards

This module is pure-Python + numpy + pandas. No randomness leaks into other
parts of the pipeline; the simulator uses a seeded RNG so re-runs reproduce.
"""
from __future__ import annotations
import re
import math
from typing import Optional
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Words -> number parser
# ---------------------------------------------------------------------------
# Handles things like:
#   "two million dollars"                          -> 2_000_000
#   "one million five hundred thousand"            -> 1_500_000
#   "350 thousand"                                 -> 350_000
#   "twenty-five thousand"                         -> 25_000
#   "$1.5 million"                                 -> 1_500_000
#   "five hundred and fifty thousand"              -> 550_000

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}


def words_to_number(text: str) -> Optional[float]:
    """
    Parse a string that may be words, digits, or a mix and return the number,
    or None if nothing parseable.

    Strategy:
      - Strip currency symbols, commas, the word "dollars", "and", and hyphens
        in compound forms ("twenty-five" -> "twenty five").
      - If the string is purely numeric (with optional decimal and 'million'/
        'thousand' suffix), parse directly. e.g. "1.5 million" -> 1_500_000.
      - Otherwise walk tokens left to right, accumulating into a 'current'
        sub-number whenever a scale word is hit; large scales (thousand,
        million, billion) flush 'current' to a running total.
    """
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None

    # Clean: strip currency symbols, commas, "dollars", "and"
    s = s.replace("$", " ")
    s = s.replace(",", "")
    s = re.sub(r"\bdollars?\b", " ", s)
    s = re.sub(r"\band\b", " ", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return None

    # Fast path: pure digits, possibly with decimal point and a scale word
    # ("1.5 million", "350 thousand", "200000", "200.000")  --
    # Note we already stripped commas, so "200.000" would parse as 200.0,
    # which is fine — that's a typo by the respondent and the validation
    # logic will catch the mismatch with the spelled-out number.
    m = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)(?:\s+(hundred|thousand|million|billion))?\s*",
        s,
    )
    if m:
        base = float(m.group(1))
        scale = _SCALES.get(m.group(2), 1) if m.group(2) else 1
        return base * scale

    # Word-token path
    tokens = s.split()
    if not tokens:
        return None

    total = 0.0
    current = 0.0
    saw_word = False

    for tok in tokens:
        # numeric token in the middle of words (e.g. "350 thousand")
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", tok):
            current += float(tok)
            saw_word = True
            continue
        if tok in _UNITS:
            current += _UNITS[tok]
            saw_word = True
        elif tok in _TENS:
            current += _TENS[tok]
            saw_word = True
        elif tok in _SCALES:
            scale = _SCALES[tok]
            if scale == 100:
                # "two hundred" -> multiply the small accumulator by 100
                current = max(current, 1) * 100
            else:
                # thousand / million / billion: flush to total
                current = max(current, 1) * scale
                total += current
                current = 0
            saw_word = True
        else:
            # unknown token — skip, but don't fail the whole parse
            continue

    if not saw_word:
        return None
    return total + current


# ---------------------------------------------------------------------------
# A/B variant detection + validation
# ---------------------------------------------------------------------------

VALIDATION_RULES = ("exact", "within_10pct", "use_words", "use_digits")
DEFAULT_VALIDATION_RULE = "use_words"


def _normalize_variant(v) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    if "a" in s and "branch" in s:
        return "A"
    if "b" in s and "branch" in s:
        return "B"
    # Also accept bare "A" / "B"
    if s in ("a", "branch a"):
        return "A"
    if s in ("b", "branch b"):
        return "B"
    return None


def reconcile_award(slider, digits, words, rule: str = DEFAULT_VALIDATION_RULE):
    """
    Given a respondent's three potential damage values (only one of slider OR
    digits+words is filled per A/B variant), return (award, source, valid):

        award:   the dollar amount to use, or None if unrecoverable
        source:  'slider' | 'open_digits' | 'open_words' | 'open_mean' | None
        valid:   True if the row passes validation, False if it should be
                 excluded from compensation analysis.

    Rule applies to Variant B respondents only (where digits+words coexist):
      exact         : digits must equal words (after rounding to nearest dollar);
                      mismatch -> invalid
      within_10pct  : digits and words within 10% of each other -> valid (use mean);
                      otherwise invalid
      use_words     : on mismatch, use the spelled-out value (and mark valid)
      use_digits    : on mismatch, use the typed digit value (and mark valid)
    """
    # Variant A: slider value present
    if slider is not None and not pd.isna(slider):
        try:
            return float(slider), "slider", True
        except (TypeError, ValueError):
            pass

    # Variant B: digits + words
    d_num = None if digits is None or (isinstance(digits, float) and pd.isna(digits)) \
                  else words_to_number(digits)
    w_num = words_to_number(words) if words is not None else None

    if d_num is None and w_num is None:
        return None, None, False

    if d_num is None:
        return w_num, "open_words", True
    if w_num is None:
        return d_num, "open_digits", True

    # Both present — apply rule
    # "Match" tolerance: within $1 (handles rounding from "1.5 million" -> 1500000.0)
    if abs(d_num - w_num) <= 1.0:
        return d_num, "open_mean", True

    if rule == "exact":
        return None, None, False
    if rule == "within_10pct":
        avg = (d_num + w_num) / 2
        if avg == 0:
            return 0.0, "open_mean", True
        if abs(d_num - w_num) / abs(avg) <= 0.10:
            return avg, "open_mean", True
        return None, None, False
    if rule == "use_words":
        return w_num, "open_words", True
    if rule == "use_digits":
        return d_num, "open_digits", True

    # Unknown rule — be conservative
    return None, None, False


# ---------------------------------------------------------------------------
# Per-respondent effective award (applies comparative-fault gate)
# ---------------------------------------------------------------------------

def compute_effective_awards(
    df: pd.DataFrame,
    award_col: str,
    liable_col: Optional[str],
    responsibility_col: Optional[str],
) -> pd.Series:
    """
    For each respondent, return their 'effective award':
        if Liable == "Yes":  award * (defendant_responsibility / 100)
        else:                 0
    Respondents missing either field get NaN (excluded from aggregates).

    award_col is a numeric column already produced by reconcile_award.
    responsibility_col is expected to be a 0-100 percentage.
    """
    awards = pd.to_numeric(df[award_col], errors="coerce")

    liable_series = pd.Series([True] * len(df), index=df.index)   # default lenient
    if liable_col and liable_col in df.columns:
        liable_text = df[liable_col].astype(str).str.strip().str.lower()
        liable_series = liable_text.isin(["yes", "y", "true", "1"])

    resp_series = pd.Series([100.0] * len(df), index=df.index)   # default full responsibility
    if responsibility_col and responsibility_col in df.columns:
        resp_series = pd.to_numeric(df[responsibility_col], errors="coerce")

    effective = pd.Series(0.0, index=df.index)
    have_data = awards.notna() & resp_series.notna()
    liable_and_have = liable_series & have_data
    effective.loc[liable_and_have] = (
        awards.loc[liable_and_have] * resp_series.loc[liable_and_have] / 100.0
    )
    # Where we don't have award data at all, return NaN so the respondent is
    # excluded from aggregates (rather than counted as a $0).
    effective.loc[~have_data] = np.nan
    return effective


# ---------------------------------------------------------------------------
# Winsorization
# ---------------------------------------------------------------------------

def winsorize(values: np.ndarray, lower_pct: float = 5, upper_pct: float = 95) -> np.ndarray:
    """Clamp values at the given percentile bounds. Drops NaN before computing
    percentiles, then re-applies the clamp to the original (non-NaN) values."""
    clean = values[~np.isnan(values)]
    if len(clean) == 0:
        return values
    lo = np.percentile(clean, lower_pct)
    hi = np.percentile(clean, upper_pct)
    out = values.copy()
    mask = ~np.isnan(out)
    out[mask] = np.clip(out[mask], lo, hi)
    return out


# ---------------------------------------------------------------------------
# Monte Carlo jury simulation
# ---------------------------------------------------------------------------

def simulate_jury_verdicts(
    effective_awards_winsorized: np.ndarray,
    liable_flags: np.ndarray,
    n_trials: int = 10_000,
    jury_size: int = 12,
    majority_required: int = 9,
    seed: int = 42,
) -> np.ndarray:
    """
    Run a Monte Carlo simulation of jury verdicts.

    Inputs:
      effective_awards_winsorized: per-respondent winsorized effective award
                                   (already comparative-fault-adjusted; $0 for
                                   non-liable respondents).
      liable_flags: per-respondent boolean (did this respondent say Liable=Yes)
      n_trials: number of simulated juries (default 10,000)
      jury_size: jurors per simulated panel (default 12)
      majority_required: votes needed for liability (default 9 = 9/12)

    For each trial:
      1. Sample `jury_size` respondents WITH replacement
         (preserves population distribution and avoids small-pool sampling bias)
      2. Count liability votes; if < majority_required, verdict = $0
         (hung or defense verdict)
      3. Otherwise verdict = mean of the LIABLE jurors' effective awards
         (deliberation proxy: high-anchor pull partly offset by reluctant jurors
          on the liable side, approximated by the mean)

    Returns an array of `n_trials` simulated jury verdicts (in dollars).
    """
    rng = np.random.default_rng(seed)

    awards = effective_awards_winsorized
    liable = liable_flags.astype(bool)

    # Filter out NaN award rows (these are excluded from the jury pool entirely
    # because we have no data for them; they don't get to "vote")
    pool_mask = ~np.isnan(awards)
    awards = awards[pool_mask]
    liable = liable[pool_mask]
    n = len(awards)

    if n == 0:
        return np.zeros(n_trials)

    verdicts = np.zeros(n_trials, dtype=float)
    for t in range(n_trials):
        idx = rng.integers(0, n, size=jury_size)
        jury_awards = awards[idx]
        jury_liable = liable[idx]
        n_liable = int(jury_liable.sum())
        if n_liable < majority_required:
            verdicts[t] = 0.0
        else:
            # Mean of LIABLE jurors' awards. Non-liable jurors' awards are
            # already $0 (by compute_effective_awards), but excluding them
            # from the average produces a more realistic deliberation outcome
            # — the liable jurors are the ones bargaining over the number.
            verdicts[t] = float(jury_awards[jury_liable].mean())

    return verdicts


# ---------------------------------------------------------------------------
# Case-value scenarios
# ---------------------------------------------------------------------------

def compute_case_value(
    df: pd.DataFrame,
    award_col: str,
    liable_col: Optional[str],
    responsibility_col: Optional[str],
    variant_col: Optional[str] = None,
    n_trials: int = 10_000,
) -> dict:
    """
    End-to-end case-value model. Produces:

      {
        "scenarios": {
            "most_likely":              {"value": ..., "label": "Most Likely"},
            "best_day_defense":         {"value": ..., "label": "Best Day (Defense)"},
            "worst_day_defense":        {"value": ..., "label": "Worst Day (Defense)"},
            "plaintiff_verdict_isolated": {"value": ..., "label": "Plaintiff Verdict in Isolation"},
        },
        "stats": {
            "n_respondents_in_pool":   ...,
            "p_defense_verdict":       ... (0-1),
            "p_plaintiff_verdict":     ... (0-1),
            "winsor_low":              ... ($),
            "winsor_high":             ... ($),
            "mean_simulated_verdict":  ...,
            "median_simulated_verdict": ...,
        },
        "config": {
            "jury_size": 12, "majority_required": 9, "n_trials": ...,
            "winsor_pct": [5, 95], "comparative_fault_rule": "...",
        },
      }
    """
    # Per-respondent effective award (winsorize numerator)
    effective = compute_effective_awards(
        df, award_col=award_col, liable_col=liable_col,
        responsibility_col=responsibility_col,
    )
    eff_arr = effective.to_numpy(dtype=float)

    # Liability flag per respondent (parallel array)
    if liable_col and liable_col in df.columns:
        liable_arr = df[liable_col].astype(str).str.strip().str.lower().isin(
            ["yes", "y", "true", "1"]
        ).to_numpy()
    else:
        liable_arr = np.ones(len(df), dtype=bool)

    # Winsorize ONLY across the non-NaN, non-zero universe to set the bounds,
    # but apply the clamp to the full array. Otherwise the huge population of
    # $0s (non-liable respondents) drags the 5th percentile to $0 and there's
    # nothing to clip.
    clean_nonzero = eff_arr[(~np.isnan(eff_arr)) & (eff_arr > 0)]
    if len(clean_nonzero) >= 10:
        winsor_low = float(np.percentile(clean_nonzero, 5))
        winsor_high = float(np.percentile(clean_nonzero, 95))
        eff_wins = eff_arr.copy()
        mask = ~np.isnan(eff_wins) & (eff_wins > 0)
        eff_wins[mask] = np.clip(eff_wins[mask], winsor_low, winsor_high)
    else:
        winsor_low = float("nan")
        winsor_high = float("nan")
        eff_wins = eff_arr.copy()

    # Run the simulation
    verdicts = simulate_jury_verdicts(
        eff_wins, liable_arr, n_trials=n_trials,
        jury_size=12, majority_required=9, seed=42,
    )

    # Build scenarios
    pos_verdicts = verdicts[verdicts > 0]
    p_defense = float((verdicts == 0).mean())

    def _money(x):
        return None if (x is None or math.isnan(x)) else round(float(x), 0)

    scenarios = {
        "most_likely": {
            "value": _money(np.median(verdicts)),
            "label": "Most Likely",
            "description": "Median of all simulated jury verdicts.",
        },
        "best_day_defense": {
            "value": _money(np.percentile(verdicts, 10)),
            "label": "Best Day (Defense)",
            "description": "10th percentile of simulated verdicts — defense gets a favorable jury.",
        },
        "worst_day_defense": {
            "value": _money(np.percentile(verdicts, 90)),
            "label": "Worst Day (Defense)",
            "description": "90th percentile of simulated verdicts — plaintiff gets a very favorable jury.",
        },
        "plaintiff_verdict_isolated": {
            "value": _money(np.median(pos_verdicts)) if len(pos_verdicts) > 0 else 0.0,
            "label": "Plaintiff Verdict in Isolation",
            "description": ("Median verdict among only the simulated juries that returned"
                            " a plaintiff verdict (>$0). Answers 'when plaintiff wins, what"
                            " is the typical award?'"),
        },
    }

    stats = {
        "n_respondents_in_pool": int((~np.isnan(eff_arr)).sum()),
        "n_liable_respondents": int((liable_arr & ~np.isnan(eff_arr)).sum()),
        "p_defense_verdict": round(p_defense, 3),
        "p_plaintiff_verdict": round(1 - p_defense, 3),
        "winsor_low":  _money(winsor_low),
        "winsor_high": _money(winsor_high),
        "mean_simulated_verdict":   _money(float(np.mean(verdicts))),
        "median_simulated_verdict": _money(float(np.median(verdicts))),
    }

    config = {
        "jury_size": 12,
        "majority_required": 9,
        "n_trials": n_trials,
        "winsor_pct": [5, 95],
        "comparative_fault_rule": "liable_yes_or_zero_then_responsibility_multiplied",
        "outlier_treatment": "winsorize_5_95_on_positive_awards",
    }

    return {"scenarios": scenarios, "stats": stats, "config": config,
            "verdict_distribution": verdicts.tolist() if n_trials <= 10_000 else None}
