"""
Phase 2: analysis.

Reads the cleaned CSV + case metadata and produces:
  - analysis_bundle.json   (every number that goes into the report)
  - charts/*.png            (every chart that goes into the report)

This module is pure deterministic computation. No AI. No LLM calls.
Phase 3 reads analysis_bundle.json + the chart filepaths.

Brand colors (Jury Analyst):
  Primary:    #0000B7
  Secondary:  #0073DF
  Mid accent: #4D4DCC
  Light:      #E6E6F7
"""
from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import schema


# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------

JA_PRIMARY = "#0000B7"
JA_SECONDARY = "#0073DF"
JA_MID = "#4D4DCC"
JA_LIGHT = "#E6E6F7"
JA_DARK = "#1A1A2E"
JA_TEXT = "#231F20"

# Chart palette for categorical data (extends from brand colors)
CATEGORICAL_PALETTE = [
    "#0000B7",   # primary
    "#0073DF",   # secondary
    "#4D4DCC",   # mid
    "#7A95E0",   # light blue
    "#B8C5E8",   # very light blue
    "#1A1A2E",   # dark
    "#8C8C9E",   # gray
]

# Defense / plaintiff coloring (mirrors the existing report)
COLOR_DEFENSE = "#C0392B"
COLOR_PLAINTIFF = "#0073DF"
COLOR_NEUTRAL = "#BDC3C7"


# ---------------------------------------------------------------------------
# Matplotlib styling
# ---------------------------------------------------------------------------

def _set_chart_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans", "DejaVu Sans", "Arial"],
        "axes.edgecolor": JA_TEXT,
        "axes.labelcolor": JA_TEXT,
        "axes.titlecolor": JA_PRIMARY,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "xtick.color": JA_TEXT,
        "ytick.color": JA_TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Numeric coercion helpers
# ---------------------------------------------------------------------------

def _numeric(df: pd.DataFrame, col: Optional[str], mapping: Optional[dict] = None) -> pd.Series:
    """Coerce a column to numeric, using a Likert mapping if provided.
    Also strips '%' and '$' signs and commas before parsing."""
    if col is None or col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index)
    if mapping:
        return schema.coerce_likert(df[col], mapping)
    # Strip common formatting before parsing
    s = df[col].astype(str).str.replace("%", "", regex=False)
    s = s.str.replace("$", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    s = s.str.strip()
    return pd.to_numeric(s, errors="coerce")


def _safe_pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 1) if denominator > 0 else 0.0


# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

def _value_counts(df: pd.DataFrame, col: Optional[str]) -> dict:
    if col is None or col not in df.columns:
        return {}
    vc = df[col].dropna().astype(str).str.strip()
    vc = vc[vc != ""]
    counts = vc.value_counts()
    total = counts.sum()
    return {
        "counts": counts.to_dict(),
        "pct": {k: _safe_pct(v, total) for k, v in counts.to_dict().items()},
        "n": int(total),
    }


def compute_demographics(df: pd.DataFrame) -> dict:
    return {
        "gender": _value_counts(df, schema.find_column(df, "sex")),
        "political_party": _value_counts(df, schema.find_column(df, "political_party")),
        "political_view": _value_counts(df, schema.find_column(df, "political_view")),
        "race": _value_counts(df, schema.find_column(df, "race")),
        "education": _value_counts(df, schema.find_column(df, "education")),
        "income": _value_counts(df, schema.find_column(df, "income")),
        "marital_status": _value_counts(df, schema.find_column(df, "marital_status")),
        "employment": _value_counts(df, schema.find_column(df, "employment")),
    }


def chart_demographic_overview(demos: dict, out_path: str):
    """Four-panel pie chart of Gender, Party, Political View, Race."""
    _set_chart_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Demographic Overview", fontsize=15, fontweight="bold", color=JA_PRIMARY)

    panels = [
        ("Gender", demos["gender"], axes[0, 0]),
        ("Political Party", demos["political_party"], axes[0, 1]),
        ("Political View", demos["political_view"], axes[1, 0]),
        ("Race/Ethnicity", demos["race"], axes[1, 1]),
    ]
    for title, data, ax in panels:
        if not data or not data.get("counts"):
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.axis("off")
            continue
        labels = list(data["counts"].keys())
        values = list(data["counts"].values())
        colors = CATEGORICAL_PALETTE[:len(labels)]
        wedges, texts, autotexts = ax.pie(
            values, labels=None, colors=colors,
            autopct=lambda p: f"{p:.0f}%" if p >= 4 else "",
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontweight("bold")
            at.set_fontsize(10)
        ax.set_title(title, fontweight="bold", fontsize=12, color=JA_PRIMARY, pad=12)
        ax.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def chart_education_income(demos: dict, out_path: str):
    _set_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Education & Income Distribution", fontsize=14, fontweight="bold", color=JA_PRIMARY)

    for ax, (key, title) in zip(axes, [("education", "Education"), ("income", "Income")]):
        data = demos.get(key, {})
        if not data.get("counts"):
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title); ax.axis("off"); continue
        items = sorted(data["counts"].items(), key=lambda kv: kv[1])
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        color = JA_PRIMARY if key == "education" else JA_SECONDARY
        bars = ax.barh(labels, values, color=color)
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    str(v), va="center", fontsize=9, color=JA_TEXT)
        ax.set_title(title, fontweight="bold", color=JA_PRIMARY)
        ax.set_xlabel("Count")
        ax.grid(axis="x", alpha=0.3); ax.grid(axis="y", visible=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Case facts
# ---------------------------------------------------------------------------

def compute_facts(df: pd.DataFrame, fact_metadata: list[dict]) -> list[dict]:
    """
    For each fact, compute importance mean, influence mean, convincingness mean,
    % plaintiff-leaning, direction, and collect juror reasoning quotes.

    fact_metadata: list of {"num": int, "text": str} from case metadata.
    """
    fact_nums = schema.detect_facts(df)
    results = []

    # Build a lookup of fact text from metadata
    text_map = {f["num"]: f["text"] for f in fact_metadata}

    # Build a lookup of "How convincing" columns by fact number. Alchemer uses
    # both "this claim" (Facts 2+) and "that claim" (Fact 1) phrasings — match
    # either, anchored on the :Fact N suffix.
    convincing_cols: dict[int, str] = {}
    for col in df.columns:
        m = schema.FACT_SUFFIX.search(col)
        if not m:
            continue
        n = int(m.group(1))
        low = col.lower()
        if "how convincing" in low:
            convincing_cols[n] = col

    # 5-point convincing scale (anchored 1 = Not at all, 5 = Extremely)
    convincing_map = {
        "Not at all convincing": 1, "Slightly convincing": 2,
        "Somewhat convincing": 3,  "Very convincing": 4,
        "Extremely convincing": 5,
    }

    for num in fact_nums:
        imp_col = schema.fact_importance_col(df, num)
        inf_col = schema.fact_influence_col(df, num)
        reason_col = schema.fact_reasoning_col(df, num)
        conv_col = convincing_cols.get(num)

        imp_numeric = _numeric(df, imp_col, schema.SIX_POINT_FACT_IMPORTANCE)
        inf_numeric = _numeric(df, inf_col, schema.SIX_POINT_FACT_INFLUENCE)
        conv_numeric = _numeric(df, conv_col, convincing_map) if conv_col else None

        imp_mean = round(float(imp_numeric.mean()), 2) if imp_numeric.notna().any() else None
        inf_mean = round(float(inf_numeric.mean()), 2) if inf_numeric.notna().any() else None
        conv_mean = (round(float(conv_numeric.mean()), 2)
                     if conv_numeric is not None and conv_numeric.notna().any() else None)

        # Influence > 3.5 = plaintiff-leaning, < 3.5 = defense-leaning
        plaintiff_count = int((inf_numeric > 3.5).sum())
        defense_count = int((inf_numeric < 3.5).sum())
        neutral_count = int((inf_numeric == 3.5).sum())
        total_valid = plaintiff_count + defense_count + neutral_count

        pct_plaintiff = _safe_pct(plaintiff_count, total_valid)
        pct_defense = _safe_pct(defense_count, total_valid)

        if inf_mean is None:
            direction = "UNKNOWN"
        elif inf_mean >= 3.5:
            direction = "PLAINTIFF"
        else:
            direction = "DEFENSE"

        quotes = []
        if reason_col and reason_col in df.columns:
            raw_quotes = df[reason_col].dropna().astype(str).str.strip()
            raw_quotes = raw_quotes[raw_quotes.str.len() > 15]
            quotes = raw_quotes.tolist()[:8]

        # Convincing distribution (counts at each numeric level 1..5)
        conv_dist = {}
        if conv_numeric is not None and conv_numeric.notna().any():
            for level in range(1, 6):
                conv_dist[level] = int((conv_numeric == level).sum())

        results.append({
            "num": num,
            "text": text_map.get(num, ""),
            "importance_mean": imp_mean,
            "influence_mean": inf_mean,
            "convincing_mean": conv_mean,
            "convincing_distribution": conv_dist,
            "pct_plaintiff": pct_plaintiff,
            "pct_defense": pct_defense,
            "n": int(inf_numeric.notna().sum()),
            "direction": direction,
            "quotes": quotes,
        })

    return results


def chart_facts_importance_influence(facts: list[dict], out_path: str):
    _set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [f"F{f['num']}" for f in facts]
    imp = [f["importance_mean"] or 0 for f in facts]
    inf = [f["influence_mean"] or 0 for f in facts]
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(x - w/2, imp, w, label="Importance (1–6)", color=JA_PRIMARY)
    b2 = ax.bar(x + w/2, inf, w, label="Influence (1=Defense → 6=Plaintiff)", color=JA_SECONDARY)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                    f"{bar.get_height():.1f}", ha="center", fontsize=9, fontweight="bold",
                    color=JA_TEXT)
    ax.axhline(3.5, color=JA_DARK, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 7); ax.set_ylabel("Mean Score")
    ax.set_title("Case Facts: Importance & Influence", fontweight="bold", color=JA_PRIMARY, fontsize=14)
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def chart_facts_direction(facts: list[dict], out_path: str):
    _set_chart_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [f"F{f['num']}" for f in facts]
    plaintiff_pct = [f["pct_plaintiff"] for f in facts]
    defense_pct = [f["pct_defense"] for f in facts]
    x = np.arange(len(labels))
    ax.bar(x, defense_pct, color=COLOR_DEFENSE, label="Defense-leaning")
    ax.bar(x, plaintiff_pct, bottom=defense_pct, color=COLOR_PLAINTIFF, label="Plaintiff-leaning")
    ax.axhline(50, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("% of Respondents"); ax.set_ylim(0, 100)
    ax.set_title("Verdict Direction per Fact", fontweight="bold", color=JA_PRIMARY, fontsize=14)
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Verdict / liability / responsibility
# ---------------------------------------------------------------------------

def compute_verdict(df: pd.DataFrame) -> dict:
    favor_col = schema.find_column(df, "favor_support")
    favor_numeric = _numeric(df, favor_col, schema.SIX_POINT_FAVOR_SUPPORT)

    plaintiff_lean = int((favor_numeric >= 4).sum())
    defense_lean = int((favor_numeric <= 3).sum())
    total_with_lean = plaintiff_lean + defense_lean

    # Liability Yes/No
    liable_col = schema.find_column(df, "liable")
    yes_count = 0; no_count = 0
    if liable_col:
        yes_count = int((df[liable_col].astype(str).str.strip().str.lower() == "yes").sum())
        no_count = int((df[liable_col].astype(str).str.strip().str.lower() == "no").sum())

    # Responsibility allocation
    plt_resp_col = schema.find_column(df, "responsibility_plaintiff")
    def_resp_col = schema.find_column(df, "responsibility_defendant")
    plt_resp = _numeric(df, plt_resp_col)
    def_resp = _numeric(df, def_resp_col)

    # Support distribution (1–6)
    support_dist = {}
    if favor_numeric.notna().any():
        labels_6pt = {
            1: "Strongly Defense", 2: "Defense", 3: "Somewhat Defense",
            4: "Somewhat Plaintiff", 5: "Plaintiff", 6: "Strongly Plaintiff",
        }
        for k, label in labels_6pt.items():
            support_dist[label] = int((favor_numeric == k).sum())

    return {
        "n": int(favor_numeric.notna().sum()),
        "pct_plaintiff_leaning": _safe_pct(plaintiff_lean, total_with_lean),
        "pct_defense_leaning": _safe_pct(defense_lean, total_with_lean),
        "pct_liable_yes": _safe_pct(yes_count, yes_count + no_count),
        "pct_liable_no": _safe_pct(no_count, yes_count + no_count),
        "liable_yes_n": yes_count,
        "liable_no_n": no_count,
        "mean_responsibility_plaintiff": round(float(plt_resp.mean()), 1) if plt_resp.notna().any() else None,
        "mean_responsibility_defendant": round(float(def_resp.mean()), 1) if def_resp.notna().any() else None,
        "support_distribution": support_dist,
        "favor_numeric": favor_numeric.tolist(),    # for later use
    }


def chart_verdict_support(verdict: dict, out_path: str):
    _set_chart_style()
    dist = verdict["support_distribution"]
    if not dist:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = list(dist.keys())
    values = list(dist.values())
    colors = ["#C0392B", "#E74C3C", "#F39C12",
              "#7FB8E0", "#0073DF", "#0000B7"]
    bars = ax.bar(labels, values, color=colors[:len(labels)])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(v), ha="center", fontsize=11, fontweight="bold", color=JA_TEXT)
    ax.axvline(2.5, color="gray", linestyle="--", alpha=0.4)
    ax.set_ylabel("Number of Jurors")
    ax.set_title("Verdict Support Distribution", fontweight="bold", color=JA_PRIMARY, fontsize=14)
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def chart_liability_responsibility(verdict: dict, out_path: str):
    _set_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Liability pie
    yes_n = verdict["liable_yes_n"]; no_n = verdict["liable_no_n"]
    if yes_n + no_n > 0:
        axes[0].pie([yes_n, no_n], labels=["Yes", "No"],
                    colors=[JA_PRIMARY, COLOR_DEFENSE],
                    autopct="%.0f%%", startangle=90,
                    wedgeprops=dict(edgecolor="white", linewidth=2),
                    textprops=dict(color="white", fontweight="bold", fontsize=12))
        axes[0].set_title(f"Found Liable ({verdict['pct_liable_yes']:.0f}% Yes)",
                          fontweight="bold", color=JA_PRIMARY, fontsize=12)
    else:
        axes[0].axis("off")
        axes[0].text(0.5, 0.5, "(no data)", ha="center", va="center", transform=axes[0].transAxes)

    # Responsibility
    plt_r = verdict["mean_responsibility_plaintiff"]
    def_r = verdict["mean_responsibility_defendant"]
    if plt_r is not None and def_r is not None:
        bars = axes[1].bar(["Plaintiff", "Defendant(s)"], [plt_r, def_r],
                           color=[COLOR_DEFENSE, JA_PRIMARY])
        for bar, v in zip(bars, [plt_r, def_r]):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                         f"{v:.0f}%", ha="center", fontsize=12, fontweight="bold", color=JA_TEXT)
        axes[1].set_ylim(0, 110); axes[1].set_ylabel("Mean % Responsibility")
        axes[1].set_title("Mean Responsibility Allocation",
                          fontweight="bold", color=JA_PRIMARY, fontsize=12)
    else:
        axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Defendant attribution
# ---------------------------------------------------------------------------

def compute_defendant_attribution(df: pd.DataFrame) -> dict:
    sup_col = schema.find_column(df, "defendant_supported")
    liable_col = schema.find_column(df, "defendant_most_liable")
    return {
        "supported_by_defense_leaners": _value_counts(df, sup_col),
        "most_liable_by_plaintiff_leaners": _value_counts(df, liable_col),
    }


# ---------------------------------------------------------------------------
# Plaintiff support by demographic subgroup
# ---------------------------------------------------------------------------

def compute_plaintiff_by_demo(df: pd.DataFrame) -> dict:
    """% plaintiff-leaning broken down by each demographic group."""
    favor_col = schema.find_column(df, "favor_support")
    favor_numeric = _numeric(df, favor_col, schema.SIX_POINT_FAVOR_SUPPORT)
    df = df.copy()
    df["_plaintiff_leaning"] = favor_numeric >= 4

    out = {}
    demo_cols = {
        "political_party": "political_party",
        "political_view": "political_view",
        "sex": "sex",
        "race": "race",
        "education": "education",
        "income": "income",
    }
    for key, canonical in demo_cols.items():
        col = schema.find_column(df, canonical)
        if not col:
            out[key] = {}
            continue
        groups = df.groupby(df[col].astype(str).str.strip())
        rows = []
        for name, sub in groups:
            if name == "" or name.lower() == "nan":
                continue
            n = len(sub)
            pct = _safe_pct(int(sub["_plaintiff_leaning"].sum()), n)
            rows.append({"group": name, "n": n, "pct_plaintiff_leaning": pct})
        rows.sort(key=lambda r: -r["pct_plaintiff_leaning"])
        out[key] = rows
    return out


def chart_plaintiff_by_demo(by_demo: dict, out_path: str):
    _set_chart_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Plaintiff Support by Demographic Subgroup",
                 fontsize=14, fontweight="bold", color=JA_PRIMARY)
    panels = [
        ("political_party", "Political Party", axes[0, 0]),
        ("political_view", "Political View", axes[0, 1]),
        ("sex", "Gender", axes[1, 0]),
        ("race", "Race/Ethnicity", axes[1, 1]),
    ]
    for key, title, ax in panels:
        rows = by_demo.get(key, [])
        if not rows:
            ax.axis("off"); ax.set_title(title); continue
        rows_sorted = sorted(rows, key=lambda r: r["pct_plaintiff_leaning"])
        labels = [f"{r['group']}" for r in rows_sorted]
        values = [r["pct_plaintiff_leaning"] for r in rows_sorted]
        ns = [r["n"] for r in rows_sorted]
        colors = [JA_SECONDARY if v >= 50 else COLOR_DEFENSE for v in values]
        bars = ax.barh(labels, values, color=colors)
        for bar, v, n in zip(bars, values, ns):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f"{v:.0f}% (n={n})", va="center", fontsize=9, color=JA_TEXT)
        ax.axvline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlim(0, 110); ax.set_xlabel("% Plaintiff-Leaning")
        ax.set_title(title, fontweight="bold", color=JA_PRIMARY)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Compensation
# ---------------------------------------------------------------------------

def compute_compensation(df: pd.DataFrame, mapping: Optional[dict] = None) -> dict:
    # The "total compensation" column. Try the canonical literal first
    # (older surveys used the variable name `Total_Compensation`); if that
    # doesn't resolve, fall back to the role-resolved combined-total column.
    comp_col = (
        schema.find_column(df, "total_compensation", mapping=mapping)
        or schema.find_column(df, "total_compensation_combined", mapping=mapping)
    )
    comp = _numeric(df, comp_col)
    liable_col = schema.find_column(df, "liable", mapping=mapping)
    liable = df[liable_col].astype(str).str.strip().str.lower() if liable_col else pd.Series([""] * len(df))

    comp_liable = comp[liable == "yes"]

    bracket_col = schema.find_column(df, "comp_bracket", mapping=mapping)
    bracket_data = _value_counts(df, bracket_col)

    demand_col = schema.find_column(df, "demand_reaction", mapping=mapping)
    demand_data = _value_counts(df, demand_col)

    deserving_col = schema.find_column(df, "deserving", mapping=mapping)
    deserving_numeric = _numeric(df, deserving_col, schema.FIVE_POINT_DESERVING)
    deserving_mean = round(float(deserving_numeric.mean()), 2) if deserving_numeric.notna().any() else None
    deserving_dist = {}
    if deserving_col:
        deserving_dist = _value_counts(df, deserving_col)

    eggshell_col = schema.find_column(df, "eggshell", mapping=mapping)
    eggshell_numeric = _numeric(df, eggshell_col, schema.SEVEN_POINT_LIKERT)
    eggshell_mean = round(float(eggshell_numeric.mean()), 2) if eggshell_numeric.notna().any() else None

    # Reasoning quotes (canonical comp_reasoning field; in newer cases this is
    # routed through the mapping UI to the "Please explain the reasoning behind
    # your total damage award" column).
    reason_col = schema.find_column(df, "comp_reasoning", mapping=mapping)
    reasoning_quotes = []
    if reason_col:
        raw = df[reason_col].dropna().astype(str).str.strip()
        raw = raw[raw.str.len() > 20]
        reasoning_quotes = raw.tolist()[:15]

    # ------------------------------------------------------------------
    # NEW: A/B split testing (sampling-design framework)
    # ------------------------------------------------------------------
    # Two patterns are supported. Pattern A: every juror filled the same
    # combined-total column. Pattern B: Branch A filled the slider-variant
    # column, Branch B filled the open-form column — these are two different
    # columns and we compare them.
    from . import damages_ab, case_value
    combined_total_col = (
        schema.find_column(df, "total_compensation_combined", mapping=mapping)
        or comp_col
    )
    slider_col = schema.find_column(df, "total_compensation_slider", mapping=mapping)
    open_col   = schema.find_column(df, "total_compensation_open", mapping=mapping)
    variant_col = damages_ab.find_variant_column(df, mapping=mapping)
    ab_analysis = damages_ab.analyze_ab_split(
        df, combined_total_col, variant_col,
        slider_col=slider_col, open_col=open_col,
    )

    # ------------------------------------------------------------------
    # NEW: Per-defendant case-value model
    # ------------------------------------------------------------------
    # find_defendant_responsibility_columns lives in damages_ab? No — keep it
    # close to where the responsibility data is resolved.  We'll detect
    # responsibility columns inline for now (single-defendant cases just resolve
    # to the canonical responsibility_defendant column).
    case_value_blocks = _compute_per_defendant_case_value(
        df, combined_total_col=combined_total_col, liable_col=liable_col,
        mapping=mapping,
    )

    return {
        "n_responses": len(comp),
        "mean_award_all": int(round(float(comp.mean()))) if comp.notna().any() else None,
        "median_award_all": int(round(float(comp.median()))) if comp.notna().any() else None,
        "mean_award_liable": int(round(float(comp_liable.mean()))) if comp_liable.notna().any() else None,
        "median_award_liable": int(round(float(comp_liable.median()))) if comp_liable.notna().any() else None,
        "n_liable": int(comp_liable.notna().sum()),
        "bracket_selection": bracket_data,
        "demand_reaction": demand_data,
        "deserving_mean": deserving_mean,
        "deserving_distribution": deserving_dist,
        "eggshell_mean": eggshell_mean,
        "award_values": comp.dropna().tolist(),
        "reasoning_quotes": reasoning_quotes,
        # New A/B + case-value fields
        "ab_analysis": ab_analysis,
        "case_value": case_value_blocks,
    }


def _compute_per_defendant_case_value(
    df: pd.DataFrame,
    combined_total_col: Optional[str],
    liable_col: Optional[str],
    mapping: Optional[dict] = None,
) -> list[dict]:
    """
    For each defendant whose responsibility column has data, run the case-value
    Monte Carlo. Each block:
      {
        "defendant_label":   "Defendant/s" | "Defendant (Nurse Thompson)" | ...,
        "n_respondents":     count of jurors with responsibility data,
        "original":          full case_value.compute_case_value output,
        "pivot":             same shape, using REASSIGNED responsibility, or None,
        "responsibility":    {"mean": ..., "median": ...} for the defendant,
      }

    Returns sorted by mean responsibility (most-exposed first).
    Empty list if no defendant has usable data.
    """
    from . import case_value
    if not combined_total_col or combined_total_col not in df.columns:
        return []

    # Find all responsibility columns. Each is either ":Responsibility" suffix
    # (original) or contains "You indicated that the facts changed" (pivot).
    party_records: dict[str, dict] = {}
    for col in df.columns:
        name = str(col)
        low = name.lower()
        is_pivot = "you indicated that the facts changed your opinion" in low
        is_orig = name.endswith(":Responsibility")
        if not (is_pivot or is_orig):
            continue
        if is_orig:
            label = name[: name.rfind(":Responsibility")].strip()
        else:
            label = name.split(":", 1)[0].strip()
        # Normalize "The Plaintiff" / "The Defendant/s" pivot labels
        norm = label.replace("The ", "").strip()
        rec = party_records.setdefault(norm, {
            "label": norm, "original_col": None, "pivot_col": None,
            "is_plaintiff": "plaintiff" in norm.lower(),
        })
        n = _parse_pct_series(df[col]).notna().sum()
        if is_orig and (rec["original_col"] is None or
                        n > _parse_pct_series(df[rec["original_col"]]).notna().sum()):
            rec["original_col"] = col
        elif is_pivot and (rec["pivot_col"] is None or
                           n > _parse_pct_series(df[rec["pivot_col"]]).notna().sum()):
            rec["pivot_col"] = col

    blocks = []
    for rec in party_records.values():
        if rec["is_plaintiff"]:
            continue
        orig_resp = _parse_pct_series(df[rec["original_col"]]) if rec["original_col"] else None
        pivot_resp = _parse_pct_series(df[rec["pivot_col"]]) if rec["pivot_col"] else None
        n_orig = int(orig_resp.notna().sum()) if orig_resp is not None else 0
        n_pivot = int(pivot_resp.notna().sum()) if pivot_resp is not None else 0
        if n_orig < 5 and n_pivot < 5:
            continue   # not enough data — skip per your earlier choice

        # Build a parsed-percent dataframe so case_value can read it
        df_for_cv = df.copy()
        if rec["original_col"]:
            df_for_cv[rec["original_col"]] = orig_resp

        block = {
            "defendant_label": rec["label"],
            "n_with_original_responsibility": n_orig,
            "n_with_pivot_responsibility":    n_pivot,
            "original": None,
            "pivot":    None,
            "responsibility": {
                "mean":   round(float(orig_resp.mean()), 1) if n_orig else None,
                "median": round(float(orig_resp.median()), 1) if n_orig else None,
            },
        }
        if n_orig >= 5:
            block["original"] = case_value.compute_case_value(
                df_for_cv,
                award_col=combined_total_col,
                liable_col=liable_col,
                responsibility_col=rec["original_col"],
                n_trials=10_000,
            )
            # Strip the verdict distribution from the bundle — it's huge and
            # only useful for charts (which we'll generate separately)
            block["original_verdict_distribution"] = \
                block["original"].pop("verdict_distribution", None)
        if n_pivot >= 5:
            df_for_cv_pivot = df.copy()
            df_for_cv_pivot[rec["pivot_col"]] = pivot_resp
            block["pivot"] = case_value.compute_case_value(
                df_for_cv_pivot,
                award_col=combined_total_col,
                liable_col=liable_col,
                responsibility_col=rec["pivot_col"],
                n_trials=10_000,
            )
            block["pivot_verdict_distribution"] = \
                block["pivot"].pop("verdict_distribution", None)
        blocks.append(block)

    # Sort by mean responsibility (most-exposed first)
    blocks.sort(key=lambda b: -(b["responsibility"]["mean"] or 0))
    return blocks


def _parse_pct_series(series: pd.Series) -> pd.Series:
    """Parse '80%', '80', 0.80, 80 etc into 0-100 float, NaN otherwise."""
    def _one(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return float("nan")
        s = str(v).strip().replace("%", "").strip()
        if not s:
            return float("nan")
        try:
            x = float(s)
        except ValueError:
            return float("nan")
        if 0 < x <= 1.0:
            x *= 100
        return x
    return series.apply(_one).astype(float)


def chart_award_distribution(comp: dict, out_path: str):
    _set_chart_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    values = comp["award_values"]
    if not values:
        ax.text(0.5, 0.5, "(no award data)", ha="center", va="center", transform=ax.transAxes)
    else:
        # Bin in 50k increments up to 500k
        bins = list(range(0, 550_000, 50_000))
        ax.hist(values, bins=bins, color=JA_SECONDARY, edgecolor="white", linewidth=1)
        mean_v = comp["mean_award_all"]
        median_v = comp["median_award_all"]
        if mean_v is not None:
            ax.axvline(mean_v, color=COLOR_DEFENSE, linestyle="--", linewidth=2,
                       label=f"Mean: ${mean_v/1000:.0f}K")
        if median_v is not None:
            ax.axvline(median_v, color=JA_DARK, linestyle=":", linewidth=2,
                       label=f"Median: ${median_v/1000:.0f}K")
        ax.legend(frameon=False)
        ax.set_xlabel("Amount ($K)")
        # Convert tick labels to K
        ax.set_xticks(bins[::2])
        ax.set_xticklabels([f"{b//1000}" for b in bins[::2]])
        ax.set_ylabel("Frequency")
    ax.set_title("Award Distribution ($0–$500K)",
                 fontweight="bold", color=JA_PRIMARY, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Predictive indicators — what predicts plaintiff support
# (Internally uses logistic regression; never exposed in the report.)
# ---------------------------------------------------------------------------

def compute_predictors(df: pd.DataFrame, extra_predictors: Optional[dict] = None) -> dict:
    """
    Predictive indicators model: outcome = plaintiff-leaning (1) vs defense-leaning (0).
    Predictors: a curated set of psychographic + demographic vars, plus any
    user-supplied extras from auto-discovered survey questions.

    Args:
        df: cleaned dataframe
        extra_predictors: optional dict {label: numeric_series} of additional
            predictors to merge in alongside the canonical ones.

    Returns predictive-strength values (standardized) sorted by absolute magnitude.
    Internally uses logistic regression but exposes a stats-free API.
    """
    favor_col = schema.find_column(df, "favor_support")
    favor_numeric = _numeric(df, favor_col, schema.SIX_POINT_FAVOR_SUPPORT)
    y = (favor_numeric >= 4).astype(int)

    # Predictor: (canonical name, label, Likert mapping or None for numeric/categorical)
    predictor_specs = [
        ("sex", "Sex", None),                                # categorical -> numeric encode
        ("damage_limits_ps", "Pro-Damage-Limits (P&S)", None),
        ("income", "Income", None),
        ("ever_sued", "Anti-Plaintiff Litigation Attitude", None),
        ("just_world_score", "Just World Belief", None),
        ("race", "Race", None),
        ("education", "Education", None),
        ("political_view", "Political_View", None),
        ("eggshell", "Eggshell Plaintiff Belief", schema.SEVEN_POINT_LIKERT),
        ("pre_info_bias", "Pre-Info Bias (Plaintiff)", None),
    ]

    # Need to construct columns sensibly
    X_data = {}

    # Sex: Female=1, Male=0
    sex_col = schema.find_column(df, "sex")
    if sex_col:
        X_data["Sex"] = (df[sex_col].astype(str).str.strip().str.lower() == "female").astype(int)

    # Damage limits P&S: Yes=1, No=0
    dl_col = schema.find_column(df, "damage_limits_ps")
    if dl_col:
        X_data["Pro-Damage-Limits (P&S)"] = (df[dl_col].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Income: ordinal encoding
    inc_col = schema.find_column(df, "income")
    if inc_col:
        inc_order = ["Less than $10,000", "$10,000 to $24,000", "$25,000 to $49,000",
                     "$50,000 to $74,000", "$75,000 to $99,000",
                     "$100,000 to $149,000", "Over $150,000"]
        # Be lenient with the dollar sign
        s = df[inc_col].astype(str).str.strip().str.replace("$", "", regex=False)
        order_clean = [v.replace("$", "") for v in inc_order]
        X_data["Income"] = s.map({v: i for i, v in enumerate(order_clean)})

    # Anti-plaintiff litigation attitude: composite of Litigation_attitudes scale (4 items, 7-pt)
    lit_cols = schema.find_scale_columns(df, "litigation_attitudes")
    if lit_cols:
        lit_numeric = pd.concat([schema.coerce_likert(df[c], schema.SEVEN_POINT_LIKERT) for c in lit_cols], axis=1)
        X_data["Anti-Plaintiff Litigation Attitude"] = lit_numeric.mean(axis=1)

    # Just world: 7-item scale mean
    jw_cols = schema.find_scale_columns(df, "just_world")
    if jw_cols:
        jw_numeric = pd.concat([schema.coerce_likert(df[c], schema.SEVEN_POINT_LIKERT) for c in jw_cols], axis=1)
        X_data["Just World Belief"] = jw_numeric.mean(axis=1)

    # Race: White=1, non-White=0 (rough — for small samples this is the safest binary)
    race_col = schema.find_column(df, "race")
    if race_col:
        X_data["Race"] = (df[race_col].astype(str).str.strip().str.startswith("White")).astype(int)

    # Education: ordinal
    edu_col = schema.find_column(df, "education")
    if edu_col:
        edu_order = ["Some High School", "High School Diploma or GED", "Some College",
                     "Associate's Degree", "Bachelor's Degree", "Master's Degree",
                     "Professional Degree (e.g., JD, DDS)", "Doctoral Degree"]
        X_data["Education"] = df[edu_col].astype(str).str.strip().map(
            {v: i for i, v in enumerate(edu_order)}
        )

    # Political_View: Conservative=0, Moderate=1, Liberal=2
    pv_col = schema.find_column(df, "political_view")
    if pv_col:
        pv_map = {"Conservative": 0, "Moderate": 1, "Liberal": 2}
        X_data["Political View"] = df[pv_col].astype(str).str.strip().map(pv_map)

    # Eggshell belief: 7-pt likert
    egg_col = schema.find_column(df, "eggshell")
    if egg_col:
        X_data["Eggshell Plaintiff Belief"] = schema.coerce_likert(df[egg_col], schema.SEVEN_POINT_LIKERT)

    # Pre-info bias: 0–100 slider, already numeric
    pib_col = schema.find_column(df, "pre_info_bias")
    if pib_col:
        X_data["Pre-Info Bias (Plaintiff)"] = pd.to_numeric(df[pib_col], errors="coerce")

    # Merge user-supplied extra predictors from auto-discovery
    if extra_predictors:
        for label, series in extra_predictors.items():
            if series is None or series.notna().sum() < 10:
                continue
            X_data[label] = series.reindex(df.index)

    # Drop rows with any NaN in predictors or outcome
    X = pd.DataFrame(X_data)
    mask = X.notna().all(axis=1) & y.notna()
    X_clean = X[mask]
    y_clean = y[mask]

    if len(X_clean) < 20 or y_clean.nunique() < 2:
        return {"accuracy": None, "coefficients": [], "n": int(len(X_clean))}

    # Standardize for comparable coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_scaled, y_clean)
    accuracy = float(model.score(X_scaled, y_clean))
    coefs = list(zip(X_clean.columns, model.coef_[0]))
    coefs.sort(key=lambda x: abs(x[1]), reverse=True)
    coef_records = [
        {
            "predictor": name,
            "coefficient": round(float(c), 3),
            "direction": "Plaintiff" if c > 0 else "Defense",
        }
        for name, c in coefs
    ]
    return {
        "accuracy": round(accuracy, 3),
        "coefficients": coef_records,
        "n": int(len(X_clean)),
    }


def chart_predictors(predictors: dict, out_path: str):
    _set_chart_style()
    coefs = predictors.get("coefficients", [])
    if not coefs:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    coefs_sorted = sorted(coefs, key=lambda c: c["coefficient"])
    labels = [c["predictor"] for c in coefs_sorted]
    values = [c["coefficient"] for c in coefs_sorted]
    colors = [JA_SECONDARY if v > 0 else COLOR_DEFENSE for v in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color=JA_TEXT, linewidth=0.8)
    ax.set_xlabel("Predictive Strength  (← Defense    Plaintiff →)")
    ax.set_title("What Predicts a Plaintiff-Leaning Juror",
                 fontweight="bold", color=JA_PRIMARY, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ---------------------------------------------------------------------------
# Open-ended response harvesting
# ---------------------------------------------------------------------------

def compute_open_ended(df: pd.DataFrame, mapping: Optional[dict] = None) -> dict:
    """Collect verbatim open-ended responses for Phase 3 thematic summarization.

    Each role can now resolve to 0, 1, or many columns (when role is multi-fill,
    like reasoning_extra). The result for a multi-fill role is a list of
    {"label": <human-readable>, "responses": [...]} entries so prompts can
    cite which prompt each batch came from.
    """
    out: dict = {}

    def _collect(col: str, min_len: int = 10) -> list[str]:
        # Defensive: skip empty strings, None, or columns not in df. An empty
        # string can arrive from a mapping payload where the user added a
        # role slot but never picked a column.
        if not col or col not in df.columns:
            return []
        raw = df[col].dropna().astype(str).str.strip()
        # Filter junk responses early — the AI never sees these
        junk = {"", "na", "n/a", "none", "nothing", "no", "n", "no.", "none.",
                "nothing.", "idk", "i don't know", "i dont know", "?", "."}
        raw = raw[raw.str.len() > min_len]
        raw = raw[~raw.str.lower().str.strip().isin(junk)]
        return raw.tolist()

    # Single-fill free-text roles
    for role in ("narrative", "evidence_gap", "unanswered",
                 "decision_reasoning", "comp_reasoning"):
        col = schema.find_column(df, role, mapping=mapping)
        out[role] = _collect(col) if col else []

    # Multi-fill role: reasoning_extra. Filter out empty strings before any
    # use — the mapping UI sometimes leaves "" entries when a user opens a
    # row but doesn't select a column.
    extra_cols = [c for c in schema.find_columns_multi(df, "reasoning_extra", mapping=mapping)
                  if c and c in df.columns]
    out["reasoning_extra"] = []
    for col in extra_cols:
        responses = _collect(col)
        if responses:
            out["reasoning_extra"].append({"label": col, "responses": responses})

    # Sponsor belief (categorical)
    sponsor_col = schema.find_column(df, "sponsor_belief", mapping=mapping)
    out["sponsor_belief"] = _value_counts(df, sponsor_col)

    # Back-compat alias: older report templates may reference `reasoning_quotes`
    # under compensation. We keep that working by exposing the comp_reasoning
    # list at the top level too (the compensation block also includes it).
    return out


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_analysis(clean_csv_path: str, case_metadata: dict, output_dir: str,
                 discovery_choices: Optional[dict] = None) -> dict:
    """
    Run the full Phase 2 analysis.

    Args:
        clean_csv_path: path to the cleaned CSV from Phase 1
        case_metadata: dict from CaseMetadata.to_dict()
        output_dir: where to write analysis_bundle.json + charts/
        discovery_choices: optional dict mapping discovery item id → {"include": bool, "predictor": bool}.
                           If None, every discovered item is included and none are predictors.

    Returns:
        The analysis bundle dict (also written to disk).
    """
    from . import auto_discover

    output_dir = Path(output_dir)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(clean_csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(clean_csv_path, encoding="latin-1")

    # Load the column mapping (mandatory: Phase 2 UI must commit it before this runs).
    # If somehow absent, fall back to auto-resolution so analysis still works,
    # but the result will only fill canonical (literal) columns reliably.
    mapping = schema.load_column_mapping(str(output_dir))
    if mapping is None:
        mapping = schema.auto_resolve_all(df)

    # --- Compute everything ---
    demos = compute_demographics(df)
    facts = compute_facts(df, case_metadata.get("facts", []))
    verdict = compute_verdict(df)
    def_attribution = compute_defendant_attribution(df)
    by_demo = compute_plaintiff_by_demo(df)
    comp = compute_compensation(df, mapping=mapping)
    open_ended = compute_open_ended(df, mapping=mapping)

    # --- Auto-discover non-canonical survey questions ---
    discovered_full = auto_discover.discover_extra_questions(
        df, str(output_dir), mapping=mapping,
    )
    # Apply user choices (include / predictor toggles, reverse-scoring overrides)
    discovered = []
    extra_predictors: dict[str, pd.Series] = {}
    for rec in discovered_full:
        choice = (discovery_choices or {}).get(rec["id"], {})
        include = choice.get("include", rec.get("include_default", True))
        as_predictor = choice.get("predictor", rec.get("predictor_default", False))
        rec["include"] = include
        rec["predictor"] = as_predictor

        # Reverse-scoring overrides for grouped scales. The UI sends:
        #   choice["reverse_overrides"] = { "<item_col>": True/False, ... }
        # We update the per_item.reverse_scored flags then recompute the
        # composite mean/median so the discovered record reflects the new
        # composition.
        reverse_overrides = choice.get("reverse_overrides") or {}
        if rec["kind"] == "grouped_scale" and reverse_overrides:
            per_item = rec.get("stats", {}).get("per_item", [])
            changed = False
            for item in per_item:
                col = item.get("col")
                if col in reverse_overrides:
                    new_val = bool(reverse_overrides[col])
                    if item.get("reverse_scored") != new_val:
                        item["reverse_scored"] = new_val
                        item["reverse_scored_manual"] = True
                        changed = True
            if changed:
                # Recompute composite using overridden flags
                series = auto_discover.composite_series_for(df, rec)
                if series is not None and series.notna().any():
                    rec["stats"]["composite_mean"] = round(float(series.mean()), 2)
                    rec["stats"]["composite_median"] = round(float(series.median()), 2)
                    rec["stats"]["n_reverse_scored"] = sum(
                        1 for it in per_item if it.get("reverse_scored")
                    )
                    rec["n"] = int(series.notna().sum())

        if include:
            discovered.append(rec)
        # If user opted this in as a predictor AND it's a numeric/binary/scale, capture series
        if as_predictor:
            series = None
            df_col = rec.get("column") or rec.get("label")
            if rec["kind"] == "grouped_scale":
                series = auto_discover.composite_series_for(df, rec)
            elif rec["kind"] == "binary":
                from .auto_discover import _try_numeric
                if df_col in df.columns:
                    num, _ = _try_numeric(df[df_col])
                    if num.notna().any():
                        series = num
                    else:
                        s = df[df_col].astype(str).str.strip().str.lower()
                        series = s.map({"yes": 1, "no": 0, "y": 1, "n": 0})
            elif rec["kind"] == "numeric_likert":
                from .auto_discover import _try_numeric
                if df_col in df.columns:
                    num, _ = _try_numeric(df[df_col])
                    series = num
            if series is not None and series.notna().sum() >= 10:
                pred_label = (rec.get("prefix") or rec["label"])[:50]
                extra_predictors[pred_label] = series

    predictors = compute_predictors(df, extra_predictors=extra_predictors)

    # Strip the favor_numeric list before serializing (it's large + not needed)
    verdict_export = {k: v for k, v in verdict.items() if k != "favor_numeric"}

    # --- Render charts ---
    chart_demographic_overview(demos, str(charts_dir / "demographics.png"))
    chart_education_income(demos, str(charts_dir / "education_income.png"))
    chart_facts_importance_influence(facts, str(charts_dir / "facts_importance_influence.png"))
    chart_facts_direction(facts, str(charts_dir / "facts_direction.png"))
    chart_verdict_support(verdict, str(charts_dir / "verdict_support.png"))
    chart_liability_responsibility(verdict, str(charts_dir / "liability_responsibility.png"))
    chart_plaintiff_by_demo(by_demo, str(charts_dir / "plaintiff_by_demo.png"))
    chart_award_distribution(comp, str(charts_dir / "award_distribution.png"))
    chart_predictors(predictors, str(charts_dir / "predictors.png"))

    # --- Bundle ---
    bundle = {
        "case_metadata": case_metadata,
        "n_respondents": int(len(df)),
        "demographics": demos,
        "facts": facts,
        "verdict": verdict_export,
        "defendant_attribution": def_attribution,
        "plaintiff_support_by_demo": by_demo,
        "compensation": comp,
        "predictors": predictors,
        "open_ended": open_ended,
        "discovered_questions": discovered,
        "discovered_questions_all": discovered_full,  # full list incl. user-deselected (for UI re-population)
        "charts": {
            "demographics": "charts/demographics.png",
            "education_income": "charts/education_income.png",
            "facts_importance_influence": "charts/facts_importance_influence.png",
            "facts_direction": "charts/facts_direction.png",
            "verdict_support": "charts/verdict_support.png",
            "liability_responsibility": "charts/liability_responsibility.png",
            "plaintiff_by_demo": "charts/plaintiff_by_demo.png",
            "award_distribution": "charts/award_distribution.png",
            "predictors": "charts/predictors.png",
        },
    }

    bundle_path = output_dir / "analysis_bundle.json"
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    return bundle
