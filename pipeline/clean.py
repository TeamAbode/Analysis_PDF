"""
Phase 1: cleaning.

Filters available:
  - status:         remove if Status != 'Complete'
  - speeder:        remove if duration < (threshold * median)
  - straightliner:  remove if any Likert grid shows the same response on >= N% of items
  - duplicate:      keep first occurrence per (RID, email)

Each exclusion is logged with: response_id, email/RID, list of reasons, and
the triggering values. The audit is exportable as CSV.

Per-person overrides: after applying filters, the UI shows flagged respondents
in a table. The user can un-check anyone to keep them in the clean dataset.
"""
from __future__ import annotations
import io
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
import pandas as pd
import numpy as np

from . import schema


# ---------------------------------------------------------------------------
# Filter configuration
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    enable_status: bool = True
    status_keep: str = "Complete"

    enable_speeder: bool = True
    speeder_threshold_pct: float = 1/3   # fraction of median duration

    enable_straightliner: bool = True
    straightliner_pct: float = 1.0        # exact match required (every item same)
    straightliner_min_items: int = 5      # only test scales with >= this many items

    enable_duplicates: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "FilterConfig":
        defaults = cls()
        for k, v in d.items():
            if hasattr(defaults, k):
                setattr(defaults, k, v)
        return defaults


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------

@dataclass
class FlagRecord:
    response_id: str
    email: str
    rid: str
    reasons: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def load_csv(path_or_buffer) -> pd.DataFrame:
    """Load Alchemer CSV. Tries utf-8 first, falls back to latin-1."""
    try:
        return pd.read_csv(path_or_buffer, encoding="utf-8")
    except UnicodeDecodeError:
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        return pd.read_csv(path_or_buffer, encoding="latin-1")


def compute_duration_seconds(df: pd.DataFrame) -> pd.Series:
    """Compute survey duration in seconds from Time Started / Date Submitted."""
    start_col = schema.find_column(df, "time_started")
    end_col = schema.find_column(df, "date_submitted")
    if not start_col or not end_col:
        return pd.Series([None] * len(df), index=df.index)
    starts = pd.to_datetime(df[start_col], errors="coerce")
    ends = pd.to_datetime(df[end_col], errors="coerce")
    return (ends - starts).dt.total_seconds()


def detect_straightlining(df: pd.DataFrame, pct_threshold: float = 1.0,
                          min_items: int = 5) -> tuple[pd.Series, dict]:
    """
    For each respondent, check whether any Likert grid (TIPI, Just World, etc.)
    has the same response on >= pct_threshold fraction of its items.

    Only scales with >= min_items items are tested (avoids false positives on
    very short scales where coincidence is likely).

    Returns (boolean Series, detail dict per index).
    """
    flags = pd.Series(False, index=df.index)
    detail = {}

    for scale_name in schema.SCALE_GROUPS:
        cols = schema.find_scale_columns(df, scale_name)
        if len(cols) < min_items:
            continue
        sub = df[cols]
        def _max_same(row):
            vals = row.dropna()
            if len(vals) == 0:
                return 0
            counts = vals.value_counts()
            return counts.iloc[0] / len(vals)
        max_pct = sub.apply(_max_same, axis=1)
        scale_flag = max_pct >= pct_threshold
        flags = flags | scale_flag
        for idx in scale_flag[scale_flag].index:
            detail.setdefault(idx, []).append(
                f"{scale_name} ({max_pct[idx]*100:.0f}% same across {len(cols)} items)"
            )

    return flags, detail


# ---------------------------------------------------------------------------
# Main flag computation
# ---------------------------------------------------------------------------

def flag_all(df: pd.DataFrame, cfg: FilterConfig) -> tuple[pd.DataFrame, list[FlagRecord]]:
    """
    Apply all filters. Return a DataFrame with flag columns AND a list of
    FlagRecord objects for any flagged respondent.

    Flag columns added to df: _flag_status, _flag_speeder, _flag_straightliner,
    _flag_duplicate, _flag_any, _flag_reasons.
    """
    df = df.copy()
    n = len(df)
    df["_flag_status"] = False
    df["_flag_speeder"] = False
    df["_flag_straightliner"] = False
    df["_flag_duplicate"] = False
    df["_flag_detail"] = [dict() for _ in range(n)]

    # --- Status ---
    if cfg.enable_status:
        status_col = schema.find_column(df, "status")
        if status_col:
            mask = df[status_col].astype(str).str.strip() != cfg.status_keep
            df.loc[mask, "_flag_status"] = True
            for idx in df.index[mask]:
                df.at[idx, "_flag_detail"]["status"] = str(df.at[idx, status_col])

    # --- Speeder ---
    duration = compute_duration_seconds(df)
    df["_duration_sec"] = duration
    if cfg.enable_speeder and duration.notna().any():
        median_dur = duration.median()
        threshold = median_dur * cfg.speeder_threshold_pct
        mask = duration < threshold
        df.loc[mask, "_flag_speeder"] = True
        for idx in df.index[mask]:
            df.at[idx, "_flag_detail"]["speeder"] = (
                f"{duration[idx]:.0f}s vs threshold {threshold:.0f}s "
                f"(median={median_dur:.0f}s)"
            )

    # --- Straightliner ---
    if cfg.enable_straightliner:
        sl_flags, sl_detail = detect_straightlining(df, cfg.straightliner_pct, cfg.straightliner_min_items)
        df["_flag_straightliner"] = sl_flags
        for idx, scales in sl_detail.items():
            df.at[idx, "_flag_detail"]["straightliner"] = "; ".join(scales)

    # --- Duplicates ---
    if cfg.enable_duplicates:
        email_col = schema.find_column(df, "email")
        rid_col = schema.find_column(df, "rid")
        keys = []
        if email_col: keys.append(email_col)
        if rid_col: keys.append(rid_col)
        if keys:
            for key in keys:
                # Skip NaN/blank for duplicate detection
                non_blank = df[key].astype(str).str.strip().replace("", pd.NA)
                dup_mask = non_blank.duplicated(keep="first") & non_blank.notna()
                df.loc[dup_mask, "_flag_duplicate"] = True
                for idx in df.index[dup_mask]:
                    prior = df.at[idx, "_flag_detail"].get("duplicate", "")
                    new = f"duplicate {key}={df.at[idx, key]}"
                    df.at[idx, "_flag_detail"]["duplicate"] = (
                        f"{prior}; {new}" if prior else new
                    )

    # --- Aggregate ---
    flag_cols = ["_flag_status", "_flag_speeder", "_flag_straightliner", "_flag_duplicate"]
    df["_flag_any"] = df[flag_cols].any(axis=1)

    def _reasons(row):
        out = []
        if row["_flag_status"]: out.append("incomplete")
        if row["_flag_speeder"]: out.append("speeder")
        if row["_flag_straightliner"]: out.append("straightliner")
        if row["_flag_duplicate"]: out.append("duplicate")
        return ", ".join(out)

    df["_flag_reasons"] = df.apply(_reasons, axis=1)

    # Build FlagRecord list
    response_col = schema.find_column(df, "response_id") or df.columns[0]
    email_col = schema.find_column(df, "email")
    rid_col = schema.find_column(df, "rid")

    records = []
    for idx in df.index[df["_flag_any"]]:
        rec = FlagRecord(
            response_id=str(df.at[idx, response_col]),
            email=str(df.at[idx, email_col]) if email_col else "",
            rid=str(df.at[idx, rid_col]) if rid_col else "",
            reasons=df.at[idx, "_flag_reasons"].split(", "),
            detail=df.at[idx, "_flag_detail"],
        )
        records.append(rec)

    return df, records


def apply_overrides(df: pd.DataFrame, keep_response_ids: list[str]) -> pd.DataFrame:
    """
    Given the flagged dataframe and a list of response_ids the user wants to
    KEEP (override their exclusion), produce the final clean dataframe.
    """
    response_col = schema.find_column(df, "response_id") or df.columns[0]
    keep_response_ids = set(str(x) for x in keep_response_ids)

    # Keep = (not flagged) OR (response_id in override list)
    final = df[(~df["_flag_any"]) | (df[response_col].astype(str).isin(keep_response_ids))].copy()

    # Drop the internal flag columns from the clean output
    cols_to_drop = [c for c in final.columns if c.startswith("_flag_") or c == "_duration_sec"]
    final = final.drop(columns=cols_to_drop)
    return final


def build_exclusion_csv(records: list[FlagRecord], overrides_kept: set[str]) -> str:
    """Build a CSV string of the final exclusions audit (after overrides)."""
    rows = []
    for rec in records:
        if rec.response_id in overrides_kept:
            continue   # user chose to keep them
        rows.append({
            "Response ID": rec.response_id,
            "RID": rec.rid,
            "Email": rec.email,
            "Reasons": ", ".join(rec.reasons),
            "Detail": json.dumps(rec.detail, ensure_ascii=False),
            "Audit Timestamp": datetime.utcnow().isoformat() + "Z",
        })
    if not rows:
        return "Response ID,RID,Email,Reasons,Detail,Audit Timestamp\n"
    out_df = pd.DataFrame(rows)
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    return buf.getvalue()


def summary_counts(df_flagged: pd.DataFrame) -> dict:
    """Counts for the Phase 1 summary widget."""
    return {
        "total_input": int(len(df_flagged)),
        "incomplete": int(df_flagged["_flag_status"].sum()),
        "speeder": int(df_flagged["_flag_speeder"].sum()),
        "straightliner": int(df_flagged["_flag_straightliner"].sum()),
        "duplicate": int(df_flagged["_flag_duplicate"].sum()),
        "flagged_any": int(df_flagged["_flag_any"].sum()),
        "would_remain": int((~df_flagged["_flag_any"]).sum()),
    }
