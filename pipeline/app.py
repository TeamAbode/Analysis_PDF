"""
Jury Analyst Pipeline — FastAPI server.

Routes:
  GET  /                      → frontend index
  GET  /phase{1,2,3}.html     → frontend pages
  GET  /static/*              → static assets
  GET  /api/cases             → list existing cases in workspace/
  POST /api/upload            → upload CSV + optional survey PDF, returns case_id
  POST /api/phase1/preview    → run filters, return flagged rows + counts
  POST /api/phase1/commit     → write clean.csv + exclusions.csv with overrides
  GET  /api/phase1/exclusions/{case_id}  → download exclusions.csv
  GET  /api/phase2/metadata/{case_id}    → parsed survey metadata (for editing)
  POST /api/phase2/run        → run analysis; returns bundle summary
  GET  /api/phase2/bundle/{case_id}      → analysis_bundle.json
  GET  /api/phase2/chart/{case_id}/{name}.png  → chart image
  POST /api/phase3/generate   → AI sections + PDF; optional regenerate list
  GET  /api/phase3/sections/{case_id}    → ai_sections.json
  GET  /api/phase3/download/{case_id}    → final report.pdf

All per-case files live under workspace/<case_id>/.
"""
from __future__ import annotations
import json
import re
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import clean as clean_mod
from pipeline import analyze as analyze_mod
from pipeline import report as report_mod
from pipeline import survey_parse


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = BASE_DIR / "workspace"
WORKSPACE.mkdir(exist_ok=True)
FRONTEND = BASE_DIR / "frontend"
STATIC = BASE_DIR / "static"
STATIC.mkdir(exist_ok=True)


app = FastAPI(title="Jury Analyst Pipeline")

# Serve frontend and static
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
app.mount("/frontend", StaticFiles(directory=str(FRONTEND)), name="frontend")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _case_dir(case_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_\-]", "", case_id)
    if not safe:
        raise HTTPException(400, "Invalid case_id")
    p = WORKSPACE / safe
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_exists(p: Path, what: str):
    if not p.exists():
        raise HTTPException(404, f"{what} not found at {p}")


# ---------------------------------------------------------------------------
# Frontend routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index():
    return (FRONTEND / "index.html").read_text(encoding="utf-8")


@app.get("/phase1.html", response_class=HTMLResponse)
def phase1_page():
    return (FRONTEND / "phase1.html").read_text(encoding="utf-8")


@app.get("/phase2.html", response_class=HTMLResponse)
def phase2_page():
    return (FRONTEND / "phase2.html").read_text(encoding="utf-8")


@app.get("/phase2_mapping.html", response_class=HTMLResponse)
def phase2_mapping_page():
    return (FRONTEND / "phase2_mapping.html").read_text(encoding="utf-8")


@app.get("/phase3.html", response_class=HTMLResponse)
def phase3_page():
    return (FRONTEND / "phase3.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Case listing
# ---------------------------------------------------------------------------

@app.get("/api/cases")
def list_cases():
    cases = []
    for child in sorted(WORKSPACE.iterdir()):
        if not child.is_dir():
            continue
        case = {
            "case_id": child.name,
            "has_raw": (child / "raw.csv").exists(),
            "has_clean": (child / "clean.csv").exists(),
            "has_bundle": (child / "analysis_bundle.json").exists(),
            "has_report": (child / "report.pdf").exists(),
        }
        # Try to read case_caption from metadata.json if it exists
        meta_path = child / "metadata.json"
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
                case["case_caption"] = m.get("case_caption", "")
            except Exception:
                case["case_caption"] = ""
        cases.append(case)
    return {"cases": cases}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload(
    case_id: str = Form(...),
    csv: UploadFile = File(...),
    survey_pdf: Optional[UploadFile] = File(None),
):
    """Upload an Alchemer CSV and (optionally) the survey export PDF."""
    cdir = _case_dir(case_id)

    # Save raw CSV
    raw_path = cdir / "raw.csv"
    with raw_path.open("wb") as f:
        shutil.copyfileobj(csv.file, f)

    info = {"case_id": case_id, "raw_csv": str(raw_path), "rows": 0}

    # Quick row count for feedback
    try:
        import pandas as pd
        df = clean_mod.load_csv(str(raw_path))
        info["rows"] = int(len(df))
        info["cols"] = int(len(df.columns))
    except Exception as e:
        info["read_error"] = str(e)

    # If survey PDF was uploaded, save + parse + cache metadata
    if survey_pdf is not None:
        pdf_path = cdir / "survey.pdf"
        with pdf_path.open("wb") as f:
            shutil.copyfileobj(survey_pdf.file, f)
        info["survey_pdf"] = str(pdf_path)
        try:
            meta = survey_parse.parse_survey_pdf(str(pdf_path))
            meta_dict = meta.to_dict()
            # The user-supplied case_id from the form is the source of truth
            # for filesystem layout. The parsed survey case_id is informational.
            meta_dict["case_id"] = case_id
            (cdir / "metadata.json").write_text(
                json.dumps(meta_dict, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            info["metadata_parsed"] = True
        except Exception as e:
            info["metadata_parse_error"] = str(e)

    return info


# ---------------------------------------------------------------------------
# Phase 1: cleaning
# ---------------------------------------------------------------------------

class FilterPayload(BaseModel):
    case_id: str
    enable_status: bool = True
    status_keep: str = "Complete"
    enable_speeder: bool = True
    speeder_threshold_pct: float = 1/3
    enable_straightliner: bool = True
    straightliner_pct: float = 1.0
    straightliner_min_items: int = 5
    enable_duplicates: bool = True


@app.post("/api/phase1/preview")
def phase1_preview(payload: FilterPayload):
    """Run filters; return flagged-respondent list + summary counts.
    Does NOT write clean.csv yet."""
    cdir = _case_dir(payload.case_id)
    raw_path = cdir / "raw.csv"
    _ensure_exists(raw_path, "raw.csv")

    df = clean_mod.load_csv(str(raw_path))
    cfg = clean_mod.FilterConfig(
        enable_status=payload.enable_status,
        status_keep=payload.status_keep,
        enable_speeder=payload.enable_speeder,
        speeder_threshold_pct=payload.speeder_threshold_pct,
        enable_straightliner=payload.enable_straightliner,
        straightliner_pct=payload.straightliner_pct,
        straightliner_min_items=payload.straightliner_min_items,
        enable_duplicates=payload.enable_duplicates,
    )
    flagged_df, records = clean_mod.flag_all(df, cfg)
    summary = clean_mod.summary_counts(flagged_df)

    # Serialize flagged records for the UI
    flagged = [
        {
            "response_id": r.response_id,
            "rid": r.rid,
            "email": r.email,
            "reasons": r.reasons,
            "detail": r.detail,
        }
        for r in records
    ]

    return {"summary": summary, "flagged": flagged}


class CommitPayload(BaseModel):
    case_id: str
    enable_status: bool = True
    status_keep: str = "Complete"
    enable_speeder: bool = True
    speeder_threshold_pct: float = 1/3
    enable_straightliner: bool = True
    straightliner_pct: float = 1.0
    straightliner_min_items: int = 5
    enable_duplicates: bool = True
    keep_response_ids: list[str] = []   # per-person overrides


@app.post("/api/phase1/commit")
def phase1_commit(payload: CommitPayload):
    """Apply filters + overrides, write clean.csv + exclusions.csv."""
    cdir = _case_dir(payload.case_id)
    raw_path = cdir / "raw.csv"
    _ensure_exists(raw_path, "raw.csv")

    df = clean_mod.load_csv(str(raw_path))
    cfg = clean_mod.FilterConfig(
        enable_status=payload.enable_status,
        status_keep=payload.status_keep,
        enable_speeder=payload.enable_speeder,
        speeder_threshold_pct=payload.speeder_threshold_pct,
        enable_straightliner=payload.enable_straightliner,
        straightliner_pct=payload.straightliner_pct,
        straightliner_min_items=payload.straightliner_min_items,
        enable_duplicates=payload.enable_duplicates,
    )
    flagged_df, records = clean_mod.flag_all(df, cfg)

    overrides = set(payload.keep_response_ids)
    clean_df = clean_mod.apply_overrides(flagged_df, list(overrides))

    # Write clean CSV
    clean_path = cdir / "clean.csv"
    clean_df.to_csv(clean_path, index=False, encoding="utf-8")

    # Write exclusions CSV
    exclusions_csv = clean_mod.build_exclusion_csv(records, overrides_kept=overrides)
    excl_path = cdir / "exclusions.csv"
    excl_path.write_text(exclusions_csv, encoding="utf-8")

    return {
        "case_id": payload.case_id,
        "clean_rows": int(len(clean_df)),
        "excluded_rows": int(len(records) - len(overrides)),
        "overrides_applied": len(overrides),
        "clean_csv": str(clean_path),
        "exclusions_csv": str(excl_path),
    }


@app.get("/api/phase1/exclusions/{case_id}")
def download_exclusions(case_id: str):
    cdir = _case_dir(case_id)
    p = cdir / "exclusions.csv"
    _ensure_exists(p, "exclusions.csv")
    return FileResponse(str(p), media_type="text/csv", filename=f"{case_id}_exclusions.csv")


# ---------------------------------------------------------------------------
# Phase 2: analysis
# ---------------------------------------------------------------------------

@app.get("/api/phase2/metadata/{case_id}")
def get_metadata(case_id: str):
    cdir = _case_dir(case_id)
    meta_path = cdir / "metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    # No parsed metadata — return blank skeleton
    return {
        "case_id": case_id,
        "case_caption": "",
        "plaintiff_name": "",
        "defendant_names": [],
        "county_state": "",
        "facts": [],
    }


class MetadataPayload(BaseModel):
    case_id: str
    case_caption: str = ""
    plaintiff_name: str = ""
    defendant_names: list[str] = []
    county_state: str = ""
    facts: list[dict] = []
    discovery_choices: dict = {}   # {auto_id: {"include": bool, "predictor": bool}}


# ---------------------------------------------------------------------------
# Phase 2: Column mapping review (MANDATORY before analysis)
# ---------------------------------------------------------------------------

@app.get("/api/phase2/mapping/{case_id}")
def get_mapping(case_id: str):
    """
    Return the column mapping for this case. If `column_mapping.json` exists,
    return that. Otherwise auto-resolve against clean.csv and return the
    suggestion (with `auto_resolved=True`) so the UI can populate.

    Also returns:
      - `role_meta`: human labels + kinds for each role (UI rendering)
      - `column_samples`: { col_name: [up to 3 sample non-null values] }
                          ONLY for columns that look like candidates (free-text,
                          short categorical, etc.) — keeps payload small.
      - `unmapped_columns`: columns not claimed by ANY canonical or role,
                            sorted so the user can scan for missed open-ends.
    """
    from pipeline import schema as schema_mod
    import pandas as pd

    cdir = _case_dir(case_id)
    clean_path = cdir / "clean.csv"
    _ensure_exists(clean_path, "clean.csv (run Phase 1 first)")

    try:
        df = pd.read_csv(clean_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(clean_path, encoding="latin-1")

    saved = schema_mod.load_column_mapping(str(cdir))
    if saved is None:
        mapping = schema_mod.auto_resolve_all(df)
    else:
        mapping = saved

    # Role metadata (labels + kinds, for the UI)
    role_meta = {
        rid: {"label": rdef["label"], "kind": rdef["kind"],
              "multi": rdef.get("multi", False)}
        for rid, rdef in schema_mod.ROLE_DEFINITIONS.items()
    }

    # Compute "known" (claimed by canonical literal patterns or scales or facts).
    # Anything NOT in 'known' is a candidate for human assignment.
    claimed = set()
    for canon in schema_mod.CANONICAL_COLUMNS:
        c = schema_mod.find_column(df, canon)
        if c:
            claimed.add(c)
    for sname in schema_mod.SCALE_GROUPS:
        for c in schema_mod.find_scale_columns(df, sname):
            claimed.add(c)
    for n in schema_mod.detect_facts(df):
        for c in [schema_mod.fact_importance_col(df, n),
                  schema_mod.fact_influence_col(df, n),
                  schema_mod.fact_reasoning_col(df, n)]:
            if c:
                claimed.add(c)

    # Light "system" filter (mirrors auto_discover's SYSTEM_COL_PATTERNS but
    # we duplicate the minimal subset here so the mapping UI doesn't need to
    # import auto_discover).
    import re
    SYS = [r"^Response ID$", r"^Status$", r"^Time Started$", r"^Time Spent$",
           r"^Date Submitted$", r"^Contact ID$", r"^SessionID$", r"^IP Address$",
           r"^Country$", r"^Region$", r"^State/Region$", r"^City$", r"^Postal$",
           r"^Latitude$", r"^Longitude$", r"^User Agent$", r"^Referer$",
           r"^Tags$", r"^Legacy Comments$", r"^Comments$", r"^Language$",
           r"^Survey Done$", r"^Group$", r"^PID$", r"^RID$",
           r"^Validity_Check\d?$", r"^18OrOlder$", r"^consent$", r"^Consent$",
           r"^County$", r"jury summons", r"^New Page Timer$", r"^New URL Redirect$"]
    def _is_sys(col):
        return any(re.search(p, col, re.IGNORECASE) for p in SYS)

    unmapped = []
    column_samples = {}
    for col in df.columns:
        if col in claimed or _is_sys(col):
            continue
        # Sample up to 3 non-null, non-empty values (truncated)
        s = df[col].dropna().astype(str).str.strip()
        s = s[s != ""]
        if len(s) == 0:
            continue
        unmapped.append(col)
        column_samples[col] = [v[:150] for v in s.head(3).tolist()]

    return {
        "case_id": case_id,
        "mapping": mapping,
        "role_meta": role_meta,
        "unmapped_columns": unmapped,
        "column_samples": column_samples,
    }


class MappingPayload(BaseModel):
    case_id: str
    # roles: { role_id: {"columns": [col, ...]} }
    roles: dict


@app.post("/api/phase2/mapping/{case_id}")
def save_mapping(case_id: str, payload: MappingPayload):
    """Persist the user-confirmed column mapping for this case."""
    from pipeline import schema as schema_mod

    cdir = _case_dir(case_id)
    clean_path = cdir / "clean.csv"
    _ensure_exists(clean_path, "clean.csv (run Phase 1 first)")

    # Normalize the payload to the on-disk shape, marking entries as
    # manually confirmed.
    roles_clean = {}
    for role_id, entry in (payload.roles or {}).items():
        cols = list(entry.get("columns", []))
        roles_clean[role_id] = {
            "columns": cols,
            "confidence": 1.0 if cols else 0.0,
            "manual": True,
        }
    mapping = {"roles": roles_clean, "auto_resolved": False}
    schema_mod.save_column_mapping(str(cdir), mapping)
    return {"case_id": case_id, "ok": True}


@app.get("/api/phase2/discover/{case_id}")
def discover_extras(case_id: str):
    """Preview the auto-discovered questions for the case's clean.csv
    WITHOUT running a full analysis. Used by the Phase 2 UI to populate
    the 'extra questions found' checkbox panel."""
    from pipeline import auto_discover
    import pandas as pd

    cdir = _case_dir(case_id)
    clean_path = cdir / "clean.csv"
    _ensure_exists(clean_path, "clean.csv (run Phase 1 first)")

    try:
        df = pd.read_csv(clean_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(clean_path, encoding="latin-1")

    records = auto_discover.discover_extra_questions(df, str(cdir))
    # Light response — strip heavy verbatim 'responses' field; UI doesn't need them yet
    light = []
    for r in records:
        rc = {k: v for k, v in r.items() if k != "responses"}
        light.append(rc)
    return {"discovered": light}


@app.post("/api/phase2/run")
def phase2_run(payload: MetadataPayload):
    """Save metadata + run analysis. Returns the analysis bundle.

    Requires a confirmed column mapping (column_mapping.json with
    auto_resolved=False). The Phase 2 UI must POST /api/phase2/mapping/{id}
    before calling this. We treat that as a hard gate so silently-missing
    open-end columns can never sneak through again.
    """
    from pipeline import schema as schema_mod

    cdir = _case_dir(payload.case_id)
    clean_path = cdir / "clean.csv"
    _ensure_exists(clean_path, "clean.csv (run Phase 1 first)")

    # MANDATORY: confirmed mapping must exist
    mapping = schema_mod.load_column_mapping(str(cdir))
    if mapping is None or mapping.get("auto_resolved", True):
        raise HTTPException(
            status_code=412,   # Precondition Failed
            detail=("Column mapping not confirmed. Open the Phase 2 mapping "
                    "review and click Confirm before running analysis."),
        )

    meta_dict = payload.model_dump()
    # Don't store discovery_choices inside metadata.json (it's a transient run-time choice)
    discovery_choices = meta_dict.pop("discovery_choices", {}) or {}
    (cdir / "metadata.json").write_text(
        json.dumps(meta_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Persist the most recent discovery choices alongside for reproducibility
    (cdir / "discovery_choices.json").write_text(
        json.dumps(discovery_choices, indent=2),
        encoding="utf-8",
    )

    bundle = analyze_mod.run_analysis(
        str(clean_path), meta_dict, str(cdir),
        discovery_choices=discovery_choices,
    )
    return {
        "case_id": payload.case_id,
        "bundle_path": str(cdir / "analysis_bundle.json"),
        "n_respondents": bundle["n_respondents"],
        "summary": {
            "pct_plaintiff_leaning": bundle["verdict"]["pct_plaintiff_leaning"],
            "pct_liable_yes": bundle["verdict"]["pct_liable_yes"],
            "mean_award_all": bundle["compensation"]["mean_award_all"],
            "median_award_all": bundle["compensation"]["median_award_all"],
            "n_facts": len(bundle["facts"]),
            "n_discovered": len(bundle.get("discovered_questions", [])),
        },
        "discovered_questions": bundle.get("discovered_questions_all", []),
    }


@app.get("/api/phase2/bundle/{case_id}")
def get_bundle(case_id: str):
    cdir = _case_dir(case_id)
    p = cdir / "analysis_bundle.json"
    _ensure_exists(p, "analysis_bundle.json")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/phase2/chart/{case_id}/{name}")
def get_chart(case_id: str, name: str):
    if not re.match(r"^[A-Za-z0-9_\-\.]+$", name):
        raise HTTPException(400, "Invalid chart name")
    cdir = _case_dir(case_id)
    p = cdir / "charts" / name
    _ensure_exists(p, f"chart {name}")
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/phase2/chart/{case_id}/auto/{name}")
def get_auto_chart(case_id: str, name: str):
    if not re.match(r"^[A-Za-z0-9_\-\.]+$", name):
        raise HTTPException(400, "Invalid chart name")
    cdir = _case_dir(case_id)
    p = cdir / "charts" / "auto" / name
    _ensure_exists(p, f"auto chart {name}")
    return FileResponse(str(p), media_type="image/png")


# ---------------------------------------------------------------------------
# Phase 3: report
# ---------------------------------------------------------------------------

class ReportPayload(BaseModel):
    case_id: str
    regenerate_sections: Optional[list[str]] = None  # None = all


@app.post("/api/phase3/generate")
def phase3_generate(payload: ReportPayload):
    """Run AI prose generation + render PDF.
    If regenerate_sections is set, only those keys are regenerated; others
    are reused from ai_sections.json if it exists."""
    cdir = _case_dir(payload.case_id)
    _ensure_exists(cdir / "analysis_bundle.json", "analysis_bundle.json (run Phase 2 first)")

    ai_sections = report_mod.run_report(str(cdir), regenerate_sections=payload.regenerate_sections)
    return {
        "case_id": payload.case_id,
        "pdf_path": str(cdir / "report.pdf"),
        "section_keys": list(ai_sections.keys()),
    }


@app.get("/api/phase3/sections/{case_id}")
def get_sections(case_id: str):
    cdir = _case_dir(case_id)
    p = cdir / "ai_sections.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/phase3/overrides/{case_id}")
def get_overrides(case_id: str):
    """Return the saved-edits payload (edits dict + skip list) if it exists.
    Returns {} when no overrides have been saved yet."""
    cdir = _case_dir(case_id)
    p = cdir / "report_overrides.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


class OverridesPayload(BaseModel):
    edits: dict = {}     # {section_key: edited_text}
    skip: list[str] = []  # section keys to omit from PDF


@app.post("/api/phase3/overrides/{case_id}")
def save_overrides_and_rerender(case_id: str, payload: OverridesPayload):
    """Persist user edits + skip flags to disk, then re-render the PDF using them.
    This does NOT call any AI; it just rebuilds the PDF from the existing AI
    sections, applying the user's edits and skip list."""
    cdir = _case_dir(case_id)
    overrides = {
        "edits": payload.edits or {},
        "skip":  list(payload.skip or []),
    }
    (cdir / "report_overrides.json").write_text(
        json.dumps(overrides, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Re-render the PDF using the existing analysis bundle + AI sections,
    # passing the overrides through to build_report_context.
    report_mod.run_report(str(cdir), regenerate_sections=[])   # [] = no AI calls; just rebuild
    return {"case_id": case_id, "ok": True,
            "skipped": overrides["skip"],
            "n_edits": len(overrides["edits"])}


@app.get("/api/phase3/download/{case_id}")
def download_report(case_id: str):
    cdir = _case_dir(case_id)
    p = cdir / "report.pdf"
    _ensure_exists(p, "report.pdf")
    caption = case_id
    meta_path = cdir / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("case_caption"):
                # safe filename
                caption = re.sub(r"[^A-Za-z0-9_\-]", "_", meta["case_caption"])[:60]
        except Exception:
            pass
    return FileResponse(str(p), media_type="application/pdf",
                        filename=f"{case_id}_{caption}_Report.pdf")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Jury Analyst Pipeline — http://localhost:8765")
    print("=" * 60)
    uvicorn.run("app:app", host="127.0.0.1", port=8765, reload=False)
