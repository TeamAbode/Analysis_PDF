"""
Parse the Alchemer survey export PDF to extract case-specific metadata:
  - Case ID (from the survey title, e.g. VFGNV2040)
  - Parties (plaintiff name + defendant names)
  - County / jurisdiction
  - Fact texts (F1..F6, however many)

This is a best-effort extraction. The Phase 2 setup UI shows extracted
values pre-filled and the user can edit before committing.
"""
from __future__ import annotations
import re
import pdfplumber
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class CaseMetadata:
    case_id: str = ""
    case_caption: str = ""           # e.g. "Ward-Brodnicki v. Potato Chip Holdings / Airbnb"
    plaintiff_name: str = ""         # e.g. "Carole Ward-Brodnicki"
    defendant_names: list = field(default_factory=list)  # list of defendant entities
    county_state: str = ""           # e.g. "Clark County, NV"
    facts: list = field(default_factory=list)  # list of {"num": 1, "text": "..."}

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_text(pdf_path: str) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def parse_survey_pdf(pdf_path: str) -> CaseMetadata:
    text = _extract_text(pdf_path)
    meta = CaseMetadata()

    # --- Case ID (looks like VFGNV2040 — uppercase letters + digits, near top) ---
    m = re.search(r"\b([A-Z]{4,6}\d{3,5})\b", text)
    if m:
        meta.case_id = m.group(1)

    # --- Plaintiff name + defendants ---
    # Look for: "The Plaintiff is X" and "she is suing the Defendants who are Y"
    m = re.search(
        r"Plaintiff\s+is\s+([A-Z][A-Za-z'\-\.]+(?:\s+[A-Z][A-Za-z'\-\.]+){0,4})",
        text
    )
    if m:
        meta.plaintiff_name = m.group(1).strip().rstrip(",")

    # Defendants — match "Defendants who are X, Y, and Z" (Alchemer comma-joined)
    m = re.search(
        r"Defendants?\s+who\s+(?:are|is)\s+([^.]+?)(?:Please\s+read|\.|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        raw = m.group(1)
        # Clean: split on commas and 'and'
        raw = re.sub(r"\s+", " ", raw).strip().rstrip(",")
        # Split on ", and " or ", " or " and "
        parts = re.split(r",\s*and\s+|,\s+|\s+and\s+", raw)
        # Strip trailing numeric IDs that Alchemer sometimes appends
        cleaned = []
        for p in parts:
            p = p.strip()
            p = re.sub(r"\s+\d{3,}$", "", p)   # strip trailing question IDs
            if p:
                cleaned.append(p)
        meta.defendant_names = cleaned

    # --- County / jurisdiction ---
    m = re.search(r"Do you live in ([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?\s+County),\s+([A-Z]{2})", text)
    if m:
        meta.county_state = f"{m.group(1)}, {m.group(2)}"

    # --- Case caption ---
    # The survey PDF doesn't reliably contain the case caption ("X v. Y" format).
    # Leave blank — user enters it on the Phase 2 setup screen.

    # --- Facts ---
    # Alchemer survey PDFs format facts in a few different ways depending on
    # whether the question was renamed in the survey designer. Try multiple
    # patterns and dedupe by fact number. All patterns capture (fact_num, text).
    fact_patterns = [
        # Pattern A: original "Shortname / Alias: Fact N" layout
        re.compile(
            r"Shortname\s*/\s*Alias:\s*Fact\s*(\d+)\s*\n?\s*\d+\s*\n(.+?)(?=\*\s*\n|How important)",
            re.DOTALL | re.IGNORECASE,
        ),
        # Pattern B: fact text appearing under a bare "Fact N" header,
        # followed by the standard importance question.
        re.compile(
            r"^\s*Fact\s+(\d+)\s*\n+(.+?)(?=\n\s*(?:How important|Importance_Outcome|Shortname))",
            re.DOTALL | re.IGNORECASE | re.MULTILINE,
        ),
        # Pattern C: Alchemer "Title: Fact N" alternative
        re.compile(
            r"Title:\s*Fact\s*(\d+)\s*\n+(.+?)(?=\*\s*\n|How important|Importance_Outcome)",
            re.DOTALL | re.IGNORECASE,
        ),
    ]
    for pat in fact_patterns:
        for m in pat.finditer(text):
            num = int(m.group(1))
            fact_text = re.sub(r"\s+", " ", m.group(2)).strip()
            fact_text = fact_text.rstrip("*").strip()
            # Drop noise like trailing question prompts that leaked into the capture
            fact_text = re.sub(r"\s+How important.*$", "", fact_text, flags=re.IGNORECASE)
            fact_text = re.sub(r"\s+How would this information.*$", "", fact_text, flags=re.IGNORECASE)
            if not fact_text or len(fact_text) < 10:
                continue
            if any(f["num"] == num for f in meta.facts):
                continue   # already captured by an earlier pattern
            meta.facts.append({"num": num, "text": fact_text})

    meta.facts.sort(key=lambda f: f["num"])

    return meta
