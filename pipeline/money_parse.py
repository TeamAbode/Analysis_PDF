"""
Money parsing for open-ended damages answers.

Jurors enter award amounts in wildly inconsistent formats: clean numbers
("2560000.0"), grouped digits ("7,000,000"), digit+scale ("5 million", "100K"),
and fully spelled-out words ("Two million five hundred and sixty thousand
dollars", "nine point five million"). This module turns any of those into a
float, or returns None when the value can't be parsed confidently.

`amounts_match` compares a numeric-field answer against the spelled-out answer,
so the compensation analysis can drop respondents whose two answers disagree.
"""
from __future__ import annotations
import re
from typing import Optional

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES_THOUSAND = {"thousand", "thousands", "grand", "k"}
_SCALES_MILLION = {"million", "millions", "m", "mm", "mil"}
_SCALES_BILLION = {"billion", "billions", "b", "bn"}

# Common misspellings seen in real juror entries -> canonical token.
_TYPO_FIX = {
    "hunderd": "hundred", "hundread": "hundred", "hundered": "hundred",
    "thousaand": "thousand", "thousnd": "thousand", "thousand,": "thousand",
    "thousands": "thousand", "millon": "million", "milion": "million",
    "millione": "million", "an": "and",
}

_ZERO_WORDS = {"nothing", "none", "n/a", "na", "nil", "zero", "0", "$0", "-", ""}


def _preprocess(raw) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in _ZERO_WORDS:
        return "0"
    # Drop currency words/symbols and filler.
    s = s.replace("$", " ").replace("usd", " ").replace("dollars", " ")
    s = s.replace("dollar", " ").replace("approximately", " ").replace("approx", " ")
    s = s.replace("about", " ").replace("only", " ").replace("~", " ")
    # Put a space between a digit and an adjacent letter: "100k" -> "100 k".
    s = re.sub(r"(?<=\d)(?=[a-z])", " ", s)
    s = re.sub(r"(?<=[a-z])(?=\d)", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def _parse_pure_numeric(s: str) -> Optional[float]:
    """Parse a string with no letters: handles ',' and ' ' thousands separators,
    and a single '.' that is a decimal only when it looks like one."""
    t = s.strip()
    if not re.fullmatch(r"[\d.,\s]+", t):
        return None
    t = t.replace(" ", "")
    has_dot, has_comma = "." in t, "," in t
    if has_dot and has_comma:
        # e.g. "7.000,000" or "1,234.56": drop the thousands sep, keep decimal.
        # Assume the LAST separator type is the decimal only if it leaves <=2 digits.
        last = max(t.rfind("."), t.rfind(","))
        tail = t[last + 1:]
        if len(tail) <= 2 and t[last] in ".,":
            t = t[:last].replace(",", "").replace(".", "") + "." + tail
        else:
            t = t.replace(",", "").replace(".", "")
    elif has_comma:
        # comma decimal only if exactly 1-2 trailing digits and one comma
        if re.fullmatch(r"\d+,\d{1,2}", t):
            t = t.replace(",", ".")
        else:
            t = t.replace(",", "")
    elif has_dot:
        # dot is a thousands separator when it groups 3-digit blocks or repeats
        if t.count(".") > 1 or re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
            t = t.replace(".", "")
        # else keep as decimal
    try:
        return float(t)
    except ValueError:
        return None


def _parse_wordish(s: str) -> Optional[float]:
    tokens = [_TYPO_FIX.get(w, w) for w in re.split(r"[\s,\-]+", s) if w and w != "and"]
    result = 0.0
    current = 0.0
    used = False
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if re.fullmatch(r"\d+(\.\d+)?", t):
            current += float(t)
            used = True
        elif t in _ONES:
            current += _ONES[t]
            used = True
        elif t in _TENS:
            current += _TENS[t]
            used = True
        elif t == "hundred":
            current = (current or 1) * 100
            used = True
        elif t in _SCALES_THOUSAND:
            result += (current or 1) * 1_000
            current = 0.0
            used = True
        elif t in _SCALES_MILLION:
            result += (current or 1) * 1_000_000
            current = 0.0
            used = True
        elif t in _SCALES_BILLION:
            result += (current or 1) * 1_000_000_000
            current = 0.0
            used = True
        elif t == "point":
            # Decimal: following single-digit words are the fractional part.
            j = i + 1
            digits = ""
            while j < len(tokens):
                w = tokens[j]
                if re.fullmatch(r"\d", w):
                    digits += w
                elif w in _ONES and _ONES[w] < 10:
                    digits += str(_ONES[w])
                else:
                    break
                j += 1
            if digits:
                current += float("0." + digits)
                used = True
            i = j
            continue
        # unknown token -> ignore (filler like leftover punctuation)
        i += 1
    if not used:
        return None
    return result + current


def parse_money(raw) -> Optional[float]:
    """Parse a free-form money answer to a float, or None if unparseable."""
    s = _preprocess(raw)
    if s is None:
        return None
    if s == "0":
        return 0.0
    if re.search(r"[a-z]", s):
        val = _parse_wordish(s)
    else:
        val = _parse_pure_numeric(s)
    if val is None or val < 0:
        return None
    return float(val)


def amounts_match(numeric_raw, words_raw, rel_tol: float = 0.01) -> Optional[bool]:
    """Do the numeric-field answer and the spelled-out answer agree?

    Returns True/False, or None when at least one side can't be parsed (so the
    caller can treat 'unverifiable' distinctly from 'verified mismatch').
    """
    a = parse_money(numeric_raw)
    b = parse_money(words_raw)
    if a is None or b is None:
        return None
    if a == 0 and b == 0:
        return True
    return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1.0)
