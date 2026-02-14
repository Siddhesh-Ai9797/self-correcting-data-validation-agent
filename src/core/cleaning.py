from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pandas as pd
from dateutil import parser as dtparser
from rapidfuzz import process, fuzz

WORD_NUMS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60
}

DEPT_CANON = {
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "ai/ml": "AI/ML",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "data science": "Data Science",
    "datascience": "Data Science",
}

LOCATION_CANON = {
    "nyc": "New York",
    "new york": "New York",
    "san francisco": "San Francisco",
    "chicago": "Chicago",
    "seattle": "Seattle",
    "boston": "Boston",
}

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

@dataclass
class CleaningReport:
    rows: int
    fixes: Dict[str, int]
    warnings: List[str]

def _clean_name(s: str) -> Optional[str]:
    if s is None or pd.isna(s):
        return None
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = " ".join([w.capitalize() for w in s.split()])
    return s if s else None

def _parse_word_number(s: str) -> Optional[int]:
    if s is None:
        return None
    t = str(s).strip().lower()
    if t == "":
        return None
    if re.fullmatch(r"\d+", t):
        return int(t)
    parts = re.split(r"[\s\-]+", t)
    total = 0
    matched = False
    for p in parts:
        if p in WORD_NUMS:
            total += WORD_NUMS[p]
            matched = True
        else:
            return None
    return total if matched else None

def _clean_email(s: str) -> Tuple[str, bool]:
    if s is None:
        return "unknown@unknown.com", True
    t = str(s).strip().lower()
    if t == "":
        return "unknown@unknown.com", True
    t = t.replace(" at ", "@").replace(" dot ", ".")
    t = t.replace("..", ".")
    t = re.sub(r"@{2,}", "@", t)
    if EMAIL_RE.match(t):
        return t, (t != str(s).strip().lower())
    if "@" in t and "." not in t.split("@", 1)[1]:
        return t, False
    return t, False

def _clean_salary(s: str) -> Tuple[Optional[float], bool]:
    if s is None:
        return None, False
    t = str(s).strip()
    if t == "" or t.lower() == "nan":
        return None, False
    t2 = re.sub(r"[,$]", "", t)
    t2 = re.sub(r"usd", "", t2, flags=re.I).strip()
    try:
        return float(t2), (t2 != t)
    except ValueError:
        return None, False

def _clean_date(s: str) -> Tuple[Optional[str], bool]:
    if s is None:
        return None, False
    t = str(s).strip()
    if t == "" or t.lower() == "nan":
        return None, False
    try:
        dt = dtparser.parse(t, dayfirst=False, fuzzy=True)
        iso = dt.date().isoformat()
        return iso, (iso != t)
    except Exception:
        return None, False

def _canon_from_map(value: str, mapping: Dict[str, str], threshold: int = 90) -> Tuple[str, bool]:
    raw = (value or "").strip()
    if raw == "":
        return raw, False
    key = raw.lower()
    if key in mapping:
        canon = mapping[key]
        return canon, canon != raw
    match = process.extractOne(key, mapping.keys(), scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        canon = mapping[match[0]]
        return canon, canon != raw
    return raw, False

def clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningReport]:
    out = df.copy()
    fixes: Dict[str, int] = {}
    warnings: List[str] = []

    out.columns = [c.strip() for c in out.columns]
    col_map = {c.lower(): c for c in out.columns}

    name_col = col_map.get("name")
    age_col = col_map.get("age")
    email_col = col_map.get("email")
    salary_col = col_map.get("salary")
    join_col = col_map.get("join_date") or col_map.get("join date")
    dept_col = col_map.get("department")
    perf_col = col_map.get("performance_score") or col_map.get("performance score")
    loc_col = col_map.get("location")

    if name_col:
        before = out[name_col].astype(str).tolist()
        out[name_col] = out[name_col].apply(_clean_name)
        fixes["name_normalized"] = sum(b != a for b, a in zip(before, out[name_col].tolist()))

    if age_col:
        def clean_age(x):
            if x is None:
                return None
            t = str(x).strip().lower()
            if t == "":
                return None
            n = _parse_word_number(t)
            if n is not None:
                return n
            try:
                return int(float(t))
            except Exception:
                return None
        before = out[age_col].tolist()
        out[age_col] = out[age_col].apply(clean_age)
        fixes["age_parsed"] = sum(b != a for b, a in zip(before, out[age_col].tolist()))

    if email_col:
        before = out[email_col].tolist()
        new_vals, changed = [], 0
        for v in before:
            cleaned, did = _clean_email(v)
            if did:
                changed += 1
            new_vals.append(cleaned)
        out[email_col] = new_vals
        fixes["email_cleaned"] = changed
        invalid = [e for e in out[email_col].tolist() if not EMAIL_RE.match(e)]
        if invalid:
            warnings.append(f"{len(invalid)} email(s) still look invalid (e.g., '{invalid[0]}').")

    if salary_col:
        before = out[salary_col].tolist()
        vals, changed = [], 0
        for v in before:
            s2, did = _clean_salary(v)
            if did:
                changed += 1
            vals.append(s2)
        out[salary_col] = vals
        fixes["salary_cleaned"] = changed

    if join_col:
        before = out[join_col].tolist()
        vals, changed = [], 0
        for v in before:
            d2, did = _clean_date(v)
            if did:
                changed += 1
            vals.append(d2)
        out[join_col] = vals
        fixes["join_date_normalized"] = changed

    if dept_col:
        before = out[dept_col].astype(str).tolist()
        new, changed = [], 0
        for v in before:
            canon, did = _canon_from_map(v, DEPT_CANON, threshold=90)
            if did:
                changed += 1
            new.append(canon)
        out[dept_col] = new
        fixes["department_standardized"] = changed

    if loc_col:
        before = out[loc_col].astype(str).tolist()
        new, changed = [], 0
        for v in before:
            canon, did = _canon_from_map(v, LOCATION_CANON, threshold=88)
            if did:
                changed += 1
            new.append(canon)
        out[loc_col] = new
        fixes["location_standardized"] = changed

    if perf_col:
        before = out[perf_col].tolist()
        def clean_perf(x):
            if x is None:
                return None
            t = str(x).strip().lower()
            if t == "" or t == "nan":
                return None
            n = _parse_word_number(t)
            if n is not None:
                return float(n)
            try:
                return float(t)
            except Exception:
                return None
        out[perf_col] = out[perf_col].apply(clean_perf)
        fixes["performance_parsed"] = sum(b != a for b, a in zip(before, out[perf_col].tolist()))

    return out, CleaningReport(rows=len(out), fixes=fixes, warnings=warnings)
