#!/usr/bin/env python3
"""
Banner search test utility.

Purpose
- Test Banner class-search behavior independently from the Tkinter app.
- Show term-code resolution, request inputs, and matching section rows.
- Help diagnose why a given term/course query does or does not return meetings.

Functional interface
1) Provide a term label (for example: Fall 2026 (Full Term)).
2) Provide a course query (for example: 2181L or ECGR 2181L).
3) The script resolves the Banner term code, runs search requests, filters results,
   and prints a readable summary of matching sections and meeting times.

Examples
- python banner_search_test.py --term "Fall 2026 (Full Term)" --query "2181L"
- python banner_search_test.py --term "Fall 2026" --query "ECGR 2181L"
- python banner_search_test.py --term "Fall 2026" --subject ECGR --number 2181L --debug
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Dict, List, Optional, Tuple

import requests

BANNER_BASE_URL = "https://selfservice.uncc.edu/StudentRegistrationSsb/ssb"

BANNER_TERM_SUFFIX_BY_SEASON = {
    "SPRING": "20",
    "SUMMER": "50",
    "FALL": "80",
    "AUTUMN": "80",
    "WINTER": "10",
}

TERM_LIKE_SUBJECTS = {
    "FALL",
    "SPRING",
    "SUMMER",
    "WINTER",
    "AUTUMN",
    "TERM",
    "SESSION",
}


def normalize_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", str(value or "").upper()).strip()


def guess_banner_term_code(term_name: str) -> Optional[str]:
    """
    Deterministic fallback for UNCC Banner term codes.

    Handles:
    - Fall 2026 (Full Term)
    - 2026 Fall
    - direct term-code strings like 202680
    """
    normalized = normalize_text(term_name)

    direct_code = re.search(r"\b(19\d{4}|20\d{4})\b", normalized)
    if direct_code:
        code = direct_code.group(1)
        if code[-2:] in {"10", "20", "50", "80"}:
            return code

    season = ""
    year = ""

    match = re.search(r"\b(SPRING|SUMMER|FALL|AUTUMN|WINTER)\s+(19\d{2}|20\d{2})\b", normalized)
    if match:
        season = match.group(1)
        year = match.group(2)
    else:
        match = re.search(r"\b(19\d{2}|20\d{2})\s+(SPRING|SUMMER|FALL|AUTUMN|WINTER)\b", normalized)
        if match:
            year = match.group(1)
            season = match.group(2)

    if not season or not year:
        return None

    suffix = BANNER_TERM_SUFFIX_BY_SEASON.get(season)
    if not suffix:
        return None
    return f"{year}{suffix}"


def parse_query(query: str) -> Tuple[str, str]:
    """
    Parse subject/number from a course query.

    Returns (subject, number), where subject may be empty when only a number is present.
    """
    candidate = normalize_text(query)
    if not candidate:
        raise ValueError("Query is empty.")

    # Prefer standalone 4-digit(+optional letter) number tokens like 2181L.
    number_only_match = re.search(r"\b(\d{4}[A-Z]?)\b", candidate)
    if number_only_match:
        number = number_only_match.group(1)
        year_like = bool(re.fullmatch(r"(19|20)\d{2}", number))
        if not year_like:
            subject_match = re.search(rf"\b([A-Z]{{2,10}})\s+{re.escape(number)}\b", candidate)
            if subject_match:
                subject = subject_match.group(1)
                if subject not in TERM_LIKE_SUBJECTS:
                    return subject, number
            return "", number

    full_match = re.search(r"\b([A-Z]{2,10})\s+([0-9]{3,4}[A-Z]?)\b", candidate)
    if full_match:
        subject = full_match.group(1)
        number = full_match.group(2)
        if subject not in TERM_LIKE_SUBJECTS:
            return subject, number

    raise ValueError(f"Could not parse query: {query}")


class BannerSearchClient:
    """
    Minimal Banner client for class-search diagnostics.

    Interface
    - resolve_term_code(term_name): tries Banner term endpoints, then deterministic fallback
    - search(subject, number, term_code): posts class-search payload and returns raw rows
    """

    def __init__(self, base_url: str = BANNER_BASE_URL, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _warm_up(self) -> None:
        self.session.get(self._url("/term/termSelection"), params={"mode": "search"}, timeout=self.timeout)

    def _fetch_terms(self) -> List[dict]:
        self._warm_up()

        # NOTE: the real Banner UI calls classSearch/getTerms (not term/getTerms).
        # term/getTerms consistently 404s; classSearch/getTerms is the working path.
        candidates = [
            (self._url("/classSearch/getTerms"), {"searchTerm": "", "offset": 1, "max": 50}),
            (self._url("/classSearch/getTerms"), {"mode": "search"}),
        ]

        for url, params in candidates:
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
            except Exception:
                continue

            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            if isinstance(data, dict):
                rows = data.get("data") or data.get("terms") or data.get("results") or data.get("rows") or []
                if isinstance(rows, list):
                    return [x for x in rows if isinstance(x, dict)]

        return []

    def _select_term(self, term_code: str) -> None:
        """
        Critical step: Banner's search results endpoint reads the active term from
        server-side session state, not from the txt_term field alone. The real
        Self-Service UI POSTs to /term/search to establish that state before
        calling searchResults. Skipping this step causes searchResults to
        silently return zero rows (no error, just an empty data list).
        """
        resp = self.session.post(
            self._url("/term/search"),
            params={"mode": "search"},
            data={"term": term_code},
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def resolve_term_code(self, term_name: str) -> Tuple[Optional[str], str]:
        normalized_target = normalize_text(term_name)
        terms = self._fetch_terms()

        for term in terms:
            code = str(term.get("code") or term.get("value") or term.get("term") or term.get("id") or "").strip()
            desc = normalize_text(str(term.get("description") or term.get("termDesc") or term.get("label") or ""))
            if not code:
                continue
            if normalized_target and (normalized_target == desc or normalized_target in desc or desc in normalized_target):
                return code, "term-endpoint"
            if normalized_target and all(token in desc for token in normalized_target.split()):
                return code, "term-endpoint-token"

        guessed = guess_banner_term_code(term_name)
        if guessed:
            return guessed, "season-year-fallback"

        return None, "unresolved"

    def _payload(self, subject: str, number: str, term_code: str) -> List[Tuple[str, str]]:
        payload = [
            ("txt_courseNumber", number),
            ("txt_term", term_code),
            ("pageOffset", "0"),
            ("pageMaxSize", "500"),
            ("sortColumn", "subjectDescription"),
            ("sortDirection", "asc"),
            ("sortOrder", "asc"),
            ("chk_openOnly", "false"),
        ]
        if subject:
            payload.insert(0, ("txt_subject", subject))
        return payload

    def search(self, subject: str, number: str, term_code: str) -> List[dict]:
        self._warm_up()
        self._select_term(term_code)
        resp = self.session.post(
            self._url("/searchResults/searchResults"),
            data=self._payload(subject, number, term_code),
            timeout=self.timeout,
        )
        resp.raise_for_status()

        try:
            data = resp.json()
        except Exception:
            text = resp.text
            match = re.search(r"window\.sectionResultsCollection\s*=\s*(\{.*?\});", text, re.DOTALL)
            if not match:
                raise RuntimeError("Search response was not JSON and did not contain sectionResultsCollection.")
            data = json.loads(match.group(1))

        if isinstance(data, dict):
            for key in ("data", "rows", "results", "items"):
                rows = data.get(key)
                if isinstance(rows, list):
                    return [x for x in rows if isinstance(x, dict)]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        return []


def extract_meeting_lines(row: dict) -> List[str]:
    lines: List[str] = []
    for entry in row.get("meetingsFaculty", []) or []:
        if not isinstance(entry, dict):
            continue
        mt = entry.get("meetingTime")
        meeting_list = mt if isinstance(mt, list) else [mt] if isinstance(mt, dict) else []
        for m in meeting_list:
            days = "".join([
                "M" if m.get("monday") else "",
                "T" if m.get("tuesday") else "",
                "W" if m.get("wednesday") else "",
                "R" if m.get("thursday") else "",
                "F" if m.get("friday") else "",
                "S" if m.get("saturday") else "",
                "U" if m.get("sunday") else "",
            ]) or "(no days)"
            begin = str(m.get("beginTime") or "----")
            end = str(m.get("endTime") or "----")
            bldg = str(m.get("building") or "")
            room = str(m.get("room") or "")
            lines.append(f"{days} {begin}-{end} {bldg} {room}".strip())
    return lines


def filter_rows(rows: List[dict], subject: str, number: str) -> List[dict]:
    filtered: List[dict] = []
    subj = subject.upper().strip()
    num = number.upper().strip()

    for row in rows:
        row_num = str(row.get("courseNumber", "")).upper().strip()
        if row_num != num:
            continue
        if subj:
            row_subj = str(row.get("subject", "")).upper().strip()
            if row_subj != subj:
                continue
        filtered.append(row)

    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Banner class search outside the GUI.")
    parser.add_argument("--term", required=True, help="Semester label, e.g. 'Fall 2026 (Full Term)'")
    parser.add_argument("--query", required=True, help="Course query, e.g. '2181L' or 'ECGR 2181L'")
    parser.add_argument("--subject", default="", help="Optional explicit subject override, e.g. ECGR")
    parser.add_argument("--number", default="", help="Optional explicit course number override, e.g. 2181L")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    parser.add_argument("--debug", action="store_true", help="Print extra request/row diagnostics")
    parser.add_argument("--max-print", type=int, default=10, help="Max result rows to print")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.subject and args.number:
        subject = args.subject.strip().upper()
        number = args.number.strip().upper()
    else:
        subject, number = parse_query(args.query)

    client = BannerSearchClient(timeout=args.timeout)

    term_code, term_source = client.resolve_term_code(args.term)
    if not term_code:
        print("ERROR: Could not resolve Banner term code from:", args.term)
        return 2

    print("Term label:", args.term)
    print("Resolved term code:", term_code, f"({term_source})")
    print("Course query:", args.query)
    print("Parsed subject:", subject or "(none)")
    print("Parsed number:", number)
    print()

    if args.debug:
        print("DEBUG payload preview:", client._payload(subject, number, term_code))
        print()

    rows = client.search(subject, number, term_code)
    filtered = filter_rows(rows, subject, number)

    print("Raw rows returned:", len(rows))
    print("Filtered rows matched:", len(filtered))
    print()

    if not filtered:
        print("No matching rows found.")
        return 1

    for idx, row in enumerate(filtered[: max(1, args.max_print)], start=1):
        crn = row.get("courseReferenceNumber")
        subj = row.get("subject")
        num = row.get("courseNumber")
        seq = row.get("sequenceNumber")
        title = row.get("courseTitle")
        print(f"[{idx}] CRN={crn}  {subj} {num}-{seq}  {title}")
        for line in extract_meeting_lines(row):
            print("    meeting:", line)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
