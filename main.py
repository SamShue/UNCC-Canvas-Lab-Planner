#!/usr/bin/env python3
"""
Canvas Lab Planner (Tkinter)
- Loads Canvas assignments and course sections
- Scrapes UNCC registrar academic calendar section to determine:
  - first day of classes
  - last day of classes (best-effort)
  - "no classes / university closed" date ranges (best-effort)
- Lets user define each section meeting pattern (days + start time)
- Auto-assigns due dates for assignments matching:
    Pre-Lab <N>, Lab <N>, Post-Lab <N>
  with rules:
    Pre-Lab: due at first meeting time of assigned week
    Lab: due at due_time_hhmm on that meeting day
    Post-Lab: due at end of week (Sunday) at due_time_hhmm
- Delay weeks shifts Lab 1 to week_after_first_week + delay
- Skips "closed/no classes" full weeks when mapping lab numbers to weeks
- Applies due dates via Canvas assignment overrides (per section)

Cross-platform:
- No Windows-only activation steps
- Uses requests + BeautifulSoup4; install deps if needed:
    pip install requests beautifulsoup4
- For Windows timezone keys with zoneinfo, you may need:
    pip install tzdata
"""

from __future__ import annotations

import configparser
import os
import datetime as dt
from difflib import SequenceMatcher
import re
import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# Third-party
try:
    import requests
except ImportError:
    raise SystemExit("Missing dependency: requests. Install with: pip install requests")

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise SystemExit("Missing dependency: beautifulsoup4. Install with: pip install beautifulsoup4")

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except Exception as ex:
    raise SystemExit(f"Your Python does not support zoneinfo properly: {ex}")


# -----------------------------
# Utilities
# -----------------------------

DAY_TOKEN_TO_WEEKDAY = {
    "M": 0,
    "MON": 0,
    "MONDAY": 0,
    "T": 1,
    "TU": 1,
    "TUE": 1,
    "TUES": 1,
    "TUESDAY": 1,
    "W": 2,
    "WED": 2,
    "WEDS": 2,
    "WEDNESDAY": 2,
    "R": 3,     # common academic shorthand for Thursday
    "TH": 3,
    "THU": 3,
    "THUR": 3,
    "THURS": 3,
    "THURSDAY": 3,
    "F": 4,
    "FRI": 4,
    "FRIDAY": 4,
    "S": 5,     # Saturday
    "SAT": 5,
    "SATURDAY": 5,
    "U": 6,     # Sunday
    "SUN": 6,
    "SUNDAY": 6,
}


def parse_hhmm(s: str) -> dt.time:
    s = s.strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if not m:
        raise ValueError(f"Time must be HH:MM (24-hour). Got: {s}")
    hh = int(m.group(1))
    mm = int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"Invalid time: {s}")
    return dt.time(hour=hh, minute=mm)


def safe_zoneinfo(tz_key: str) -> ZoneInfo:
    """
    On Windows, ZoneInfo may raise ZoneInfoNotFoundError unless tzdata is installed.
    """
    try:
        return ZoneInfo(tz_key)
    except ZoneInfoNotFoundError as e:
        # Give a clean, actionable message
        msg = (
            f"Timezone '{tz_key}' was not found by zoneinfo.\n\n"
            "On Windows, Python often needs the IANA tz database installed.\n"
            "Fix: install tzdata:\n\n"
            "    pip install tzdata\n\n"
            "Then re-run the tool.\n\n"
            f"Original error: {e}"
        )
        raise ZoneInfoNotFoundError(msg) from e


def isoformat_z(dt_aware: dt.datetime) -> str:
    # Canvas expects ISO 8601 with offset, e.g. 2026-01-15T23:59:00-05:00
    return dt_aware.isoformat(timespec="seconds")


def monday_of_week(d: dt.date) -> dt.date:
    return d - dt.timedelta(days=d.weekday())


def sunday_of_week(d: dt.date) -> dt.date:
    return d + dt.timedelta(days=(6 - d.weekday()))


def overlaps_days(a0: dt.date, a1: dt.date, b0: dt.date, b1: dt.date) -> int:
    """
    Inclusive overlap day count between [a0,a1] and [b0,b1].
    """
    start = max(a0, b0)
    end = min(a1, b1)
    if end < start:
        return 0
    return (end - start).days + 1


def normalize_banner_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()


TERM_LIKE_SUBJECTS = {
    "FALL",
    "SPRING",
    "SUMMER",
    "WINTER",
    "AUTUMN",
    "TERM",
    "SESSION",
}

YEAR_MIN = 1900
YEAR_MAX = 2099

BANNER_TERM_SUFFIX_BY_SEASON = {
    "SPRING": "20",
    "SUMMER": "50",
    "FALL": "80",
    "AUTUMN": "80",
    "WINTER": "10",
}


def _looks_like_term_marker(subject: str, number: str) -> bool:
    if subject in TERM_LIKE_SUBJECTS:
        return True
    if re.fullmatch(r"(19|20)\d{2}[A-Z]?", number):
        return subject in TERM_LIKE_SUBJECTS or subject.endswith("TERM")
    return False


def _is_year_like_number(number: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", number)) and YEAR_MIN <= int(number) <= YEAR_MAX


def _extract_banner_course_number(value: str) -> Optional[str]:
    candidate = str(value or "").upper()
    if not candidate:
        return None

    # Prefer standalone 4-digit(+optional letter) tokens like 2181L.
    for token in re.findall(r"\b(\d{4}[A-Z]?)\b", candidate):
        if _is_year_like_number(token):
            continue
        return token
    return None


def guess_banner_term_code(term_name: str) -> Optional[str]:
    normalized = normalize_banner_text(term_name)

    # Prefer a direct Banner-style 6-digit term code if present (e.g., 202680).
    direct_code = re.search(r"\b(19\d{4}|20\d{4})\b", normalized)
    if direct_code:
        code = direct_code.group(1)
        if code[-2:] in {"10", "20", "50", "80"}:
            return code

    season = ""
    year = ""

    # Common format: FALL 2026 (FULL TERM)
    match = re.search(r"\b(SPRING|SUMMER|FALL|AUTUMN|WINTER)\s+(19\d{2}|20\d{2})\b", normalized)
    if match:
        season = match.group(1)
        year = match.group(2)
    else:
        # Alternate format: 2026 FALL (FULL TERM)
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


def parse_banner_course_code(course: CourseOption) -> Optional[Tuple[str, str]]:
    return parse_banner_course_query(course.course_code) or parse_banner_course_query(course.name)


def parse_banner_course_query(value: str) -> Optional[Tuple[str, str]]:
    candidate = str(value or "").strip().upper()
    if not candidate:
        return None

    # If we can find a plain course number, use that first; subject is optional.
    number_only = _extract_banner_course_number(candidate)
    if number_only:
        subject_match = re.search(rf"\b([A-Z]{{2,10}})\s*[- ]\s*{re.escape(number_only)}\b", candidate)
        if subject_match and not _looks_like_term_marker(subject_match.group(1), number_only):
            return subject_match.group(1), number_only
        return "", number_only

    matches = re.finditer(r"\b([A-Z]{2,10})\s*[- ]?\s*([0-9]{3,4}[A-Z]?)\b", candidate)
    for match in matches:
        subject = match.group(1)
        number = match.group(2)
        if _looks_like_term_marker(subject, number):
            continue
        if _is_year_like_number(number):
            continue
        return subject, number

    return None


def parse_banner_time(value: object) -> Optional[dt.time]:
    value = str(value or "").strip()
    if not re.fullmatch(r"\d{4}", value):
        return None
    hour = int(value[:2])
    minute = int(value[2:])
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return dt.time(hour=hour, minute=minute)
    return None


def banner_weekdays_from_meeting_time(meeting_time: dict) -> List[int]:
    day_fields = [
        (0, "monday"),
        (1, "tuesday"),
        (2, "wednesday"),
        (3, "thursday"),
        (4, "friday"),
        (5, "saturday"),
        (6, "sunday"),
    ]
    weekdays: List[int] = []
    for weekday, key in day_fields:
        if meeting_time.get(key) or meeting_time.get(f"{key}Flag") or meeting_time.get(f"{key}_flag"):
            weekdays.append(weekday)
    return weekdays


# -----------------------------
# Data models
# -----------------------------

@dataclass
class CanvasConfig:
    base_url: str
    api_key: str
    course_id: Optional[str] = None


@dataclass
class SemesterConfig:
    calendar_url: str
    anchor: str


@dataclass
class TimeConfig:
    timezone: str
    due_time: dt.time


@dataclass
class CourseOption:
    id: int
    name: str
    course_code: str = ""
    term_name: str = ""
    workflow_state: str = ""

    def label(self) -> str:
        parts = [self.name]
        if self.course_code:
            parts.append(self.course_code)
        if self.term_name:
            parts.append(self.term_name)
        parts.append(f"ID {self.id}")
        return " | ".join(part for part in parts if part)


@dataclass
class SectionMeeting:
    section_id: int
    section_name: str
    weekdays: List[int]      # 0=Mon
    start_time: dt.time      # local


@dataclass
class SemesterCalendar:
    classes_begin: dt.date
    classes_end: Optional[dt.date]
    # Ranges that indicate "no classes / university closed" etc.
    closures: List[Tuple[dt.date, dt.date]]  # inclusive
    # Optional human-friendly term name or heading parsed from the page
    term_name: Optional[str] = None

    def instructional_weeks(self) -> List[dt.date]:
        """
        Returns list of Mondays (week anchors) for instructional weeks.
        Skips "closure weeks" that appear to cancel most weekdays.
        """
        start_monday = monday_of_week(self.classes_begin)
        end_date = self.classes_end or (self.classes_begin + dt.timedelta(days=7 * 16))
        end_monday = monday_of_week(end_date)

        mondays: List[dt.date] = []
        cur = start_monday
        while cur <= end_monday:
            week_start = cur
            week_end = cur + dt.timedelta(days=6)

            # Determine if this week should be skipped:
            # skip if closures cover >= 1 weekday of this week (Mon-Fri)
            weekdays_covered = 0
            for c0, c1 in self.closures:
                # Only count overlap with Mon-Fri
                wk_mon = week_start
                wk_fri = week_start + dt.timedelta(days=4)
                overlap = overlaps_days(wk_mon, wk_fri, c0, c1)
                weekdays_covered = max(weekdays_covered, overlap)

            if weekdays_covered >= 1:
                # treat as "week cancelled" when any weekday is affected
                pass
            else:
                mondays.append(week_start)

            cur += dt.timedelta(days=7)

        return mondays

    def skipped_weeks(self) -> List[Tuple[dt.date, dt.date, List[Tuple[dt.date, dt.date]]]]:
        """
        Returns list of skipped weeks as tuples: (week_monday, week_sunday, list_of_closure_ranges)
        A week is considered skipped if closures cover >= 1 weekday (Mon-Fri) of that week.
        """
        start_monday = monday_of_week(self.classes_begin)
        end_date = self.classes_end or (self.classes_begin + dt.timedelta(days=7 * 16))
        end_monday = monday_of_week(end_date)

        skipped: List[Tuple[dt.date, dt.date, List[Tuple[dt.date, dt.date]]]] = []
        cur = start_monday
        while cur <= end_monday:
            wk_mon = cur
            wk_fri = cur + dt.timedelta(days=4)
            overlaps: List[Tuple[dt.date, dt.date]] = []
            max_overlap = 0
            for c0, c1 in self.closures:
                overlap = overlaps_days(wk_mon, wk_fri, c0, c1)
                if overlap > 0:
                    overlaps.append((c0, c1))
                    max_overlap = max(max_overlap, overlap)

            if max_overlap >= 1:
                wk_sun = cur + dt.timedelta(days=6)
                skipped.append((wk_mon, wk_sun, overlaps))

            cur += dt.timedelta(days=7)

        return skipped


@dataclass
class Assignment:
    id: int
    name: str
    overrides: List[dict] = None  # Store assignment overrides with due dates
    
    def __post_init__(self):
        if self.overrides is None:
            self.overrides = []


@dataclass
class DueSuggestion:
    assignment_id: int
    assignment_name: str
    # per section due datetime
    due_by_section: Dict[int, dt.datetime]


# -----------------------------
# Registrar scraping (best-effort)
# -----------------------------

MONTHS = {
    "JANUARY": 1, "JAN": 1,
    "FEBRUARY": 2, "FEB": 2,
    "MARCH": 3, "MAR": 3,
    "APRIL": 4, "APR": 4,
    "MAY": 5,
    "JUNE": 6, "JUN": 6,
    "JULY": 7, "JUL": 7,
    "AUGUST": 8, "AUG": 8,
    "SEPTEMBER": 9, "SEP": 9, "SEPT": 9,
    "OCTOBER": 10, "OCT": 10,
    "NOVEMBER": 11, "NOV": 11,
    "DECEMBER": 12, "DEC": 12,
}


def parse_date_like(s: str) -> Optional[dt.date]:
    """
    Parse common registrar formats (best-effort):
      - January 13, 2026
      - Jan 13, 2026
      - 1/13/2026
      - 13 Jan 2026 (rare)
    """
    s = s.strip().replace("\u00a0", " ")
    # MM/DD/YYYY
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{4})", s)
    if m:
        mm, dd, yyyy = map(int, m.groups())
        return dt.date(yyyy, mm, dd)

    # Month DD, YYYY
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})", s)
    if m:
        mon = MONTHS.get(m.group(1).upper())
        if mon:
            return dt.date(int(m.group(3)), mon, int(m.group(2)))

    # DD Month YYYY
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s)
    if m:
        mon = MONTHS.get(m.group(2).upper())
        if mon:
            return dt.date(int(m.group(3)), mon, int(m.group(1)))

    return None


def parse_date_range(s: str) -> Optional[Tuple[dt.date, dt.date]]:
    """
    Handles:
      - January 13, 2026
      - January 13-17, 2026
      - January 13 - January 17, 2026
      - 1/13/2026 - 1/17/2026
    """
    # Normalize whitespace and common dash characters; strip parenthetical notes
    s = s.strip().replace("\u00a0", " ")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    # remove parenthetical notes like '(Spring Recess)'
    s = re.sub(r"\(.*?\)", "", s)
    s = " ".join(s.split())
    # Try split on '-'
    if "-" not in s:
        d = parse_date_like(s)
        if d:
            return (d, d)
        return None

    parts = [p.strip() for p in s.split("-")]
    if len(parts) != 2:
        return None

    left, right = parts

    # Case: "January 13-17, 2026" => left has month+day, right has day+year
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})", left)
    m2 = re.fullmatch(r"(\d{1,2}),\s*(\d{4})", right)
    if m and m2:
        mon = MONTHS.get(m.group(1).upper())
        if mon:
            y = int(m2.group(2))
            d0 = dt.date(y, mon, int(m.group(2)))
            d1 = dt.date(y, mon, int(m2.group(1)))
            if d1 >= d0:
                return (d0, d1)

    # Otherwise parse both sides as full dates (or right may omit year)
    d0 = parse_date_like(left)
    d1 = parse_date_like(right)

    # If right omits year but includes Month DD, infer year from left
    if d0 and not d1:
        # try Month DD, YYYY stripped? already tried; now try Month DD (no year)
        m = re.fullmatch(r"([A-Za-z]+)\s+(\d{1,2})", right)
        if m:
            mon = MONTHS.get(m.group(1).upper())
            if mon:
                d1 = dt.date(d0.year, mon, int(m.group(2)))

    if d0 and d1 and d1 >= d0:
        return (d0, d1)

    return None


def scrape_semester_calendar(url: str, anchor: str) -> SemesterCalendar:
    """
    Best-effort scrape of the registrar page. It looks for tables near the anchor section
    and tries to identify key rows (classes begin/end, closures).
    """
    # New approach: find candidate tables across the page and parse each as a candidate semester
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    debug = bool(os.environ.get("CANVAS_LAB_PLANNER_DEBUG"))
    if debug:
        print(f"[debug] Scraping URL: {url}  anchor: {anchor}")

    tables = soup.find_all("table")
    candidates: List[SemesterCalendar] = []

    # Helper heuristics reused from original implementation
    def looks_like_begin(event: str) -> bool:
        e = event.lower()
        return ("first day of classes" in e) or ("classes begin" in e) or ("instruction begins" in e)

    def looks_like_end(event: str) -> bool:
        e = event.lower()
        return ("last day of classes" in e) or ("classes end" in e) or ("instruction ends" in e)

    def looks_like_no_class(event: str) -> bool:
        e = event.lower()

        # Strong direct indicators
        if "no classes" in e or "university closed" in e:
            return True

        # Generic 'closed' (but not the phrase already matched above)
        if "closed" in e and "university closed" not in e:
            return True

        # Explicit recess / break keywords
        if any(k in e for k in ("recess", "spring break", "fall break", "winter break")):
            return True

        # Holidays that commonly imply no classes
        if "labor day" in e or "thanksgiving" in e:
            return True

        # Martin Luther King / MLK
        if "martin luther king" in e or ("mlk" in e and "day" in e):
            return True

        # 'Holiday' together with 'no classes'
        if "holiday" in e and "no classes" in e:
            return True

        # 'Cancel' / 'cancellation' words are ambiguous; only treat as closure
        # when they appear in a context that relates to classes/academic operations.
        if ("cancel" in e or "cancellation" in e):
            if any(ctx in e for ctx in ("class", "classes", "university", "instruction", "school")):
                return True

        return False

    for tbl in tables:
        # For each table, try to find a nearby heading (term name).
        # Primary: nearest previous header in document order (closest before the table).
        term_name: Optional[str] = None
        h = tbl.find_previous(["h1", "h2", "h3", "h4"])
        if h and h.get_text(strip=True):
            term_name = h.get_text(" ", strip=True)
        else:
            # Fallback: look for a header among the table's parent's immediate children
            parent = tbl.parent
            if parent:
                for child in parent.find_all(["h1", "h2", "h3", "h4"], recursive=False):
                    if child and child.get_text(strip=True):
                        term_name = child.get_text(" ", strip=True)
                        break

        # parse rows for this table only
        rows: List[Tuple[str, str]] = []
        for tr in tbl.find_all("tr"):
            cols = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if debug:
                print(f"[debug] raw cols: {cols}")
            # Some tables put both date and event in a single cell (separated by tab/dash).
            # If there's only one column, try to split it heuristically into date + event.
            if len(cols) == 1 and cols[0]:
                single = cols[0]
                # try splitting on common separators: tab, em/en dash, hyphen with optional spaces
                parts = re.split(r"\t|\s*[\u2013\u2014\-]\s*", single, maxsplit=1)
                if len(parts) == 2:
                    cols = [parts[0].strip(), parts[1].strip()]

            if len(cols) >= 2:
                # Try to detect which column contains the date by testing parse_date_range()
                date_idx: Optional[int] = None
                for i, col in enumerate(cols):
                    try:
                        if parse_date_range(col):
                            date_idx = i
                            break
                    except Exception:
                        continue

                if date_idx is None:
                    # fallback: assume first column is date
                    date_idx = 0

                left = cols[date_idx]
                # join remaining columns to form the event/description text
                right_parts = [c for j, c in enumerate(cols) if j != date_idx]
                right = " ".join(right_parts)
                if debug:
                    print(f"[debug] detected date_idx={date_idx}, left={left!r}, right={right!r}")

                if left.lower() in ("date", "dates") and right.lower() in ("event", "description"):
                    if debug:
                        print("[debug] skipping header-like row")
                    continue
                if left and right:
                    rows.append((left, right))
                    if debug:
                        print(f"[debug] Accepted row -> date: {left!r}, event: {right!r}")
                else:
                    if debug:
                        print(f"[debug] Rejected row -> cols={cols}")
            else:
                if debug:
                    print(f"[debug] Skipped tr with cols={cols}")
                continue

        if not rows:
            if debug:
                print(f"[debug] no parsable rows found for table with term_name={term_name!r}")
            continue
        if debug:
            print(f"[debug] Parsed rows for table (term_name={term_name!r}):")
            for d, e in rows:
                print(f"    -> date_text={d!r}, event_text={e!r}")

        classes_begin: Optional[dt.date] = None
        classes_end: Optional[dt.date] = None
        closures: List[Tuple[dt.date, dt.date]] = []

        for date_text, event_text in rows:
            rng = parse_date_range(date_text)
            if not rng:
                continue
            d0, d1 = rng

            if classes_begin is None and looks_like_begin(event_text):
                classes_begin = d0

            if classes_end is None and looks_like_end(event_text):
                classes_end = d0

            if looks_like_no_class(event_text):
                closures.append((d0, d1))

        if classes_begin is None:
            # heuristic: earliest date that mentions "class"
            for date_text, event_text in rows:
                rng = parse_date_range(date_text)
                if not rng:
                    continue
                d0, _ = rng
                if "class" in event_text.lower():
                    classes_begin = d0
                    break

        if classes_begin is None:
            # skip this table - couldn't find a begin date
            continue

        # merge closures
        closures.sort(key=lambda x: x[0])
        merged: List[Tuple[dt.date, dt.date]] = []
        for a, b in closures:
            if not merged:
                merged.append((a, b))
            else:
                p0, p1 = merged[-1]
                if a <= (p1 + dt.timedelta(days=1)):
                    merged[-1] = (p0, max(p1, b))
                else:
                    merged.append((a, b))

        candidates.append(SemesterCalendar(classes_begin=classes_begin, classes_end=classes_end, closures=merged, term_name=term_name))

    if not candidates:
        raise ValueError("Could not parse any semester calendars from the registrar page.")

    return candidates


# -----------------------------
# Canvas API (requests)
# -----------------------------

class CanvasClient:
    def __init__(self, cfg: CanvasConfig):
        self.base_url = cfg.base_url.rstrip("/")
        self.course_id = str(cfg.course_id).strip() if cfg.course_id else None
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {cfg.api_key}"})

    def set_course_id(self, course_id: Optional[str]) -> None:
        self.course_id = str(course_id).strip() if course_id else None

    def _require_course_id(self) -> str:
        if not self.course_id:
            raise ValueError("No Canvas course is selected.")
        return self.course_id

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _paginate(self, url: str, params: Optional[dict] = None) -> List[dict]:
        out: List[dict] = []
        next_url = url
        while next_url:
            r = self.s.get(next_url, params=params, timeout=30)
            r.raise_for_status()
            out.extend(r.json())
            # Canvas uses Link header for pagination
            link = r.headers.get("Link", "")
            next_url = None
            for part in link.split(","):
                if 'rel="next"' in part:
                    m = re.search(r"<([^>]+)>", part)
                    if m:
                        next_url = m.group(1)
                        break
            params = None  # only for first call
        return out

    def list_assignments(self) -> List[Assignment]:
        course_id = self._require_course_id()
        url = self._url(f"/api/v1/courses/{course_id}/assignments")
        data = self._paginate(url, params={"per_page": 100, "include[]": "overrides"})
        assignments = []
        for a in data:
            overrides = a.get("overrides", [])
            assignments.append(Assignment(
                id=int(a["id"]), 
                name=str(a.get("name", "")).strip(),
                overrides=overrides
            ))
        return assignments

    def list_sections(self) -> List[Tuple[int, str]]:
        course_id = self._require_course_id()
        url = self._url(f"/api/v1/courses/{course_id}/sections")
        data = self._paginate(url, params={"per_page": 100})
        return [(int(s["id"]), str(s.get("name", "")).strip()) for s in data]

    def get_course(self) -> dict:
        """Return course details from Canvas API for the configured course id."""
        course_id = self._require_course_id()
        url = self._url(f"/api/v1/courses/{course_id}")
        r = self.s.get(url, timeout=30)
        r.raise_for_status()
        return r.json()

    def list_courses(self) -> List[CourseOption]:
        url = self._url("/api/v1/courses")
        data = self._paginate(
            url,
            params={
                "per_page": 100,
                "enrollment_state": "active",
                "include[]": "term",
            },
        )
        courses: List[CourseOption] = []
        for course in data:
            term_value = course.get("term") or {}
            if isinstance(term_value, dict):
                term_name = str(term_value.get("name", "")).strip()
            else:
                term_name = str(term_value).strip()
            courses.append(
                CourseOption(
                    id=int(course["id"]),
                    name=str(course.get("name", "")).strip(),
                    course_code=str(course.get("course_code", "")).strip(),
                    term_name=term_name,
                    workflow_state=str(course.get("workflow_state", "")).strip(),
                )
            )
        return courses

    def list_assignment_overrides(self, assignment_id: int) -> List[dict]:
        course_id = self._require_course_id()
        url = self._url(f"/api/v1/courses/{course_id}/assignments/{assignment_id}/overrides")
        return self._paginate(url, params={"per_page": 100})

    def create_override(self, assignment_id: int, section_id: int, due_at_iso: str) -> dict:
        course_id = self._require_course_id()
        url = self._url(f"/api/v1/courses/{course_id}/assignments/{assignment_id}/overrides")
        payload = {
            "assignment_override[course_section_id]": section_id,
            "assignment_override[due_at]": due_at_iso,
        }
        r = self.s.post(url, data=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def update_override(self, assignment_id: int, override_id: int, due_at_iso: str) -> dict:
        course_id = self._require_course_id()
        url = self._url(f"/api/v1/courses/{course_id}/assignments/{assignment_id}/overrides/{override_id}")
        payload = {
            "assignment_override[due_at]": due_at_iso,
        }
        r = self.s.put(url, data=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def set_due_for_section(self, assignment_id: int, section_id: int, due_at_iso: str) -> None:
        # Update existing override if present, else create new
        overrides = self.list_assignment_overrides(assignment_id)
        for o in overrides:
            if int(o.get("course_section_id", -1)) == int(section_id):
                self.update_override(assignment_id, int(o["id"]), due_at_iso)
                return
        self.create_override(assignment_id, section_id, due_at_iso)


class BannerClient:
    def __init__(self, base_url: str = "https://selfservice.uncc.edu/StudentRegistrationSsb/ssb"):
        self.base_url = base_url.rstrip("/")
        self.s = requests.Session()
        self._term_cache: Optional[List[dict]] = None

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.base_url}{path}"

    def _warm_up(self) -> None:
        self.s.get(self._url("/term/termSelection"), params={"mode": "search"}, timeout=30)

    def _json_candidates(self, urls: List[str], params: Optional[dict] = None) -> object:
        last_error: Optional[Exception] = None
        for url in urls:
            try:
                response = self.s.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
        if last_error:
            raise last_error
        raise RuntimeError("Banner request returned no data.")

    def _fetch_terms(self) -> List[dict]:
        if self._term_cache is not None:
            return self._term_cache

        self._warm_up()
        data: object = []
        last_error: Optional[Exception] = None

        # NOTE: term/getTerms consistently 404s on UNCC's Banner instance.
        # classSearch/getTerms is the endpoint the real Self-Service UI uses.
        for params in ({"searchTerm": "", "offset": 1, "max": 50}, {"mode": "search"}):
            try:
                data = self._json_candidates([self._url("/classSearch/getTerms")], params=params)
                if data:
                    break
            except Exception as exc:
                last_error = exc
                continue

        if not data and last_error is not None:
            # Do not raise here; resolve_term_code can still use deterministic fallback.
            self._term_cache = []
            return self._term_cache

        if isinstance(data, list):
            terms = data
        elif isinstance(data, dict):
            terms = (
                data.get("data")
                or data.get("terms")
                or data.get("results")
                or data.get("rows")
                or []
            )
        else:
            terms = []

        self._term_cache = [term for term in terms if isinstance(term, dict)]
        return self._term_cache

    def _term_display(self, term: dict) -> str:
        pieces = [
            str(term.get("description") or term.get("termDesc") or term.get("label") or "").strip(),
            str(term.get("code") or term.get("value") or term.get("term") or term.get("id") or "").strip(),
        ]
        return " ".join(piece for piece in pieces if piece)

    def resolve_term_code(self, term_name: Optional[str]) -> Optional[str]:
        if not term_name:
            return None

        normalized_target = normalize_banner_text(term_name)
        exact_tokens = set(normalized_target.split())

        terms: List[dict] = []
        try:
            terms = self._fetch_terms()
        except Exception:
            terms = []

        for term in terms:
            display = normalize_banner_text(self._term_display(term))
            code = str(term.get("code") or term.get("value") or term.get("term") or term.get("id") or "").strip()
            if not code:
                continue
            if normalized_target and (normalized_target == display or normalized_target in display or display in normalized_target):
                return code
            display_tokens = set(display.split())
            if exact_tokens and exact_tokens <= display_tokens:
                return code

        if terms:
            for term in terms:
                display = self._term_display(term)
                if normalized_target and normalized_target in normalize_banner_text(display):
                    return str(term.get("code") or term.get("value") or term.get("term") or term.get("id") or "").strip() or None

        # Fallback: if term text includes a season/year phrase, match Banner terms by those tokens.
        season_year = re.search(r"\b(SPRING|SUMMER|FALL|WINTER)\s+(19\d{2}|20\d{2})\b", normalized_target)
        if season_year and terms:
            season = season_year.group(1)
            year = season_year.group(2)
            for term in terms:
                display = normalize_banner_text(self._term_display(term))
                code = str(term.get("code") or term.get("value") or term.get("term") or term.get("id") or "").strip()
                if not code:
                    continue
                if season in display and year in display:
                    return code

        guessed = guess_banner_term_code(term_name)
        if guessed:
            return guessed

        return None

    def _search_payload(self, subject: str, course_number: str, term_code: str) -> List[Tuple[str, str]]:
        payload = [
            ("txt_courseNumber", course_number),
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

    def search_sections(self, subject: str, course_number: str, term_code: str) -> List[dict]:
        self._warm_up()

        # Critical step: Banner's search results endpoint reads the active term from
        # server-side session state, not just from the txt_term field. The real
        # Self-Service UI POSTs to /term/search to establish that state before
        # calling searchResults. Skipping this step causes searchResults to
        # silently return zero rows (no error, just an empty data list).
        select_resp = self.s.post(
            self._url("/term/search"),
            params={"mode": "search"},
            data={"term": term_code},
            timeout=30,
        )
        select_resp.raise_for_status()

        response = self.s.post(
            self._url("/searchResults/searchResults"),
            data=self._search_payload(subject, course_number, term_code),
            timeout=30,
        )
        response.raise_for_status()

        try:
            data = response.json()
        except Exception:
            text = response.text
            match = re.search(r"window\.sectionResultsCollection\s*=\s*(\{.*?\});", text, re.DOTALL)
            if not match:
                raise ValueError("Banner search response did not include JSON results.")
            import json

            data = json.loads(match.group(1))

        if isinstance(data, dict):
            for key in ("data", "rows", "results", "items"):
                rows = data.get(key)
                if isinstance(rows, list):
                    return [row for row in rows if isinstance(row, dict)]

        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]

        return []

    def _meeting_time_from_row(self, row: dict) -> Optional[Tuple[List[int], dt.time]]:
        meeting_times: List[dict] = []
        for faculty_entry in row.get("meetingsFaculty", []) or []:
            if not isinstance(faculty_entry, dict):
                continue
            mt = faculty_entry.get("meetingTime")
            if isinstance(mt, list):
                meeting_times.extend([entry for entry in mt if isinstance(entry, dict)])
            elif isinstance(mt, dict):
                meeting_times.append(mt)

        best: Optional[Tuple[int, dt.time, List[int]]] = None
        for meeting_time in meeting_times:
            weekdays = banner_weekdays_from_meeting_time(meeting_time)
            begin_time = parse_banner_time(
                meeting_time.get("beginTime")
                or meeting_time.get("startTime")
                or meeting_time.get("begin")
                or meeting_time.get("start")
            )
            if not weekdays or not begin_time:
                continue
            score = (-len(weekdays), begin_time.hour * 60 + begin_time.minute)
            if best is None or score < (best[0], best[1].hour * 60 + best[1].minute):
                best = (score[0], begin_time, weekdays)

        if best is None:
            return None
        return best[2], best[1]

    def _row_section_number(self, row: dict) -> str:
        for key in ("sequenceNumber", "sequence_number", "sectionNumber", "section_number", "courseReferenceNumber", "course_reference_number"):
            value = str(row.get(key, "")).strip()
            if value:
                return value
        return ""

    def _canvas_section_suffix(self, section_name: str, course_number: str = "") -> Optional[str]:
        """
        Extract the Banner-style section/sequence identifier (e.g. "L01", "001")
        from a Canvas section name.

        Canvas section names for combined/cross-listed sections often embed the
        course number itself (e.g. "ECGR-2181L-L01"), so a naive regex can match
        the course number instead of the actual section suffix. To avoid that,
        the course number is stripped out first, then the last remaining
        alphanumeric token containing a digit is used as the section suffix.
        """
        normalized = str(section_name or "").upper()
        if course_number:
            normalized = re.sub(re.escape(course_number.upper()), " ", normalized)

        tokens = re.findall(r"[A-Z0-9]+", normalized)
        for token in reversed(tokens):
            if re.search(r"\d", token):
                return token
        return None

    def _subject_hints_from_sections(self, sections: List[Tuple[int, str]]) -> List[str]:
        hints: List[str] = []
        for _section_id, section_name in sections:
            normalized = str(section_name or "").upper()
            match = re.search(r"\b([A-Z]{2,10})[-_\s]*\d{4}[A-Z]?\b", normalized)
            if not match:
                continue
            subject = match.group(1)
            if subject in TERM_LIKE_SUBJECTS:
                continue
            if subject not in hints:
                hints.append(subject)
        return hints

    def autofill_section_meetings(self, course: CourseOption, sections: List[Tuple[int, str]]) -> Dict[int, SectionMeeting]:
        parsed = parse_banner_course_code(course)
        if not parsed:
            return {}

        return self.autofill_section_meetings_for_query(f"{parsed[0]} {parsed[1]}", course.term_name, sections)

    def autofill_section_meetings_for_query(self, query_text: str, term_name: Optional[str], sections: List[Tuple[int, str]]) -> Dict[int, SectionMeeting]:
        parsed = parse_banner_course_query(query_text)
        if not parsed:
            return {}

        subject, course_number = parsed
        term_code = self.resolve_term_code(term_name)
        if not term_code:
            return {}

        subject_hints = self._subject_hints_from_sections(sections)
        subject_candidates: List[str] = []
        normalized_subject = subject.upper().strip()
        if normalized_subject:
            subject_candidates.append(normalized_subject)
        else:
            subject_candidates.extend(subject_hints)

        rows: List[dict] = []
        tried_subjects: set[str] = set()
        if subject_candidates:
            for subject_candidate in subject_candidates:
                if subject_candidate in tried_subjects:
                    continue
                tried_subjects.add(subject_candidate)
                rows.extend(self.search_sections(subject_candidate, course_number, term_code))
        else:
            rows.extend(self.search_sections("", course_number, term_code))

        if not rows and subject_candidates:
            rows.extend(self.search_sections("", course_number, term_code))
        if not rows:
            return {}

        normalized_course_number = course_number.upper().strip()
        inferred_subject = normalized_subject
        if not inferred_subject and len(subject_hints) == 1:
            inferred_subject = subject_hints[0]

        filtered_rows: List[dict] = []
        for row in rows:
            row_course_number = str(row.get("courseNumber", "")).upper().strip()
            if row_course_number != normalized_course_number:
                continue
            if inferred_subject:
                row_subject = str(row.get("subject", "")).upper().strip()
                if row_subject != inferred_subject:
                    continue
            filtered_rows.append(row)
        rows = filtered_rows
        if not rows:
            return {}

        rows_by_section: Dict[str, dict] = {}
        rows_by_digits: Dict[str, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            section_number = self._row_section_number(row)
            if section_number:
                rows_by_section[section_number] = row
                digits = re.sub(r"[^0-9]", "", section_number)
                if digits:
                    rows_by_digits.setdefault(digits, row)

        if not rows_by_section:
            return {}

        autofilled: Dict[int, SectionMeeting] = {}
        for section_id, section_name in sections:
            section_suffix = self._canvas_section_suffix(section_name, course_number)
            candidate_row: Optional[dict] = None
            if section_suffix:
                candidate_row = rows_by_section.get(section_suffix)
                if not candidate_row:
                    digits = re.sub(r"[^0-9]", "", section_suffix)
                    if digits:
                        candidate_row = rows_by_digits.get(digits)
            if not candidate_row and len(rows_by_section) == 1:
                candidate_row = next(iter(rows_by_section.values()))

            if not candidate_row:
                continue

            parsed_time = self._meeting_time_from_row(candidate_row)
            if not parsed_time:
                continue

            weekdays, start_time = parsed_time
            autofilled[section_id] = SectionMeeting(
                section_id=section_id,
                section_name=section_name,
                weekdays=weekdays,
                start_time=start_time,
            )

        return autofilled


# -----------------------------
# Due date automation logic
# -----------------------------

LAB_NAME_RE = re.compile(
    r"\b(?P<kind>pre[-\s]*lab|post[-\s]*lab|lab)\b\s*(?P<num>\d+)\b",
    re.IGNORECASE,
)


def classify_assignment(name: str) -> Optional[Tuple[str, int]]:
    """
    Returns (kind, num) where kind in {'pre', 'lab', 'post'}, else None.
    """
    m = LAB_NAME_RE.search(name)
    if not m:
        return None
    kind_raw = m.group("kind").lower().replace(" ", "").replace("-", "")
    num = int(m.group("num"))
    if kind_raw.startswith("pre"):
        return ("pre", num)
    if kind_raw.startswith("post"):
        return ("post", num)
    return ("lab", num)


def first_meeting_date_in_week(week_monday: dt.date, meeting_weekdays: List[int]) -> dt.date:
    """
    Given a week Monday and a list of weekdays, return the earliest meeting date in that week.
    """
    candidates = []
    for wd in sorted(set(meeting_weekdays)):
        candidates.append(week_monday + dt.timedelta(days=(wd - 0)))
    return min(candidates)


def build_due_suggestions(
    assignments: List[Assignment],
    section_meetings: List[SectionMeeting],
    calendar: SemesterCalendar,
    tz: ZoneInfo,
    due_time: dt.time,
    delay_weeks: int,
    lab_base: int = 1,
) -> Tuple[List[DueSuggestion], List[int]]:
    """
    Maps lab numbers to instructional weeks:
      lab1_week_index = 1 + delay_weeks   (week after first week) + delay
    and increments for each lab number, skipping cancelled weeks because
    instructional_weeks() already filtered them out.
    """
    # Filter to only assignments we can classify
    classified: List[Tuple[Assignment, str, int]] = []
    for a in assignments:
        c = classify_assignment(a.name)
        if c:
            kind, num = c
            classified.append((a, kind, num))

    if not classified:
        return []

    # Determine max lab number we need to map
    max_n = max(num for _, _, num in classified)

    weeks = calendar.instructional_weeks()
    base_index = 1 + max(0, delay_weeks)  # "week after first week" + delay

    # lab_num -> week_monday (compute per actual lab number present)
    lab_week: Dict[int, dt.date] = {}
    for n in range(min(num for _, _, num in classified), max_n + 1):
        idx = base_index + (n - lab_base)
        if idx < 0 or idx >= len(weeks):
            continue
        lab_week[n] = weeks[idx]

    suggestions: List[DueSuggestion] = []
    unschedulable_assignment_ids: List[int] = []

    for a, kind, num in classified:
        wk = lab_week.get(num)
        if not wk:
            unschedulable_assignment_ids.append(a.id)
            continue

        per_section: Dict[int, dt.datetime] = {}

        for sm in section_meetings:
            meet_date = first_meeting_date_in_week(wk, sm.weekdays)

            if kind == "pre":
                # due at meeting start time
                due_dt = dt.datetime.combine(meet_date, sm.start_time).replace(tzinfo=tz)

            elif kind == "lab":
                # due at due_time on that meeting day
                due_dt = dt.datetime.combine(meet_date, due_time).replace(tzinfo=tz)

            else:  # post
                # end of that week (Sunday) at due_time
                end_date = sunday_of_week(wk)
                due_dt = dt.datetime.combine(end_date, due_time).replace(tzinfo=tz)

            per_section[sm.section_id] = due_dt

        suggestions.append(DueSuggestion(
            assignment_id=a.id,
            assignment_name=a.name,
            due_by_section=per_section
        ))

    # Sort by assignment name for stable display
    suggestions.sort(key=lambda x: x.assignment_name.lower())
    return suggestions, unschedulable_assignment_ids


# -----------------------------
# Tkinter UI
# -----------------------------

class SectionMeetingDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, section_id: int, section_name: str, existing_meeting: Optional[SectionMeeting] = None):
        super().__init__(parent)
        self.title(f"Meeting time: {section_name}")
        self.resizable(False, False)
        self.section_id = section_id
        self.section_name = section_name
        self.result: Optional[Tuple[List[int], dt.time]] = None

        frm = ttk.Frame(self, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text=f"Section: {section_name}", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        ttk.Label(frm, text="Meeting days (e.g., M W F or Tu Th):").grid(row=1, column=0, sticky="w")
        self.days_entry = ttk.Entry(frm, width=24)
        self.days_entry.grid(row=1, column=1, sticky="w", padx=(8, 0))
        
        # Pre-populate with existing data if available
        if existing_meeting:
            # Convert weekday numbers to day abbreviations
            day_map = {0: "M", 1: "T", 2: "W", 3: "R", 4: "F", 5: "S", 6: "U"}
            days_str = " ".join(day_map.get(wd, str(wd)) for wd in existing_meeting.weekdays)
            self.days_entry.insert(0, days_str)
        else:
            self.days_entry.insert(0, "M W")

        ttk.Label(frm, text="Start time (24h HH:MM):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.time_entry = ttk.Entry(frm, width=10)
        self.time_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        
        # Pre-populate with existing time if available
        if existing_meeting:
            self.time_entry.insert(0, existing_meeting.start_time.strftime("%H:%M"))
        else:
            self.time_entry.insert(0, "09:00")

        btns = ttk.Frame(frm)
        btns.grid(row=3, column=0, columnspan=4, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Cancel", command=self._cancel).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="OK", command=self._ok).grid(row=0, column=1)

        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self._cancel())

        self.transient(parent)
        self.grab_set()
        self.days_entry.focus_set()

    def _parse_days(self, s: str) -> List[int]:
        tokens = re.split(r"[,\s]+", s.strip())
        wds: List[int] = []
        for t in tokens:
            if not t:
                continue
            key = t.upper()
            if key not in DAY_TOKEN_TO_WEEKDAY:
                raise ValueError(f"Unknown day token: '{t}'. Use M, T, W, R, F (etc).")
            wds.append(DAY_TOKEN_TO_WEEKDAY[key])
        wds = sorted(set(wds))
        if not wds:
            raise ValueError("At least one meeting day is required.")
        return wds

    def _ok(self):
        try:
            wds = self._parse_days(self.days_entry.get())
            tm = parse_hhmm(self.time_entry.get())
            self.result = (wds, tm)
            self.destroy()
        except Exception as e:
            messagebox.showerror("Invalid meeting time", str(e), parent=self)

    def _cancel(self):
        self.result = None
        self.destroy()


class SkippedWeeksDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, skipped: List[Tuple[dt.date, dt.date, List[Tuple[dt.date, dt.date]]]]):
        super().__init__(parent)
        self.title("Skipped Weeks")
        self.resizable(False, False)
        frm = ttk.Frame(self, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="Weeks skipped due to closures", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.listbox = tk.Listbox(frm, width=90, height=10)
        self.listbox.grid(row=1, column=0, sticky="nsew")

        for wk_mon, wk_sun, closures in skipped:
            closures_str = "; ".join(f"{c0.isoformat()} to {c1.isoformat()}" for c0, c1 in closures) if closures else "(none)"
            line = f"Week {wk_mon.isoformat()} — {wk_sun.isoformat()}  |  closures: {closures_str}"
            self.listbox.insert("end", line)

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="OK", command=self._ok).grid(row=0, column=0)

        self.bind("<Return>", lambda _e: self._ok())
        self.bind("<Escape>", lambda _e: self._ok())

        self.transient(parent)
        self.grab_set()
        self.listbox.focus_set()

    def _ok(self):
        self.destroy()


class SemesterSelectionDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, candidates: List[SemesterCalendar]):
        super().__init__(parent)
        self.title("Select semester")
        self.resizable(False, False)
        self.result: Optional[SemesterCalendar] = None

        frm = ttk.Frame(self, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="Select the semester to use", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.listbox = tk.Listbox(frm, width=80, height=8)
        self.listbox.grid(row=1, column=0, sticky="nsew")

        # populate
        for i, c in enumerate(candidates):
            name = c.term_name or f"Semester {i+1}"
            rr = f"{name} — {c.classes_begin.isoformat()} to {c.classes_end.isoformat() if c.classes_end else '(unknown)'}"
            self.listbox.insert("end", rr)

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Cancel", command=self._cancel).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="OK", command=lambda: self._ok(candidates)).grid(row=0, column=1)

        self.bind("<Return>", lambda _e: self._ok(candidates))
        self.bind("<Escape>", lambda _e: self._cancel())

        self.transient(parent)
        self.grab_set()
        self.listbox.focus_set()

    def _ok(self, candidates: List[SemesterCalendar]):
        try:
            sel = self.listbox.curselection()
            if not sel:
                messagebox.showerror("No selection", "Please select a semester.", parent=self)
                return
            idx = int(sel[0])
            self.result = candidates[idx]
            self.destroy()
        except Exception as e:
            messagebox.showerror("Selection error", str(e), parent=self)

    def _cancel(self):
        self.result = None
        self.destroy()


class ConfigDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, config_path: str):
        super().__init__(parent)
        self.title("Edit config")
        self.resizable(False, False)
        self.parent = parent
        self.config_path = config_path

        frm = ttk.Frame(self, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="Edit configuration", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        cfg = configparser.ConfigParser()
        cfg.read(self.config_path)

        # Canvas
        ttk.Label(frm, text="Canvas base_url:").grid(row=1, column=0, sticky="w")
        self.base_url_e = ttk.Entry(frm, width=60)
        self.base_url_e.grid(row=1, column=1, sticky="w")
        self.base_url_e.insert(0, cfg.get("canvas", "base_url", fallback=""))

        ttk.Label(frm, text="Canvas API key:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.api_key_e = ttk.Entry(frm, width=60)
        self.api_key_e.grid(row=2, column=1, sticky="w", pady=(6, 0))
        self.api_key_e.insert(0, cfg.get("canvas", "api_key", fallback=""))

        # Semester
        ttk.Label(frm, text="Calendar URL:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.cal_url_e = ttk.Entry(frm, width=60)
        self.cal_url_e.grid(row=3, column=1, sticky="w", pady=(8, 0))
        self.cal_url_e.insert(0, cfg.get("semester", "calendar_url", fallback=""))

        ttk.Label(frm, text="Anchor:").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.anchor_e = ttk.Entry(frm, width=40)
        self.anchor_e.grid(row=4, column=1, sticky="w", pady=(6, 0))
        self.anchor_e.insert(0, cfg.get("semester", "anchor", fallback=""))

        # lab_base is auto-detected; not editable via this dialog

        # Time
        ttk.Label(frm, text="Timezone (IANA):").grid(row=6, column=0, sticky="w", pady=(8, 0))
        self.tz_e = ttk.Entry(frm, width=40)
        self.tz_e.grid(row=6, column=1, sticky="w", pady=(8, 0))
        self.tz_e.insert(0, cfg.get("time", "timezone", fallback="America/New_York"))

        ttk.Label(frm, text="Due time (HH:MM):").grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.due_time_e = ttk.Entry(frm, width=10)
        self.due_time_e.grid(row=7, column=1, sticky="w", pady=(6, 0))
        self.due_time_e.insert(0, cfg.get("time", "due_time_hhmm", fallback="23:59"))

        btns = ttk.Frame(frm)
        btns.grid(row=8, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="Cancel", command=self._cancel).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btns, text="Save", command=self._save).grid(row=0, column=1)

        self.bind("<Return>", lambda _e: self._save())
        self.bind("<Escape>", lambda _e: self._cancel())

        self.transient(parent)
        self.grab_set()
        self.base_url_e.focus_set()

    def _save(self):
        # Validate and write the config
        try:
            base_url = self.base_url_e.get().strip()
            api_key = self.api_key_e.get().strip()
            cal_url = self.cal_url_e.get().strip()
            anchor = self.anchor_e.get().strip()
            tz = self.tz_e.get().strip()
            due_time = self.due_time_e.get().strip()

            # basic validation
            if not base_url or not api_key:
                raise ValueError("Canvas base_url and api_key are required.")
            # validate due_time
            _ = parse_hhmm(due_time)

            cfg = configparser.ConfigParser()
            cfg["canvas"] = {
                "base_url": base_url,
                "api_key": api_key,
            }
            existing = configparser.ConfigParser()
            existing.read(self.config_path)
            saved_course_id = existing.get("canvas", "course_id", fallback="").strip()
            if saved_course_id:
                cfg["canvas"]["course_id"] = saved_course_id
            cfg["semester"] = {
                "calendar_url": cal_url,
                "anchor": anchor,
            }
            cfg["time"] = {
                "timezone": tz,
                "due_time_hhmm": due_time,
            }

            with open(self.config_path, "w", encoding="utf-8") as fh:
                cfg.write(fh)

            # reload config in parent
            try:
                self.parent._load_config_and_init()
            except Exception:
                pass

            self.destroy()
        except Exception as e:
            messagebox.showerror("Config save error", str(e), parent=self)

    def _cancel(self):
        self.destroy()


class App(tk.Tk):
    def __init__(self, config_path: str = "config.ini"):
        super().__init__()
        self.title("Canvas Lab Planner")

        self.config_path = config_path
        self.canvas_cfg: Optional[CanvasConfig] = None
        self.sem_cfg: Optional[SemesterConfig] = None
        self.time_cfg: Optional[TimeConfig] = None
        self.course_name: Optional[str] = None  # Store course name from Canvas

        self.tz: Optional[ZoneInfo] = None
        self.calendar: Optional[SemesterCalendar] = None
        self.client: Optional[CanvasClient] = None
        self.banner_client: BannerClient = BannerClient()
        self.banner_search_var = tk.StringVar(value="")

        self.course_options: List[CourseOption] = []
        self.course_options_by_id: Dict[str, CourseOption] = {}
        self.course_options_by_label: Dict[str, CourseOption] = {}

        self.sections: List[Tuple[int, str]] = []
        self.section_meetings: Dict[int, SectionMeeting] = {}
        self.assignments: List[Assignment] = []
        self.suggestions: List[DueSuggestion] = []
        self.unschedulable_assignment_ids: set[int] = set()

        self._build_ui()

        # Size the window to fit everything the UI needs (Controls panel, meeting
        # list, treeview, footer) instead of a fixed guess that can clip content
        # as more controls are added. Capped to the available screen size.
        self.update_idletasks()
        req_w = self.winfo_reqwidth()
        req_h = self.winfo_reqheight()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(req_w, screen_w - 40)
        win_h = min(req_h, screen_h - 80)
        self.minsize(min(req_w, screen_w - 40), min(req_h, screen_h - 80))
        self.geometry(f"{win_w}x{win_h}")

        self._load_config_and_init()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="both", expand=True)

        # Controls
        ctrl = ttk.LabelFrame(top, text="Controls", padding=10)
        ctrl.pack(fill="x", padx=5, pady=5)

        label_font = ("Segoe UI", 9, "bold")
        row = 0

        ttk.Label(ctrl, text="Config file:").grid(row=row, column=0, sticky="w")
        self.cfg_label = ttk.Label(ctrl, text=self.config_path)
        self.cfg_label.grid(row=row, column=1, sticky="w", padx=(8, 0))
        row += 1

        ttk.Label(ctrl, text="Delay weeks (Lab 1 starts week after first week + delay):").grid(row=row, column=0, sticky="w", pady=(6, 0))
        self.delay_var = tk.IntVar(value=0)
        self.delay_spin = ttk.Spinbox(ctrl, from_=0, to=20, textvariable=self.delay_var, width=5)
        self.delay_spin.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        row += 1

        ttk.Separator(ctrl, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 8))
        row += 1

        # --- Semester info (populated after scraping) ---
        ttk.Label(ctrl, text="Semester", font=label_font).grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Label(ctrl, text="Term:").grid(row=row, column=0, sticky="w", pady=(4, 0))
        self.term_var = tk.StringVar(value="(not loaded)")
        ttk.Label(ctrl, textvariable=self.term_var).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        row += 1

        ttk.Label(ctrl, text="First day:").grid(row=row, column=0, sticky="w")
        self.begin_var = tk.StringVar(value="(not loaded)")
        ttk.Label(ctrl, textvariable=self.begin_var).grid(row=row, column=1, sticky="w", padx=(8, 0))
        row += 1

        ttk.Label(ctrl, text="Last day:").grid(row=row, column=0, sticky="w")
        self.end_var = tk.StringVar(value="(not loaded)")
        ttk.Label(ctrl, textvariable=self.end_var).grid(row=row, column=1, sticky="w", padx=(8, 0))
        row += 1

        ttk.Separator(ctrl, orient="horizontal").grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 8))
        row += 1

        # --- Course selection + Banner search ---
        ttk.Label(ctrl, text="Course", font=label_font).grid(row=row, column=0, sticky="w")
        row += 1

        ttk.Label(ctrl, text="Canvas course:").grid(row=row, column=0, sticky="w", pady=(4, 0))
        self.course_var = tk.StringVar(value="(not loaded)")
        ttk.Label(ctrl, textvariable=self.course_var).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        row += 1

        # Course selector dropdown with fuzzy filtering against the API-backed list.
        # Clicking into it auto-selects existing text so a new query can replace it immediately.
        ttk.Label(ctrl, text="Course search (fuzzy):").grid(row=row, column=0, sticky="w", pady=(4, 0))
        self.course_selector = ttk.Combobox(ctrl, width=72, state="normal")
        self.course_selector.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        self.course_selector.bind("<KeyRelease>", self._on_course_query_changed)
        self.course_selector.bind("<<ComboboxSelected>>", self._on_course_selected)
        self.course_selector.bind("<FocusIn>", self._on_course_selector_focus_in)
        row += 1

        # Single editable field for the Banner course number, pre-filled with an
        # estimate parsed from the selected Canvas course (no separate read-only display).
        ttk.Label(ctrl, text="Banner course number:").grid(row=row, column=0, sticky="w", pady=(4, 0))
        self.banner_search_entry = ttk.Entry(ctrl, width=30, textvariable=self.banner_search_var)
        self.banner_search_entry.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=(4, 0))
        row += 1

        # Buttons, arranged in a single column to keep the panel visually simple.
        btnrow = ttk.Frame(ctrl)
        btnrow.grid(row=0, column=2, rowspan=row, sticky="ne", padx=(24, 0))
        button_specs = [
            ("Reload semester", self.on_reload_semester),
            ("Load Canvas data", self.on_load_canvas),
            ("Search section times", self.on_search_section_times),
            ("Set section meeting times", self.on_set_meetings),
            ("Auto-compute due dates", self.on_compute),
            ("Apply to Canvas", self.on_apply),
            ("Show skipped weeks", self.on_show_skipped_weeks),
            ("Edit config", self.on_edit_config),
        ]
        for btn_row, (text, command) in enumerate(button_specs):
            ttk.Button(btnrow, text=text, command=command).grid(row=btn_row, column=0, padx=5, pady=2, sticky="ew")

        # Status
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(top, textvariable=self.status_var).pack(fill="x", padx=8, pady=(0, 8))

        # Section meetings display
        meetings_frame = ttk.LabelFrame(top, text="Section Meeting Times", padding=10)
        meetings_frame.pack(fill="x", padx=5, pady=(0, 5))
        
        self.meetings_listbox = tk.Listbox(meetings_frame, height=4, font=("Consolas", 9))
        self.meetings_listbox.pack(fill="both", expand=True)

        # Treeview
        cols = ("assignment", "existing", "preview")
        self.tree = ttk.Treeview(top, columns=cols, show="headings", height=18)
        self.tree.heading("assignment", text="Assignment")
        self.tree.heading("existing", text="Current Canvas Due Dates")
        self.tree.heading("preview", text="Auto Due Preview (per section)")
        self.tree.column("assignment", width=280, anchor="w")
        self.tree.column("existing", width=380, anchor="w")
        self.tree.column("preview", width=380, anchor="w")
        self.tree.column("preview", width=640, anchor="w")
        self.tree.tag_configure("unschedulable", background="#fff3cd")
        self.tree.pack(fill="both", expand=True, padx=5, pady=5)

        # Help footer
        footer = ttk.LabelFrame(top, text="Naming rule used for automation", padding=10)
        footer.pack(fill="x", padx=5, pady=(5, 0))
        ttk.Label(
            footer,
            text="Assignments matching: Pre-Lab <N>, Lab <N>, Post-Lab <N> (case-insensitive). "
                 "Pre-Lab due at meeting start; Lab due at due_time_hhmm on meeting day; "
                 "Post-Lab due Sunday at due_time_hhmm.",
            wraplength=1050
        ).pack(anchor="w")

    def _set_status(self, s: str):
        self.status_var.set(s)
        self.update_idletasks()

    def _estimated_banner_course_text(self, course: Optional[CourseOption]) -> str:
        if not course:
            return "(not estimated)"
        parsed = parse_banner_course_code(course)
        if not parsed:
            return "(could not estimate)"
        subject, course_number = parsed
        return f"{subject} {course_number}".strip()

    def _update_banner_search_term(self, course: Optional[CourseOption]) -> None:
        if not course:
            self.banner_search_var.set("")
            return
        estimated = self._estimated_banner_course_text(course)
        if estimated not in {"(not estimated)", "(could not estimate)"}:
            self.banner_search_var.set(estimated)

    def _on_course_selector_focus_in(self, event=None):
        """
        Auto-select the existing text when the fuzzy course search box gains focus,
        so typing a new query immediately replaces it. The selection is applied via
        after_idle because the widget's own click handling would otherwise reset it.
        """
        def select_all():
            self.course_selector.selection_range(0, "end")
            self.course_selector.icursor("end")
        self.after_idle(select_all)

    def _normalize_course_query(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    def _course_search_blob(self, course: CourseOption) -> str:
        return " ".join(part for part in (course.name, course.course_code, course.term_name, str(course.id)) if part)

    def _score_course_option(self, course: CourseOption, query: str) -> float:
        normalized_query = self._normalize_course_query(query)
        if not normalized_query:
            return 0.0

        haystack = self._normalize_course_query(self._course_search_blob(course))
        score = SequenceMatcher(None, normalized_query, haystack).ratio()
        if normalized_query in haystack:
            score += 0.8
        if self._normalize_course_query(course.name).startswith(normalized_query):
            score += 0.5
        if course.course_code and self._normalize_course_query(course.course_code).startswith(normalized_query):
            score += 0.7
        for token in normalized_query.split():
            if token in haystack:
                score += 0.15
        if str(course.id) == query.strip():
            score += 1.0
        return score

    def _sorted_course_options(self, query: str = "") -> List[CourseOption]:
        if not self.course_options:
            return []

        query = query.strip()
        if not query:
            return sorted(
                self.course_options,
                key=lambda course: (
                    self._normalize_course_query(course.term_name),
                    self._normalize_course_query(course.course_code),
                    self._normalize_course_query(course.name),
                    course.id,
                ),
            )

        scored: List[Tuple[float, CourseOption]] = []
        for course in self.course_options:
            score = self._score_course_option(course, query)
            if score > 0:
                scored.append((score, course))

        if not scored:
            return sorted(
                self.course_options,
                key=lambda course: (
                    self._normalize_course_query(course.term_name),
                    self._normalize_course_query(course.course_code),
                    self._normalize_course_query(course.name),
                    course.id,
                ),
            )

        scored.sort(
            key=lambda item: (
                -item[0],
                self._normalize_course_query(item[1].term_name),
                self._normalize_course_query(item[1].name),
                item[1].id,
            )
        )
        return [course for _, course in scored]

    def _update_course_selector(self, query: str = ""):
        if not hasattr(self, "course_selector"):
            return

        active_query = query.strip()
        matches = self._sorted_course_options(active_query)
        labels = [course.label() for course in matches[:75]]
        self.course_selector["values"] = labels

    def _persist_selected_course_id(self, course_id: Optional[str]) -> None:
        cfg = configparser.ConfigParser()
        if os.path.exists(self.config_path):
            cfg.read(self.config_path)
        if "canvas" not in cfg:
            cfg["canvas"] = {}
        if course_id:
            cfg["canvas"]["course_id"] = str(course_id)
        else:
            cfg["canvas"].pop("course_id", None)

        with open(self.config_path, "w", encoding="utf-8") as fh:
            cfg.write(fh)

    def _set_active_course(self, course_id: Optional[str], persist: bool = True) -> None:
        course_key = str(course_id).strip() if course_id else ""

        self.suggestions = []
        self.unschedulable_assignment_ids = set()
        self.sections = []
        self.assignments = []
        self.course_name = None
        self._refresh_tree_empty()

        if not course_key:
            if self.canvas_cfg:
                self.canvas_cfg.course_id = None
            if self.client:
                self.client.set_course_id(None)
            self.course_var.set("(not selected)")
            self._update_banner_search_term(None)
            self.course_selector.set("")
            self.section_meetings.clear()
            self._refresh_meetings_display()
            if persist:
                self._persist_selected_course_id(None)
            self._set_status("No Canvas course selected.")
            return

        course_option = self.course_options_by_id.get(course_key)
        display = course_option.label() if course_option else f"Course ID {course_key}"

        if self.canvas_cfg:
            self.canvas_cfg.course_id = course_key
        if self.client:
            self.client.set_course_id(course_key)

        self.course_var.set(display)
        self.course_selector.set(display)
        self._update_banner_search_term(course_option)
        self._load_section_meetings()

        if persist:
            self._persist_selected_course_id(course_key)

        self._set_status(f"Selected {display}.")

    def _on_course_query_changed(self, event):
        if getattr(self, "_course_selector_refreshing", False):
            return
        self._update_course_selector(self.course_selector.get())

    def _on_course_selected(self, event=None):
        selected_text = self.course_selector.get().strip()
        if not selected_text:
            return

        course_option = self.course_options_by_label.get(selected_text)
        if not course_option and selected_text.isdigit():
            course_option = self.course_options_by_id.get(selected_text)

        if not course_option:
            ranked = self._sorted_course_options(selected_text)
            if ranked:
                top = ranked[0]
                if self._score_course_option(top, selected_text) >= 0.5:
                    course_option = top

        if not course_option:
            return

        if self.canvas_cfg and self.canvas_cfg.course_id == str(course_option.id):
            self.course_selector.set(course_option.label())
            return

        self._set_active_course(str(course_option.id))

    def _load_courses_async(self):
        if not self.client:
            return
        self._set_status("Loading Canvas course list...")
        self._run_threaded(self._load_courses_worker)

    def _load_courses_worker(self):
        try:
            courses = self.client.list_courses()
            self.after(0, lambda courses=courses: self._apply_loaded_courses(courses))
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Course list error", str(e)))
            self.after(0, lambda: self._set_status("Failed to load Canvas course list."))

    def _apply_loaded_courses(self, courses: List[CourseOption]):
        self.course_options = courses
        self.course_options_by_id = {str(course.id): course for course in courses}
        self.course_options_by_label = {course.label(): course for course in courses}

        self._update_course_selector()

        if self.canvas_cfg and self.canvas_cfg.course_id:
            current_course = self.course_options_by_id.get(str(self.canvas_cfg.course_id))
            if current_course:
                self.course_selector.set(current_course.label())
                self.course_var.set(current_course.label())
                self._update_banner_search_term(current_course)
                self._load_section_meetings()
            else:
                self.course_selector.set("")
                self.course_var.set("(selected course not found)")
                self._update_banner_search_term(None)
        elif len(courses) == 1:
            self._set_active_course(str(courses[0].id))
        else:
            self.course_selector.set("")
            self.course_var.set("(not selected)")
            self._update_banner_search_term(None)

        self._set_status(f"Loaded {len(courses)} Canvas courses. Type to search, then pick a course.")

    def _resolve_course_selection(self) -> Optional[CourseOption]:
        selected_text = self.course_selector.get().strip()
        if not selected_text:
            if self.canvas_cfg and self.canvas_cfg.course_id:
                cached_id = str(self.canvas_cfg.course_id)
                cached_option = self.course_options_by_id.get(cached_id)
                if cached_option:
                    return cached_option
                if cached_id.isdigit():
                    return CourseOption(id=int(cached_id), name=f"Course ID {cached_id}")
            return None

        course_option = self.course_options_by_label.get(selected_text)
        if course_option:
            return course_option

        if selected_text.isdigit():
            course_option = self.course_options_by_id.get(selected_text)
            if course_option:
                return course_option

        ranked = self._sorted_course_options(selected_text)
        if ranked and self._score_course_option(ranked[0], selected_text) >= 0.5:
            return ranked[0]

        if self.canvas_cfg and self.canvas_cfg.course_id:
            cached_id = str(self.canvas_cfg.course_id)
            cached_option = self.course_options_by_id.get(cached_id)
            if cached_option:
                return cached_option
            if cached_id.isdigit():
                return CourseOption(id=int(cached_id), name=f"Course ID {cached_id}")

        return None

    def _refresh_meetings_display(self):
        """Update the section meetings listbox with current meeting times"""
        self.meetings_listbox.delete(0, "end")
        
        if not self.section_meetings:
            self.meetings_listbox.insert("end", "(No section meeting times set yet)")
            return
        
        # Sort by section name for consistent display
        sorted_meetings = sorted(self.section_meetings.values(), 
                                key=lambda m: m.section_name.lower())
        
        for meeting in sorted_meetings:
            # Convert weekday numbers to day abbreviations
            day_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            days_str = " ".join(day_map.get(wd, str(wd)) for wd in meeting.weekdays)
            time_str = meeting.start_time.strftime("%H:%M")
            
            display_line = f"{meeting.section_name:<30} | {days_str:<15} | {time_str}"
            self.meetings_listbox.insert("end", display_line)

    def _load_config_and_init(self):
        try:
            cfg = configparser.ConfigParser()
            read_files = cfg.read(self.config_path)
            if not read_files:
                # Create a default/dummy config file so users have a starting point.
                cfg['canvas'] = {
                    'base_url': 'https://uncc.instructure.com/',
                    'api_key': '<YOUR_TOKEN_HERE>',
                }
                cfg['semester'] = {
                    'calendar_url': 'https://registrar.example.edu/academic-calendar',
                    'anchor': 'Academic Calendar',
                }
                cfg['time'] = {
                    'timezone': 'America/New_York',
                    'due_time_hhmm': '23:59',
                }
                try:
                    with open(self.config_path, 'w', encoding='utf-8') as fh:
                        cfg.write(fh)
                    messagebox.showinfo(
                        'Config created',
                        f"No config found. A default config was created at {self.config_path}.\n\nPlease edit it with real values and then click 'Reload semester' or restart the app."
                    )
                    self._set_status(f"Created default config at {self.config_path}. Edit it and reload.")
                    return
                except Exception as e:
                    raise FileNotFoundError(f"Config not found and failed to create default: {e}")

            # Canvas
            base_url = cfg.get("canvas", "base_url").strip()
            api_key = cfg.get("canvas", "api_key").strip()
            course_id = cfg.get("canvas", "course_id", fallback="").strip() or None
            self.canvas_cfg = CanvasConfig(base_url=base_url, api_key=api_key, course_id=course_id)

            # Semester
            cal_url = cfg.get("semester", "calendar_url").strip()
            anchor = cfg.get("semester", "anchor").strip()
            self.sem_cfg = SemesterConfig(calendar_url=cal_url, anchor=anchor)

            # Time
            tz_key = cfg.get("time", "timezone").strip()
            due_time = parse_hhmm(cfg.get("time", "due_time_hhmm").strip())
            self.time_cfg = TimeConfig(timezone=tz_key, due_time=due_time)

            self.tz = safe_zoneinfo(self.time_cfg.timezone)
            self.client = CanvasClient(self.canvas_cfg)

            if self.canvas_cfg.course_id:
                self.course_var.set(f"Course ID {self.canvas_cfg.course_id}")
                self._update_banner_search_term(None)
            else:
                self.course_var.set("(not selected)")
                self._update_banner_search_term(None)

            # Lab numbering base (optional in config under [semester] -> lab_base)
            try:
                raw_lab_base = cfg.get("semester", "lab_base", fallback=None)
                if raw_lab_base is not None:
                    self.lab_base = int(raw_lab_base)
                else:
                    # not specified in config: defer to auto-detect after loading assignments
                    self.lab_base = None
            except Exception:
                self.lab_base = None

            # Load saved section meeting times if available
            self._load_section_meetings()

            # Load the Canvas course list in the background so the picker is searchable.
            self._load_courses_async()

            self._set_status("Config loaded. Click 'Reload semester' then 'Load Canvas data'.")
        except Exception as e:
            messagebox.showerror("Config error", str(e))
            self._set_status("Config load failed.")

    def _save_section_meetings(self):
        """Save current section meeting times to section_meetings.ini"""
        if not self.section_meetings or not self.canvas_cfg or not self.canvas_cfg.course_id:
            return

        meetings_config_path = "section_meetings.ini"
        cfg = configparser.ConfigParser()
        
        # Read existing file to preserve other courses' data
        if os.path.exists(meetings_config_path):
            cfg.read(meetings_config_path)

        course_id = str(self.canvas_cfg.course_id)
        
        # Remove old sections for this course
        sections_to_remove = [s for s in cfg.sections() if s.startswith(f"course_{course_id}_section_")]
        for section_key in sections_to_remove:
            cfg.remove_section(section_key)

        # Add/update course metadata section
        course_meta_key = f"course_{course_id}_metadata"
        if course_meta_key in cfg:
            cfg.remove_section(course_meta_key)
        cfg[course_meta_key] = {
            'course_id': course_id,
            'course_name': self.course_name or '(unknown)'
        }

        # Add current sections for this course
        for section_id, meeting in self.section_meetings.items():
            section_key = f"course_{course_id}_section_{section_id}"
            # Convert weekdays list to comma-separated string
            weekdays_str = ",".join(str(wd) for wd in meeting.weekdays)
            # Convert time to HH:MM string
            time_str = meeting.start_time.strftime("%H:%M")

            cfg[section_key] = {
                'course_id': course_id,
                'section_id': str(meeting.section_id),
                'section_name': meeting.section_name,
                'weekdays': weekdays_str,
                'start_time': time_str
            }

        try:
            with open(meetings_config_path, 'w', encoding='utf-8') as fh:
                cfg.write(fh)
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save section meetings: {e}")

    def _load_section_meetings(self):
        """Load section meeting times from section_meetings.ini if it exists for current course"""
        # Clear existing section meetings first
        self.section_meetings.clear()
        
        if not self.canvas_cfg or not self.canvas_cfg.course_id:
            self._refresh_meetings_display()
            return
        
        meetings_config_path = "section_meetings.ini"
        if not os.path.exists(meetings_config_path):
            self._refresh_meetings_display()
            return  # No saved meetings yet

        try:
            cfg = configparser.ConfigParser()
            cfg.read(meetings_config_path)
            
            course_id = str(self.canvas_cfg.course_id)
            prefix = f"course_{course_id}_section_"

            for section_key in cfg.sections():
                if not section_key.startswith(prefix):
                    continue

                section_id = int(cfg.get(section_key, "section_id"))
                section_name = cfg.get(section_key, "section_name")
                weekdays_str = cfg.get(section_key, "weekdays")
                time_str = cfg.get(section_key, "start_time")

                # Parse weekdays from comma-separated string
                weekdays = [int(wd.strip()) for wd in weekdays_str.split(",") if wd.strip()]
                # Parse time
                start_time = parse_hhmm(time_str)

                self.section_meetings[section_id] = SectionMeeting(
                    section_id=section_id,
                    section_name=section_name,
                    weekdays=weekdays,
                    start_time=start_time
                )

            if self.section_meetings:
                self._set_status(f"Loaded {len(self.section_meetings)} saved section meeting times.")
            self._refresh_meetings_display()
        except Exception as e:
            # Don't show error dialog on load, just log it
            print(f"Warning: Failed to load section meetings: {e}")
            self._refresh_meetings_display()

    def on_reload_semester(self):
        if not self.sem_cfg:
            return
        self._set_status("Loading semester calendar...")
        self._run_threaded(self._load_semester_worker)

    def _load_semester_worker(self):
        try:
            cands = scrape_semester_calendar(self.sem_cfg.calendar_url, self.sem_cfg.anchor)
            # scrape_semester_calendar now returns a list of candidates
            def handle_candidates():
                try:
                    if len(cands) == 1:
                        chosen = cands[0]
                    else:
                        # ask user to pick from list
                        dlg = SemesterSelectionDialog(self, cands)
                        self.wait_window(dlg)
                        chosen = dlg.result

                    if not chosen:
                        self._set_status("Semester load cancelled by user.")
                        return

                    self.calendar = chosen
                    weeks = chosen.instructional_weeks()
                    msg = (
                        f"Semester loaded. Classes begin: {chosen.classes_begin.isoformat()}"
                        + (f", Classes end: {chosen.classes_end.isoformat()}" if chosen.classes_end else ", Classes end: (not found; using fallback)")
                        + f". Instructional weeks: {len(weeks)}."
                    )
                    self._set_status(msg)
                    term_display = chosen.term_name or "(not found)"
                    begin_display = chosen.classes_begin.isoformat()
                    end_display = chosen.classes_end.isoformat() if chosen.classes_end else "(not found)"
                    self.term_var.set(term_display)
                    self.begin_var.set(begin_display)
                    self.end_var.set(end_display)
                except Exception as e:
                    messagebox.showerror("Semester load error", str(e))
                    self._set_status("Semester load failed.")

            # schedule interaction on main thread
            self.after(0, handle_candidates)
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Semester load error", str(e)))
            self.after(0, lambda: self._set_status("Semester load failed."))

    def on_load_canvas(self):
        if not self.client:
            return
        resolved_course = self._resolve_course_selection()
        if not resolved_course:
            messagebox.showinfo("No course selected", "Type to search the course list and pick a Canvas course first.")
            return
        if not self.canvas_cfg or self.canvas_cfg.course_id != str(resolved_course.id):
            self._set_active_course(str(resolved_course.id))
        self._set_status("Loading Canvas sections + assignments...")
        self._run_threaded(self._load_canvas_worker)

    def on_edit_config(self):
        dlg = ConfigDialog(self, self.config_path)
        self.wait_window(dlg)

    def on_search_section_times(self):
        if not self.client:
            return

        resolved_course = self._resolve_course_selection()
        if not resolved_course:
            messagebox.showerror("No course selected", "Type to search the course list and pick a Canvas course first.")
            return
        if not self.sections:
            messagebox.showerror("No sections loaded", "Load Canvas data first, then search for section times.")
            return
        if not self.calendar:
            messagebox.showerror("No semester loaded", "Click 'Reload semester' first so Banner term selection is implicit from the selected semester.")
            return

        query_text = self.banner_search_var.get().strip()
        parsed_query = parse_banner_course_query(query_text)
        if not parsed_query:
            messagebox.showerror("Invalid Banner search term", "Enter a course number like 2181L, or subject + number like ECGR 2181L.")
            return

        self._set_status("Searching Banner for section times...")
        term_hint = (self.calendar.term_name or "").strip() or (resolved_course.term_name or "").strip()
        self._run_threaded(lambda: self._search_section_times_worker(resolved_course, query_text, term_hint, list(self.sections)))

    def _search_section_times_worker(self, course_option: CourseOption, query_text: str, term_hint: str, sections: List[Tuple[int, str]]):
        try:
            resolved_term = self.banner_client.resolve_term_code(term_hint)
            if not resolved_term:
                raise ValueError(
                    f"Could not map semester term '{term_hint or '(empty)'}' to a Banner term code. Try reloading semester and selecting the correct term."
                )
            autofilled = self.banner_client.autofill_section_meetings_for_query(query_text, term_hint, sections)
            if not autofilled:
                raise ValueError(
                    f"No Banner section times were found for {query_text} in {term_hint or 'the selected term'}."
                )

            def apply_results():
                for section_id, meeting in autofilled.items():
                    self.section_meetings[section_id] = meeting
                self._save_section_meetings()
                self._refresh_meetings_display()
                self._set_status(f"Loaded {len(autofilled)} section meeting time(s) from Banner.")

            self.after(0, apply_results)
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Banner search error", str(e)))
            self.after(0, lambda: self._set_status("Banner section search failed."))

    def _load_canvas_worker(self):
        try:
            sections = self.client.list_sections()
            assignments = self.client.list_assignments()
            # Auto-detect lab numbering base: if any classified assignment uses 0, prefer base 0.
            # If config explicitly set lab_base, do not override it.
            try:
                any_zero = any((classify_assignment(a.name) and classify_assignment(a.name)[1] == 0) for a in assignments)
                if self.lab_base is None:
                    if any_zero:
                        self.lab_base = 0
                        self.after(0, lambda: self._set_status("Detected Lab 0 in assignments; using lab base = 0."))
                    else:
                        # default to 1 when no Lab 0 present
                        self.lab_base = 1
                        # do not spam status for normal default behavior
                else:
                    # lab_base explicitly set in config; leave as-is
                    pass
            except Exception:
                # non-fatal; ignore detection failures and ensure a sensible default
                if self.lab_base is None:
                    self.lab_base = 1
            # Fetch course title for display
            try:
                course_info = self.client.get_course()
                course_name = str(course_info.get("name", "")).strip()
                self.course_name = course_name  # Store for later use
            except Exception:
                course_name = "(unknown)"
                self.course_name = None

            # Keep all assignments; automation will filter display
            self.sections = sections
            self.assignments = assignments

            # Reload section meetings for this course (don't clear them)
            # They were already loaded in _load_config_and_init or when switching courses
            self.after(0, self._refresh_tree_empty)
            self.after(0, self._refresh_meetings_display)
            # update course title label
            self.after(0, lambda: self.course_var.set(course_name))
            # Update course selector in case we have a new course name
            self.after(0, self._update_course_selector)
            self.after(0, lambda: self._set_status(f"Loaded {len(sections)} sections and {len(assignments)} assignments."))
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Canvas load error", str(e)))
            self.after(0, lambda: self._set_status("Canvas load failed."))

    def _format_existing_due_dates(self, assignment: Assignment) -> str:
        """Format existing Canvas due dates for display"""
        if not assignment.overrides:
            return "(no overrides set)"
        
        parts = []
        for override in assignment.overrides:
            section_id = override.get("course_section_id")
            due_at = override.get("due_at")
            
            if section_id and due_at:
                # Get section name if available
                sname = None
                for sid, name in self.sections:
                    if sid == section_id:
                        sname = name
                        break
                
                if not sname:
                    sname = f"Section {section_id}"
                
                # Parse ISO datetime and format it
                try:
                    from dateutil import parser as date_parser
                    due_dt = date_parser.parse(due_at)
                    formatted = due_dt.strftime('%Y-%m-%d %H:%M %Z')
                    parts.append(f"{sname}: {formatted}")
                except Exception:
                    parts.append(f"{sname}: {due_at}")
        
        return " | ".join(parts) if parts else "(no overrides set)"

    def _refresh_tree_empty(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        # Show just the auto-eligible assignments
        eligible = [a for a in self.assignments if classify_assignment(a.name)]
        eligible.sort(key=lambda x: x.name.lower())
        for a in eligible:
            existing_dates = self._format_existing_due_dates(a)
            self.tree.insert("", "end", values=(a.name, existing_dates, "(not computed yet)"))

    def on_set_meetings(self):
        if not self.sections:
            messagebox.showinfo("No sections", "Load Canvas data first.")
            return

        for sid, sname in self.sections:
            # Get existing meeting for this section if available
            existing_meeting = self.section_meetings.get(sid)
            dlg = SectionMeetingDialog(self, sid, sname, existing_meeting)
            self.wait_window(dlg)
            if dlg.result is None:
                # user canceled; keep any previously entered and move on
                continue
            wds, tm = dlg.result
            self.section_meetings[sid] = SectionMeeting(
                section_id=sid,
                section_name=sname,
                weekdays=wds,
                start_time=tm
            )

        # Save section meetings to config file after they have been entered
        self._save_section_meetings()
        
        # Update the display
        self._refresh_meetings_display()
        
        # Update course selector in case this is a new course
        self._update_course_selector()

        self._set_status(f"Meeting times set for {len(self.section_meetings)}/{len(self.sections)} sections.")

    def on_show_skipped_weeks(self):
        if not self.calendar:
            messagebox.showinfo("No semester", "Load a semester first (Reload semester).")
            return

        skipped = self.calendar.skipped_weeks()
        if not skipped:
            messagebox.showinfo("No skipped weeks", "No weeks appear to be skipped due to closures.")
            return

        dlg = SkippedWeeksDialog(self, skipped)
        self.wait_window(dlg)

    def on_compute(self):
        if not self.calendar:
            messagebox.showinfo("Semester not loaded", "Click 'Reload semester' first.")
            return
        if not self.tz or not self.time_cfg:
            messagebox.showinfo("Time config missing", "Time configuration not loaded.")
            return
        if not self.assignments:
            messagebox.showinfo("No assignments", "Load Canvas data first.")
            return
        if len(self.section_meetings) == 0:
            messagebox.showinfo("No meeting times", "Set section meeting times first.")
            return

        delay = int(self.delay_var.get())
        self._set_status("Computing due dates...")
        self._run_threaded(lambda: self._compute_worker(delay))

    def _compute_worker(self, delay: int):
        try:
            suggestions = build_due_suggestions(
                assignments=self.assignments,
                section_meetings=list(self.section_meetings.values()),
                calendar=self.calendar,
                tz=self.tz,
                due_time=self.time_cfg.due_time,
                delay_weeks=delay,
                lab_base=(int(self.lab_base) if getattr(self, "lab_base", None) is not None else 1),
            )
            self.suggestions, unschedulable_ids = suggestions
            self.unschedulable_assignment_ids = set(unschedulable_ids)
            self.after(0, self._refresh_tree_with_suggestions)

            def set_compute_status():
                msg = f"Computed due dates for {len(self.suggestions)} assignments."
                if self.unschedulable_assignment_ids:
                    msg += f" Skipped {len(self.unschedulable_assignment_ids)} assignment(s) that exceed available instructional weeks."
                self._set_status(msg)

            self.after(0, set_compute_status)
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Compute error", str(e)))
            self.after(0, lambda: self._set_status("Compute failed."))

    def _refresh_tree_with_suggestions(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)

        suggestions_by_id = {s.assignment_id: s for s in self.suggestions}
        eligible = [a for a in self.assignments if classify_assignment(a.name)]
        eligible.sort(key=lambda x: x.name.lower())

        for a in eligible:
            sug = suggestions_by_id.get(a.id)
            if sug is not None:
                parts = []
                for sid, due_dt in sorted(
                    sug.due_by_section.items(),
                    key=lambda x: self.section_meetings.get(x[0], SectionMeeting(x[0], str(x[0]), [], dt.time())).section_name.lower()
                ):
                    sname = self.section_meetings[sid].section_name if sid in self.section_meetings else str(sid)
                    parts.append(f"{sname}: {due_dt.strftime('%Y-%m-%d %H:%M %Z')}")
                preview = " | ".join(parts)
                
                # Format existing due dates for display
                existing_dates = self._format_existing_due_dates(a)
                
                self.tree.insert("", "end", values=(a.name, existing_dates, preview))
            elif a.id in self.unschedulable_assignment_ids:
                self.tree.insert(
                    "",
                    "end",
                    values=(a.name, "(not scheduled: exceeds available instructional weeks)", ""),
                    tags=("unschedulable",),
                )

    def on_apply(self):
        if not self.client:
            return
        if not self.suggestions:
            messagebox.showinfo("Nothing to apply", "Compute due dates first.")
            return

        if not messagebox.askyesno("Apply to Canvas", "This will set/overwrite per-section due dates (assignment overrides). Continue?"):
            return

        self._set_status("Applying due dates to Canvas...")
        self._run_threaded(self._apply_worker)

    def _apply_worker(self):
        try:
            total = 0
            for sug in self.suggestions:
                for sid, due_dt in sug.due_by_section.items():
                    due_iso = isoformat_z(due_dt)
                    self.client.set_due_for_section(sug.assignment_id, sid, due_iso)
                    total += 1
            self.after(0, lambda: self._set_status(f"Applied {total} due dates (overrides) to Canvas."))
            self.after(0, lambda: messagebox.showinfo("Done", f"Applied {total} due dates (overrides) to Canvas."))
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Apply error", str(e)))
            self.after(0, lambda: self._set_status("Apply failed."))

    def _run_threaded(self, fn):
        t = threading.Thread(target=fn, daemon=True)
        t.start()


def main():
    config_path = "config.ini"
    if len(sys.argv) >= 2:
        config_path = sys.argv[1]
    app = App(config_path=config_path)
    app.mainloop()


if __name__ == "__main__":
    main()
