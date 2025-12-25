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


# -----------------------------
# Data models
# -----------------------------

@dataclass
class CanvasConfig:
    base_url: str
    api_key: str
    course_id: str


@dataclass
class SemesterConfig:
    calendar_url: str
    anchor: str


@dataclass
class TimeConfig:
    timezone: str
    due_time: dt.time


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
        self.course_id = cfg.course_id
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {cfg.api_key}"})

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
        url = self._url(f"/api/v1/courses/{self.course_id}/assignments")
        data = self._paginate(url, params={"per_page": 100})
        return [Assignment(id=int(a["id"]), name=str(a.get("name", "")).strip()) for a in data]

    def list_sections(self) -> List[Tuple[int, str]]:
        url = self._url(f"/api/v1/courses/{self.course_id}/sections")
        data = self._paginate(url, params={"per_page": 100})
        return [(int(s["id"]), str(s.get("name", "")).strip()) for s in data]

    def get_course(self) -> dict:
        """Return course details from Canvas API for the configured course id."""
        url = self._url(f"/api/v1/courses/{self.course_id}")
        r = self.s.get(url, timeout=30)
        r.raise_for_status()
        return r.json()

    def list_assignment_overrides(self, assignment_id: int) -> List[dict]:
        url = self._url(f"/api/v1/courses/{self.course_id}/assignments/{assignment_id}/overrides")
        return self._paginate(url, params={"per_page": 100})

    def create_override(self, assignment_id: int, section_id: int, due_at_iso: str) -> dict:
        url = self._url(f"/api/v1/courses/{self.course_id}/assignments/{assignment_id}/overrides")
        payload = {
            "assignment_override[course_section_id]": section_id,
            "assignment_override[due_at]": due_at_iso,
        }
        r = self.s.post(url, data=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def update_override(self, assignment_id: int, override_id: int, due_at_iso: str) -> dict:
        url = self._url(f"/api/v1/courses/{self.course_id}/assignments/{assignment_id}/overrides/{override_id}")
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
) -> List[DueSuggestion]:
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
    # Need enough weeks to place labs
    base_index = 1 + max(0, delay_weeks)  # "week after first week" + delay

    # Compute how far into `weeks` our highest-numbered lab will land,
    # taking into account whether numbering starts at 0 or 1.
    needed_last_index = base_index + (max_n - lab_base)
    if needed_last_index >= len(weeks) or (base_index < 0):
        raise ValueError(
            f"Not enough instructional weeks to schedule up to Lab {max_n} (lab base={lab_base}). "
            f"Have {len(weeks)} instructional weeks, need at least {needed_last_index + 1}."
        )

    # lab_num -> week_monday (compute per actual lab number present)
    lab_week: Dict[int, dt.date] = {}
    for n in range(min(num for _, _, num in classified), max_n + 1):
        idx = base_index + (n - lab_base)
        if idx < 0 or idx >= len(weeks):
            continue
        lab_week[n] = weeks[idx]

    suggestions: List[DueSuggestion] = []

    for a, kind, num in classified:
        wk = lab_week.get(num)
        if not wk:
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
    return suggestions


# -----------------------------
# Tkinter UI
# -----------------------------

class SectionMeetingDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, section_id: int, section_name: str):
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
        self.days_entry.insert(0, "M W")

        ttk.Label(frm, text="Start time (24h HH:MM):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.time_entry = ttk.Entry(frm, width=10)
        self.time_entry.grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
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

        ttk.Label(frm, text="Course ID:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.course_id_e = ttk.Entry(frm, width=20)
        self.course_id_e.grid(row=3, column=1, sticky="w", pady=(6, 0))
        self.course_id_e.insert(0, cfg.get("canvas", "course_id", fallback=""))

        # Semester
        ttk.Label(frm, text="Calendar URL:").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.cal_url_e = ttk.Entry(frm, width=60)
        self.cal_url_e.grid(row=4, column=1, sticky="w", pady=(8, 0))
        self.cal_url_e.insert(0, cfg.get("semester", "calendar_url", fallback=""))

        ttk.Label(frm, text="Anchor:").grid(row=5, column=0, sticky="w", pady=(6, 0))
        self.anchor_e = ttk.Entry(frm, width=40)
        self.anchor_e.grid(row=5, column=1, sticky="w", pady=(6, 0))
        self.anchor_e.insert(0, cfg.get("semester", "anchor", fallback=""))

        # lab_base is auto-detected; not editable via this dialog

        # Time
        ttk.Label(frm, text="Timezone (IANA):").grid(row=7, column=0, sticky="w", pady=(8, 0))
        self.tz_e = ttk.Entry(frm, width=40)
        self.tz_e.grid(row=7, column=1, sticky="w", pady=(8, 0))
        self.tz_e.insert(0, cfg.get("time", "timezone", fallback="America/New_York"))

        ttk.Label(frm, text="Due time (HH:MM):").grid(row=8, column=0, sticky="w", pady=(6, 0))
        self.due_time_e = ttk.Entry(frm, width=10)
        self.due_time_e.grid(row=8, column=1, sticky="w", pady=(6, 0))
        self.due_time_e.insert(0, cfg.get("time", "due_time_hhmm", fallback="23:59"))

        btns = ttk.Frame(frm)
        btns.grid(row=9, column=0, columnspan=2, sticky="e", pady=(12, 0))
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
            course_id = self.course_id_e.get().strip()
            cal_url = self.cal_url_e.get().strip()
            anchor = self.anchor_e.get().strip()
            tz = self.tz_e.get().strip()
            due_time = self.due_time_e.get().strip()
            tz = self.tz_e.get().strip()
            due_time = self.due_time_e.get().strip()

            # basic validation
            if not base_url or not api_key or not course_id:
                raise ValueError("Canvas base_url, api_key and course_id are required.")
            # validate due_time
            _ = parse_hhmm(due_time)

            cfg = configparser.ConfigParser()
            cfg["canvas"] = {
                "base_url": base_url,
                "api_key": api_key,
                "course_id": course_id,
            }
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
        self.geometry("1100x650")

        self.config_path = config_path
        self.canvas_cfg: Optional[CanvasConfig] = None
        self.sem_cfg: Optional[SemesterConfig] = None
        self.time_cfg: Optional[TimeConfig] = None

        self.tz: Optional[ZoneInfo] = None
        self.calendar: Optional[SemesterCalendar] = None
        self.client: Optional[CanvasClient] = None

        self.sections: List[Tuple[int, str]] = []
        self.section_meetings: Dict[int, SectionMeeting] = {}
        self.assignments: List[Assignment] = []
        self.suggestions: List[DueSuggestion] = []

        self._build_ui()
        self._load_config_and_init()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="both", expand=True)

        # Controls
        ctrl = ttk.LabelFrame(top, text="Controls", padding=10)
        ctrl.pack(fill="x", padx=5, pady=5)

        ttk.Label(ctrl, text="Config file:").grid(row=0, column=0, sticky="w")
        self.cfg_label = ttk.Label(ctrl, text=self.config_path)
        self.cfg_label.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(ctrl, text="Delay weeks (Lab 1 starts week after first week + delay):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.delay_var = tk.IntVar(value=0)
        self.delay_spin = ttk.Spinbox(ctrl, from_=0, to=20, textvariable=self.delay_var, width=5)
        self.delay_spin.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        # Term / calendar info (populated after scraping)
        ttk.Label(ctrl, text="Term:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.term_var = tk.StringVar(value="(not loaded)")
        ttk.Label(ctrl, textvariable=self.term_var).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(ctrl, text="First day:").grid(row=3, column=0, sticky="w")
        self.begin_var = tk.StringVar(value="(not loaded)")
        ttk.Label(ctrl, textvariable=self.begin_var).grid(row=3, column=1, sticky="w", padx=(8, 0))

        ttk.Label(ctrl, text="Last day:").grid(row=4, column=0, sticky="w")
        self.end_var = tk.StringVar(value="(not loaded)")
        ttk.Label(ctrl, textvariable=self.end_var).grid(row=4, column=1, sticky="w", padx=(8, 0))

        # Canvas course title (populated after loading Canvas data)
        ttk.Label(ctrl, text="Course:").grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.course_var = tk.StringVar(value="(not loaded)")
        ttk.Label(ctrl, textvariable=self.course_var).grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        # Lab numbering is auto-detected (Lab 0 if present); no manual radio control.

        btnrow = ttk.Frame(ctrl)
        btnrow.grid(row=0, column=2, rowspan=6, sticky="e", padx=(20, 0))
        ttk.Button(btnrow, text="Reload semester", command=self.on_reload_semester).grid(row=0, column=0, padx=5)
        ttk.Button(btnrow, text="Load Canvas data", command=self.on_load_canvas).grid(row=0, column=1, padx=5)
        ttk.Button(btnrow, text="Set section meeting times", command=self.on_set_meetings).grid(row=0, column=2, padx=5)
        ttk.Button(btnrow, text="Auto-compute due dates", command=self.on_compute).grid(row=1, column=0, padx=5, pady=(8, 0))
        ttk.Button(btnrow, text="Apply to Canvas", command=self.on_apply).grid(row=1, column=1, padx=5, pady=(8, 0))
        ttk.Button(btnrow, text="Show skipped weeks", command=self.on_show_skipped_weeks).grid(row=1, column=2, padx=5, pady=(8, 0))
        ttk.Button(btnrow, text="Edit config", command=self.on_edit_config).grid(row=2, column=0, padx=5, pady=(8, 0))

        # Status
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(top, textvariable=self.status_var).pack(fill="x", padx=8, pady=(0, 8))

        # Treeview
        cols = ("assignment", "preview")
        self.tree = ttk.Treeview(top, columns=cols, show="headings", height=22)
        self.tree.heading("assignment", text="Assignment")
        self.tree.heading("preview", text="Auto Due Preview (per section)")
        self.tree.column("assignment", width=420, anchor="w")
        self.tree.column("preview", width=640, anchor="w")
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

    def _load_config_and_init(self):
        try:
            cfg = configparser.ConfigParser()
            read_files = cfg.read(self.config_path)
            if not read_files:
                # Create a default/dummy config file so users have a starting point.
                cfg['canvas'] = {
                    'base_url': 'https://canvas.example.edu',
                    'api_key': '<YOUR_TOKEN_HERE>',
                    'course_id': '12345',
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
            course_id = cfg.get("canvas", "course_id").strip()
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

            self._set_status("Config loaded. Click 'Reload semester' then 'Load Canvas data'.")
        except Exception as e:
            messagebox.showerror("Config error", str(e))
            self._set_status("Config load failed.")

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
        self._set_status("Loading Canvas sections + assignments...")
        self._run_threaded(self._load_canvas_worker)

    def on_edit_config(self):
        dlg = ConfigDialog(self, self.config_path)
        self.wait_window(dlg)

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
            except Exception:
                course_name = "(unknown)"

            # Keep all assignments; automation will filter display
            self.sections = sections
            self.assignments = assignments

            # Default meeting info empty; user sets it
            self.section_meetings.clear()
            self.after(0, self._refresh_tree_empty)
            # update course title label
            self.after(0, lambda: self.course_var.set(course_name))
            self.after(0, lambda: self._set_status(f"Loaded {len(sections)} sections and {len(assignments)} assignments."))
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Canvas load error", str(e)))
            self.after(0, lambda: self._set_status("Canvas load failed."))

    def _refresh_tree_empty(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        # Show just the auto-eligible assignments
        eligible = [a for a in self.assignments if classify_assignment(a.name)]
        eligible.sort(key=lambda x: x.name.lower())
        for a in eligible:
            self.tree.insert("", "end", values=(a.name, "(not computed yet)"))

    def on_set_meetings(self):
        if not self.sections:
            messagebox.showinfo("No sections", "Load Canvas data first.")
            return

        for sid, sname in self.sections:
            dlg = SectionMeetingDialog(self, sid, sname)
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
            self.suggestions = suggestions
            self.after(0, self._refresh_tree_with_suggestions)
            self.after(0, lambda: self._set_status(f"Computed due dates for {len(suggestions)} assignments."))
        except Exception as e:
            self.after(0, lambda e=e: messagebox.showerror("Compute error", str(e)))
            self.after(0, lambda: self._set_status("Compute failed."))

    def _refresh_tree_with_suggestions(self):
        for iid in self.tree.get_children():
            self.tree.delete(iid)

        # Pretty preview per section
        for sug in self.suggestions:
            parts = []
            for sid, due_dt in sorted(sug.due_by_section.items(), key=lambda x: self.section_meetings.get(x[0], SectionMeeting(x[0], str(x[0]), [], dt.time())).section_name.lower()):
                sname = self.section_meetings[sid].section_name if sid in self.section_meetings else str(sid)
                parts.append(f"{sname}: {due_dt.strftime('%Y-%m-%d %H:%M %Z')}")
            preview = " | ".join(parts)
            self.tree.insert("", "end", values=(sug.assignment_name, preview))

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
