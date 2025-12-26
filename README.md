# UNCC Canvas Lab Planner

Canvas Lab Planner is a Tkinter GUI tool to help instructors set per-section due dates for
lab-related Canvas assignments based on the UNCC academic calendar. It scrapes the registrar
calendar (best-effort), identifies instructional weeks while skipping university-closed weeks,
and maps assignments named with the patterns `Pre-Lab <N>`, `Lab <N>`, and `Post-Lab <N>` to
per-section due dates (applied as Canvas assignment overrides).

**Key features**
- Scrapes the registrar academic calendar to determine classes begin/end and closure ranges.
- Lets you define meeting days and start times per Canvas section.
- Auto-computes per-section due dates for Pre-Lab/Lab/Post-Lab assignments and applies them
  via Canvas assignment overrides.
- Skips weeks with closures (holidays/breaks) so due dates do not fall on cancelled weeks.
- Lab-numbering is auto-detected: the tool will use `Lab 0` as the base if any assignment named
  `Lab 0` exists; otherwise it defaults to base 1. You can also set a fixed `lab_base` manually
  by editing `config.ini` (see Configuration below).
- Debug output can be enabled via the `CANVAS_LAB_PLANNER_DEBUG` environment variable.

Prerequisites
- Python 3.9+ (3.11 recommended).
- Install Python packages from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

On Windows, `zoneinfo` may require the `tzdata` package:

```powershell
pip install tzdata
```

Configuration
- Copy and edit `config.ini` (example below).
- Required sections/keys:
  - `[canvas]`
    - `base_url` — your Canvas URL (e.g. `https://uncc.instructure.com/`)
    - `api_key` — Canvas API token with permissions to edit assignment overrides
    - `course_id` — Canvas course ID to operate on
  - `[semester]`
    - `calendar_url` — registrar calendar page URL
    - `anchor` — (best-effort) anchor or section identifier used by the scraper
    - `lab_base` — optional (0 or 1). If present in the file it will be used. If omitted, the
      application will auto-detect `Lab 0` presence when Canvas assignments are loaded and set
      the lab base to 0 if found; otherwise it defaults to 1.
  - `[time]`
    - `timezone` — IANA tz key (e.g. `America/New_York`)
    - `due_time_hhmm` — default due time for Lab/Post-Lab in `HH:MM` 24-hour format

Example `config.ini`:

```ini
[canvas]
base_url = https://uncc.instructure.com/
api_key = <YOUR_TOKEN>
course_id = 12345

[semester]
calendar_url = https://registrar.charlotte.edu/calendars-schedules/academic-year-fall-2025-spring-2026/#spring-full-term
anchor = spring-full-term

[time]
timezone = America/New_York
due_time_hhmm = 23:59
```

Usage
- Launch the GUI:

```powershell
python main.py         # or: python main.py path\to\config.ini
```

Typical workflow:
- Click `Reload semester` to scrape the registrar calendar and populate Term / First day / Last day.
- Click `Load Canvas data` to fetch sections and assignments. The tool will auto-detect `Lab 0`
  and use base 0 if present; to override detection, set `lab_base` in `config.ini` manually.
- Click `Set section meeting times` to enter meeting days and start time for each section.
- Click `Auto-compute due dates` to compute per-section suggestions.
- Review the preview column in the UI; click `Apply to Canvas` to push per-section due-date overrides.

Config editor and notes
- The app includes an `Edit config` dialog that lets you edit common fields and saves `config.ini`.
  Note: the in-app editor no longer exposes `lab_base`; if you want to force a specific lab base,
  edit `config.ini` manually and add `lab_base = 0` or `lab_base = 1` under the `[semester]`
  section.
- The config dialog writes a simplified `config.ini` and does not preserve arbitrary extra
  sections/keys. If you have extra custom keys you wish to keep, edit the file manually or
  back it up before using the dialog.

Debugging
- Enable verbose debug prints for scraper and heuristics:

```powershell
$env:CANVAS_LAB_PLANNER_DEBUG = '1'; python main.py
```

Troubleshooting
- If the scraper cannot find term dates, the registrar page may have an unexpected structure.
  Enable debug output to see parsed table rows and refine heuristics or provide the page URL
  for tweaking.
- On Windows, if you see timezone errors from `zoneinfo`, install `tzdata`.

Notes and limitations
- Scraping registrar pages is best-effort; different institutions or page layouts may require
  additional heuristics.
- The tool matches assignment names using the pattern `Pre-Lab <N>`, `Lab <N>`, `Post-Lab <N>`.
  If your naming differs, adjust assignment names in Canvas or extend `LAB_NAME_RE` in `main.py`.
- The app currently stores no state beyond the `config.ini` file; meeting times are entered per-run.

Contributing
- Bug reports and PRs welcome. Please open issues/PRs in the repository with sample registrar
  rows or assignment-name examples that fail heuristics.

License
- (Add your preferred license here.)

Contact
- For questions, provide the registrar URL and a sample assignment list/rows that are problematic
  so heuristics can be improved.
