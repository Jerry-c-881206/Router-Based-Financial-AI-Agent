from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime


_RE_QUARTER = re.compile(r"^(?P<year>\d{4})-Q(?P<q>[1-4])$")
_RE_MONTH = re.compile(r"^(?P<year>\d{4})-(?P<month>0[1-9]|1[0-2])$")
_RE_YEAR = re.compile(r"^(?P<year>\d{4})$")


@dataclass(frozen=True)
class TimeBounds:
    time_range: str
    start_date: date
    end_date: date


def _last_day_of_month(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def parse_time_range(time_range: str, *, today: date | None = None) -> TimeBounds:
    """
    Parse normalized `time_range` from SDD Query Decomposer into date bounds.
    Expected: `YYYY-QN` / `YYYY-MM` / `YYYY`.
    """

    if today is None:
        today = date.today()

    time_range = time_range.strip()

    m_q = _RE_QUARTER.match(time_range)
    if m_q:
        year = int(m_q.group("year"))
        q = int(m_q.group("q"))
        start_month = (q - 1) * 3 + 1
        end_month = q * 3
        start_date = date(year, start_month, 1)
        end_date = date(year, end_month, _last_day_of_month(year, end_month))
        return TimeBounds(time_range=time_range, start_date=start_date, end_date=end_date)

    m_m = _RE_MONTH.match(time_range)
    if m_m:
        year = int(m_m.group("year"))
        month = int(m_m.group("month"))
        start_date = date(year, month, 1)
        end_date = date(year, month, _last_day_of_month(year, month))
        return TimeBounds(time_range=time_range, start_date=start_date, end_date=end_date)

    m_y = _RE_YEAR.match(time_range)
    if m_y:
        year = int(m_y.group("year"))
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        return TimeBounds(time_range=time_range, start_date=start_date, end_date=end_date)

    # Defensive fallback: if the LLM returned a non-normalized value, do not guess silently.
    raise ValueError(f"Unsupported time_range format: {time_range!r} (expected YYYY-QN / YYYY-MM / YYYY)")


def date_to_iso(d: date) -> str:
    return d.isoformat()


def is_older_than_three_months(data_date: date, *, today: date | None = None) -> bool:
    if today is None:
        today = date.today()
    # Roughly aligned with "3 months" concept.
    cutoff_year = today.year
    cutoff_month = today.month - 3
    while cutoff_month <= 0:
        cutoff_month += 12
        cutoff_year -= 1
    cutoff = date(cutoff_year, cutoff_month, 1)
    return data_date < cutoff


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

