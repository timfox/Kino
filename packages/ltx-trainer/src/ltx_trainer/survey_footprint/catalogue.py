"""CSV/TSV catalogue ingest and augmentation (Sec. 2.5)."""

from __future__ import annotations

import csv
import io
import re
from typing import Any

from ltx_trainer.survey_footprint.moc_engine import filter_coordinates, load_builtin_moc

_RA_ALIASES = re.compile(r"^(ra|radeg|ra_deg|raj2000|ra_icrs|alpha)$", re.I)
_DEC_ALIASES = re.compile(r"^(dec|de|decl|decdeg|dec_deg|dej2000|dec_icrs|delta)$", re.I)
_RA_HR = re.compile(r"ra.*hr|ra_hours", re.I)


def _match_column(headers: list[str], pattern: re.Pattern[str]) -> str | None:
    for h in headers:
        if pattern.match(h.strip()):
            return h
    return None


def detect_ra_dec_columns(headers: list[str]) -> tuple[str, str, bool]:
    """Return (ra_col, dec_col, ra_in_hours)."""
    ra_col = _match_column(headers, _RA_ALIASES)
    dec_col = _match_column(headers, _DEC_ALIASES)
    if ra_col is None or dec_col is None:
        raise ValueError("catalogue must contain ra and dec columns (fuzzy match supported)")
    ra_hours = bool(_RA_HR.search(ra_col))
    return ra_col, dec_col, ra_hours


def parse_catalogue_text(text: str, *, delimiter: str | None = None) -> tuple[list[str], list[dict[str, str]]]:
    """Parse CSV or TSV with header row."""
    if delimiter is None:
        delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if reader.fieldnames is None:
        raise ValueError("empty catalogue")
    headers = list(reader.fieldnames)
    rows = [dict(r) for r in reader]
    return headers, rows


def rows_to_coords(
    rows: list[dict[str, str]],
    ra_col: str,
    dec_col: str,
    *,
    ra_in_hours: bool = False,
) -> tuple[list[float], list[float]]:
    sample = [float(row[ra_col]) for row in rows[: min(20, len(rows))]]
    use_hours = ra_in_hours or _RA_HR.search(ra_col) is not None
    if not use_hours and sample and max(sample) <= 24.0 and min(sample) >= 0.0:
        use_hours = "deg" not in ra_col.lower()
    ra_list: list[float] = []
    dec_list: list[float] = []
    for row in rows:
        ra = float(row[ra_col])
        dec = float(row[dec_col])
        if use_hours:
            ra *= 15.0
        ra_list.append(ra % 360.0)
        dec_list.append(dec)
    return ra_list, dec_list


def augment_catalogue(
    headers: list[str],
    rows: list[dict[str, str]],
    survey_ids: list[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Append boolean in_<survey_id> columns (Sec. 2.5)."""
    ra_col, dec_col, ra_hr = detect_ra_dec_columns(headers)
    ra, dec = rows_to_coords(rows, ra_col, dec_col, ra_in_hours=ra_hr)
    new_headers = list(headers)
    for sid in survey_ids:
        col = f"in_{sid}"
        if col not in new_headers:
            new_headers.append(col)
    mocs = {sid: load_builtin_moc(sid) for sid in survey_ids}
    memberships = {sid: filter_coordinates(ra, dec, mocs[sid]) for sid in survey_ids}
    out_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        out = dict(row)
        for sid in survey_ids:
            out[f"in_{sid}"] = memberships[sid][i]
        out_rows.append(out)
    return new_headers, out_rows


def catalogue_to_csv(headers: list[str], rows: list[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()
