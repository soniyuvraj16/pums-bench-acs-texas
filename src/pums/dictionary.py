"""Lookup tool for the official PUMS data dictionary.

The dictionary CSV has two row shapes:

    NAME,<var>,<C|N>,<width>,"<description>"
    VAL,<var>,<C|N>,<width>,"<lo>","<hi>","<label>"

A VAL row with lo == hi is a single code; lo != hi is a range (e.g. AGEP
0..99, or the "bbbb" blank marker). Blanks are written as a run of 'b'.

Usage:
    python -m pums.dictionary AGEP ESR WKHP
    python -m pums.dictionary --year 2023 HINCP
"""
from __future__ import annotations

import argparse
import csv
import sys
from functools import lru_cache

from .config import ROOT

DICT_PATH = {
    2024: ROOT / "PUMS_Data_Dictionary_2024.csv",
    2023: ROOT / "PUMS_Data_Dictionary_2023.csv",
}


@lru_cache(maxsize=None)
def _parse(year: int):
    """-> {var: {"desc": str, "type": str, "width": int, "values": [(lo, hi, label)]}}"""
    entries: dict[str, dict] = {}
    with open(DICT_PATH[year], newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh):
            if not row or len(row) < 5:
                continue
            tag, var = row[0].strip(), row[1].strip()
            if tag == "NAME":
                entries[var] = {
                    "desc": row[4],
                    "type": row[2].strip(),
                    "width": int(row[3]) if row[3].strip().isdigit() else None,
                    "values": [],
                }
            elif tag == "VAL" and var in entries:
                lo, hi = row[4], row[5] if len(row) > 5 else row[4]
                label = row[6] if len(row) > 6 else ""
                entries[var]["values"].append((lo, hi, label))
    return entries


def describe(var: str, year: int = 2024) -> str:
    entries = _parse(year)
    var = var.upper()
    if var not in entries:
        near = [k for k in entries if k.startswith(var[:3])][:10]
        return f"{var}: NOT FOUND in {year} dictionary. Similar: {near}"

    e = entries[var]
    lines = [f"{var} ({e['type']}{e['width']})  {e['desc']}"]
    for lo, hi, label in e["values"]:
        code = lo if lo == hi else f"{lo}..{hi}"
        lines.append(f"    {code:<24} {label}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Print PUMS data dictionary entries.")
    ap.add_argument("vars", nargs="+")
    ap.add_argument("--year", type=int, default=2024, choices=[2023, 2024])
    args = ap.parse_args()
    for v in args.vars:
        print(describe(v, args.year))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
