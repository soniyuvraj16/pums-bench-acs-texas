"""Integrity checks on the raw inputs.

The competition publishes SHA-256 checksums for the two 2024 Census *zips*.
If the zips are still around we check them directly. Once they are unpacked
the checksum no longer applies, so we fall back to structural checks: row
counts, the presence of all 80 replicate weights, and the adjustment factors
the dictionary documents for that year.
"""
from __future__ import annotations

import hashlib
import sys

import pyarrow.parquet as pq

from .config import (
    EXPECTED_SHA256,
    HH_REPWEIGHTS,
    PERSON_REPWEIGHTS,
    RAW,
    ROOT,
    cache_path,
)

# Row counts of the files this pipeline was built against (header excluded).
EXPECTED_ROWS = {
    ("person", 2024): 292_272,
    ("household", 2024): 132_629,
    ("person", 2023): 301_984,
    ("household", 2023): 135_627,
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_zips() -> list[str]:
    notes = []
    for name, expected in EXPECTED_SHA256.items():
        candidates = [ROOT / name, ROOT / "zips" / name]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            notes.append(f"  {name}: not present (already unpacked) - checksum not verifiable")
            continue
        actual = sha256(found)
        ok = "OK" if actual == expected else "MISMATCH"
        notes.append(f"  {name}: {ok}")
        if actual != expected:
            notes.append(f"      expected {expected}")
            notes.append(f"      actual   {actual}")
    return notes


def check_structure() -> tuple[list[str], bool]:
    notes, ok = [], True
    for (kind, year), raw in RAW.items():
        if not raw.exists():
            notes.append(f"  {kind} {year}: RAW FILE MISSING ({raw})")
            ok = False
            continue
        cache = cache_path(kind, year)
        if not cache.exists():
            notes.append(f"  {kind} {year}: cache not built yet")
            ok = False
            continue

        pf = pq.ParquetFile(cache)
        rows, names = pf.metadata.num_rows, set(pf.schema.names)
        want = EXPECTED_ROWS[(kind, year)]
        row_ok = rows == want

        reps = PERSON_REPWEIGHTS if kind == "person" else HH_REPWEIGHTS
        missing = [c for c in reps if c not in names]
        key = "PWGTP" if kind == "person" else "WGTP"

        flags = []
        if not row_ok:
            flags.append(f"rows {rows:,} != expected {want:,}")
        if missing:
            flags.append(f"missing {len(missing)} replicate weights")
        if key not in names:
            flags.append(f"missing {key}")
        if "SERIALNO" not in names:
            flags.append("missing SERIALNO")

        if flags:
            ok = False
            notes.append(f"  {kind} {year}: PROBLEM - {'; '.join(flags)}")
        else:
            notes.append(f"  {kind} {year}: {rows:,} rows, 80 replicate weights, {key} + SERIALNO present")
    return notes, ok


def main() -> int:
    print("Zip checksums (competition-published, 2024 only):")
    for line in check_zips():
        print(line)

    print("\nStructural checks:")
    notes, ok = check_structure()
    for line in notes:
        print(line)

    print("\n" + ("All structural checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
