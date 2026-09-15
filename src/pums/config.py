"""Paths and constants for the PUMS-Bench pipeline."""
from pathlib import Path

# Repository root (two levels up from src/pums/)
ROOT = Path(__file__).resolve().parents[2]

# Raw Census microdata, as extracted from the Census Bureau zips.
RAW = {
    ("person", 2024): ROOT / "csv_ptx" / "psam_p48.csv",
    ("household", 2024): ROOT / "csv_htx" / "psam_h48.csv",
    ("person", 2023): ROOT / "csv_ptx (2023)" / "psam_p48.csv",
    ("household", 2023): ROOT / "csv_htx (2023)" / "psam_h48.csv",
}

# Columnar cache; parquet lets us read only the columns a question needs.
CACHE = ROOT / "cache"

# Competition inputs / outputs
QUESTIONS = ROOT / "questions.csv"
SUBMISSION = ROOT / "submission.csv"

# SHA-256 of the exact 2024 Census zips named in the competition overview.
EXPECTED_SHA256 = {
    "csv_ptx.zip": "42ad7127988234d7dd30c23df8ada371b2135b5db912d2e634a6f901477afe63",
    "csv_htx.zip": "59274006389de6f8d6ee5f6e2eb48f7f1ccd4d3c9f730df857be641d5ef65b94",
}

# The 80 successive-difference replicate weights.
N_REPLICATES = 80
PERSON_REPWEIGHTS = [f"PWGTP{i}" for i in range(1, N_REPLICATES + 1)]
HH_REPWEIGHTS = [f"WGTP{i}" for i in range(1, N_REPLICATES + 1)]


def cache_path(kind: str, year: int) -> Path:
    return CACHE / f"{kind}_{year}.parquet"
