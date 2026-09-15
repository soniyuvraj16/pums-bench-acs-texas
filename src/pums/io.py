"""Loading PUMS columns out of the parquet cache.

Everything is stored as text (see build_cache). These helpers do the explicit
conversion to numbers, and they keep PUMS's "blank means outside the universe"
rule intact: a blank becomes NaN, never 0.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import cache_path


@lru_cache(maxsize=None)
def columns_available(kind: str, year: int) -> tuple[str, ...]:
    return tuple(pq.ParquetFile(cache_path(kind, year)).schema.names)


def load(kind: str, year: int, columns) -> pd.DataFrame:
    """Read the named columns as raw strings.

    Reading only what a question needs keeps memory low - the person file has
    286 columns and we rarely want more than a handful.
    """
    columns = list(dict.fromkeys(columns))  # de-dupe, keep order
    available = set(columns_available(kind, year))
    missing = [c for c in columns if c not in available]
    if missing:
        raise KeyError(f"{kind} {year} has no column(s): {missing}")
    return pq.read_table(cache_path(kind, year), columns=columns).to_pandas()


def num(series: pd.Series) -> pd.Series:
    """Text column -> float, with blanks as NaN.

    PUMS writes a blank field for records outside a variable's universe. Those
    must stay NaN so they can be excluded, not silently counted as zero.
    """
    s = series.astype("string").str.strip()
    s = s.replace({"": pd.NA})
    return pd.to_numeric(s, errors="coerce").astype("float64")


def ints(series: pd.Series) -> pd.Series:
    """Text column -> nullable integer (for codes used in comparisons)."""
    return num(series).astype("Int64")


def codes(series: pd.Series) -> pd.Series:
    """Text column -> stripped string, for character-coded variables.

    Variables like NAICSP, SOCP and OCCP are genuinely character codes with
    meaningful leading zeros, so they must not go through num().
    """
    return series.astype("string").str.strip()


def person(year: int, columns) -> pd.DataFrame:
    return load("person", year, columns)


def household(year: int, columns) -> pd.DataFrame:
    return load("household", year, columns)


def adj_factor(raw) -> pd.Series | float:
    """ADJINC / ADJHSG are stored as 7-digit implied-6-decimal integers.

    The dictionary gives e.g. ADJINC = 1055457, meaning multiply by 1.055457.
    """
    if isinstance(raw, pd.Series):
        return num(raw) / 1_000_000.0
    return float(raw) / 1_000_000.0


def as_float_matrix(df: pd.DataFrame, columns) -> np.ndarray:
    """Stack replicate-weight columns into an (n_rows, n_reps) float array."""
    return np.column_stack([num(df[c]).to_numpy() for c in columns])
