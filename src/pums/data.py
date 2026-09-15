"""Typed, cached frames for the columns the questions actually use.

Loading is cached per (kind, year) so each parquet file is touched once even
though 38 solvers ask for it.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import io

PERSON_COLS = ["SERIALNO", "SPORDER", "PWGTP", "AGEP", "SEX", "ESR", "WAGP", "SCHL", "ADJINC"]
HH_COLS = ["SERIALNO", "WGTP", "NP", "TYPEHUGQ", "TEN", "HINCP", "GRNTP", "VALP",
           "ADJINC", "ADJHSG"]


@lru_cache(maxsize=None)
def persons(year: int) -> pd.DataFrame:
    raw = io.person(year, PERSON_COLS)
    out = pd.DataFrame({"SERIALNO": raw.SERIALNO.astype("string").str.strip()})
    for col in PERSON_COLS[1:]:
        out[col] = io.num(raw[col])
    out["ADJINC_F"] = out["ADJINC"] / 1_000_000.0
    return out


@lru_cache(maxsize=None)
def households(year: int) -> pd.DataFrame:
    raw = io.household(year, HH_COLS)
    out = pd.DataFrame({"SERIALNO": raw.SERIALNO.astype("string").str.strip()})
    for col in HH_COLS[1:]:
        out[col] = io.num(raw[col])
    out["ADJINC_F"] = out["ADJINC"] / 1_000_000.0
    out["ADJHSG_F"] = out["ADJHSG"] / 1_000_000.0
    return out


@lru_cache(maxsize=None)
def adjinc(year: int) -> float:
    """The income adjustment factor carried by that year's own file."""
    values = households(year)["ADJINC"].dropna().unique()
    assert len(values) == 1, f"ADJINC not constant in {year}: {values}"
    return float(values[0]) / 1_000_000.0


@lru_cache(maxsize=None)
def adjhsg(year: int) -> float:
    """The housing adjustment factor - used by GRNTP and VALP, not by income."""
    values = households(year)["ADJHSG"].dropna().unique()
    assert len(values) == 1, f"ADJHSG not constant in {year}: {values}"
    return float(values[0]) / 1_000_000.0


# --------------------------------------------------------------------------
# Universes
# --------------------------------------------------------------------------

def occupied_hu(year: int) -> pd.DataFrame:
    """Occupied housing units: TYPEHUGQ == 1 and NP >= 1."""
    h = households(year)
    keep = (h.TYPEHUGQ == 1) & (h.NP >= 1)
    return h[keep.fillna(False).to_numpy()]


@lru_cache(maxsize=None)
def householders(year: int) -> pd.DataFrame:
    """One row per household: the SPORDER == 1 person's characteristics.

    Every occupied housing unit has exactly one such record (verified in
    diagnose.py), so this is a clean 1:1 key.
    """
    p = persons(year)
    hh = p[(p.SPORDER == 1).fillna(False).to_numpy()]
    return hh[["SERIALNO", "AGEP", "SEX"]].rename(
        columns={"AGEP": "HH_AGEP", "SEX": "HH_SEX"}
    )


@lru_cache(maxsize=None)
def max_age_by_serial(year: int) -> pd.Series:
    """Oldest person in each household, for 'at least one member aged X' tests."""
    p = persons(year)
    return p.groupby("SERIALNO", observed=True)["AGEP"].max()


def persons_with_household(year: int, hh_columns) -> pd.DataFrame:
    """Left-join every person to their household record on SERIALNO.

    SERIALNO is a string key; both sides are read as text so a 13-character
    id like '2024HU0000123' never gets mangled into a float.
    """
    p = persons(year)
    h = households(year)[["SERIALNO", *hh_columns]]
    return p.merge(h, on="SERIALNO", how="left", validate="many_to_one")


# --------------------------------------------------------------------------
# Replicate weights
# --------------------------------------------------------------------------

def replicate_matrix(kind: str, year: int, rep_columns, mask=None,
                     chunk: int = 20) -> np.ndarray:
    """Load the 80 replicate weights as an (n_rows, 80) float array.

    Read in column blocks and subset as we go: materialising all 80 columns of
    the person file at once is avoidable memory pressure.
    """
    blocks = []
    for start in range(0, len(rep_columns), chunk):
        names = list(rep_columns[start:start + chunk])
        frame = io.load(kind, year, names)
        block = np.column_stack([io.num(frame[c]).to_numpy() for c in names])
        blocks.append(block if mask is None else block[mask])
        del frame, block
    return np.hstack(blocks)
