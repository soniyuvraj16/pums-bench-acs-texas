"""Checks on the data properties the question universes depend on.

Several questions name a universe without saying what to do about blanks
inside it (q013 says "occupied housing units" with no blank clause, while
q027 explicitly says "non-blank GRNTP"). This script establishes which
variables actually have blanks inside each universe, so the solvers can
handle them on evidence rather than assumption.
"""
from __future__ import annotations

import pandas as pd

from . import io


def blanks(series: pd.Series) -> int:
    s = series.astype("string").str.strip()
    return int((s.isna() | (s == "")).sum())


def main() -> None:
    pd.set_option("display.width", 120)

    print("=== adjustment factors by year ===")
    for year in (2023, 2024):
        p = io.person(year, ["ADJINC"])
        h = io.household(year, ["ADJINC", "ADJHSG"])
        print(f"  {year}: person ADJINC={p.ADJINC.unique()}  "
              f"hh ADJINC={h.ADJINC.unique()}  ADJHSG={h.ADJHSG.unique()}")

    print("\n=== 2024 household universe: occupied HUs (TYPEHUGQ==1 & NP>=1) ===")
    h = io.household(2024, ["SERIALNO", "WGTP", "NP", "TYPEHUGQ", "TEN",
                            "HINCP", "GRNTP", "VALP"])
    typ, np_ = io.ints(h.TYPEHUGQ), io.ints(h.NP)
    occ = ((typ == 1) & (np_ >= 1)).fillna(False)
    print(f"  occupied HU records: {int(occ.sum()):,}")
    print(f"  TYPEHUGQ==1 (all):   {int((typ == 1).sum()):,}")
    print(f"  vacant (T==1,NP==0): {int(((typ == 1) & (np_ == 0)).fillna(False).sum()):,}")
    sub = h[occ.to_numpy()]
    for col in ("HINCP", "GRNTP", "VALP", "TEN", "WGTP"):
        print(f"  blanks in {col:<6} within occupied HUs: {blanks(sub[col]):,}")
    print(f"  WGTP==0 within occupied HUs: {int((io.num(sub.WGTP) == 0).sum()):,}")
    ten = io.ints(sub.TEN)
    print(f"  TEN distribution: {ten.value_counts(dropna=False).sort_index().to_dict()}")

    print("\n=== 2024 person universe: civilian employed (ESR in 1,2) ===")
    p = io.person(2024, ["SERIALNO", "SPORDER", "PWGTP", "AGEP", "ESR",
                         "WAGP", "SCHL", "SEX"])
    esr = io.ints(p.ESR)
    emp = esr.isin([1, 2]).fillna(False)
    print(f"  records: {int(emp.sum()):,}")
    pe = p[emp.to_numpy()]
    print(f"  blanks in WAGP within universe: {blanks(pe.WAGP):,}")
    print(f"  WAGP==0 within universe:        {int((io.num(pe.WAGP) == 0).sum()):,}")

    print("\n=== 2024 person: SCHL / AGEP blanks ===")
    print(f"  AGEP blanks (all persons): {blanks(p.AGEP):,}")
    age = io.num(p.AGEP)
    a25 = (age >= 25).fillna(False)
    print(f"  persons 25+: {int(a25.sum()):,}, SCHL blanks within: {blanks(p[a25.to_numpy()].SCHL):,}")

    print("\n=== person <-> household join on SERIALNO (2024) ===")
    hh_keys = set(h.SERIALNO)
    p_keys = set(p.SERIALNO)
    print(f"  household SERIALNOs: {len(hh_keys):,}   person SERIALNOs: {len(p_keys):,}")
    print(f"  person SERIALNOs not in household file: {len(p_keys - hh_keys):,}")
    print(f"  occupied-HU SERIALNOs with no person record: "
          f"{len(set(h[occ.to_numpy()].SERIALNO) - p_keys):,}")

    sp1 = p[(io.ints(p.SPORDER) == 1).fillna(False).to_numpy()]
    occ_serials = set(h[occ.to_numpy()].SERIALNO)
    sp1_serials = set(sp1.SERIALNO)
    print(f"  SPORDER==1 records: {len(sp1):,} (unique SERIALNO: {len(sp1_serials):,})")
    print(f"  occupied HUs lacking a SPORDER==1 person: {len(occ_serials - sp1_serials):,}")

    print("\n=== group quarters (2024) ===")
    gq = h[(typ.isin([2, 3])).fillna(False).to_numpy()]
    print(f"  GQ household records: {len(gq):,}")
    print(f"  GQ WGTP values: {io.num(gq.WGTP).unique()[:5]}")
    gq_persons = p[p.SERIALNO.isin(set(gq.SERIALNO)).to_numpy()]
    print(f"  persons in GQ: {len(gq_persons):,}, weighted: "
          f"{io.num(gq_persons.PWGTP).sum():,.0f}")


if __name__ == "__main__":
    main()
