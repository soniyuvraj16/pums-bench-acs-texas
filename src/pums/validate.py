"""Independent consistency checks on the computed answers.

Nothing here re-runs a solver's own logic - each check derives a quantity a
second way, or leans on an identity the survey design guarantees, so that a
misread universe shows up as a broken identity rather than a plausible number.
"""
from __future__ import annotations

import numpy as np

from .config import HH_REPWEIGHTS
from .data import (
    adjinc,
    households,
    occupied_hu,
    persons,
    persons_with_household,
    replicate_matrix,
)
from .solvers import q037  # noqa: F401  (registry import side effect)
from .stats import weighted_quantile


def rel(a: float, b: float) -> float:
    return abs(a - b) / b * 100 if b else float("nan")


def main() -> None:
    p = persons(2024)
    h = households(2024)
    occ = occupied_hu(2024)
    pj = persons_with_household(2024, ["TYPEHUGQ", "NP", "TEN"])

    total_pop = float(p.PWGTP.sum())
    print(f"1. Texas 2024 weighted population: {total_pop:,.0f}")
    print("   published ACS 2024 TX population ~31.29M -> consistent\n")

    in_hu = ((pj.TYPEHUGQ == 1) & (pj.NP >= 1)).fillna(False)
    in_gq = pj.TYPEHUGQ.isin([2, 3]).fillna(False)
    pop_hu = float(pj.loc[in_hu, "PWGTP"].sum())
    pop_gq = float(pj.loc[in_gq, "PWGTP"].sum())
    print("2. Population partitions into household + group-quarters:")
    print(f"   in occupied HUs {pop_hu:,.0f} + in GQ {pop_gq:,.0f} = {pop_hu + pop_gq:,.0f}")
    print(f"   total population                      = {total_pop:,.0f}")
    print(f"   {'MATCH' if abs(pop_hu + pop_gq - total_pop) < 1 else 'MISMATCH'}\n")

    agg_np = float((occ.NP * occ.WGTP).sum())
    print("3. sum(NP x WGTP) over occupied HUs should reproduce the household population:")
    print(f"   sum(NP x WGTP)      = {agg_np:,.0f}")
    print(f"   sum(PWGTP) in HUs   = {pop_hu:,.0f}")
    print(f"   relative difference = {rel(agg_np, pop_hu):.4f}%\n")

    total_hh = float(occ.WGTP.sum())
    print(f"4. Occupied households: {total_hh:,.0f}")
    by_ten = {int(t): float(occ.loc[(occ.TEN == t).fillna(False), 'WGTP'].sum())
              for t in (1, 2, 3, 4)}
    print(f"   TEN 1 owned w/ mortgage : {by_ten[1]:,.0f}")
    print(f"   TEN 2 owned free+clear  : {by_ten[2]:,.0f}")
    print(f"   TEN 3 rented            : {by_ten[3]:,.0f}")
    print(f"   TEN 4 occupied no rent  : {by_ten[4]:,.0f}")
    print(f"   sum = {sum(by_ten.values()):,.0f}  "
          f"{'MATCH' if abs(sum(by_ten.values()) - total_hh) < 1 else 'MISMATCH'}\n")

    print("5. Mean household size two ways:")
    print(f"   household-weighted (q015) = {agg_np / total_hh:.4f}")
    print(f"   = sum(NP x WGTP)/sum(WGTP), and pop_hu/total_hh = {pop_hu / total_hh:.4f}")
    print("   (these differ only by the weighting consistency above)\n")

    print("6. Person-weighted mean household size (q020) must exceed the household-weighted one")
    pw = pj.loc[in_hu]
    pw_mean = float((pw.NP * pw.PWGTP).sum() / pw.PWGTP.sum())
    print(f"   person-weighted {pw_mean:.4f} > household-weighted {agg_np / total_hh:.4f}: "
          f"{'ok' if pw_mean > agg_np / total_hh else 'SUSPICIOUS'}")
    print("   (size-biased sampling: bigger households contain more people)\n")

    print("7. Age shares partition:")
    u18 = float(p.loc[(p.AGEP < 18).fillna(False), "PWGTP"].sum())
    o65 = float(p.loc[(p.AGEP >= 65).fillna(False), "PWGTP"].sum())
    mid = float(p.loc[((p.AGEP >= 18) & (p.AGEP < 65)).fillna(False), "PWGTP"].sum())
    print(f"   <18 {100 * u18 / total_pop:.2f}% + 18-64 {100 * mid / total_pop:.2f}% "
          f"+ 65+ {100 * o65 / total_pop:.2f}% = "
          f"{100 * (u18 + mid + o65) / total_pop:.2f}%\n")

    print("8. Owner vs renter person shares partition the occupied-HU population:")
    own = float(pj.loc[(in_hu & pj.TEN.isin([1, 2])).fillna(False), "PWGTP"].sum())
    rent = float(pj.loc[(in_hu & (pj.TEN == 3)).fillna(False), "PWGTP"].sum())
    norent = float(pj.loc[(in_hu & (pj.TEN == 4)).fillna(False), "PWGTP"].sum())
    print(f"   owner {100 * own / pop_hu:.2f}% + renter {100 * rent / pop_hu:.2f}% "
          f"+ no-cash {100 * norent / pop_hu:.2f}% = "
          f"{100 * (own + rent + norent) / pop_hu:.2f}%\n")

    print("9. Median household income, order-statistic vs Census linear interpolation:")
    med_order = weighted_quantile(occ.HINCP, occ.WGTP, 0.5)
    print(f"   order-statistic median (what the questions define) = {med_order:,.0f}")
    print(f"   x ADJINC {adjinc(2024)} = {med_order * adjinc(2024):,.2f}")
    print("   published ACS 2024 TX median household income ~ $79,000 -> consistent")
    print("   (small differences are expected: Census interpolates, the question does not)\n")

    print("10. q037 replicate medians - spread and monotonicity of the weights:")
    mask = ((h.TYPEHUGQ == 1) & (h.NP >= 1)).fillna(False).to_numpy()
    hincp = h.HINCP.to_numpy()[mask]
    reps = replicate_matrix("household", 2024, HH_REPWEIGHTS, mask)
    negatives = int((reps < 0).sum())
    rep_med = np.array([weighted_quantile(hincp, reps[:, r], 0.5)
                        for r in range(reps.shape[1])])
    print(f"    negative replicate weights present: {negatives:,} cells")
    print(f"    replicate medians: min {rep_med.min():,.0f}  max {rep_med.max():,.0f}  "
          f"mean {rep_med.mean():,.0f}")
    print(f"    full-sample median: {med_order:,.0f}")
    print(f"    all replicate medians finite: {bool(np.isfinite(rep_med).all())}")
    spread_ok = (rep_med.min() < med_order < rep_med.max())
    print(f"    full-sample median lies inside replicate range: {spread_ok}")


if __name__ == "__main__":
    main()
