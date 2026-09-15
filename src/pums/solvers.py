"""One solver per competition question.

Each docstring records the universe, weight and adjustment taken from the
question text, so a wrong answer can be traced to a misreading rather than
hunted for in the arithmetic. Solvers return raw numbers; run.py formats them
using the answer_format column of questions.csv.
"""
from __future__ import annotations

import numpy as np

from .config import HH_REPWEIGHTS, PERSON_REPWEIGHTS
from .data import (
    adjhsg,
    adjinc,
    householders,
    households,
    max_age_by_serial,
    occupied_hu,
    persons,
    persons_with_household,
    replicate_matrix,
)
from .registry import solver
from .stats import replicate_se, weighted_mean, weighted_quantile

# ==========================================================================
# 1-6: unweighted record counts
# ==========================================================================

@solver("q001", "All rows of the 2024 person file.")
def q001() -> float:
    return float(len(persons(2024)))


@solver("q002", "2024 person rows with AGEP >= 65, unweighted.")
def q002() -> float:
    p = persons(2024)
    return float(((p.AGEP >= 65).fillna(False)).sum())


@solver("q003", "2024 household rows with TYPEHUGQ == 1, unweighted.")
def q003() -> float:
    h = households(2024)
    return float(((h.TYPEHUGQ == 1).fillna(False)).sum())


@solver("q004", "2024 person rows with SEX == 2, unweighted.")
def q004() -> float:
    p = persons(2024)
    return float(((p.SEX == 2).fillna(False)).sum())


@solver("q005", "2024 person rows with ESR exactly 3. Blank ESR (under 16) is not counted.")
def q005() -> float:
    p = persons(2024)
    # ESR is NaN for under-16s; == 3 is False there, so blanks drop out.
    return float(((p.ESR == 3).fillna(False)).sum())


@solver("q006", "Vacant units: TYPEHUGQ == 1 and NP == 0, unweighted.")
def q006() -> float:
    h = households(2024)
    return float((((h.TYPEHUGQ == 1) & (h.NP == 0)).fillna(False)).sum())


# ==========================================================================
# 7-8: weighted totals
# ==========================================================================

@solver("q007", "Sum PWGTP over persons with AGEP >= 65.")
def q007() -> float:
    p = persons(2024)
    return float(p.loc[(p.AGEP >= 65).fillna(False), "PWGTP"].sum())


@solver("q008", "Sum WGTP over occupied HUs with TEN == 3 (rented).")
def q008() -> float:
    h = occupied_hu(2024)
    return float(h.loc[(h.TEN == 3).fillna(False), "WGTP"].sum())


# ==========================================================================
# 9-17: weighted means, medians, percentiles, shares
# ==========================================================================

@solver("q009", "PWGTP-weighted mean AGEP over all person records.")
def q009() -> float:
    p = persons(2024)
    return weighted_mean(p.AGEP, p.PWGTP)


@solver("q010", "Mean WAGP*ADJINC over ESR in (1,2). WAGP == 0 stays in the universe.")
def q010() -> float:
    p = persons(2024)
    uni = p[p.ESR.isin([1, 2]).fillna(False).to_numpy()]
    adjusted = uni.WAGP * adjinc(2024)
    return weighted_mean(adjusted, uni.PWGTP)


@solver("q011", "PWGTP-weighted median AGEP over all person records.")
def q011() -> float:
    p = persons(2024)
    return weighted_quantile(p.AGEP, p.PWGTP, 0.5)


@solver("q012", "Median of UNADJUSTED WAGP over ESR in (1,2) and WAGP > 0, then x ADJINC.")
def q012() -> float:
    p = persons(2024)
    keep = (p.ESR.isin([1, 2]) & (p.WAGP > 0)).fillna(False)
    uni = p[keep.to_numpy()]
    median_unadjusted = weighted_quantile(uni.WAGP, uni.PWGTP, 0.5)
    return median_unadjusted * adjinc(2024)


@solver("q013", "WGTP-weighted median HINCP over occupied HUs, x ADJINC.")
def q013() -> float:
    h = occupied_hu(2024)
    return weighted_quantile(h.HINCP, h.WGTP, 0.5) * adjinc(2024)


@solver("q014", "PWGTP-weighted 75th percentile of AGEP over all person records.")
def q014() -> float:
    p = persons(2024)
    return weighted_quantile(p.AGEP, p.PWGTP, 0.75)


@solver("q015", "WGTP-weighted mean NP over occupied HUs.")
def q015() -> float:
    h = occupied_hu(2024)
    return weighted_mean(h.NP, h.WGTP)


@solver("q016", "100 * PWGTP(AGEP < 18) / PWGTP(all persons).")
def q016() -> float:
    p = persons(2024)
    under18 = p.loc[(p.AGEP < 18).fillna(False), "PWGTP"].sum()
    return 100.0 * float(under18) / float(p.PWGTP.sum())


@solver("q017", "100 * PWGTP(AGEP >= 25 & SCHL >= 21) / PWGTP(AGEP >= 25).")
def q017() -> float:
    p = persons(2024)
    base = (p.AGEP >= 25).fillna(False)
    grad = (base & (p.SCHL >= 21)).fillna(False)
    return 100.0 * float(p.loc[grad, "PWGTP"].sum()) / float(p.loc[base, "PWGTP"].sum())


# ==========================================================================
# 18-25: person <-> household joins
# ==========================================================================

@solver("q018", "Sum PWGTP over persons whose household is an occupied rented HU.")
def q018() -> float:
    pj = persons_with_household(2024, ["TYPEHUGQ", "NP", "TEN"])
    keep = ((pj.TYPEHUGQ == 1) & (pj.NP >= 1) & (pj.TEN == 3)).fillna(False)
    return float(pj.loc[keep, "PWGTP"].sum())


@solver("q019", "PERSON-weighted median of household HINCP over persons in occupied HUs, x ADJINC.")
def q019() -> float:
    pj = persons_with_household(2024, ["TYPEHUGQ", "NP", "HINCP"])
    keep = ((pj.TYPEHUGQ == 1) & (pj.NP >= 1)).fillna(False)
    uni = pj[keep.to_numpy()]
    return weighted_quantile(uni.HINCP, uni.PWGTP, 0.5) * adjinc(2024)


@solver("q020", "PERSON-weighted mean of household NP over persons in occupied HUs.")
def q020() -> float:
    pj = persons_with_household(2024, ["TYPEHUGQ", "NP"])
    keep = ((pj.TYPEHUGQ == 1) & (pj.NP >= 1)).fillna(False)
    uni = pj[keep.to_numpy()]
    return weighted_mean(uni.NP, uni.PWGTP)


@solver("q021", "Sum WGTP over occupied HUs whose SPORDER == 1 person has SEX == 2.")
def q021() -> float:
    h = occupied_hu(2024).merge(householders(2024), on="SERIALNO",
                                how="left", validate="one_to_one")
    return float(h.loc[(h.HH_SEX == 2).fillna(False), "WGTP"].sum())


@solver("q022", "WGTP-weighted median of householder AGEP over occupied HUs.")
def q022() -> float:
    h = occupied_hu(2024).merge(householders(2024), on="SERIALNO",
                                how="left", validate="one_to_one")
    return weighted_quantile(h.HH_AGEP, h.WGTP, 0.5)


@solver("q023", "Sum WGTP over occupied HUs where any same-SERIALNO person has AGEP >= 65.")
def q023() -> float:
    h = occupied_hu(2024).copy()
    oldest = h.SERIALNO.map(max_age_by_serial(2024))
    return float(h.loc[(oldest >= 65).fillna(False), "WGTP"].sum())


@solver("q024", "Sum PWGTP over persons whose household record has TYPEHUGQ in (2,3).")
def q024() -> float:
    pj = persons_with_household(2024, ["TYPEHUGQ"])
    keep = pj.TYPEHUGQ.isin([2, 3]).fillna(False)
    return float(pj.loc[keep, "PWGTP"].sum())


@solver("q025", "100 * PWGTP(household TEN in (1,2)) / PWGTP(persons in occupied HUs).")
def q025() -> float:
    pj = persons_with_household(2024, ["TYPEHUGQ", "NP", "TEN"])
    base = ((pj.TYPEHUGQ == 1) & (pj.NP >= 1)).fillna(False)
    owner = (base & pj.TEN.isin([1, 2])).fillna(False)
    return 100.0 * float(pj.loc[owner, "PWGTP"].sum()) / float(pj.loc[base, "PWGTP"].sum())


# ==========================================================================
# 26-28: conditioned household medians, with the right adjustment factor
# ==========================================================================

@solver("q026", "WGTP-weighted median HINCP over occupied HUs with householder AGEP >= 65, x ADJINC.")
def q026() -> float:
    h = occupied_hu(2024).merge(householders(2024), on="SERIALNO",
                                how="left", validate="one_to_one")
    uni = h[(h.HH_AGEP >= 65).fillna(False).to_numpy()]
    return weighted_quantile(uni.HINCP, uni.WGTP, 0.5) * adjinc(2024)


@solver("q027", "WGTP-weighted median GRNTP over occupied HUs with non-blank GRNTP, x ADJHSG.")
def q027() -> float:
    h = occupied_hu(2024)
    uni = h[h.GRNTP.notna().to_numpy()]
    # Gross rent takes the HOUSING adjustment factor, not the income one.
    return weighted_quantile(uni.GRNTP, uni.WGTP, 0.5) * adjhsg(2024)


@solver("q028", "WGTP-weighted median VALP over owner-occupied HUs with non-blank VALP, x ADJHSG.")
def q028() -> float:
    h = occupied_hu(2024)
    keep = (h.TEN.isin([1, 2]) & h.VALP.notna()).fillna(False)
    uni = h[keep.to_numpy()]
    return weighted_quantile(uni.VALP, uni.WGTP, 0.5) * adjhsg(2024)


# ==========================================================================
# 29-33: cross-year comparisons, each year using its own ADJINC
# ==========================================================================

@solver("q029", "2023 PWGTP-weighted median AGEP over all person records.")
def q029() -> float:
    p = persons(2023)
    return weighted_quantile(p.AGEP, p.PWGTP, 0.5)


@solver("q030", "sum(PWGTP) 2024 minus sum(PWGTP) 2023, all person records.")
def q030() -> float:
    return float(persons(2024).PWGTP.sum()) - float(persons(2023).PWGTP.sum())


def _median_hincp_adjusted(year: int) -> float:
    """Weighted median HINCP over that year's occupied HUs, times that year's ADJINC."""
    h = occupied_hu(year)
    return weighted_quantile(h.HINCP, h.WGTP, 0.5) * adjinc(year)


@solver("q031", "2023 WGTP-weighted median HINCP x the 2023 file's own ADJINC.")
def q031() -> float:
    return _median_hincp_adjusted(2023)


@solver("q032", "Percent change in adjusted median HINCP, 2023 -> 2024, unrounded intermediates.")
def q032() -> float:
    m23 = _median_hincp_adjusted(2023)
    m24 = _median_hincp_adjusted(2024)
    return 100.0 * (m24 - m23) / m23


def _renter_households(year: int) -> float:
    h = occupied_hu(year)
    return float(h.loc[(h.TEN == 3).fillna(False), "WGTP"].sum())


@solver("q033", "Renter-occupied household total, 2024 minus 2023.")
def q033() -> float:
    return _renter_households(2024) - _renter_households(2023)


# ==========================================================================
# 34-38: replicate-weight standard errors (SDR)
# ==========================================================================

@solver("q034", "SDR SE of sum(PWGTP) over persons AGEP >= 65.")
def q034() -> float:
    p = persons(2024)
    mask = (p.AGEP >= 65).fillna(False).to_numpy()
    X = float(p.loc[mask, "PWGTP"].sum())
    reps = replicate_matrix("person", 2024, PERSON_REPWEIGHTS, mask)
    return replicate_se(X, reps.sum(axis=0))


@solver("q035", "SDR SE of sum(WGTP) over occupied rented HUs.")
def q035() -> float:
    h = households(2024)
    mask = ((h.TYPEHUGQ == 1) & (h.NP >= 1) & (h.TEN == 3)).fillna(False).to_numpy()
    X = float(h.loc[mask, "WGTP"].sum())
    reps = replicate_matrix("household", 2024, HH_REPWEIGHTS, mask)
    return replicate_se(X, reps.sum(axis=0))


@solver("q036", "SDR SE of the mean adjusted WAGP over ESR in (1,2); replicate weight in both parts of the ratio.")
def q036() -> float:
    p = persons(2024)
    mask = p.ESR.isin([1, 2]).fillna(False).to_numpy()
    wage = (p.WAGP * adjinc(2024)).to_numpy()[mask]
    X = float(np.sum(wage * p.PWGTP.to_numpy()[mask]) / np.sum(p.PWGTP.to_numpy()[mask]))
    reps = replicate_matrix("person", 2024, PERSON_REPWEIGHTS, mask)
    rep_est = (wage @ reps) / reps.sum(axis=0)
    return replicate_se(X, rep_est)


@solver("q037", "SDR SE of the adjusted median HINCP; each replicate re-runs the identical median definition.")
def q037() -> float:
    h = households(2024)
    mask = ((h.TYPEHUGQ == 1) & (h.NP >= 1)).fillna(False).to_numpy()
    hincp = h.HINCP.to_numpy()[mask]
    factor = adjinc(2024)

    X = weighted_quantile(hincp, h.WGTP.to_numpy()[mask], 0.5) * factor
    reps = replicate_matrix("household", 2024, HH_REPWEIGHTS, mask)
    # Replicate weights can be negative, so the cumulative sum is not
    # monotonic; weighted_quantile scans rather than binary-searches.
    rep_est = [weighted_quantile(hincp, reps[:, r], 0.5) * factor
               for r in range(reps.shape[1])]
    return replicate_se(X, rep_est)


@solver("q038", "SDR SE (in percentage points) of the percent of 25+ with SCHL >= 21.")
def q038() -> float:
    p = persons(2024)
    base = (p.AGEP >= 25).fillna(False).to_numpy()
    grad = (base & (p.SCHL >= 21).fillna(False).to_numpy())
    # Restrict to the denominator universe, flagging the numerator inside it.
    grad_in_base = grad[base]

    w = p.PWGTP.to_numpy()[base]
    X = 100.0 * float(w[grad_in_base].sum()) / float(w.sum())

    reps = replicate_matrix("person", 2024, PERSON_REPWEIGHTS, base)
    rep_est = 100.0 * reps[grad_in_base].sum(axis=0) / reps.sum(axis=0)
    return replicate_se(X, rep_est)
