"""Weighted estimators and replicate-weight standard errors."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .config import N_REPLICATES


# --------------------------------------------------------------------------
# Point estimators
# --------------------------------------------------------------------------

def weighted_total(weights) -> float:
    """Population total: the sum of the weights over the universe."""
    w = np.asarray(weights, dtype="float64")
    return float(np.nansum(w))


def weighted_sum(values, weights) -> float:
    """Weighted sum of a quantity, e.g. aggregate income."""
    v = np.asarray(values, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    ok = ~np.isnan(v) & ~np.isnan(w)
    return float(np.sum(v[ok] * w[ok]))


def weighted_mean(values, weights) -> float:
    v = np.asarray(values, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    ok = ~np.isnan(v) & ~np.isnan(w)
    denom = np.sum(w[ok])
    if denom == 0:
        return float("nan")
    return float(np.sum(v[ok] * w[ok]) / denom)


def weighted_share(mask, weights) -> float:
    """Weighted proportion of the universe satisfying `mask` (0..1)."""
    m = np.asarray(mask, dtype="bool")
    w = np.asarray(weights, dtype="float64")
    ok = ~np.isnan(w)
    denom = np.sum(w[ok])
    if denom == 0:
        return float("nan")
    return float(np.sum(w[ok & m]) / denom)


def weighted_quantile(values, weights, q: float) -> float:
    """The quantile definition the questions state, implemented literally:

        the smallest value v such that the sum of the weights over records
        with value <= v is at least q * (total weight of the universe)

    Two details this has to get right:

    1. Ties. The condition is about *distinct* values, so all records sharing
       a value are accumulated before the test is applied. Testing per record
       could stop partway through a tie group.

    2. Negative weights. Replicate weights may be negative (see PUMS Accuracy
       of the Data), so the cumulative sum is NOT necessarily monotonic and a
       binary search over it would be wrong. We scan the distinct values in
       ascending order and take the first that satisfies the condition, which
       is what "the smallest value v such that ..." asks for.

    Records whose value is NaN (blank: outside the variable's universe) are
    dropped, and the total is taken over the records that remain. Where a
    question intends blanks to be excluded it says so ("non-blank GRNTP");
    where it does not, the data has no blanks in that universe anyway.
    """
    v = np.asarray(values, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    ok = ~np.isnan(v) & ~np.isnan(w)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return float("nan")

    order = np.argsort(v, kind="mergesort")
    v, w = v[order], w[order]

    total = float(np.sum(w))
    target = q * total

    # Cumulative weight at the END of each distinct-value run.
    cum = np.cumsum(w)
    last_of_run = np.r_[np.nonzero(np.diff(v))[0], v.size - 1]
    distinct = v[last_of_run]
    cum_at_value = cum[last_of_run]

    hits = np.nonzero(cum_at_value >= target)[0]
    if hits.size == 0:
        # Only reachable if the weights sum to <= 0; fall back to the maximum.
        return float(distinct[-1])
    return float(distinct[hits[0]])


def weighted_median(values, weights) -> float:
    return weighted_quantile(values, weights, 0.5)


# --------------------------------------------------------------------------
# Replicate-weight standard errors
# --------------------------------------------------------------------------

def replicate_se(estimate: float, replicate_estimates: Sequence[float]) -> float:
    """Successive difference replication standard error.

    From PUMS Accuracy of the Data:

        SE(X) = sqrt( (4/80) * sum_{r=1..80} (X_r - X)^2 )

    where X is the estimate from the full weight and X_r the same estimate
    recomputed with replicate weight r. Replicate weights may be negative;
    nothing here clips them.
    """
    reps = np.asarray(list(replicate_estimates), dtype="float64")
    if reps.size != N_REPLICATES:
        raise ValueError(f"expected {N_REPLICATES} replicate estimates, got {reps.size}")
    return float(np.sqrt((4.0 / N_REPLICATES) * np.sum((reps - float(estimate)) ** 2)))


def replicate_estimates(
    df: pd.DataFrame,
    rep_columns: Sequence[str],
    estimator: Callable[[pd.Series], float],
) -> list[float]:
    """Recompute `estimator` once per replicate weight column.

    `estimator` receives the replicate weight as a float Series aligned to df
    and returns that replicate's estimate.
    """
    from .io import num

    return [estimator(num(df[c])) for c in rep_columns]


def se_and_estimate(
    df: pd.DataFrame,
    weight_column: str,
    rep_columns: Sequence[str],
    estimator: Callable[[pd.Series], float],
) -> tuple[float, float]:
    """Return (point estimate, standard error) for an estimator over df."""
    from .io import num

    point = estimator(num(df[weight_column]))
    reps = replicate_estimates(df, rep_columns, estimator)
    return point, replicate_se(point, reps)


def margin_of_error(se: float, z: float = 1.645) -> float:
    """ACS publishes 90% margins of error, i.e. 1.645 * SE."""
    return z * se


def coefficient_of_variation(estimate: float, se: float) -> float:
    """CV as a percentage."""
    if estimate == 0:
        return float("nan")
    return 100.0 * se / estimate
