"""Tests for the weighted quantile and the replicate SE formula.

The quantile is checked against a deliberately slow, literal reading of the
spec sentence: "the smallest value v such that the sum of the weights over
records with value <= v is at least q * total". If the fast implementation
and the literal one ever disagree, the fast one is wrong.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pums.stats import replicate_se, weighted_median, weighted_quantile  # noqa: E402


def brute_force_quantile(values, weights, q):
    """Literal transcription of the spec sentence. O(n^2), for testing only."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    total = w.sum()
    target = q * total
    for cand in sorted(set(v.tolist())):          # ascending distinct values
        if w[v <= cand].sum() >= target:
            return cand
    return max(v.tolist())


CASES = [
    # (values, weights, label)
    ([1, 2, 3, 4], [1, 1, 1, 1], "uniform, even count"),
    ([1, 2, 3], [1, 1, 1], "uniform, odd count"),
    ([5, 5, 5, 9], [2, 3, 1, 4], "heavy ties"),
    ([1, 2, 3, 4], [10, 1, 1, 1], "mass at the bottom"),
    ([1, 2, 3, 4], [1, 1, 1, 10], "mass at the top"),
    ([3, 1, 2], [1, 5, 2], "unsorted input"),
    ([7], [3], "single record"),
    ([1, 2, 3, 4, 5], [1, -2, 4, 1, 1], "negative weight (replicate-like)"),
    ([1, 2, 3, 4, 5], [2, -1, -1, 3, 2], "several negatives"),
    ([10, 20, 30], [0, 5, 5], "zero weight present"),
]


def main() -> int:
    failures = []

    print("weighted_quantile vs literal spec transcription")
    for values, weights, label in CASES:
        for q in (0.5, 0.75, 0.25):
            got = weighted_quantile(values, weights, q)
            want = brute_force_quantile(values, weights, q)
            ok = got == want
            if not ok:
                failures.append((label, q, want, got))
            print(f"  q={q}  {label:<34} got={got:<6g} want={want:<6g} "
                  f"{'ok' if ok else 'FAIL'}")

    print("\nweighted_median is quantile at 0.5")
    assert weighted_median([1, 2, 3], [1, 1, 1]) == weighted_quantile([1, 2, 3], [1, 1, 1], 0.5)
    print("  ok")

    print("\nNaN values are dropped, not treated as zero")
    got = weighted_median([1.0, np.nan, 3.0], [1.0, 50.0, 1.0])
    want = brute_force_quantile([1.0, 3.0], [1.0, 1.0], 0.5)
    print(f"  got={got}  want={want}  {'ok' if got == want else 'FAIL'}")
    if got != want:
        failures.append(("nan handling", 0.5, want, got))

    print("\nreplicate_se matches the documented SDR formula")
    rng = np.random.default_rng(0)
    X = 1000.0
    reps = (X + rng.normal(0, 25, 80)).tolist()
    want = float(np.sqrt((4.0 / 80.0) * sum((r - X) ** 2 for r in reps)))
    got = replicate_se(X, reps)
    print(f"  got={got:.6f}  want={want:.6f}  {'ok' if abs(got - want) < 1e-9 else 'FAIL'}")
    if abs(got - want) >= 1e-9:
        failures.append(("replicate_se", None, want, got))

    # Zero variance -> zero SE; wrong replicate count -> error.
    assert replicate_se(5.0, [5.0] * 80) == 0.0
    try:
        replicate_se(5.0, [5.0] * 79)
        failures.append(("replicate count guard", None, "raise", "no raise"))
    except ValueError:
        pass
    print("  guards ok")

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures:
            print("   ", f)
        return 1
    print("\nall stats tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
