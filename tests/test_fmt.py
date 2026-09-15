"""Formatting is scored by exact string match, so it gets its own tests.

The rule that matters here is "round half away from zero": Python's built-in
round() would give integer(2.5) == "2", which is wrong for this competition.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pums.fmt import apply_format, decimal2, integer, percent1  # noqa: E402

INTEGER_CASES = [
    (0.5, "1"), (1.5, "2"), (2.5, "3"), (3.5, "4"),
    (-0.5, "-1"), (-2.5, "-3"),
    (52300.0, "52300"), (52300, "52300"),
    (0.4, "0"), (0.6, "1"), (0.0, "0"),
    (1234567.0, "1234567"),          # no thousands separators
]

DECIMAL2_CASES = [
    (0.4, "0.40"),                   # always two places
    (0.005, "0.01"), (2.675, "2.68"), (1.0, "1.00"),
    (-0.125, "-0.13"), (0.0, "0.00"),
    (12.344, "12.34"), (12.345, "12.35"),
]

PERCENT1_CASES = [
    (12.3, "12.3"), (12.25, "12.3"), (9.95, "10.0"),
    (0.0, "0.0"), (100.0, "100.0"), (7.0, "7.0"),
    (-3.45, "-3.5"),
]


def _check(fn, cases, label):
    bad = []
    for value, expected in cases:
        got = fn(value)
        status = "ok" if got == expected else "FAIL"
        if got != expected:
            bad.append((value, expected, got))
        print(f"  {label}({value!r:>12}) = {got!r:>10}  expect {expected!r:>10}  {status}")
    return bad


def main() -> int:
    failures = []
    print("integer")
    failures += _check(integer, INTEGER_CASES, "integer")
    print("\ndecimal2")
    failures += _check(decimal2, DECIMAL2_CASES, "decimal2")
    print("\npercent1")
    failures += _check(percent1, PERCENT1_CASES, "percent1")

    print("\ndispatch")
    assert apply_format(0.4, "decimal2") == "0.40"
    assert apply_format(2.5, "integer") == "3"
    assert apply_format(12.25, "percent1") == "12.3"
    print("  apply_format routes by name: ok")

    # No scientific notation for large values.
    assert "e" not in integer(1e12).lower(), integer(1e12)
    print(f"  integer(1e12) = {integer(1e12)}: ok")

    if failures:
        print(f"\n{len(failures)} FAILURES: {failures}")
        return 1
    print("\nall formatting tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
