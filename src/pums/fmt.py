"""Answer formatting.

Scoring is exact string match, so formatting is part of the answer. The
competition specifies three formats and one rounding rule:

    integer   no separators, no decimal point, no currency symbol -> 52300
    decimal2  exactly two decimal places, always                  -> 0.40
    percent1  a number, one decimal, no % sign                    -> 12.3

    Round half away from zero.

Python's built-in round() is banker's rounding (round-half-to-even), which
gives 0.5 -> 0 and 2.5 -> 2. decimal.ROUND_HALF_UP is Python's name for
round-half-away-from-zero, which is what the rules ask for.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def _dec(value) -> Decimal:
    """Convert to Decimal via str() so we round the shortest repr of the float.

    Decimal(float) would expose binary representation error: Decimal(2.675)
    is 2.67499999999999982236431605997495353221893310546875, which rounds
    down. Decimal("2.675") rounds up, matching how the value reads.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(repr(float(value)))


def _quantize(value, places: int) -> Decimal:
    exp = Decimal(1).scaleb(-places)  # 0 -> 1, 1 -> 0.1, 2 -> 0.01
    return _dec(value).quantize(exp, rounding=ROUND_HALF_UP)


def integer(value) -> str:
    """Whole number, no separators, no decimal point."""
    q = _quantize(value, 0)
    # Normalise -0 to 0; format without exponent or trailing '.'
    if q == 0:
        q = Decimal(0)
    return f"{q:f}"


def decimal2(value) -> str:
    """Exactly two decimal places, always."""
    q = _quantize(value, 2)
    if q == 0:
        q = Decimal("0.00")
    return f"{q:.2f}"


def percent1(value) -> str:
    """One decimal place, no percent sign. Value is already in percent units."""
    q = _quantize(value, 1)
    if q == 0:
        q = Decimal("0.0")
    return f"{q:.1f}"


FORMATTERS = {
    "integer": integer,
    "decimal2": decimal2,
    "percent1": percent1,
}


def apply_format(value, fmt: str) -> str:
    """Format `value` according to the question's declared answer format."""
    key = (fmt or "").strip().lower()
    if key not in FORMATTERS:
        raise ValueError(f"unknown answer format: {fmt!r}")
    return FORMATTERS[key](value)
