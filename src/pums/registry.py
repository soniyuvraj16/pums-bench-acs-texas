"""Question solver registry.

Each question gets one function, registered under its question_id. A solver
returns a raw number; the runner applies the format declared by questions.csv,
so the arithmetic and the presentation stay separate.
"""
from __future__ import annotations

from typing import Callable

SOLVERS: dict[str, Callable[[], float]] = {}
NOTES: dict[str, str] = {}


def solver(qid: str, note: str = ""):
    """Register the solver for `qid`.

    `note` records the reading of the question that the code implements -
    which universe, which weight, which adjustment - so a wrong answer can be
    traced to a wrong interpretation rather than a wrong formula.
    """
    def decorate(fn: Callable[[], float]) -> Callable[[], float]:
        if qid in SOLVERS:
            raise ValueError(f"duplicate solver for {qid}")
        SOLVERS[qid] = fn
        NOTES[qid] = note or (fn.__doc__ or "").strip()
        return fn
    return decorate
