"""Build submission.csv by running every registered solver.

Reads questions.csv for the id list and each question's declared answer
format, runs the matching solver, formats the result, and writes the CSV in
the order the questions file gives.
"""
from __future__ import annotations

import argparse
import sys
import traceback

import pandas as pd

from . import solvers  # noqa: F401  - importing registers every solver
from .config import QUESTIONS, SUBMISSION
from .fmt import apply_format
from .registry import NOTES, SOLVERS


def load_questions() -> pd.DataFrame:
    if not QUESTIONS.exists():
        raise SystemExit(f"missing {QUESTIONS} - download it from the competition Data tab")
    q = pd.read_csv(QUESTIONS, dtype=str).fillna("")
    q.columns = [c.strip().lower() for c in q.columns]
    return q


def _format_column(questions: pd.DataFrame) -> str:
    for cand in ("answer_format", "format", "answer_type"):
        if cand in questions.columns:
            return cand
    raise SystemExit(f"questions.csv has no answer-format column; saw {list(questions.columns)}")


def _id_column(questions: pd.DataFrame) -> str:
    for cand in ("question_id", "id", "qid"):
        if cand in questions.columns:
            return cand
    raise SystemExit(f"questions.csv has no id column; saw {list(questions.columns)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run just these question ids")
    ap.add_argument("--verbose", action="store_true", help="show raw values and notes")
    args = ap.parse_args()

    questions = load_questions()
    idcol, fmtcol = _id_column(questions), _format_column(questions)

    rows, failures = [], []
    for _, q in questions.iterrows():
        qid, fmt = q[idcol].strip(), q[fmtcol].strip()
        if args.only and qid not in args.only:
            continue

        fn = SOLVERS.get(qid)
        if fn is None:
            failures.append((qid, "no solver registered"))
            rows.append({"question_id": qid, "answer": ""})
            continue
        try:
            raw = fn()
            answer = apply_format(raw, fmt)
            rows.append({"question_id": qid, "answer": answer})
            if args.verbose:
                print(f"{qid}  {answer:>16}   (raw={raw!r})")
                if NOTES.get(qid):
                    print(f"        {NOTES[qid]}")
            else:
                print(f"{qid}  {answer}")
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures.append((qid, f"{type(exc).__name__}: {exc}"))
            rows.append({"question_id": qid, "answer": ""})
            if args.verbose:
                traceback.print_exc()

    out = pd.DataFrame(rows)
    if not args.only:
        out.to_csv(SUBMISSION, index=False, lineterminator="\n")
        print(f"\nwrote {SUBMISSION}  ({len(out)} rows)")

    if failures:
        print(f"\n{len(failures)} UNRESOLVED:", file=sys.stderr)
        for qid, msg in failures:
            print(f"  {qid}: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
