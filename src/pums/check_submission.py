"""Validate submission.csv against the competition's formatting rules.

Scoring is exact string match, so a correct number in the wrong shape scores
zero. This checks the shape independently of the code that produced it.
"""
from __future__ import annotations

import csv
import re
import sys

import pandas as pd

from .config import QUESTIONS, SUBMISSION

PATTERNS = {
    # no separators, no decimal point, no currency symbol; optional leading minus
    "integer": re.compile(r"^-?\d+$"),
    # exactly two decimal places, always
    "decimal2": re.compile(r"^-?\d+\.\d{2}$"),
    # a number with exactly one decimal, no percent sign
    "percent1": re.compile(r"^-?\d+\.\d$"),
}


def main() -> int:
    questions = pd.read_csv(QUESTIONS, dtype=str).fillna("")
    expected_ids = questions.question_id.str.strip().tolist()
    formats = dict(zip(expected_ids, questions.answer_format.str.strip()))

    with open(SUBMISSION, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    problems = []

    header, body = rows[0], rows[1:]
    if header != ["question_id", "answer"]:
        problems.append(f"header is {header}, expected ['question_id', 'answer']")

    if len(body) != 38:
        problems.append(f"{len(body)} data rows, expected 38")

    got_ids = [r[0] for r in body]
    if got_ids != expected_ids:
        problems.append("question_id column does not match questions.csv order/content")
        missing = set(expected_ids) - set(got_ids)
        extra = set(got_ids) - set(expected_ids)
        if missing:
            problems.append(f"  missing ids: {sorted(missing)}")
        if extra:
            problems.append(f"  unexpected ids: {sorted(extra)}")

    print(f"{'id':<6} {'format':<9} {'answer':>16}  check")
    print("-" * 46)
    for row in body:
        if len(row) != 2:
            problems.append(f"row {row} does not have exactly 2 fields")
            continue
        qid, answer = row
        fmt = formats.get(qid, "")
        pattern = PATTERNS.get(fmt)

        if answer == "":
            status = "EMPTY - solver failed"
            problems.append(f"{qid}: empty answer")
        elif pattern is None:
            status = f"unknown format {fmt!r}"
            problems.append(f"{qid}: {status}")
        elif not pattern.match(answer):
            status = f"FAILS {fmt} pattern"
            problems.append(f"{qid}: {answer!r} fails {fmt}")
        else:
            status = "ok"

        # Catch the formatting mistakes the rules single out.
        for bad, label in ((",", "thousands separator"), ("$", "currency symbol"),
                           ("%", "percent sign"), (" ", "whitespace")):
            if bad in answer:
                problems.append(f"{qid}: answer contains a {label}: {answer!r}")
                status = f"HAS {label}"

        print(f"{qid:<6} {fmt:<9} {answer:>16}  {status}")

    print()
    if problems:
        print(f"{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("submission.csv is well-formed: 38 rows, ids match, all formats valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
