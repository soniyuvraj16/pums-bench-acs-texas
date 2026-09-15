# PUMS-Bench: Deterministic Analytics over ACS PUMS (Texas)

Answers to the 38 questions in *PUMS-Bench*, computed from the U.S. Census Bureau's
2024 and 2023 ACS 1-year PUMS files for Texas. No machine learning and no
prediction: every answer is a deterministic function of the published microdata.

The output is [`submission.csv`](submission.csv) — 38 rows, `question_id,answer`.

## Reproducing the answers

```bash
pip install -r requirements.txt

# 1. Put the raw Census data in place (see "Data" below)
# 2. Put questions.csv in the repo root (Kaggle competition Data tab)

export PYTHONPATH=src            # Windows: $env:PYTHONPATH="src"

python -m pums.build_cache       # CSV -> parquet (~1 min, once)
python -m pums.verify            # integrity + structural checks
python -m pums.run               # writes submission.csv
python -m pums.check_submission  # validates the output format
```

Tests and cross-checks:

```bash
python tests/test_fmt.py         # answer formatting, incl. rounding rule
python tests/test_stats.py       # weighted quantile + SDR standard error
python -m pums.diagnose          # blanks/universes the questions depend on
python -m pums.validate          # independent consistency identities
```

## Data

Not committed (≈580 MB, and Census data is better fetched from the source).
Download from the U.S. Census Bureau and extract so the layout is:

```
csv_ptx/psam_p48.csv           2024 person      https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_ptx.zip
csv_htx/psam_h48.csv           2024 household   https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_htx.zip
csv_ptx (2023)/psam_p48.csv    2023 person      https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_ptx.zip
csv_htx (2023)/psam_h48.csv    2023 household   https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_htx.zip
```

Published SHA-256 for the 2024 zips (checked by `pums.verify` when the zips are present):

| file | sha256 |
|---|---|
| `csv_ptx.zip` | `42ad7127988234d7dd30c23df8ada371b2135b5db912d2e634a6f901477afe63` |
| `csv_htx.zip` | `59274006389de6f8d6ee5f6e2eb48f7f1ccd4d3c9f730df857be641d5ef65b94` |

Once the zips are unpacked the checksum no longer applies, so `pums.verify`
falls back to structural checks — row counts, all 80 replicate weights present,
`SERIALNO` and the full-sample weight present:

| file | rows |
|---|---|
| 2024 person | 292,272 |
| 2024 household | 132,629 |
| 2023 person | 301,984 |
| 2023 household | 135,627 |

Reference documents used: PUMS Data Dictionary 2024 and 2023, and
*PUMS Accuracy of the Data (2024)* for the replicate-weight methodology.

## Layout

```
src/pums/
  config.py            paths, replicate-weight names, expected checksums
  build_cache.py       CSV -> parquet, streamed, every column kept as text
  io.py                column loading; explicit text -> number conversion
  data.py              typed cached frames, universes, joins, replicate loading
  stats.py             weighted mean/quantile, SDR standard error
  fmt.py               integer / decimal2 / percent1, round half away from zero
  solvers.py           one function per question, each stating its universe
  registry.py          maps question_id -> solver
  run.py               builds submission.csv
  dictionary.py        CLI lookup of official data dictionary entries
  verify.py            input integrity checks
  diagnose.py          blank/universe/join properties the questions rely on
  validate.py          independent consistency identities
  check_submission.py  output format validation
tests/
  test_fmt.py          formatting and rounding
  test_stats.py        quantile vs a literal transcription of the spec; SDR
```

## Decisions that affect correctness

**Everything is read as text, then converted explicitly.** PUMS writes a blank
for "outside the universe". A CSV reader that guesses types turns those blanks
into `0.0` and coerces `SERIALNO` into a float. The parquet cache stores every
column as a string and each solver converts what it needs, so blanks stay `NaN`
and never get counted as zeros.

**The quantile is the one the questions define, not a library default.** Each
median question specifies: *the smallest value `v` such that the sum of the
weights over records with value `<= v` is at least half the universe's total
weight.* That is an order statistic, not an interpolated median — so it differs
slightly from the figures Census publishes, which interpolate. `tests/test_stats.py`
checks the implementation against a deliberately slow, literal transcription of
that sentence.

**Replicate weights can be negative**, which the Census documentation states
explicitly. That means the cumulative weight sum is *not* monotonic, so a binary
search over it would be invalid. `weighted_quantile` scans distinct values in
ascending order and returns the first that satisfies the condition — correct
whether or not the weights are non-negative. This matters for q037, the standard
error of a median. (The 2024 Texas household replicate weights do contain
negative cells.)

**Ties are accumulated before the test.** The condition is about distinct values,
so all records sharing a value are summed before the threshold is checked.

**Each year carries its own adjustment factor.** `ADJINC` is 1.015250 for 2024
and 1.019518 for 2023; the cross-year questions use each file's own value.
`ADJHSG` (1.000000 in both years) applies to `GRNTP` and `VALP`, *not* to income —
q027 and q028 use it, the income questions do not.

**Round half away from zero.** Python's built-in `round()` is banker's rounding
(`round(2.5) == 2`), which is the wrong rule here. Formatting goes through
`decimal.ROUND_HALF_UP` instead.

**Where a question's definition diverges from the published ACS figure, the
question wins.** q015 defines mean household size as `sum(NP*WGTP)/sum(WGTP)`,
which gives 2.61. Census publishes 2.68, computed as household population over
household count. Both are correct for their own definition; the spec is binding.
The gap is the documented difference between person weights (raked to population
controls) and household weights (raked to housing-unit controls) — `pums.validate`
shows the ratio is 1.0277.

## Checks that passed

- `q001 = 292272` reproduces the worked example in the competition overview.
- Weighted Texas population 31,290,831, matching the published 2024 ACS estimate.
- Household population + group-quarters population = total population, exactly.
- Tenure categories sum to the occupied-household total, exactly.
- Age bands (<18, 18–64, 65+) sum to 100.00%.
- `NP` equals the actual person-record count for all 110,443 occupied housing
  units, so the `SERIALNO` join loses nothing.
- Every person `SERIALNO` appears in the household file; every occupied unit has
  exactly one `SPORDER == 1` householder.
- q027's non-blank `GRNTP` universe is 29,044 records — exactly the
  renter-occupied count, as the variable's universe implies.
- Median age is 35 in both years independently (2023 crosses 50% at 50.136%,
  2024 at 50.154%).

## Notes on specific questions

| q | point |
|---|---|
| q005 | `ESR` is blank under 16; blanks are excluded rather than counted as non-3. |
| q010 | `WAGP == 0` stays in the universe, as the question states. |
| q012 | The median is taken on **unadjusted** `WAGP`, then multiplied by `ADJINC`. |
| q015 / q020 | Household-weighted vs person-weighted mean household size: 2.61 vs 3.52. The person-weighted figure is larger because larger households contain more people. |
| q019 | Person-weighted median household income — deliberately not the household-weighted one. |
| q027 / q028 | Adjusted with `ADJHSG`, not `ADJINC`. |
| q031 / q032 | Use the 2023 file's own `ADJINC`; intermediates are not rounded. |
| q034–q038 | SDR: `SE = sqrt((4/80) * Σ(X_r − X)²)`, replicate weights used exactly as stored. |
