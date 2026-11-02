# Shared data-preparation checks (Teammate 1)

PLAN.md assigns Teammate 1 to lead the shared data-preparation checks: snapshot verification, group membership, transformations, scaling, and window indexing. This note checks the frozen prepared data shipped in the team seed. It compares that data against an independent re-derivation from the unchanged raw files.

**Result: every check passed. No discrepancies or exclusions were found, and nothing was regenerated or changed.**

## How to run

From this folder:

    .venv/bin/python study.py verify
    .venv/bin/python -m pytest tests -q

`tests/test_shared_data_checks.py` holds the eight independent checks. It does not import `fredmd_study.preparation`. Transformations, scaling statistics and window indices are recomputed there with separate code (windows use a cumulative finite-count method) and compared with `data/prepared/`.

Run on 2026-10-31: Apple M4 Pro, macOS, Python 3.11.7, `requirements.lock` installed. `verify` passed (frozen hashes, BasicTS commit `c2bb6e31`, package versions). All 15 tests passed: 7 existing and 8 new.

To confirm that the new checks can fail, I applied four deliberate errors once during this session. These mutations are not part of the suite:
- standard deviation with ddof=1;
- PAYEMS without the ×100;
- a training boundary moved to 2000;
- an injected missing month in UNRATE.

Each one was detected.

## Findings

| Check | Result |
|---|---|
| Snapshot | `data/raw/fredmd_2026-09.csv` SHA-256 `d6e239038afbdbf4667c2a63c13c942817bc196fedab4355cd7e0d353ecdd65d` matches `sources.json`, `freeze.json` and the prepared metadata, and its recorded URL is the one in PLAN.md. The separately distributed copy `../2026-09-MD.csv` has the identical hash. |
| Calendar | 812 contiguous monthly rows, 1959-01 to 2026-08. The CSV's first row (transformation codes) is used only as metadata. |
| Group membership | All four groups equal the official appendix (`FRED-MD_updated_appendix.csv` in `fred-md_appendix0324.zip`): output and income 15 (INDPRO excluded), labor 31, housing 10, interest and exchange rates 22. One case-only alias is applied (`IPB51222s` → `IPB51222S`). No identifiers are missing from the CSV, and no series are excluded. INDPRO is in no auxiliary group. |
| Transformation codes | For all 79 series, the CSV code row, the appendix `tcode` and the prepared metadata agree. INDPRO = 5, PAYEMS = 5, UNRATE = 2. |
| Transformed values | The independent recomputation reproduces every prepared series after undoing standardization, with identical missing-value positions. INDPRO and PAYEMS are 100 × log difference. UNRATE is the monthly difference in percentage points. Other code-5 series use the documented FRED-MD log difference without ×100; this scale cancels under per-series standardization. |
| Invalid log inputs | No series with a log code (4, 5, 6) has a non-positive raw value in this snapshot, so the rule never applies to real data. The existing synthetic test covers the shared code's handling, and any violation would show up as a mismatch in missing-value positions. |
| Missing raw values | Labor group: only 2025-10 is missing, for CE16OV, CLF16OV, HWIURATIO, UEMP15OV, UEMP15T26, UEMP27OV, UEMP5TO14, UEMPLT5, UEMPMEAN and UNRATE. INDPRO has none. That month falls after the 2019 test boundary, so no study window is affected. |
| Scaling | For all 79 series, mean and population standard deviation (`ddof=0`) use only finite transformed observations from 1960-01 to 1999-12, match the independent computation, and are positive. |
| Window indexing | Recomputed train, validation and test indices equal the prepared indices for all 79 series. Every forecast month lies inside its split, and every 27-month window is finite. Each series is windowed on its own, so auxiliary missingness cannot change the INDPRO windows. INDPRO has 465 train, 118 validation and 118 test windows. Labor series have 464–466 training windows. |

## Smoke check

The plan requires a shared smoke check before full training. Three-update smoke runs for group 2 (labor) passed for linear, MLP and PatchTST on this machine: exactly 3 optimizer updates, validation at the final update, and finite predictions. They never evaluate the test set. Outputs are in `results/smoke/group2/random/<model>/seed1/`.

## Out of scope here

Batch exposure and schedule checks are led by Teammate 3, BasicTS configuration by Teammate 2, and the results format by Teammate 4.
