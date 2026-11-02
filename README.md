# FRED-MD team experiment

Completed seed-1 study: four complete economic groups, three forecasting models, four training schedules, and two shared target-only controls per model (54 runs).

- [Combined report](fredmd-group-curriculum-study/reports/team/REPORT.md)
- [All 54 results](fredmd-group-curriculum-study/reports/team/results.csv)
- [Housing study — Teammate 3](fredmd-group-curriculum-study/reports/housing/REPORT.md)
- [Verification audit](fredmd-group-curriculum-study/reports/team/verification.json)
- [Setup instructions](fredmd-group-curriculum-study/TEAM_START_HERE.md)
- [Agreed protocol](fredmd-group-curriculum-study/PLAN.md)

Clone with `--recurse-submodules`, create the pinned Python 3.11 environment, and run `study.py verify` from the research folder. Rebuild reports with `.venv/bin/python reports/build_team_report.py`. Do not regenerate frozen inputs or rerun completed conditions into existing output folders.

Labor had the lowest random-mixing MSE across all three models in this single seed. Findings are preliminary. See the combined report for limitations, provenance, and details of unavailable labor checkpoints. Original teammate reports are preserved as historical records; their pending-status notes are superseded by the combined report.

Group 1, 2, and 3 outputs are under `fredmd-group-curriculum-study/results/study/`. Imported group 6 outputs remain in their original `group6_results_seed1_20261031T203006Z_154216/` directory.
