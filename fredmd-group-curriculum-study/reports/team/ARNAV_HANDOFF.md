# Draft handoff to Arnav

We finished the seed-1 experiment and combined all four groups: 48 augmented runs plus six shared target-only controls, 54 runs total. The combined report is `fredmd-group-curriculum-study/reports/team/REPORT.md`, with all metrics in the adjacent `results.csv`.

Labor had the lowest random-mixing INDPRO test MSE for Linear, MLP, and PatchTST. Housing random mixing improved over both trained target-only controls for all three models, but only MLP narrowly beat zero growth on both MSE and MAE. Scheduling effects depended on group and model.

We verified matching frozen setup/snapshot/initialization hashes and recomputed MSE/MAE from saved predictions for all 54 runs. Housing completed exact 800-update budgets; available checkpoint counters were verified. Labor checkpoints were not committed, so their counters could not be independently inspected. This is a preliminary single-seed retrospective study; hardware timings are not directly compared.
