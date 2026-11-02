# Team experiment: economic groups and training order

**Complete: 48 augmented runs plus six shared controls, seed 1.**

## Method

September 2026 FRED-MD; 24 monthly inputs predict three months. Auxiliary examples predict their own series. Training/validation/test forecast months fall in 1960–1999 / 2000–2009 / 2010–2019. Each series uses training-only scaling (ddof=0).

Each augmented condition uses the same 300 INDPRO batches and 500 auxiliary batches, batch size 32. Linear, MLP, and PatchTST start from their common saved weights with fresh Adam optimizers (learning rate 0.001). Seed 1, CPU, float32, one thread, no dropout or early stopping. Final weights are evaluated after exactly 800 updates; controls use 300 or 800.

MSE and MAE average all 118 forecast origins × three horizons after reversing INDPRO standardization. Zero growth means zero in original growth units.

## Group selection under random mixing

| model | group_name | test_mse | test_mae | mse_improvement_vs_target300_pct | mse_improvement_vs_target800_pct |
|---|---|---|---|---|---|
| linear | Labor market | 0.218851 | 0.386242 | 17.897375 | 12.877921 |
| linear | Interest and exchange rates | 0.236844 | 0.406573 | 11.147292 | 5.715162 |
| linear | Housing | 0.237806 | 0.396821 | 10.786409 | 5.332215 |
| linear | Output and income | 0.246089 | 0.415466 | 7.678994 | 2.034825 |
| mlp | Labor market | 0.209866 | 0.369995 | 14.644020 | 18.529088 |
| mlp | Interest and exchange rates | 0.226601 | 0.397351 | 7.837304 | 12.032188 |
| mlp | Output and income | 0.230304 | 0.398233 | 6.331363 | 10.594791 |
| mlp | Housing | 0.235172 | 0.393838 | 4.351254 | 8.704809 |
| patchtst | Labor market | 0.214746 | 0.375569 | 27.402760 | 46.966485 |
| patchtst | Housing | 0.251749 | 0.416510 | 14.893467 | 37.828234 |
| patchtst | Output and income | 0.259885 | 0.413886 | 12.142969 | 35.818949 |
| patchtst | Interest and exchange rates | 0.287484 | 0.445036 | 2.812853 | 29.003140 |

Positive percentages mean lower MSE than the indicated control.

## Shared target-only controls

| model | order | test_mse | test_mae |
|---|---|---|---|
| linear | target_only_300 | 0.266558 | 0.429928 |
| linear | target_only_800 | 0.251200 | 0.418753 |
| mlp | target_only_300 | 0.245871 | 0.411060 |
| mlp | target_only_800 | 0.257596 | 0.413130 |
| patchtst | target_only_300 | 0.295804 | 0.454636 |
| patchtst | target_only_800 | 0.404924 | 0.522091 |

Zero growth: MSE 0.237266; MAE 0.394043.

## Findings

- Under random mixing, labor market has the lowest observed test MSE for linear: 0.218851. This is a descriptive single-seed comparison.
- Under random mixing, labor market has the lowest observed test MSE for mlp: 0.209866. This is a descriptive single-seed comparison.
- Under random mixing, labor market has the lowest observed test MSE for patchtst: 0.214746. This is a descriptive single-seed comparison.

Every random-mixing group/model condition improves test MSE over both trained target-only controls in this seed. Labor random mixing also beats zero growth on both MSE and MAE for all three models. This comparison controls optimizer updates with the 800-update baseline and target exposure with the 300-update baseline.

The best observed schedule varies by group and model. Target-first has the lowest MSE for all three labor models; housing favors interleaving for Linear/PatchTST and random mixing for MLP. Auxiliary-first is the highest-MSE schedule for every housing, labor, and output/income model. Rates/FX PatchTST is an exception: random mixing is worse than the other schedules. There is no universal winning order.

## Ordering within each group

### Output and income

| model | order | test_mse | test_mae |
|---|---|---|---|
| linear | auxiliary_first | 0.248394 | 0.417364 |
| mlp | auxiliary_first | 0.249804 | 0.411953 |
| patchtst | auxiliary_first | 0.379460 | 0.501997 |
| linear | interleaved | 0.243279 | 0.411905 |
| mlp | interleaved | 0.235875 | 0.400296 |
| patchtst | interleaved | 0.240598 | 0.398055 |
| linear | random | 0.246089 | 0.415466 |
| mlp | random | 0.230304 | 0.398233 |
| patchtst | random | 0.259885 | 0.413886 |
| linear | target_first | 0.241793 | 0.410385 |
| mlp | target_first | 0.234434 | 0.400741 |
| patchtst | target_first | 0.244851 | 0.403610 |

### Labor market

| model | order | test_mse | test_mae |
|---|---|---|---|
| linear | auxiliary_first | 0.243803 | 0.412900 |
| mlp | auxiliary_first | 0.249139 | 0.411473 |
| patchtst | auxiliary_first | 0.307611 | 0.453955 |
| linear | interleaved | 0.220244 | 0.388311 |
| mlp | interleaved | 0.213281 | 0.376125 |
| patchtst | interleaved | 0.214828 | 0.373728 |
| linear | random | 0.218851 | 0.386242 |
| mlp | random | 0.209866 | 0.369995 |
| patchtst | random | 0.214746 | 0.375569 |
| linear | target_first | 0.209902 | 0.374423 |
| mlp | target_first | 0.203182 | 0.364631 |
| patchtst | target_first | 0.211020 | 0.369042 |

### Housing

| model | order | test_mse | test_mae |
|---|---|---|---|
| linear | auxiliary_first | 0.251102 | 0.417644 |
| mlp | auxiliary_first | 0.263680 | 0.426635 |
| patchtst | auxiliary_first | 0.319130 | 0.466989 |
| linear | interleaved | 0.237394 | 0.398221 |
| mlp | interleaved | 0.237232 | 0.396699 |
| patchtst | interleaved | 0.247621 | 0.403698 |
| linear | random | 0.237806 | 0.396821 |
| mlp | random | 0.235172 | 0.393838 |
| patchtst | random | 0.251749 | 0.416510 |
| linear | target_first | 0.239898 | 0.398949 |
| mlp | target_first | 0.243803 | 0.397016 |
| patchtst | target_first | 0.284555 | 0.434234 |

### Interest and exchange rates

| model | order | test_mse | test_mae |
|---|---|---|---|
| linear | auxiliary_first | 0.251133 | 0.418650 |
| mlp | auxiliary_first | 0.248597 | 0.416308 |
| patchtst | auxiliary_first | 0.275389 | 0.431348 |
| linear | interleaved | 0.238874 | 0.406950 |
| mlp | interleaved | 0.231098 | 0.400677 |
| patchtst | interleaved | 0.250034 | 0.418535 |
| linear | random | 0.236844 | 0.406573 |
| mlp | random | 0.226601 | 0.397351 |
| patchtst | random | 0.287484 | 0.445036 |
| linear | target_first | 0.238801 | 0.404629 |
| mlp | target_first | 0.234091 | 0.398809 |
| patchtst | target_first | 0.255576 | 0.411489 |

## Verification and limits

All 54 saved summaries and prediction files passed provenance, validation-cadence, finite-value, identical-target, and direct MSE/MAE checks. Batch identity/multiplicity and the exact interleaving cycle were checked for every group. Available final checkpoints were checked against optimizer step counters. Labor checkpoints were not committed, so their optimizer counters cannot be independently inspected here.

Hardware differs across teammates; training times are retained but are not pooled or ranked. Local hardware and installed versions are recorded in ../housing_execution/. Python 3.11.18 is used locally rather than the original 3.11.7 patch release; all pinned packages and frozen scientific files verify unchanged.

One seed provides preliminary results without uncertainty estimates or significance claims. Revised historical data do not establish real-time forecasting performance. These comparisons do not establish a general ranking of categories. Test results describe the prespecified conditions and were not used to tune or retrain.

Snapshot SHA-256: d6e239038afbdbf4667c2a63c13c942817bc196fedab4355cd7e0d353ecdd65d

Freeze SHA-256: 5b095334de6612a9be6002a1f8d72b0a7f795e938bb699b05ca2dd85dd3d737b

Source branches: output-and-income-jayden (group 1 and 800 controls), labor-market-lisa (group 2 and 300 controls), Fardin_group6_results (group 6), plus locally completed housing runs. Imported files were preserved. Original teammate reports may describe comparisons as pending; this report supersedes those pending-status notes.

See results.csv for all 54 runs, verification.json for the audit, group_selection.png for comparisons, and ../housing/REPORT.md for the housing study.
