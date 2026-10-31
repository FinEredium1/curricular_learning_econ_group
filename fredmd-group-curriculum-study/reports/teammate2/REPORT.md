# Teammate 2: Output-and-Income Training for INDPRO Forecasting

## Question and scope

Does supplementing industrial-production histories with the complete output-and-income group improve INDPRO forecasting, and does training order matter? This is the agreed brief, single-seed study. Results are preliminary.

## Shared method

- September 2026 FRED-MD snapshot; official group 1 excluding INDPRO provides 15 auxiliary series.
- Global univariate forecasting: 24 monthly inputs and three outputs. Auxiliary examples predict their own series.
- Forecast months: training 1960–1999, validation 2000–2009, testing 2010–2019. Standardization uses transformed training observations only, with ddof=0.
- INDPRO has 465 valid training windows, 118 validation windows, and 118 test windows.
- Each augmented run uses the same 300 INDPRO batches plus the same 500 auxiliary batches, batch size 32. Orders change only batch sequence.
- Linear, MLP, and PatchTST use the exact agreed architectures. All runs use Adam at 0.001, no weight decay/dropout/early stopping, and seed 1.
- CPU, float32, one PyTorch thread; 800 optimizer updates with final-checkpoint evaluation.

## Final test results

The zero-growth baseline has MSE **0.237266** and MAE **0.394043**. MSE is in squared monthly-growth units; MAE is in monthly-growth units.

| Model | Random mixing | Auxiliary first | Target first | Interleaved | INDPRO only, 800 |
|---|---:|---:|---:|---:|---:|
| linear | 0.246089 | 0.248394 | 0.241793 | 0.243279 | 0.251200 |
| mlp | 0.230304 | 0.249804 | 0.234434 | 0.235875 | 0.257596 |
| patchtst | 0.259885 | 0.379460 | 0.244851 | 0.240598 | 0.404924 |

The table reports test MSE. The accompanying results.csv contains MAE, compute times, provenance, and comparisons to the 800-update control.

## Findings

- **linear:** lowest augmented test MSE occurs with target first (0.241793). Random mixing improves MSE over the 800-update target-only control by 2.03%.
- **mlp:** lowest augmented test MSE occurs with random mixing (0.230304). Random mixing improves MSE over the 800-update target-only control by 10.59%.
- **patchtst:** lowest augmented test MSE occurs with regular interleaving (0.240598). Random mixing improves MSE over the 800-update target-only control by 35.82%.

All twelve augmented runs have lower MSE than their model's 800-update target-only control in this seed. However, every Linear and PatchTST run remains above the zero-growth MSE baseline. MLP random mixing, target-first, and interleaving beat zero growth on MSE; every trained run has higher MAE than zero growth.

Auxiliary-first has higher MSE than random mixing for all three models. The lowest-MSE order differs by model, so the results do not support a universal advantage for auxiliary-first training.

These are final-test descriptions of the prespecified conditions, not grounds for changing the protocol or selecting hyperparameters against the test set.

## Verification

All 15 runs completed exactly 800 optimizer steps. Checkpoint optimizer counters independently equal 800. Each run includes all scheduled validations (30, 60, …, 780, 800), and saved prediction errors reproduce the reported MSE/MAE. All runs share the same snapshot, frozen setup, per-model initial weights, and test targets. Target-first and the target-only control have identical validation errors through their common first 300 target batches.

Seven setup checks and three short BasicTS smoke runs were completed before research training. The frozen experimental code and artifacts were not changed after the teammate setup package was shared.

## Limits and pending comparisons

- Teammate 1's three 300-update controls are still needed for the fixed-target-exposure comparison. They have not been simulated or substituted.
- Other teammates' results are needed to compare different economic groups.
- One seed does not support variance estimates, statistical-significance claims, or a general ranking of schedules.
- The 800-update comparison holds total updates fixed; the target-only control receives more INDPRO batches.
- Revised historical observations and these retrospective splits do not establish real-time forecasting performance.
- Finishing on INDPRO, optimizer history, and group composition may all affect scheduling outcomes. The study does not identify a unique mechanism.
- Compute times exclude batch fetching, preparation, validation, testing, and plotting. Compare timings only on equivalent hardware.

## Files

- results.csv: all 15 results and relative MSE changes.
- validation_comparison.png: validation curves for all conditions.
- test_errors.png: MSE and MAE with zero-growth references.
- training_schedules.png: batch sources and cumulative target exposure.
- forecasts_linear.png, forecasts_mlp.png, forecasts_patchtst.png: identical four prespecified test examples.
- verification.json: checks and completion status.

Snapshot SHA-256: d6e239038afbdbf4667c2a63c13c942817bc196fedab4355cd7e0d353ecdd65d
Frozen setup SHA-256: 5b095334de6612a9be6002a1f8d72b0a7f795e938bb699b05ca2dd85dd3d737b
