# Housing study — Teammate 3

## Question and method

Does complete-group housing training help INDPRO forecasting, and does changing the order of identical batches affect the result?

September 2026 FRED-MD; 24 monthly inputs predict three months. Auxiliary examples predict their own series. Training/validation/test forecast months fall in 1960–1999 / 2000–2009 / 2010–2019. Each series uses training-only scaling (ddof=0).

Each augmented condition uses the same 300 INDPRO batches and 500 auxiliary batches, batch size 32. Linear, MLP, and PatchTST start from their common saved weights with fresh Adam optimizers (learning rate 0.001). Seed 1, CPU, float32, one thread, no dropout or early stopping. Final weights are evaluated after exactly 800 updates; controls use 300 or 800.

Housing includes all ten series: HOUST, HOUSTMW, HOUSTNE, HOUSTS, HOUSTW, PERMIT, PERMITMW, PERMITNE, PERMITS, PERMITW. Each receives exactly 50 auxiliary batches. Shared frozen inputs were reused; preparation was not rerun.

## Final test results

MSE and MAE average all 118 forecast origins × three horizons after reversing INDPRO standardization. Zero growth means zero in original growth units.

Zero-growth MSE: 0.237266; MAE: 0.394043.

| model | order | test_mse | test_mae | mse_improvement_vs_target300_pct | mse_improvement_vs_target800_pct |
|---|---|---|---|---|---|
| linear | auxiliary_first | 0.251102 | 0.417644 | 5.798437 | 0.039298 |
| mlp | auxiliary_first | 0.263680 | 0.426635 | -7.243430 | -2.362131 |
| patchtst | auxiliary_first | 0.319130 | 0.466989 | -7.885649 | 21.187703 |
| linear | interleaved | 0.237394 | 0.398221 | 10.940868 | 5.496118 |
| mlp | interleaved | 0.237232 | 0.396699 | 3.513513 | 7.905198 |
| patchtst | interleaved | 0.247621 | 0.403698 | 16.289011 | 38.847703 |
| linear | random | 0.237806 | 0.396821 | 10.786409 | 5.332215 |
| mlp | random | 0.235172 | 0.393838 | 4.351254 | 8.704809 |
| patchtst | random | 0.251749 | 0.416510 | 14.893467 | 37.828234 |
| linear | target_first | 0.239898 | 0.398949 | 10.001413 | 4.499228 |
| mlp | target_first | 0.243803 | 0.397016 | 0.841065 | 5.354389 |
| patchtst | target_first | 0.284555 | 0.434234 | 3.803044 | 29.726492 |

Positive improvement percentages indicate lower MSE than the named shared control. Controls are in the combined team results.

## Findings

- linear: lowest housing test MSE is 0.237394 with interleaved. Random mixing changes MSE by 10.79% improvement versus the 300-update control and 5.33% versus the 800-update control.
- mlp: lowest housing test MSE is 0.235172 with random. Random mixing changes MSE by 4.35% improvement versus the 300-update control and 8.70% versus the 800-update control.
- patchtst: lowest housing test MSE is 0.247621 with interleaved. Random mixing changes MSE by 14.89% improvement versus the 300-update control and 37.83% versus the 800-update control.

Random mixing improves MSE over both trained controls for all three models. Against zero growth, MLP random mixing improves MSE by about 0.88% and MAE by about 0.05%; MLP interleaving also narrowly improves MSE. The other housing runs do not beat zero growth on MSE. These small differences should not be treated as reliable gains from one seed.

Auxiliary-first has the highest housing test MSE for every model. The best schedule differs by model, so housing does not support a universal ordering advantage.

## Training-stage transitions

Auxiliary-first switches after update 500: the saved checkpoints at 480 and 510 bracket that switch. Target-first switches after 300: compare 300 and 330. These measurements also include training between checkpoints, so they do not isolate an instantaneous causal effect.

| model | order | before_update | before_mse | after_update | after_mse | final_mse |
|---|---|---|---|---|---|---|
| linear | auxiliary_first | 480 | 0.607992 | 510 | 0.593545 | 0.558458 |
| linear | target_first | 300 | 0.602551 | 330 | 0.580734 | 0.535932 |
| mlp | auxiliary_first | 480 | 0.530581 | 510 | 0.515061 | 0.567530 |
| mlp | target_first | 300 | 0.550285 | 330 | 0.531583 | 0.531301 |
| patchtst | auxiliary_first | 480 | 0.542952 | 510 | 0.532260 | 0.651553 |
| patchtst | target_first | 300 | 0.595365 | 330 | 0.535117 | 0.574326 |

At the auxiliary-to-target transition, validation MSE initially falls for all models, but MLP and PatchTST finish with higher validation error than at update 480. After target-to-housing switching, all three models initially improve from update 300 to 330. The later curves fluctuate, especially for PatchTST; these observations do not isolate a mechanism.

## Verification and limitations

All 54 saved summaries and prediction files passed provenance, validation-cadence, finite-value, identical-target, and direct MSE/MAE checks. Batch identity/multiplicity and the exact interleaving cycle were checked for every group. Available final checkpoints were checked against optimizer step counters. Labor checkpoints were not committed, so their optimizer counters cannot be independently inspected here.

Hardware differs across teammates; training times are retained but are not pooled or ranked. Local hardware and installed versions are recorded in ../housing_execution/. Python 3.11.18 is used locally rather than the original 3.11.7 patch release; all pinned packages and frozen scientific files verify unchanged.

One seed provides preliminary results without uncertainty estimates or significance claims. Revised historical data do not establish real-time forecasting performance. These comparisons do not establish a general ranking of categories. Test results describe the prespecified conditions and were not used to tune or retrain.

Preparation time is unavailable because frozen artifacts were reused. Window counts are in ../../data/prepared/window_counts.csv. Figures show validation curves, batch sources, and the same four prespecified evenly spaced test windows. Full outputs and final checkpoints are under ../../results/study/group3/.
