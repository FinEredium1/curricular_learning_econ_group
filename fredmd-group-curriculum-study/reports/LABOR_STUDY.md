# Labor-market study (Teammate 1)

**Status:** all 15 Teammate 1 runs are complete: 12 labor-group augmented runs and 3 shared 300-update INDPRO-only controls. The 800-update INDPRO-only control belongs to Teammate 2 and is **pending**, so no comparison against it is made here. These are single-seed results, so they are preliminary.

## Question

Does training a shared model on the complete FRED-MD labor-market group (group 2, 31 series) help forecast industrial production (INDPRO)? And does the presentation order of those same batches change the result?

## Method

The method follows PLAN.md without changes.

- **Data.** September 2026 FRED-MD snapshot. Each example uses 24 months of input to forecast the next 3. The forecast months fall in 1960–1999 for training, 2000–2009 for validation, and 2010–2019 for testing. Each series is standardized on its own 1960–1999 values (`ddof=0`). Labor examples predict their own series and are not extra inputs to INDPRO. Data checks: [LABOR_DATA_CHECKS.md](LABOR_DATA_CHECKS.md).
- **Batches.** These were frozen before training, with 32 windows per batch sampled with replacement using seed 1. Every augmented run uses the same 300 INDPRO batches and the same 500 labor batches (16–17 per series).
- **Conditions.**
  - Random mixing: all 800 batches shuffled once.
  - Auxiliary first: 500 labor batches, then 300 INDPRO batches.
  - Target first: 300 INDPRO batches, then 500 labor batches.
  - Regular interleaving: T, A, A, T, A, A, T, A, repeated 100 times.
  - Control: the 300 INDPRO batches alone.
- **Models.**
  - Linear.
  - MLP with 64 ReLU hidden units.
  - PatchTST: patch length 6, stride 3, hidden size 32, 4 heads, 2 layers, feed-forward size 128, layer norm, no dropout, RevIN or decomposition.
- **Training.**
  - Every run of a model starts from that model's saved initial weights.
  - Optimizer: Adam, learning rate 0.001, no weight decay, no learning-rate scheduler, no early stopping.
  - float32, seed 1, BasicTS 1.1.0 (`c2bb6e31`), CPU with one thread.
  - Validation every 30 updates and at the final update.
  - The test set is scored with the final weights.
- **Metrics.** Test errors are in INDPRO monthly growth units (100 × log difference), averaged over 118 forecast origins × 3 horizons. The always-zero-growth forecast is the reference.

## Results

INDPRO test MSE. Percentages are the change against the same model's 300-update control.

| Condition | Linear | MLP | PatchTST |
|---|---:|---:|---:|
| INDPRO only, 300 updates | 0.2666 | 0.2459 | 0.2958 |
| Random mixing | 0.2189 (-17.9%) | 0.2099 (-14.6%) | 0.2147 (-27.4%) |
| Auxiliary first | 0.2438 (-8.5%) | 0.2491 (+1.3%) | 0.3076 (+4.0%) |
| Target first | 0.2099 (-21.3%) | 0.2032 (-17.4%) | 0.2110 (-28.7%) |
| Regular interleaving | 0.2202 (-17.4%) | 0.2133 (-13.3%) | 0.2148 (-27.4%) |
| Always zero growth | 0.2373 | 0.2373 | 0.2373 |
| INDPRO only, 800 updates | pending | pending | pending |

INDPRO test MAE:

| Condition | Linear | MLP | PatchTST |
|---|---:|---:|---:|
| INDPRO only, 300 updates | 0.4299 | 0.4111 | 0.4546 |
| Random mixing | 0.3862 | 0.3700 | 0.3756 |
| Auxiliary first | 0.4129 | 0.4115 | 0.4540 |
| Target first | 0.3744 | 0.3646 | 0.3690 |
| Regular interleaving | 0.3883 | 0.3761 | 0.3737 |
| Always zero growth | 0.3940 | 0.3940 | 0.3940 |

More detail:
- Full table, with final validation MSE, training times, hardware and per-series window counts: [labor_results.md](labor_results.md) (machine-readable version: `labor_results.csv`).
- Figures:
  - [validation curves](labor_validation_curves.png);
  - [batch source by update](labor_batch_sources.png);
  - [four test-window forecasts](labor_forecast_windows.png): windows at evenly spaced positions in the test list, fixed before errors were inspected.

## Findings

1. **Labor-market training helped against the 300-update control in three of four schedules, for every model.** Random mixing, target first and regular interleaving lowered test MSE by 13.3–28.7% and MAE by 8.5–18.8%, with the same INDPRO exposure. All nine of these runs beat the always-zero-growth forecast on MSE (0.2032–0.2202 against 0.2373) and on MAE. None of the three 300-update controls did (MSE 0.2459–0.2958).

2. **Order mattered mostly because of auxiliary first.** It was the worst schedule for every model:
   - test MSE changed by -8.5% (linear), +1.3% (MLP) and +4.0% (PatchTST) against the control;
   - it was worse than zero growth for all three models.

   The other three schedules were close together: their test MSE spans 0.0103 (linear), 0.0101 (MLP) and 0.0038 (PatchTST). Target first had the lowest test MSE for every model, but the evidence for that ranking differs by model:
   - **Linear:** validation curves are smooth (over updates 600–800, consecutive checkpoints differ by about 0.002), and target first has the lowest validation MSE of the three at every checkpoint from update 330 on.
   - **MLP and PatchTST:** the three curves cross repeatedly. Target first is lowest at only 8 and 13 of 27 checkpoints. Over updates 600–800, consecutive checkpoints of the same run differ by about 0.005–0.010 (MLP) and 0.016–0.048 (PatchTST), so one seed with one final checkpoint does not rank these three schedules.

3. **What happened at the stage transitions** (validation MSE):
   - **Auxiliary first:** 500 labor-only updates brought INDPRO validation MSE below the control's final value for every model. At update 480, it was 0.5646 against 0.6026 (linear), 0.5044 against 0.5503 (MLP), and 0.5181 against 0.5954 (PatchTST). The 300 INDPRO updates that followed did not improve it further. Linear ended at 0.5605, and MLP rose to 0.5483. PatchTST ended at 0.6661, a spike at the final checkpoint; its lowest value after the switch was 0.5164.
   - **Target first:** switching to labor batches lowered validation MSE for every model. From update 300 to 330, it went from 0.6026 to 0.5800 (linear), 0.5503 to 0.5061 (MLP), and 0.5954 to 0.5215 (PatchTST).

   PLAN.md warns that auxiliary-first gains might come from finishing on INDPRO. In this run auxiliary first produced no gain, and finishing on INDPRO coincided with higher INDPRO error.

4. **A possible explanation, not tested.** The 2010–2019 test period averaged 0.112 monthly INDPRO growth, well below the 1960–1999 training mean of 0.283. Runs that end on INDPRO batches (the control and auxiliary first) have average test forecasts of 0.25–0.31, near the training mean. Runs with labor batches throughout or at the end average 0.17–0.23, closer to the test-period level. This level difference may account for part of the gain. Separating it from other effects would need further work beyond this plan.

5. **Pending: more updates or more labor data?** The linear 300-update control's validation error was still falling at update 300. So part of the gain over that control may come from 500 extra optimizer updates rather than from labor-market content. The plan's 800-update INDPRO-only control (Teammate 2) answers this question. Until it is available, the results above show only that labor training helped while keeping INDPRO exposure fixed. They do not show that labor training beats spending the same updates on INDPRO.

## Limitations

- **Single seed (1).** The findings are preliminary. Final-checkpoint evaluation makes MLP and PatchTST results sensitive to the exact last update.
- **Revised data.** The snapshot holds revised historical data, so this is not a real-time forecasting test.
- **Narrow scope.**
  - Results cover the complete labor-market group and these four schedules only. They do not rank economic groups in general.
  - Regular interleaving tests only the specified T, A, A, T, A, A, T, A cycle.
- **Timing.**
  - Training times (0.06–2.36 s per run) are from one Apple M4 Pro CPU on one thread, and should be compared only with runs on matching hardware.
  - Preparation time is not reported: the shared seed's prepared data were verified, not regenerated, and the seed does not record preparation time.
- **Provisional format.** The table and figure format is provisional until Teammate 4's shared conventions are agreed.

## Reproduce

Run these from `fredmd-group-curriculum-study/`, using a Python 3.11 environment with `requirements.lock` installed in `.venv`:

    .venv/bin/python study.py verify
    .venv/bin/python -m pytest tests -q
    for m in linear mlp patchtst; do .venv/bin/python study.py smoke --group 2 --model $m; done
    for m in linear mlp patchtst; do for o in random target_only_300; do .venv/bin/python study.py run --group 2 --model $m --order $o; done; done
    for m in linear mlp patchtst; do for o in auxiliary_first target_first interleaved; do .venv/bin/python study.py run --group 2 --model $m --order $o; done; done
    PYTHONPATH=src .venv/bin/python -m fredmd_study.group_report --group 2

The runs were executed one at a time in this order: selection comparisons first, then ordering comparisons. Each run writes to `results/study/group2/<condition>/<model>/seed1/`:
- `summary.json`
- `validation.csv`
- `predictions.csv` (forecast origin, forecast month, horizon, actual, prediction)
- `final.pt`
- BasicTS logs and config

The 300-update controls are stored under `group2/target_only_300/`. They are shared controls and do not depend on the group. Run outputs are committed except checkpoints (`*.pt`) and tensorboard files. BasicTS configs and logs are committed with `<study_root>` in place of the local path; configure the filter in `.gitattributes` once per clone before adding results. The report script picks up Teammate 2's `target_only_800` runs automatically once they are placed under `results/study/group*/target_only_800/`.
