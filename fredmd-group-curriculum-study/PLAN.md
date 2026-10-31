# Brief Team Experiment: Economic Groups and Training Order

## 1. Purpose and scope

Each teammate will study whether training on a complete economic group helps forecast **industrial production (INDPRO)**, and whether changing the presentation order of those same training examples affects forecasting.

This is a **brief whole-group extension of the economic starter exercise**, rather than the complete semester project. The starter’s forecasting setup, models, and optimizer settings come from pages 5–6 of :codex-file-citation{path="/Users/jquang/Documents/College/004-Fall-2026/VIP/1st-Semester-Outline.pdf" purpose="source"}.

Our deliberate extensions are complete auxiliary groups, an 800-update augmented-training budget, two target-only controls, and two additional schedules.

**Immediate scope:**

- One evaluation target: **INDPRO**.
- Four complete auxiliary economic groups.
- Three models: **Linear, MLP, and PatchTST**.
- Four training orders per group.
- Training seed **1**.
- **54 distinct training runs** across the team.

Random-group membership controls, additional evaluation targets, statistical difficulty rankings, and repeated seeds are outside this first experiment. Our results will concern the specified groups and schedules; they will not establish a general ranking of economic categories.

**LLM handoff:** Give the LLM this entire plan and identify the assigned teammate. Every implementation must follow the common protocol. Everyone should extend the same shared codebase.

## 2. Common setup: data, models, and training

### One shared implementation and environment

Before full training, freeze and distribute:

- One FRED-MD snapshot and its SHA-256 checksum.
- Official group membership and transformation metadata.
- One shared preprocessing and window-building implementation.
- Saved sampling and scheduling manifests.
- One initial-weight checkpoint per model.
- One tested BasicTS version or commit and a dependency lockfile.
- A common results format and documented run commands.

**Use BasicTS for training, validation, and evaluation.** Custom datasets, small PyTorch models, and scheduling adapters should integrate with BasicTS. Keep one shared training implementation and change conditions through configuration.

Use the same Python, PyTorch, NumPy, BasicTS, precision, and device backend across teammates. Prefer a shared compute environment and the same GPU type. Record hardware differences; compare training times only when hardware and timing procedures match.

### Data snapshot and group assignments

Use the **September 2026 FRED-MD snapshot**, downloaded once and shared:

[September 2026 FRED-MD CSV](https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2026-09-md.csv)

Use the group definitions and transformation documentation available through the [official FRED-MD resources](https://www.stlouisfed.org/research/economists/mccracken/fred-databases).

| Teammate | Complete auxiliary group |
|---|---|
| **Teammate 1** | Group 2: Labor market |
| **Teammate 2** | Group 1: Output and income, excluding INDPRO |
| **Teammate 3** | Group 3: Housing |
| **Teammate 4** | Group 6: Interest and exchange rates |

Include every eligible series in the assigned group. Record missing identifiers, exclusions, and reasons. INDPRO always belongs to the separate target-training pool, including for Teammate 2.

### Preprocessing and forecasting examples

- Treat the CSV’s transformation-code row as metadata.
- For **INDPRO and PAYEMS**, use \(100[\log(x_t)-\log(x_{t-1})]\).
- For **UNRATE**, use the monthly difference in percentage points.
- For other series, apply their documented FRED-MD transformation codes.
- Preserve the monthly calendar. Invalid transformations and missing observations remain unavailable.
- Exclude windows containing missing or nonfinite values. Do not fill gaps or join observations across missing months.
- Build valid windows independently for each series. Auxiliary missingness must not change the shared INDPRO evaluation windows.

Each example uses **24 consecutive monthly observations to predict the following three observations**, advancing one month at a time.

| Split | All three forecast months must fall within |
|---|---|
| Training | January 1960–December 1999 |
| Validation | January 2000–December 2009 |
| Testing | January 2010–December 2019 |

Discard windows whose forecast months cross these boundaries. Input histories may extend into an earlier period.

Standardize each series separately using its finite transformed observations from **1960–1999 only**, with population standard deviation (`ddof=0`). Apply those saved statistics to validation and testing. Exclude and report series with insufficient data, zero standard deviation, or no valid training windows.

An auxiliary example predicts **its own series**. The shared model learns across examples; auxiliary series do not become extra input columns for INDPRO.

Validate and test only on INDPRO.

### Models

| Model | Configuration |
|---|---|
| **Linear** | Direct mapping from 24 inputs to three outputs; bias enabled |
| **MLP** | 24 inputs → 64-unit hidden layer → ReLU → three outputs; biases enabled |
| **PatchTST** | Input length 24; output length 3; one input feature; patch length 6; stride 3; hidden size 32; four attention heads; two encoder layers; feed-forward size 128 |

Disable all dropout. For PatchTST, disable RevIN and decomposition, and use layer normalization. Provide the actual shared implementation if an adapter is needed.

Leave other architecture settings at the pinned version’s defaults and save the resolved configuration. Use the same architecture implementation across all groups and schedules.

### Training settings

| Setting | Required value |
|---|---|
| Batch size | 32 examples |
| Loss | Ordinary MSE over all examples and forecast steps |
| Optimizer | Adam |
| Learning rate | Constant `0.001` |
| Adam betas | `(0.9, 0.999)` |
| Adam epsilon | `1e-8` |
| Weight decay | `0` |
| AMSGrad | Disabled |
| Learning-rate scheduler | None |
| Early stopping | Disabled |
| Precision | Float32; mixed precision disabled |
| Training seed | `1` |
| Validation | Every 30 updates, plus the final update |
| Final evaluation | Final weights at the prescribed budget |

Within each model, every run starts from identical saved initial weights and a fresh optimizer. Each run starts independently.

Keep sampling, ordering, and initialization randomness separate. Use seed 1 for separately instantiated generators, record their roles, and save their generated manifests before training. Use dedicated sampling generators per series so adding a series does not change the sampled windows for existing series.

## 3. Batch preparation, controls, and schedules

### Shared INDPRO batches

Prepare one master collection of **800 INDPRO batches**, sampling valid training windows uniformly with replacement.

Define a saved target block containing the first **300 batches**. Every augmented run uses exactly this target block.

| Condition | INDPRO batches | Auxiliary batches | Updates |
|---|---:|---:|---:|
| Target-only control | 300 | 0 | 300 |
| Augmented training | 300 | 500 | 800 |
| Longer target-only control | 800 | 0 | 800 |

The 300-update control uses the target block in its saved order. The 800-update control uses that same block followed by the additional 500 INDPRO batches.

Run both controls separately from the common initialization, once per model, and share their results.

### Auxiliary batches

Each teammate prepares **500 auxiliary batches** for their complete group.

- Divide 500 as evenly as possible among eligible series.
- Allocate remainder batches using the saved sampling generator.
- Each batch contains examples from one series.
- Sample 32 valid training windows with replacement for each batch.
- Shuffle the resulting 500-batch auxiliary block once and save its order.

Group size therefore affects exposure per auxiliary series, while the total auxiliary budget remains equal across groups.

Save unique batch IDs, series IDs, and exact window references. Repeated windows are permitted because sampling uses replacement.

### Four schedules per teammate

Reuse the same 300 target batches and 500 auxiliary batches in all four schedules.

| Schedule | Presentation |
|---|---|
| **Random mixing** | Shuffle all 800 batches together once |
| **Auxiliary first** | Saved 500-batch auxiliary block → saved 300-batch target block |
| **Target first** | Saved 300-batch target block → saved 500-batch auxiliary block |
| **Regular interleaving** | Repeat the cycle below 100 times |

For interleaving, `T` consumes the next unused batch from the saved target block and `A` consumes the next unused batch from the saved auxiliary block:

```text
T, A, A, T, A, A, T, A
```

This consumes exactly 300 target and 500 auxiliary batches.

Apply each schedule once across the full run. Configure the BasicTS loader to preserve it. Months inside forecasting examples always remain chronological.

The ordering comparisons must differ only in batch sequence. Membership, multiplicities, initialization, optimizer, and total exposure stay fixed.

### How to interpret the controls

- **Augmented versus 300-update target-only:** Does additional auxiliary training help while preserving target exposure?
- **Augmented versus 800-update target-only:** Is auxiliary training more useful than spending the additional updates on INDPRO?
- **Groups under random mixing:** Which complete-group choice works better under equal budgets?
- **Orders within one group:** Does scheduling the same batches differently affect forecasting?

## 4. Teammate assignments and code requirements

### Teammate 1: Labor-market study

Run all four schedules for the complete labor-market group across all three models.

Lead the shared data-preparation checks, including snapshot verification, group membership, transformations, scaling, and window indexing.

Produce the **300-update target-only controls** for all three models and share them.

Write a short study explaining whether labor-market training helps INDPRO forecasting and whether its scheduling changes the result.

### Teammate 2: Output-and-income study

Run all four schedules for the complete output-and-income group, excluding INDPRO from auxiliary membership.

Lead the shared BasicTS configuration, model implementations, environment lockfile, and final-checkpoint evaluation procedure.

Produce the **800-update target-only controls** for all three models and share them.

Write a short study explaining the selection and ordering results for output-and-income training.

### Teammate 3: Housing study

Run all four schedules for the complete housing group.

Lead checks that schedules preserve identical batch contents and exposure, and verify that the regular cycle consumes the intended batch counts.

Write a short study explaining the housing results, including changes in INDPRO validation error at training-stage transitions.

### Teammate 4: Interest-and-exchange-rate study

Run all four schedules for the complete interest-and-exchange-rate group.

Lead the shared results-table format, figure conventions, and aggregation procedure.

Write a short study explaining the group’s selection and ordering results, with particular attention to performance after training moves away from INDPRO.

Each teammate owns **12 augmented runs**. Teammates 1 and 2 each additionally own three shared control runs.

### Instructions for every coding LLM

Prioritize **readable, robust, reproducible code**:

- Use small named functions, meaningful variable names, docstrings, and useful type hints.
- Separate preprocessing, window construction, batch preparation, scheduling, model configuration, evaluation, and plotting.
- Reuse shared modules; represent experimental differences through configuration.
- Provide documented commands for preparation, verification, individual runs, and aggregation.
- Keep notebooks optional. Experiments must run without hidden notebook state.
- Validate assumptions and fail with actionable errors. Report exclusions explicitly.
- Save outputs separately by group, model, schedule, and seed.
- Preserve the raw snapshot and existing results.
- Inspect the pinned BasicTS API rather than inventing configuration fields.
- Save actual outputs and logs; do not fabricate unavailable results.

The shared run interface must accept group, model, schedule/control, seed, prepared-artifact location, and output location. Full runs must load the frozen artifacts rather than resampling automatically.

## 5. Verification, reporting, and execution sequence

### Required checks

Before accepting results, verify:

- Identical snapshot checksums, dependencies, and preprocessing across teammates.
- One common INDPRO training-window index and evaluation-window set.
- Training-only scaling and correct chronological boundaries.
- Correct transformation handling, including invalid log inputs.
- Identical batch IDs and multiplicities across each group’s four schedules.
- Three finite predictions per example and finite losses.
- Exactly 300 or 800 optimizer updates, with final validation included.
- Final-checkpoint evaluation rather than best-validation selection.
- Ordinary MSE/MAE that count valid zero observations.
- Agreement between reported metrics and a direct calculation from saved predictions.

Complete one shared smoke check before full training.

### Outputs from each study

Submit:

- Reproducible code, run instructions, configurations, and provenance.
- Final INDPRO test MSE and MAE after reversing standardization.
- INDPRO validation MSE curves against actual optimizer updates.
- Training-batch source plots against update number.
- Training time, hardware, and valid-window counts.
- Forecast plots for the same four INDPRO test windows across conditions.
- A short written account of the question, method, findings, and limitations.

Select the four example windows once from evenly spaced positions in the valid test-window list, before inspecting errors.

Report errors in INDPRO monthly growth units. Include an always-zero-growth forecast, meaning zero in the **original growth units**, rather than standardized zero.

Save predictions with forecast origin, forecast month, horizon, actual value, and predicted value. Aggregate test errors over all forecast origins and all three horizons consistently.

Measure training time excluding preparation, validation, testing, and plotting. Record preparation time separately.

### Execution sequence

1. **Freeze the common setup.** Prepare the snapshot, metadata, windows, batches, initialization checkpoints, environment, and output format.
2. **Run shared checks.** Every teammate verifies the same artifacts.
3. **Complete selection comparisons first.** Run random mixing for four groups across three models, plus the six shared controls: **18 runs**.
4. **Complete ordering comparisons.** Run the remaining three schedules per group and model: **36 additional runs**.
5. **Compile the short report.** Compare group selection under random mixing first, then scheduling within each group.

Keep the protocol fixed during this pass. Use validation results for troubleshooting and reserve test evaluation for the final comparisons.

The report should state that single-seed findings are preliminary, revised historical data do not establish real-time forecasting performance, and auxiliary-first gains may reflect finishing on INDPRO. Regular interleaving tests the specified schedule. Improvements, deterioration, and little difference are all valid outcomes.
