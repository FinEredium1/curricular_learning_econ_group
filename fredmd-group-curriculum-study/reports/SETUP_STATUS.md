# Teammate 2: setup and next steps

## Assignment

Your auxiliary group is complete output and income, excluding INDPRO: CUMFNS, IPB51222S, IPBUSEQ, IPCONGD, IPDCONGD, IPDMAT, IPFINAL, IPFPNSS, IPFUELS, IPMANSICS, IPMAT, IPNCONGD, IPNMAT, RPI, W875RX1.

INDPRO is the common target. Your assigned study is 12 augmented runs (three models, four orders), plus three shared 800-update target-only controls. Teammate 1 owns the 300-update controls. Interest and exchange rates is already assigned to another teammate.

## Completed setup

- Agreed PLAN.md copied unchanged into this project.
- Fresh isolated Python environment and complete requirements.lock.
- BasicTS 1.1.0 pinned to c2bb6e31e591167e84459775a21a62e70a5893ce.
- September 2026 CSV and official appendix downloaded unchanged; provenance is in data/raw/sources.json.
- Snapshot SHA-256: d6e239038afbdbf4667c2a63c13c942817bc196fedab4355cd7e0d353ecdd65d.
- Official group membership verified: 15 output/income auxiliaries, 31 labor, 10 housing, 22 interest/exchange; 79 distinct series including INDPRO.
- Training-only per-series preprocessing with ddof=0, independent missing-data windows, and no excluded group members.
- Common INDPRO windows: 465 training, 118 validation, 118 test.
- Saved common batches, per-group allocations, all four schedules, common model initialization, and frozen file hashes.
- Seven correctness tests passed; pip check found no broken requirements.
- Linear, MLP, and PatchTST each passed an exact three-update BasicTS smoke run, with final validation and finite predictions.
- No full research runs or test-set evaluations have been performed.

## Share before comparing studies

Share the same source code, PLAN.md, configs, requirements files, raw data, prepared data, and artifacts. Teammates should use the same BasicTS commit and frozen initialization/batch files. Do not independently regenerate the setup or copy the machine-specific .venv directory.

The current reference backend is CPU, float32, one PyTorch thread. Agree on this shared runtime or one reviewed common replacement before accepting cross-teammate comparisons. Record hardware, and compare timings only on comparable hardware.

The interface supports every assigned group through --group (1, 2, 3, 6); teammate number and official group number differ. Your --group is 1.

## First research batch

For each model (linear, mlp, patchtst), run random mixing for group 1 and target_only_800. These six runs establish your selection reference and shared longer target controls. Then run auxiliary_first, target_first, and interleaved for all three models (nine more runs).

Use the shared target_only_300 results from Teammate 1 in the eventual report. Keep smoke outputs separate from research results. Full commands are in README.md.

## Important implementation notes

- The installed PatchTST supports layer normalization directly; no extra normalization adapter is needed.
- The pinned BasicTS runner has a step-mode stopping-boundary issue. The documented ExactBudgetRunner wrapper preserves its training loop and independently verifies the number of optimizer updates.
- All research runs load explicit final weights. No best-validation checkpoint is substituted.
- The accepted PLAN.md uses ddof=0; the alternate teammate document's ddof=1 is not used.
- Non-target code-5 series use ordinary log differences; INDPRO and PAYEMS retain the starter's factor of 100. This factor cancels under per-series standardization and is recorded in metadata.
