# Jayden: output-and-income experiment

This branch contains Teammate 2's completed **single-seed** experiment. Start with reports/teammate2/REPORT.md and reports/teammate2/results.csv when compiling the team report.

## Contents

- Complete output-and-income group: 15 auxiliaries plus INDPRO as the common evaluation target.
- Three models, four augmented schedules, and three shared 800-update INDPRO-only controls: 15 completed runs.
- Official September 2026 snapshot, frozen preprocessing, batch manifests, and initial weights.
- Per-run predictions, validation curves, final checkpoints, logs, figures, and verification records.
- The exact BasicTS revision as a Git submodule.

## Reproduce from this GitHub branch

From a new checkout:

    git clone --branch output-and-income-jayden --recurse-submodules https://github.com/FinEredium1/curricular_learning_econ_group.git
    cd curricular_learning_econ_group/fredmd-group-curriculum-study
    python3.11 -m venv .venv
    .venv/bin/python -m pip install -r requirements.lock
    .venv/bin/python study.py verify
    .venv/bin/python -m pytest tests -q

If you already cloned without submodules, run "git submodule update --init --recursive" from the repository root first.

The installed environment and caches are deliberately excluded. All scientific files covered by artifacts/freeze.json are included unchanged. The snapshot, prepared data, and starting weights should not be regenerated for team comparisons.

## Main findings

- Added output-and-income training lowered MSE compared with each model's 800-update target-only control in this seed.
- The lowest-MSE order differed by model: target-first for Linear, random mixing for MLP, and regular interleaving for PatchTST.
- MLP random mixing had MSE about 0.230304, versus 0.237266 for zero growth.
- Zero growth had lower MAE than every trained run.
- These are preliminary single-seed results, not statistically conclusive or real-time forecasting evidence.

## Combining branches

Use the frozen snapshot/setup hashes, model, condition, and seed columns to match comparisons. Take the shared 800-update controls once per model, rather than treating their appearances in several reports as independent runs.

Teammate 1's 300-update controls and other economic groups remain separate pending inputs. Do not label this branch a complete cross-group comparison.

The original setup ZIP included BasicTS source directly. On GitHub, the same revision is represented by a submodule so a second repository's internal Git files are not uploaded as ordinary project files.
