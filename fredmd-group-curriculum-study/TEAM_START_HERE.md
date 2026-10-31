# Team setup: start here

This package contains the common implementation and frozen inputs for the brief FRED-MD experiment. Use this same package before comparing results.

## What everyone shares

- BasicTS 1.1.0 at commit c2bb6e31e591167e84459775a21a62e70a5893ce; source is included in vendor/BasicTS.
- Python 3.11 and the pinned requirements.lock.
- The unchanged September 2026 CSV and official appendix.
- Per-series preprocessing with ddof=0, the same INDPRO windows, saved initial weights, and exact batch/schedule manifests.
- CPU, float32, one PyTorch thread for the current reference setup.

The README and configs describe the package author's assignment as Teammate 2. Those labels do not restrict the shared code. Select your assigned official group with --group; do not edit frozen files to change your assignment.

## Setup on macOS or Linux

Unzip the package. Open a terminal in the extracted fredmd-group-curriculum-study folder. Install Python 3.11 and Git if needed, then run:

    python3.11 -m venv .venv
    .venv/bin/python -m pip install -r requirements.lock
    .venv/bin/python study.py verify
    .venv/bin/python -m pytest tests -q

For Windows, create the environment with "py -3.11 -m venv .venv" and replace ".venv/bin/python" with ".venv\\Scripts\\python.exe" in the remaining commands.

Package installation requires internet access. If a pinned dependency fails on your platform, share the error before changing versions; use a common reviewed environment.

Do not rerun "study.py prepare": the package already contains the frozen prepared inputs.

A successful verification reports 79 total series and INDPRO windows of 465 training, 118 validation, and 118 test.

## Choose your group

| Teammate | Official group flag | Auxiliary data |
|---|---|---|
| 1 | --group 2 | Labor market |
| 2 | --group 1 | Output and income, excluding INDPRO |
| 3 | --group 3 | Housing |
| 4 | --group 6 | Interest and exchange rates |

## Run the technical checks first

Replace the group flag with your own:

    .venv/bin/python study.py smoke --group 2 --model linear
    .venv/bin/python study.py smoke --group 2 --model mlp
    .venv/bin/python study.py smoke --group 2 --model patchtst

These are three-update checks with validation only, not research results. Each must report updates=3 and smoke_only=True. Test-set errors are not evaluated.

## Then run your assigned study

Example for Teammate 1:

    .venv/bin/python study.py run --group 2 --model linear --order random

Repeat for models linear, mlp, and patchtst, and schedules random, auxiliary_first, target_first, and interleaved.

Teammate 1 also owns target_only_300 for all three models. Teammate 2 owns target_only_800 for all three models. Share those controls once produced.

Full runs evaluate final weights on the test set, so confirm the shared setup before starting them. Begin with random mixing and the common controls, as specified in PLAN.md.

## Keep the comparison consistent

- Use the packaged snapshot, prepared inputs, saved batches, initial weights, and pinned source unchanged.
- Do not silently alter scaling, transformations, architecture, learning rate, budgets, or schedules.
- Use the CPU reference backend unless the team agrees on a reviewed common replacement.
- Record hardware; do not compare training times across different hardware.
- Keep smoke outputs separate from final study outputs.
- Return each research run's summary.json, validation.csv, predictions.csv, validation.png, and your short write-up. Keep the original run directory and final checkpoint for audit.
- Include the snapshot and freeze checksums from summary.json when exchanging results.

Share changes to the common implementation with everyone before using them in comparisons. A changed scientific input or source file makes verification fail until the team reviews a new freeze.

The archive excludes the author's installed environment, caches, and smoke-run results. Everyone creates a fresh local environment.
