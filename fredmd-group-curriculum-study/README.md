# FRED-MD Group Curriculum Study

Assigned: **Teammate 2 — complete output-and-income group, excluding INDPRO.**

Setup verified: seven correctness checks and three exact-update BasicTS smoke runs passed. Your 15 assigned research runs are complete and verified. See reports/teammate2/REPORT.md. Initial setup details remain in reports/SETUP_STATUS.md.

The agreed protocol is PLAN.md. This is a brief, single-seed experiment. Your scope is 12 group-1 augmented runs and three shared 800-update INDPRO-only controls. Teammate 4 is already assigned.

## Shared reference setup

- Python 3.11; PyTorch 2.5.1; NumPy 1.26.4; pandas 2.2.3.
- BasicTS source pinned at c2bb6e31e591167e84459775a21a62e70a5893ce (1.1.0).
- Dated September 2026 snapshot; unchanged inputs and checksums in data/raw/.
- Per-series training-only scaling with ddof=0, as specified in the accepted PLAN.md.
- CPU, float32, one PyTorch thread for the reference setup. Agree on the same backend with teammates before accepting comparable study runs. Hardware-dependent times must not be pooled.

BasicTS 1.1.0 directly supports norm_type="layer_norm"; no PatchTST normalization patch is needed.

The pinned runner stops one step early in step mode. ExactBudgetRunner applies a small documented boundary adapter around the existing BasicTS loop, counts real optimizer steps, and requires exact completion. The vendored source stays unchanged. All three models must pass a three-update smoke run.

## Commands

Run commands from this folder with the project environment:

    .venv/bin/python study.py prepare
    .venv/bin/python study.py verify
    .venv/bin/python -m pytest tests -q
    .venv/bin/python study.py smoke --group 1 --model linear
    .venv/bin/python study.py smoke --group 1 --model mlp
    .venv/bin/python study.py smoke --group 1 --model patchtst

Preparation freezes all four groups, common target batches, all schedules, and all initial weights for a shareable team starter. It does not train other teammates' studies.

After the shared setup and smoke checks are accepted:

    .venv/bin/python study.py run --group 1 --model linear --order random
    .venv/bin/python study.py run --group 1 --model linear --order auxiliary_first
    .venv/bin/python study.py run --group 1 --model linear --order target_first
    .venv/bin/python study.py run --group 1 --model linear --order interleaved
    .venv/bin/python study.py run --group 1 --model linear --order target_only_800

Repeat those study commands for mlp and patchtst. A run refuses to overwrite its output directory; choose a new --output path for an explicitly documented rerun.

Smoke runs perform three updates with validation only. They never inspect test errors, and are excluded from study results. Research run commands use the full 300/800 budgets and explicitly evaluate final weights.

## Layout

- src/fredmd_study/: shared preparation, scheduling, model, dataset, and BasicTS adapter code.
- configs/: frozen protocol choices.
- data/raw/: official unchanged CSV, appendix archive, provenance and checksums.
- data/prepared/: separate per-series windows/scaling and standardized arrays.
- artifacts/: frozen batch manifests, initial weights and integrity manifest.
- results/smoke/: technical verification only.
- results/study/: final research runs.
- reports/: transformed-series figures, checks, study write-up, and handoff notes.
- tests/: focused correctness checks.

## Reproduction

Clone BasicTS into vendor/BasicTS, check out the exact recorded commit, create a Python 3.11 environment, and install requirements.lock from this folder. The editable BasicTS requirement uses its relative local vendor directory. Share the frozen raw data, prepared data and artifacts with teammates rather than independently regenerating them.

Every training command verifies frozen hashes and package versions before starting. Changing scientific source files requires a newly reviewed freeze, not an unnoticed rerun.

## Metric and timing details

MSE/MAE include legitimate zero targets. Test predictions and targets are inverse-scaled into INDPRO growth units; the zero-growth comparator is zero in those original units. Test errors count every origin/horizon prediction.

Training timing measures each step after batch fetch and before validation; it excludes data preparation, batch fetching, validation, testing, and plotting. Record hardware separately.

## Deliberate implementation choices

Named RNGs use seed 1 plus a stable SHA256-derived role/series/group key. These exact frozen manifests must be shared.

Only INDPRO and PAYEMS code-5 values use the starter's factor of 100; other code-5 values use ordinary log-differences. This multiplier cancels during separate per-series standardization. The effective transformation is recorded for each series.

Full groups are derived from the official appendix, with the explicit IPB51222s -> IPB51222S spelling mapping. Missing or unusable group members cause preparation to stop for resolution.
