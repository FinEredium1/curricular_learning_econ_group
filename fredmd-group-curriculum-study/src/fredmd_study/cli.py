"""Preparation, artifact verification, and explicit research/smoke commands."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import subprocess

import numpy as np

from .protocol import (
    GROUP_NAMES, INPUT_LEN, MODELS, ORDERS, OUTPUT_LEN, SPLITS, TARGET,
    read_json, sha256, write_json,
)


def freeze(root: Path) -> dict:
    """Record scientific inputs and code; source changes invalidate subsequent runs."""
    paths = [root / "PLAN.md", root / "requirements.in", root / "requirements.lock",
             root / "configs/protocol.json", root / "study.py"]
    for pattern in ("data/raw/*", "data/prepared/*", "artifacts/initial*.pt",
                    "artifacts/initializations.json", "artifacts/batches_seed1.json",
                    "src/fredmd_study/*.py"):
        paths.extend(path for path in root.glob(pattern) if path.is_file())
    commit = subprocess.check_output(
        ["git", "-C", str(root / "vendor/BasicTS"), "rev-parse", "HEAD"], text=True
    ).strip()
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "basicts_commit": commit, "python": platform.python_version(),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("BasicTS", "torch", "numpy", "pandas", "easy-torch")},
        "files": {str(path.relative_to(root)): sha256(path) for path in sorted(set(paths))},
    }
    write_json(root / "artifacts/freeze.json", result)
    return result


def verify(root: Path) -> dict:
    """Verify the frozen scientific setup plus window and schedule invariants."""
    from .schedules import verify_schedules
    frozen = read_json(root / "artifacts/freeze.json")
    for relative, expected in frozen["files"].items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Frozen file changed: {relative}; resolve before training.")
    commit = subprocess.check_output(
        ["git", "-C", str(root / "vendor/BasicTS"), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != frozen["basicts_commit"]:
        raise ValueError("BasicTS checkout no longer matches frozen commit.")
    for name, expected in frozen["versions"].items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"{name}: expected version {expected}, got {actual}.")
    metadata = read_json(root / "data/prepared/metadata.json")
    manifest = read_json(root / "artifacts/batches_seed1.json")
    verify_schedules(manifest, metadata)
    values = dict(np.load(root / "data/prepared/standardized.npz", allow_pickle=False))
    years = np.array([int(date[:4]) for date in metadata["dates"]])
    for name, info in metadata["series"].items():
        fit = values[name][(years >= 1960) & (years <= 1999)]
        fit = fit[np.isfinite(fit)]
        assert np.isclose(fit.mean(), 0.0, atol=1e-10), name
        assert np.isclose(fit.std(ddof=0), 1.0, atol=1e-10), name
        for split, indices in info["windows"].items():
            start, end = SPLITS[split]
            for first in indices:
                assert start <= years[first] and years[first + OUTPUT_LEN - 1] <= end
                assert np.isfinite(values[name][first - INPUT_LEN:first + OUTPUT_LEN]).all()
    summary = {
        "verified": True, "series": len(values),
        "groups": {group: len(names) for group, names in metadata["groups"].items()},
        "indpro_windows": metadata["series"][TARGET]["window_counts"],
        "snapshot_sha256": metadata["snapshot_sha256"],
    }
    return summary


def main(root: Path) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify", "smoke", "run"])
    parser.add_argument("--group", type=int, choices=list(GROUP_NAMES), default=1)
    parser.add_argument("--model", choices=MODELS, default="linear")
    parser.add_argument("--order", choices=[*ORDERS, "target_only_300", "target_only_800"], default="random")
    parser.add_argument("--seed", type=int, choices=[1], default=1)
    parser.add_argument("--output", type=Path, help="New output directory inside this research folder.")
    args = parser.parse_args()
    if args.command == "prepare":
        from .preparation import prepare_data
        from .schedules import prepare_schedules
        from .models import prepare_initializations
        # Preparation is a single freeze step. Fail on partial/existing artifacts.
        if (root / "artifacts/freeze.json").exists():
            raise FileExistsError("Frozen artifacts already exist. Use verify.")
        metadata = prepare_data(root)
        prepare_schedules(root, metadata)
        prepare_initializations(root)
        freeze(root)
        print(verify(root))
    elif args.command == "verify":
        print(verify(root))
    else:
        from .runner import run_condition
        verify(root)
        smoke = args.command == "smoke"
        output = args.output or (
            root / "results" / ("smoke" if smoke else "study")
            / f"group{args.group}" / args.order / args.model / "seed1"
        )
        output = output.resolve()
        if not output.is_relative_to(root.resolve()):
            raise ValueError("All outputs must stay inside the research folder.")
        print(run_condition(root, args.group, args.model, args.order, output, smoke=smoke))
