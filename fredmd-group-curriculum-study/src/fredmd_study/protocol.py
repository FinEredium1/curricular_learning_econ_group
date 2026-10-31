"""Protocol constants and reproducible, named random streams."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

INPUT_LEN = 24
OUTPUT_LEN = 3
BATCH_SIZE = 32
SEED = 1
TARGET = "INDPRO"
GROUP_NAMES = {1: "output_income", 2: "labor", 3: "housing", 6: "interest_exchange"}
EXPECTED_COUNTS = {1: 15, 2: 31, 3: 10, 6: 22}
SPLITS = {"train": (1960, 1999), "val": (2000, 2009), "test": (2010, 2019)}
ORDERS = ("random", "auxiliary_first", "target_first", "interleaved")
MODELS = ("linear", "mlp", "patchtst")
# Case differs between the official appendix and dated CSV; no other aliases are inferred.
SERIES_ALIASES = {"IPB51222s": "IPB51222S"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rng(role: str, series: str = "", group: int = 0) -> np.random.Generator:
    """Stable, independent streams; Python's randomized hash() is never used."""
    key = f"seed={SEED}|role={role}|series={series}|group={group}".encode()
    words = np.frombuffer(hashlib.sha256(key).digest(), dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence([SEED, *map(int, words)]))
