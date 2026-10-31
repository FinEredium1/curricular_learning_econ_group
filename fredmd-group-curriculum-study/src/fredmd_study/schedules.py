"""Save batch contents once, then change only their presentation order."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import numpy as np

from .protocol import BATCH_SIZE, GROUP_NAMES, ORDERS, SEED, TARGET, read_json, rng, write_json


def allocate_batches(series: list[str], total: int = 500, group: int = 0) -> dict[str, int]:
    if not series or total < len(series):
        raise ValueError("The batch budget must expose every assigned series.")
    count, remainder = divmod(total, len(series))
    allocation = dict.fromkeys(series, count)
    for name in rng("allocation", group=group).permutation(series)[:remainder]:
        allocation[str(name)] += 1
    return allocation


def build_orders(target: list[str], auxiliary: list[str], group: int) -> dict[str, list[str]]:
    mixed = target + auxiliary
    random_order = [mixed[int(i)] for i in rng("random_order", group=group).permutation(len(mixed))]
    interleaved = []
    t, a = iter(target), iter(auxiliary)
    for _ in range(100):
        interleaved.extend([next(t), next(a), next(a), next(t), next(a), next(a), next(t), next(a)])
    orders = {
        "random": random_order,
        "auxiliary_first": auxiliary + target,
        "target_first": target + auxiliary,
        "interleaved": interleaved,
    }
    for order in orders.values():
        if len(order) != 800 or Counter(order) != Counter(mixed):
            raise AssertionError("An ordering changed batch contents or exposure.")
    return orders


def prepare_schedules(root: Path, metadata: dict) -> dict:
    destination = root / "artifacts" / "batches_seed1.json"
    if destination.exists():
        raise FileExistsError("Batch manifests already exist; verify rather than overwrite.")
    batches = {}

    def sample(name: str, number: int, prefix: str) -> list[str]:
        candidates = metadata["series"][name]["windows"]["train"]
        generator = rng("window_sampling", series=name)
        ids = []
        for position in range(number):
            identifier = f"{prefix}:{name}:{position:04d}"
            starts = generator.choice(candidates, size=BATCH_SIZE, replace=True)
            batches[identifier] = {"series": name, "forecast_indices": starts.astype(int).tolist()}
            ids.append(identifier)
        return ids

    master_target = sample(TARGET, 800, "target")
    target = master_target[:300]
    groups = {}
    for group in GROUP_NAMES:
        allocation = allocate_batches(metadata["groups"][str(group)], group=group)
        auxiliary = [identifier for name, count in allocation.items()
                     for identifier in sample(name, count, f"group{group}")]
        auxiliary = [auxiliary[int(i)] for i in rng("auxiliary_block", group=group).permutation(500)]
        groups[str(group)] = {
            "allocation": allocation, "auxiliary": auxiliary,
            "orders": build_orders(target, auxiliary, group),
        }
    manifest = {
        "seed": SEED, "batch_size": BATCH_SIZE,
        "rng_scheme": "SHA256 of seed/role/series/group -> NumPy SeedSequence; independent PCG64 streams",
        "target_300": target, "target_800": master_target,
        "batches": batches, "groups": groups,
    }
    write_json(destination, manifest)
    return manifest


def verify_schedules(manifest: dict, metadata: dict) -> None:
    """Check every saved sample reference, allocation, and schedule."""
    assert manifest["target_800"][:300] == manifest["target_300"]
    for identifier, batch in manifest["batches"].items():
        assert len(batch["forecast_indices"]) == BATCH_SIZE, identifier
        allowed = set(metadata["series"][batch["series"]]["windows"]["train"])
        assert set(batch["forecast_indices"]) <= allowed, identifier
    for group, record in manifest["groups"].items():
        assert set(record["allocation"]) == set(metadata["groups"][group])
        actual = Counter(manifest["batches"][identifier]["series"] for identifier in record["auxiliary"])
        assert actual == Counter(record["allocation"])
        combined = manifest["target_300"] + record["auxiliary"]
        assert set(record["orders"]) == set(ORDERS)
        for order in record["orders"].values():
            assert len(order) == 800
            assert Counter(order) == Counter(combined)
        assert record["orders"]["auxiliary_first"] == record["auxiliary"] + manifest["target_300"]
        assert record["orders"]["target_first"] == manifest["target_300"] + record["auxiliary"]
