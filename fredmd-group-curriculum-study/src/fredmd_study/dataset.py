"""Thin BasicTS dataset adapter for frozen univariate sample references."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from basicts.data.base_dataset import BasicTSDataset

from .protocol import INPUT_LEN, OUTPUT_LEN, TARGET, read_json


class FrozenForecastDataset(BasicTSDataset):
    def __init__(self, dataset_name: str, root: str, batch_ids: list[str], mode: str):
        super().__init__(dataset_name, mode)
        project = Path(root)
        self.metadata = read_json(project / "data/prepared/metadata.json")
        self.values = dict(np.load(project / "data/prepared/standardized.npz", allow_pickle=False))
        manifest = read_json(project / "artifacts/batches_seed1.json")
        if str(mode) == "train":
            self.samples = [
                (manifest["batches"][identifier]["series"], index)
                for identifier in batch_ids
                for index in manifest["batches"][identifier]["forecast_indices"]
            ]
        else:
            split = "val" if str(mode) == "val" else "test"
            self.samples = [(TARGET, index) for index in self.metadata["series"][TARGET]["windows"][split]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        series, first = self.samples[item]
        values = self.values[series]
        return {
            "inputs": torch.tensor(values[first - INPUT_LEN:first, None], dtype=torch.float32),
            "targets": torch.tensor(values[first:first + OUTPUT_LEN, None], dtype=torch.float32),
        }
