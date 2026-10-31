"""Exact small models and the pinned BasicTS PatchTST implementation."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn
from basicts.configs import BasicTSModelConfig
from basicts.models.PatchTST import PatchTSTConfig, PatchTSTForForecasting

from .protocol import INPUT_LEN, OUTPUT_LEN, MODELS, SEED, sha256, write_json


class Linear(nn.Module):
    def __init__(self, config: BasicTSModelConfig):
        super().__init__()
        self.mapping = nn.Linear(config.input_len, config.output_len, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mapping(inputs.squeeze(-1)).unsqueeze(-1)


class MLP(nn.Module):
    def __init__(self, config: BasicTSModelConfig):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.input_len, 64, bias=True), nn.ReLU(),
            nn.Linear(64, config.output_len, bias=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs.squeeze(-1)).unsqueeze(-1)


def model_spec(name: str) -> tuple[type[nn.Module], object]:
    if name in {"linear", "mlp"}:
        return (Linear if name == "linear" else MLP), BasicTSModelConfig(input_len=INPUT_LEN, output_len=OUTPUT_LEN)
    if name != "patchtst":
        raise ValueError(f"Unknown model: {name}")
    return PatchTSTForForecasting, PatchTSTConfig(
        input_len=INPUT_LEN, output_len=OUTPUT_LEN, num_features=1,
        patch_len=6, patch_stride=3, hidden_size=32, n_heads=4,
        num_layers=2, intermediate_size=128, norm_type="layer_norm",
        attn_dropout=0.0, fc_dropout=0.0, head_dropout=0.0,
        use_revin=False, decomp=False,
    )


def prepare_initializations(root: Path) -> dict:
    records = {}
    for name in MODELS:
        path = root / "artifacts" / f"initial_{name}_seed{SEED}.pt"
        if path.exists():
            raise FileExistsError(f"Initial checkpoint already exists: {path}")
        torch.manual_seed(SEED)
        cls, config = model_spec(name)
        model = cls(config).float()
        prediction = model(torch.zeros(2, INPUT_LEN, 1))
        assert prediction.shape == (2, OUTPUT_LEN, 1)
        assert torch.isfinite(prediction).all()
        assert all(module.p == 0 for module in model.modules() if isinstance(module, nn.Dropout))
        torch.save({"model_state_dict": model.state_dict()}, path)
        resolved = asdict(config) if name == "patchtst" else dict(config)
        records[name] = {
            "checkpoint": path.name, "sha256": sha256(path),
            "class": cls.__name__, "config": resolved,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    write_json(root / "artifacts" / "initializations.json", records)
    return records
