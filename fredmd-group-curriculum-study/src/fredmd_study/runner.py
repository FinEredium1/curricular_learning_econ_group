"""BasicTS runner adapter: exact budgets, frozen initialization, final-weight evaluation."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

from basicts.configs import BasicTSForecastingConfig
from basicts.runners import BasicTSRunner
from basicts.utils import RunnerStatus
from easytorch.device import set_device_type

from .dataset import FrozenForecastDataset
from .models import model_spec
from .protocol import BATCH_SIZE, SEED, TARGET, read_json, sha256, write_json


def mse(prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Ordinary MSE: legitimate zero targets are included."""
    return (prediction - targets).square().mean()


def mae(prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (prediction - targets).abs().mean()


class ExactBudgetRunner(BasicTSRunner):
    """Keep BasicTS training/evaluation, with a tested upstream boundary correction.

    Pinned BasicTS c2bb6e31 increments global_steps before its >= stopping check.
    Its step-mode loop therefore stops one optimizer update early. Raising that
    internal bound by one preserves the requested scientific budget; an independent
    counter and exact batch-count checks verify the number of real optimizer steps.
    The vendored source is not modified.
    """

    def __init__(self, cfg):
        self.requested_updates = cfg.num_steps
        self.actual_updates = 0
        self.training_seconds = 0.0
        self.validation_rows = []
        self.test_predictions = []
        self.test_targets = []
        super().__init__(cfg)
        checkpoint = torch.load(cfg.initial_checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    def _train_loop(self):
        self.num_steps = self.requested_updates + 1
        try:
            super()._train_loop()
        finally:
            self.num_steps = self.requested_updates

    def _optimizer_step(self):
        super()._optimizer_step()
        self.actual_updates += 1

    def on_step_start(self, step):
        self._compute_started = time.perf_counter()

    def on_step_end(self, step):
        self.training_seconds += time.perf_counter() - self._compute_started
        if step % 30 == 0 or step == self.requested_updates:
            self.validate(train_step=step)
            self.reset_meters()

    def _forward(self, model, data, step, epoch=None):
        result = super()._forward(model, data, step, epoch)
        if result["prediction"].shape != result["targets"].shape:
            raise RuntimeError("Prediction shape does not match three-month targets.")
        if not torch.isfinite(result["prediction"]).all():
            raise FloatingPointError("Nonfinite predictions; stop rather than save invalid results.")
        if self.status == RunnerStatus.EVALUATING:
            self.test_predictions.append(result["prediction"].detach().cpu().numpy())
            self.test_targets.append(result["targets"].detach().cpu().numpy())
        return result

    def on_validate_end(self, train_step=None, train_epoch=None):
        std = self.cfg.target_std
        row = {
            "update": int(train_step),
            "mse_standardized": self.meter_pool.get_value("val/MSE"),
            "mae_standardized": self.meter_pool.get_value("val/MAE"),
        }
        row["mse_growth_units"] = row["mse_standardized"] * std ** 2
        row["mae_growth_units"] = row["mae_standardized"] * std
        self.validation_rows.append(row)
        pd.DataFrame(self.validation_rows).to_csv(Path(self.cfg.output_dir) / "validation.csv", index=False)
        super().on_validate_end(train_step=train_step, train_epoch=train_epoch)

    def on_train_end(self):
        if self.actual_updates != self.requested_updates:
            raise RuntimeError(f"Expected {self.requested_updates} updates, got {self.actual_updates}.")
        if self.validation_rows[-1]["update"] != self.requested_updates:
            raise RuntimeError("Final-update validation is missing.")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "step": self.actual_updates,
        }, Path(self.cfg.output_dir) / "final.pt")
        super().on_train_end()


def run_condition(root: Path, group: int, model_name: str, order: str,
                  output: Path, smoke: bool = False) -> dict:
    """Execute a frozen condition. Smoke runs validate but never inspect test errors."""
    metadata = read_json(root / "data/prepared/metadata.json")
    manifest = read_json(root / "artifacts/batches_seed1.json")
    if order == "target_only_300":
        batch_ids = manifest["target_300"]
    elif order == "target_only_800":
        batch_ids = manifest["target_800"]
    else:
        batch_ids = manifest["groups"][str(group)]["orders"][order]
    if smoke:
        batch_ids = batch_ids[:3]
    output.mkdir(parents=True, exist_ok=False)
    model_cls, model_config = model_spec(model_name)
    initial = root / "artifacts" / f"initial_{model_name}_seed{SEED}.pt"
    target = metadata["series"][TARGET]
    torch.set_num_threads(1)
    set_device_type("cpu")
    cfg = BasicTSForecastingConfig(
        model=model_cls, model_config=model_config,
        dataset_name="FREDMD_2026_09", dataset_type=FrozenForecastDataset,
        dataset_params={"dataset_name": "FREDMD_2026_09", "root": str(root), "batch_ids": batch_ids},
        scaler=None, rescale=False, null_val=np.nan,
        input_len=24, output_len=3,
        train_batch_size=BATCH_SIZE, val_batch_size=128, test_batch_size=128,
        train_data_shuffle=False, train_data_num_workers=0,
        val_data_shuffle=False, test_data_shuffle=False,
        num_epochs=None, num_steps=len(batch_ids), val_interval=30,
        test_interval=None, eval_after_train=False, save_results=False,
        loss=mse, metrics=[("MSE", mse), ("MAE", mae)], target_metric="MSE",
        optimizer_params={"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-8,
                          "weight_decay": 0.0, "amsgrad": False},
        lr_scheduler=None, callbacks=[], gpus=None, seed=SEED,
        deterministic=True, tf32=False, cudnn_benchmark=False,
        model_dtype="float32", ckpt_save_dir=str(output / "basicts"),
    )
    cfg.initial_checkpoint = str(initial)
    cfg.output_dir = str(output)
    cfg.target_std = target["std"]
    cfg.save()
    runner = ExactBudgetRunner(cfg)
    runner.init_logger(logger_name="BasicTS-study", log_file_name="training_log")
    runner.train()
    summary = {
        "group": group, "model": model_name, "order": order, "seed": SEED,
        "smoke_only": smoke, "updates": runner.actual_updates,
        "training_seconds": runner.training_seconds, "device": "cpu",
        "torch_threads": 1, "initial_checkpoint_sha256": sha256(initial),
        "snapshot_sha256": metadata["snapshot_sha256"],
        "freeze_sha256": sha256(root / "artifacts/freeze.json"),
        "training_timing": "step computation after batch fetch; excludes validation, testing, and plotting",
    }
    if not smoke:
        runner.eval(str(output / "final.pt"))
        prediction = np.concatenate(runner.test_predictions, axis=0).squeeze(-1).astype(np.float64)
        actual = np.concatenate(runner.test_targets, axis=0).squeeze(-1).astype(np.float64)
        prediction = prediction * target["std"] + target["mean"]
        actual = actual * target["std"] + target["mean"]
        summary.update({
            "test_windows": len(actual),
            "test_mse": float(np.square(prediction - actual).mean()),
            "test_mae": float(np.abs(prediction - actual).mean()),
            "zero_growth_mse": float(np.square(actual).mean()),
            "zero_growth_mae": float(np.abs(actual).mean()),
        })
        expected_mse = runner.meter_pool.get_value("test/MSE") * target["std"] ** 2
        if not np.isclose(summary["test_mse"], expected_mse, rtol=1e-5, atol=1e-8):
            raise RuntimeError("Direct and BasicTS test metrics disagree.")
        rows = []
        dates = metadata["dates"]
        for row, first in enumerate(target["windows"]["test"]):
            for horizon in range(3):
                rows.append({
                    "forecast_origin": dates[first - 1], "forecast_month": dates[first + horizon],
                    "horizon": horizon + 1, "actual": actual[row, horizon],
                    "prediction": prediction[row, horizon],
                })
        pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False)
    write_json(output / "summary.json", summary)
    plot_validation(output, runner.validation_rows)
    return summary


def plot_validation(output: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    axis.plot([r["update"] for r in rows], [r["mse_growth_units"] for r in rows], marker=".")
    axis.set(xlabel="Optimizer updates", ylabel="INDPRO validation MSE (growth units squared)")
    figure.savefig(output / "validation.png", dpi=140)
    plt.close(figure)
