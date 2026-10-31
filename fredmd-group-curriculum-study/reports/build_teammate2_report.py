"""Verify and summarize Teammate 2 results without retraining or changing the protocol."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import platform
import zipfile

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch

REPORTS = ROOT / "reports/teammate2"
MODELS = ["linear", "mlp", "patchtst"]
ORDERS = ["random", "auxiliary_first", "target_first", "interleaved", "target_only_800"]
LABELS = {
    "random": "Random mixing", "auxiliary_first": "Auxiliary first",
    "target_first": "Target first", "interleaved": "Regular interleaving",
    "target_only_800": "INDPRO only (800)",
}
COLORS = dict(zip(ORDERS, ["#2878b5", "#e1812c", "#3a923a", "#9467bd", "#696969"]))


def load_json(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_and_collect():
    frozen = load_json(ROOT / "artifacts/freeze.json")
    for relative, expected in frozen["files"].items():
        assert digest(ROOT / relative) == expected, relative
    metadata = load_json(ROOT / "data/prepared/metadata.json")
    manifest = load_json(ROOT / "artifacts/batches_seed1.json")
    expected_validation = [*range(30, 800, 30), 800]
    records, runs = [], {}
    reference_actual = None
    for model in MODELS:
        initial_hashes = set()
        for order in ORDERS:
            directory = ROOT / f"results/study/group1/{order}/{model}/seed1"
            summary = load_json(directory / "summary.json")
            assert summary["updates"] == 800 and not summary["smoke_only"]
            assert summary["seed"] == 1 and summary["group"] == 1
            assert summary["snapshot_sha256"] == metadata["snapshot_sha256"]
            assert summary["freeze_sha256"] == digest(ROOT / "artifacts/freeze.json")
            initial_hashes.add(summary["initial_checkpoint_sha256"])
            curve = pd.read_csv(directory / "validation.csv")
            assert curve["update"].tolist() == expected_validation
            assert np.isfinite(curve.select_dtypes(include="number").to_numpy()).all()
            predictions = pd.read_csv(directory / "predictions.csv")
            assert len(predictions) == 118 * 3
            assert set(predictions["horizon"]) == {1, 2, 3}
            assert not predictions.duplicated(["forecast_origin", "horizon"]).any()
            assert np.isfinite(predictions[["actual", "prediction"]].to_numpy()).all()
            identity = predictions[["forecast_origin", "forecast_month", "horizon", "actual"]]
            if reference_actual is None:
                reference_actual = identity
            else:
                pd.testing.assert_frame_equal(identity, reference_actual)
            residual = predictions["prediction"] - predictions["actual"]
            assert np.isclose(np.mean(residual ** 2), summary["test_mse"], rtol=1e-12)
            assert np.isclose(np.mean(abs(residual)), summary["test_mae"], rtol=1e-12)
            assert np.isclose(np.mean(predictions["actual"] ** 2), summary["zero_growth_mse"], rtol=1e-12)
            checkpoint = torch.load(directory / "final.pt", map_location="cpu", weights_only=True)
            assert checkpoint["step"] == 800
            steps = {int(state["step"].item()) for state in checkpoint["optim_state_dict"]["state"].values()}
            assert steps == {800}, (model, order, steps)
            assert all(torch.isfinite(value).all() for value in checkpoint["model_state_dict"].values())
            records.append(summary)
            runs[(model, order)] = (directory, curve, predictions)
        assert len(initial_hashes) == 1, model
        # These schedules start with the same 300 INDPRO batches and initial weights.
        reference = runs[(model, "target_only_800")][1]
        target_first = runs[(model, "target_first")][1]
        np.testing.assert_allclose(
            reference.loc[reference["update"] <= 300, "mse_growth_units"],
            target_first.loc[target_first["update"] <= 300, "mse_growth_units"],
            rtol=0, atol=1e-12,
        )
    assert len(records) == 15
    for order in ORDERS[:-1]:
        assert Counter(manifest["groups"]["1"]["orders"][order]) == Counter(
            manifest["target_300"] + manifest["groups"]["1"]["auxiliary"]
        )
    table = pd.DataFrame(records)
    reference_mse = table[table["order"] == "target_only_800"].set_index("model")["test_mse"]
    table["mse_improvement_vs_target800_pct"] = [
        100 * (1 - row.test_mse / reference_mse[row.model]) for row in table.itertuples()
    ]
    return metadata, manifest, table, runs


def plot_curves(runs):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for axis, model in zip(axes, MODELS):
        for order in ORDERS:
            curve = runs[(model, order)][1]
            axis.plot(curve["update"], curve["mse_growth_units"],
                      color=COLORS[order], label=LABELS[order], linewidth=1.4)
        axis.set(title=model.upper() if model == "mlp" else model.capitalize(),
                 xlabel="Optimizer updates", ylabel="INDPRO validation MSE")
        axis.grid(alpha=0.2)
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.savefig(REPORTS / "validation_comparison.png", dpi=170)
    plt.close(figure)


def plot_test_errors(table):
    zero_mse = table.iloc[0]["zero_growth_mse"]
    zero_mae = table.iloc[0]["zero_growth_mae"]
    figure, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for column, model in enumerate(MODELS):
        subset = table[table["model"] == model].set_index("order")
        for row, (metric, zero) in enumerate([("test_mse", zero_mse), ("test_mae", zero_mae)]):
            axis = axes[row, column]
            axis.bar(range(5), [subset.loc[o, metric] for o in ORDERS], color=[COLORS[o] for o in ORDERS])
            axis.axhline(zero, color="#111111", linestyle="--", linewidth=1.2, label="Zero growth")
            axis.set_xticks(range(5), ["Random", "Aux first", "Target first", "Interleave", "Target only"],
                            rotation=25, ha="right")
            axis.set_ylabel("Test MSE" if row == 0 else "Test MAE")
            if row == 0:
                axis.set_title(model.upper() if model == "mlp" else model.capitalize())
            axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Final-update INDPRO errors; dashed line = zero-growth baseline")
    figure.savefig(REPORTS / "test_errors.png", dpi=170)
    plt.close(figure)


def plot_schedules(manifest):
    schedules = manifest["groups"]["1"]["orders"]
    target_ids = set(manifest["target_300"])
    rows = np.array([[int(identifier in target_ids) for identifier in schedules[o]] for o in ORDERS[:-1]])
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), layout="constrained",
                              gridspec_kw={"height_ratios": [1, 1.5]})
    axes[0].imshow(rows, aspect="auto", interpolation="nearest",
                   cmap=ListedColormap(["#d8e6f3", "#214c75"]), extent=(0, 800, 3.5, -0.5))
    axes[0].set_yticks(range(4), [LABELS[o] for o in ORDERS[:-1]])
    axes[0].set(title="Batch source: dark blue = INDPRO; light blue = auxiliary", xlabel="Update")
    for row, order in zip(rows, ORDERS[:-1]):
        axes[1].plot(np.arange(801), np.r_[0, np.cumsum(row)],
                     label=LABELS[order], color=COLORS[order])
    axes[1].set(xlabel="Optimizer updates", ylabel="Cumulative INDPRO batches")
    axes[1].grid(alpha=0.2)
    axes[1].legend(ncol=2, frameon=False)
    figure.savefig(REPORTS / "training_schedules.png", dpi=170)
    plt.close(figure)


def plot_forecasts(metadata, runs):
    indices = metadata["series"]["INDPRO"]["windows"]["test"]
    # This evenly-spaced rule was specified in PLAN.md before any research runs.
    chosen_positions = np.linspace(0, len(indices) - 1, 4, dtype=int)
    chosen = [indices[int(position)] for position in chosen_positions]
    selection = {"rule": "Four evenly spaced valid test windows, as prespecified in PLAN.md",
                 "positions": chosen_positions.tolist(),
                 "forecast_origins": [metadata["dates"][first - 1] for first in chosen]}
    (REPORTS / "forecast_examples.json").write_text(json.dumps(selection, indent=2) + "\n")
    arrays = np.load(ROOT / "data/prepared/standardized.npz", allow_pickle=False)
    info = metadata["series"]["INDPRO"]
    values = arrays["INDPRO"] * info["std"] + info["mean"]
    dates = pd.to_datetime(metadata["dates"])
    for model in MODELS:
        figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        for axis, first in zip(axes.flat, chosen):
            axis.plot(dates[first - 6:first + 3], values[first - 6:first + 3],
                      color="black", marker=".", label="Observed", linewidth=1.5)
            axis.axvline(dates[first] - pd.Timedelta(days=15), color="#bbbbbb", linestyle=":")
            axis.axhline(0, color="#444444", linestyle="--", label="Zero growth", linewidth=0.8)
            origin = metadata["dates"][first - 1]
            for order in ORDERS:
                predictions = runs[(model, order)][2]
                sample = predictions[predictions["forecast_origin"] == origin]
                assert len(sample) == 3
                axis.plot(pd.to_datetime(sample["forecast_month"]), sample["prediction"],
                          color=COLORS[order], marker="o", markersize=3,
                          linewidth=1.2, label=LABELS[order])
            axis.set(title=f"Origin: {origin}", ylabel="Monthly growth (%)")
            axis.tick_params(axis="x", rotation=25)
            axis.grid(alpha=0.15)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
        figure.suptitle(f"{model}: same four test windows; display shows final 6 of 24 input months")
        figure.savefig(REPORTS / f"forecasts_{model}.png", dpi=160)
        plt.close(figure)


def write_report(metadata, table):
    zero_mse = table.iloc[0]["zero_growth_mse"]
    zero_mae = table.iloc[0]["zero_growth_mae"]
    lines = [
        "# Teammate 2: Output-and-Income Training for INDPRO Forecasting",
        "",
        "## Question and scope",
        "",
        "Does supplementing industrial-production histories with the complete output-and-income group improve INDPRO forecasting, and does training order matter? This is the agreed brief, single-seed study. Results are preliminary.",
        "",
        "## Shared method",
        "",
        "- September 2026 FRED-MD snapshot; official group 1 excluding INDPRO provides 15 auxiliary series.",
        "- Global univariate forecasting: 24 monthly inputs and three outputs. Auxiliary examples predict their own series.",
        "- Forecast months: training 1960–1999, validation 2000–2009, testing 2010–2019. Standardization uses transformed training observations only, with ddof=0.",
        "- INDPRO has 465 valid training windows, 118 validation windows, and 118 test windows.",
        "- Each augmented run uses the same 300 INDPRO batches plus the same 500 auxiliary batches, batch size 32. Orders change only batch sequence.",
        "- Linear, MLP, and PatchTST use the exact agreed architectures. All runs use Adam at 0.001, no weight decay/dropout/early stopping, and seed 1.",
        "- CPU, float32, one PyTorch thread; 800 optimizer updates with final-checkpoint evaluation.",
        "",
        "## Final test results",
        "",
        f"The zero-growth baseline has MSE **{zero_mse:.6f}** and MAE **{zero_mae:.6f}**. MSE is in squared monthly-growth units; MAE is in monthly-growth units.",
        "",
        "| Model | Random mixing | Auxiliary first | Target first | Interleaved | INDPRO only, 800 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        values = table[table["model"] == model].set_index("order")
        lines.append("| " + model + " | " + " | ".join(f"{values.loc[o, 'test_mse']:.6f}" for o in ORDERS) + " |")
    lines += ["", "The table reports test MSE. The accompanying results.csv contains MAE, compute times, provenance, and comparisons to the 800-update control.", "",
              "## Findings", ""]
    for model in MODELS:
        augmented = table[(table["model"] == model) & (table["order"] != "target_only_800")]
        best = augmented.loc[augmented["test_mse"].idxmin()]
        random = augmented[augmented["order"] == "random"].iloc[0]
        lines.append(
            f"- **{model}:** lowest augmented test MSE occurs with {LABELS[best['order']].lower()} "
            f"({best['test_mse']:.6f}). Random mixing improves MSE over the 800-update target-only control "
            f"by {random['mse_improvement_vs_target800_pct']:.2f}%."
        )
    lines += [
        "",
        "All twelve augmented runs have lower MSE than their model's 800-update target-only control in this seed. However, every Linear and PatchTST run remains above the zero-growth MSE baseline. MLP random mixing, target-first, and interleaving beat zero growth on MSE; every trained run has higher MAE than zero growth.",
        "",
        "Auxiliary-first has higher MSE than random mixing for all three models. The lowest-MSE order differs by model, so the results do not support a universal advantage for auxiliary-first training.",
        "",
        "These are final-test descriptions of the prespecified conditions, not grounds for changing the protocol or selecting hyperparameters against the test set.",
        "",
        "## Verification",
        "",
        "All 15 runs completed exactly 800 optimizer steps. Checkpoint optimizer counters independently equal 800. Each run includes all scheduled validations (30, 60, …, 780, 800), and saved prediction errors reproduce the reported MSE/MAE. All runs share the same snapshot, frozen setup, per-model initial weights, and test targets. Target-first and the target-only control have identical validation errors through their common first 300 target batches.",
        "",
        "Seven setup checks and three short BasicTS smoke runs were completed before research training. The frozen experimental code and artifacts were not changed after the teammate setup package was shared.",
        "",
        "## Limits and pending comparisons",
        "",
        "- Teammate 1's three 300-update controls are still needed for the fixed-target-exposure comparison. They have not been simulated or substituted.",
        "- Other teammates' results are needed to compare different economic groups.",
        "- One seed does not support variance estimates, statistical-significance claims, or a general ranking of schedules.",
        "- The 800-update comparison holds total updates fixed; the target-only control receives more INDPRO batches.",
        "- Revised historical observations and these retrospective splits do not establish real-time forecasting performance.",
        "- Finishing on INDPRO, optimizer history, and group composition may all affect scheduling outcomes. The study does not identify a unique mechanism.",
        "- Compute times exclude batch fetching, preparation, validation, testing, and plotting. Compare timings only on equivalent hardware.",
        "",
        "## Files",
        "",
        "- results.csv: all 15 results and relative MSE changes.",
        "- validation_comparison.png: validation curves for all conditions.",
        "- test_errors.png: MSE and MAE with zero-growth references.",
        "- training_schedules.png: batch sources and cumulative target exposure.",
        "- forecasts_linear.png, forecasts_mlp.png, forecasts_patchtst.png: identical four prespecified test examples.",
        "- verification.json: checks and completion status.",
        "",
        f"Snapshot SHA-256: {metadata['snapshot_sha256']}",
        f"Frozen setup SHA-256: {digest(ROOT / 'artifacts/freeze.json')}",
    ]
    (REPORTS / "REPORT.md").write_text("\n".join(lines) + "\n")


def package_results():
    distribution = ROOT / "distribution"
    distribution.mkdir(exist_ok=True)
    for filename, orders in [
        ("teammate2-output-income-seed1-results.zip", ORDERS),
        ("shared-indpro-800-controls-seed1.zip", ["target_only_800"]),
    ]:
        destination = distribution / filename
        if destination.exists():
            raise FileExistsError(destination)
        paths = [ROOT / "artifacts/freeze.json", ROOT / "PLAN.md"]
        for order in orders:
            for model in MODELS:
                directory = ROOT / f"results/study/group1/{order}/{model}/seed1"
                paths.extend(path for path in directory.rglob("*") if path.is_file())
        if len(orders) > 1:
            paths.extend(path for path in REPORTS.rglob("*") if path.is_file())
            paths.append(Path(__file__))
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(set(paths)):
                archive.write(path, str(path.relative_to(ROOT)))
        with zipfile.ZipFile(destination) as archive:
            assert archive.testzip() is None
        destination.with_suffix(".zip.sha256").write_text(f"{digest(destination)}  {filename}\n")


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    metadata, manifest, table, runs = verify_and_collect()
    table.to_csv(REPORTS / "results.csv", index=False)
    plot_curves(runs)
    plot_test_errors(table)
    plot_schedules(manifest)
    plot_forecasts(metadata, runs)
    write_report(metadata, table)
    status = {
        "research_runs_completed": 15, "actual_updates_per_run": 800,
        "validation_updates": [*range(30, 800, 30), 800],
        "all_prediction_metrics_reproduced": True,
        "all_optimizer_step_counters_verified": True,
        "shared_initial_weights_verified": True, "shared_test_targets_verified": True,
        "common_first_300_target_validation_matches": True,
        "frozen_setup_unchanged": True,
        "pending": ["Teammate 1 target_only_300 results", "other economic groups"],
        "platform": platform.platform(), "python": platform.python_version(),
    }
    (REPORTS / "verification.json").write_text(json.dumps(status, indent=2) + "\n")
    readme = ROOT / "README.md"
    content = readme.read_text().replace(
        "Full research runs have not started. See reports/SETUP_STATUS.md.",
        "Your 15 assigned research runs are complete and verified. See reports/teammate2/REPORT.md. Initial setup details remain in reports/SETUP_STATUS.md.",
    )
    readme.write_text(content)
    package_results()
    print(table[["model", "order", "test_mse", "test_mae", "mse_improvement_vs_target800_pct"]].to_string(index=False))
    print("Reports and result packages created; frozen scientific setup unchanged.")


if __name__ == "__main__":
    main()
