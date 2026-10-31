"""Results table and figures for one auxiliary group's study runs (provisional team format).

Run from the research folder after the study runs finish:

    PYTHONPATH=src .venv/bin/python -m fredmd_study.group_report --group 2

Every reported metric is recomputed from saved predictions and checked against summary.json.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from .protocol import GROUP_NAMES, MODELS, ORDERS, OUTPUT_LEN, TARGET, read_json

CONDITIONS = ("target_only_300", *ORDERS, "target_only_800")
BUDGETS = {"target_only_300": (300, 0), "target_only_800": (800, 0), **{order: (300, 500) for order in ORDERS}}
LABELS = {
    "target_only_300": "INDPRO only (300)", "random": "Random mixing",
    "auxiliary_first": "Auxiliary first", "target_first": "Target first",
    "interleaved": "Regular interleaving", "target_only_800": "INDPRO only (800)",
}
# Validated categorical slots (light surface); color follows the condition, not its rank.
COLORS = {
    "target_only_300": "#2a78d6", "random": "#eb6834", "auxiliary_first": "#1baf7a",
    "target_first": "#eda100", "interleaved": "#e87ba4", "target_only_800": "#008300",
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
EXAMPLE_WINDOWS = 4


def find_run(root: Path, group: int, condition: str, model: str) -> Path | None:
    """Augmented runs live under their group; a shared control may live under its owner's group."""
    study = root / "results" / "study"
    if condition in ORDERS:
        candidates = [study / f"group{group}" / condition / model / "seed1"]
    else:
        candidates = sorted(study.glob(f"group*/{condition}/{model}/seed1"))
    found = [path for path in candidates if (path / "summary.json").exists()]
    if len(found) > 1:
        raise ValueError(f"Several {condition}/{model} runs found {found}; keep exactly one shared control.")
    return found[0] if found else None


def checked_run(path: Path, condition: str, metadata: dict, values: np.ndarray) -> dict:
    """Confirm the exact budget, then recompute test errors directly from saved predictions."""
    summary = read_json(path / "summary.json")
    validation = pd.read_csv(path / "validation.csv")
    predictions = pd.read_csv(path / "predictions.csv")
    updates = sum(BUDGETS[condition])
    if summary["smoke_only"] or summary["updates"] != updates or validation["update"].iloc[-1] != updates:
        raise ValueError(f"{path}: expected a full {updates}-update run with final validation.")
    target = metadata["series"][TARGET]
    if len(predictions) != len(target["windows"]["test"]) * OUTPUT_LEN:
        raise ValueError(f"{path}: predictions.csv must hold every test origin and horizon.")
    if not np.isfinite(predictions[["actual", "prediction"]].to_numpy()).all():
        raise ValueError(f"{path}: nonfinite saved predictions.")
    month_index = {date: index for index, date in enumerate(metadata["dates"])}
    stored_actual = values[predictions["forecast_month"].map(month_index).to_numpy()] * target["std"] + target["mean"]
    if not np.allclose(predictions["actual"], stored_actual, rtol=0, atol=1e-5):
        raise ValueError(f"{path}: saved actual values do not match prepared INDPRO growth.")
    error = predictions["prediction"] - predictions["actual"]
    direct = {
        "test_mse": float(np.square(error).mean()), "test_mae": float(np.abs(error).mean()),
        "zero_growth_mse": float(np.square(predictions["actual"]).mean()),
        "zero_growth_mae": float(np.abs(predictions["actual"]).mean()),
    }
    for key, value in direct.items():
        if not np.isclose(value, summary[key], rtol=1e-9, atol=1e-12):
            raise ValueError(f"{path}: {key} in summary.json disagrees with predictions.csv.")
    target_batches, auxiliary_batches = BUDGETS[condition]
    return {
        "condition": condition, "indpro_batches": target_batches, "auxiliary_batches": auxiliary_batches,
        "updates": updates, **direct,
        "final_validation_mse": float(validation["mse_growth_units"].iloc[-1]),
        "training_seconds": summary["training_seconds"],
    }


def collect_results(root: Path, group: int, metadata: dict) -> tuple[pd.DataFrame, dict, list[str]]:
    """Gather every completed condition; only shared controls owned elsewhere may be pending."""
    values = np.load(root / "data/prepared/standardized.npz", allow_pickle=False)[TARGET]
    rows, run_paths, pending = [], {}, []
    for model in MODELS:
        for condition in CONDITIONS:
            path = find_run(root, group, condition, model)
            if path is None:
                if condition in ORDERS:
                    raise FileNotFoundError(f"Missing group {group} run: {condition}/{model}. Run study.py first.")
                pending.append(f"{condition}/{model}")
                continue
            rows.append({"model": model, **checked_run(path, condition, metadata, values),
                         "run_directory": str(path.relative_to(root))})
            run_paths[(model, condition)] = path
    table = pd.DataFrame(rows)
    zero = table[["zero_growth_mse", "zero_growth_mae"]].drop_duplicates()
    if len(zero) != 1:
        raise ValueError("Zero-growth errors differ across runs; test targets are not identical.")
    baseline = table[table["condition"] == "target_only_300"].set_index("model")
    for metric in ("mse", "mae"):
        reference = table["model"].map(baseline[f"test_{metric}"])
        table[f"{metric}_change_vs_300_pct"] = 100.0 * (table[f"test_{metric}"] / reference - 1.0)
    return table, run_paths, pending


def example_window_indices(metadata: dict) -> list[int]:
    """Four evenly spaced positions in the valid INDPRO test-window list, fixed before errors."""
    windows = metadata["series"][TARGET]["windows"]["test"]
    positions = np.linspace(0, len(windows) - 1, EXAMPLE_WINDOWS).round().astype(int)
    return [windows[position] for position in positions]


def hardware_description() -> str:
    try:
        processor = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        processor = platform.processor() or platform.machine()
    return f"{processor}; {platform.platform()}; Python {platform.python_version()}"


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(frame[c]) else "---" for c in frame.columns) + "|"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *body])


def write_markdown(path: Path, group: int, table: pd.DataFrame, pending: list[str],
                   metadata: dict, manifest: dict, windows: list[int]) -> None:
    name = GROUP_NAMES[group]
    zero = table.iloc[0]
    lines = [
        f"# {name.replace('_', ' ').title()} group (group {group}): results table",
        "",
        "Generated by `PYTHONPATH=src .venv/bin/python -m fredmd_study.group_report "
        f"--group {group}`. Provisional format until the shared team table is agreed.",
        "",
        "INDPRO test errors (January 2010-December 2019 forecast months; 118 origins x 3 horizons)"
        " in monthly growth units (100 x log difference), final weights, seed 1.",
        "Change columns are relative to the 300-update INDPRO-only control of the same model.",
        "",
    ]
    for model in MODELS:
        rows = table[table["model"] == model]
        view = pd.DataFrame({
            "Condition": rows["condition"].map(LABELS),
            "INDPRO / aux batches": [f"{t} / {a}" for t, a in zip(rows["indpro_batches"], rows["auxiliary_batches"])],
            "Test MSE": rows["test_mse"].map("{:.4f}".format),
            "Test MAE": rows["test_mae"].map("{:.4f}".format),
            "MSE vs 300 (%)": rows["mse_change_vs_300_pct"].map("{:+.1f}".format),
            "MAE vs 300 (%)": rows["mae_change_vs_300_pct"].map("{:+.1f}".format),
            "Final val MSE": rows["final_validation_mse"].map("{:.4f}".format),
            "Train s": rows["training_seconds"].map("{:.2f}".format),
        })
        zero_row = {"Condition": "Always zero growth", "INDPRO / aux batches": "-",
                    "Test MSE": f"{zero['zero_growth_mse']:.4f}", "Test MAE": f"{zero['zero_growth_mae']:.4f}",
                    "MSE vs 300 (%)": "-", "MAE vs 300 (%)": "-", "Final val MSE": "-", "Train s": "-"}
        view = pd.concat([view, pd.DataFrame([zero_row])], ignore_index=True)
        lines += [f"## {model}", "", markdown_table(view), ""]
        lines += [f"- Pending: {LABELS[item.split('/')[0]]} (shared control not yet available)"
                  for item in pending if item.endswith(f"/{model}")]
        lines.append("")
    target = metadata["series"][TARGET]
    allocation = manifest["groups"][str(group)]["allocation"]
    counts = pd.DataFrame([
        {"Series": series, "FRED-MD code": metadata["series"][series]["code"],
         "Train windows": metadata["series"][series]["window_counts"]["train"],
         "Auxiliary batches": allocation[series]}
        for series in metadata["groups"][str(group)]
    ])
    test_windows = target["windows"]["test"]
    positions = [test_windows.index(first) for first in windows]
    origins = [f"{metadata['dates'][first - 1][:7]} -> {metadata['dates'][first][:7]}" for first in windows]
    lines += [
        "## Run environment and window counts",
        "",
        f"- Hardware: {hardware_description()}; CPU backend, float32, one PyTorch thread.",
        "- Training seconds: optimizer-step computation after batch fetch; excludes preparation, "
        "batch fetching, validation, testing, and plotting.",
        f"- INDPRO valid windows: {target['window_counts']['train']} train, "
        f"{target['window_counts']['val']} validation, {target['window_counts']['test']} test.",
        f"- Example forecast windows (positions {', '.join(map(str, positions))} of the "
        f"{len(test_windows)} test windows), as origin -> first forecast month: {', '.join(origins)}.",
        f"- Metric check: all {len(table)} runs' summary.json errors match a direct recomputation "
        "from predictions.csv.",
        "",
        f"Auxiliary series ({len(counts)}; {sum(allocation.values())} batches of 32 windows):",
        "",
        markdown_table(counts),
        "",
    ]
    path.write_text("\n".join(lines))


def styled_figure(rows: int, columns: int, size: tuple[float, float], **options):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    })
    figure, axes = plt.subplots(rows, columns, figsize=size, layout="constrained", squeeze=False, **options)
    return plt, figure, axes


def plot_validation_curves(path: Path, run_paths: dict) -> None:
    plt, figure, axes = styled_figure(1, len(MODELS), (13, 4))
    for axis, model in zip(axes[0], MODELS):
        for condition in CONDITIONS:
            if (model, condition) not in run_paths:
                continue
            curve = pd.read_csv(run_paths[(model, condition)] / "validation.csv")
            # Target-first matches the 300-update control through update 300, so controls are
            # dashed and drawn on top to keep both visible where trajectories coincide.
            control = condition not in ORDERS
            axis.plot(curve["update"], curve["mse_growth_units"], color=COLORS[condition],
                      linewidth=1.0, marker="o", markersize=2.5, label=LABELS[condition],
                      linestyle="--" if control else "-", zorder=3 if control else 2)
        axis.set(title=model, xlabel="Optimizer updates", xlim=(0, 810))
    axes[0][0].set_ylabel("INDPRO validation MSE (growth units squared)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_batch_sources(path: Path, group: int, manifest: dict, conditions: list[str]) -> None:
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    orders = {"target_only_300": manifest["target_300"], "target_only_800": manifest["target_800"],
              **manifest["groups"][str(group)]["orders"]}
    name = GROUP_NAMES[group].replace("_", " ")
    plt, figure, axes = styled_figure(len(conditions), 2, (13, 0.55 * len(conditions) + 1.2),
                                      gridspec_kw={"width_ratios": [4, 1]})
    colormap = ListedColormap([COLORS["target_only_300"], COLORS["random"]])
    colormap.set_bad(SURFACE)
    for row, condition in zip(axes, conditions):
        sources = np.full(800, np.nan)
        sources[:len(orders[condition])] = [manifest["batches"][identifier]["series"] != TARGET
                                            for identifier in orders[condition]]
        for axis, limit in zip(row, (800, 40)):
            axis.imshow(sources[None, :limit], aspect="auto", cmap=colormap, vmin=0, vmax=1,
                        interpolation="nearest", extent=(0.5, limit + 0.5, 0, 1))
            axis.set(yticks=[], xlim=(0.5, limit + 0.5))
            axis.grid(False)
        row[0].set_ylabel(LABELS[condition], rotation=0, ha="right", va="center")
    axes[0][0].set_title("Batch source by optimizer update")
    axes[0][1].set_title("First 40 updates")
    axes[-1][0].set_xlabel("Optimizer update")
    axes[-1][1].set_xlabel("Optimizer update")
    figure.legend(handles=[Patch(color=COLORS["target_only_300"], label="INDPRO batch"),
                           Patch(color=COLORS["random"], label=f"{name.capitalize()} auxiliary batch")],
                  loc="outside lower center", ncol=2, frameon=False)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_forecast_windows(path: Path, root: Path, run_paths: dict, metadata: dict, windows: list[int]) -> None:
    target = metadata["series"][TARGET]
    growth = np.load(root / "data/prepared/standardized.npz")[TARGET] * target["std"] + target["mean"]
    dates = pd.to_datetime(metadata["dates"])
    predictions = {key: pd.read_csv(run / "predictions.csv", parse_dates=["forecast_origin", "forecast_month"])
                   for key, run in run_paths.items()}
    import matplotlib.dates as mdates
    # Share y within a window so models compare directly; the 2008 crash would flatten other windows.
    plt, figure, axes = styled_figure(len(MODELS), len(windows), (14, 8.5), sharey="col")
    for row, model in zip(axes, MODELS):
        for axis, first in zip(row, windows):
            span = slice(first - 24, first + OUTPUT_LEN)
            axis.axhline(0.0, color=MUTED, linewidth=0.8, linestyle="--", label="Always zero growth")
            axis.axvline(dates[first - 1], color=GRID, linewidth=1.0)
            axis.plot(dates[span], growth[span], color=INK, linewidth=1.0, label="Actual INDPRO growth")
            for condition in CONDITIONS:
                if (model, condition) not in predictions:
                    continue
                frame = predictions[(model, condition)]
                frame = frame[frame["forecast_origin"] == dates[first - 1]].sort_values("horizon")
                axis.plot(frame["forecast_month"], frame["prediction"], color=COLORS[condition],
                          linewidth=1.0, marker="o", markersize=3, label=LABELS[condition])
            axis.set_title(f"{model}: forecasts {dates[first]:%Y-%m} to {dates[first + OUTPUT_LEN - 1]:%Y-%m}")
            axis.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        row[0].set_ylabel("INDPRO growth (100 x log diff)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def build_report(root: Path, group: int) -> pd.DataFrame:
    metadata = read_json(root / "data/prepared/metadata.json")
    manifest = read_json(root / "artifacts/batches_seed1.json")
    windows = example_window_indices(metadata)
    table, run_paths, pending = collect_results(root, group, metadata)
    prefix = root / "reports" / GROUP_NAMES[group]
    table.to_csv(f"{prefix}_results.csv", index=False)
    write_markdown(Path(f"{prefix}_results.md"), group, table, pending, metadata, manifest, windows)
    plot_validation_curves(Path(f"{prefix}_validation_curves.png"), run_paths)
    completed = [condition for condition in CONDITIONS if condition in set(table["condition"])]
    plot_batch_sources(Path(f"{prefix}_batch_sources.png"), group, manifest, completed)
    plot_forecast_windows(Path(f"{prefix}_forecast_windows.png"), root, run_paths, metadata, windows)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", type=int, choices=list(GROUP_NAMES), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if Path.cwd().resolve() != root:
        raise RuntimeError(f"Run from the research folder: {root}")
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    table = build_report(root, args.group)
    print(f"Wrote reports for {len(table)} runs of group {args.group} ({GROUP_NAMES[args.group]}).")


if __name__ == "__main__":
    main()
