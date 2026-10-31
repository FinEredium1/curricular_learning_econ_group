"""Prepare chronological, separately scaled univariate forecasting examples."""
from __future__ import annotations

from io import StringIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .protocol import (
    EXPECTED_COUNTS, GROUP_NAMES, INPUT_LEN, OUTPUT_LEN, SERIES_ALIASES,
    SPLITS, TARGET, read_json, sha256, write_json,
)


def transform(values: pd.Series, code: int, series: str = "") -> pd.Series:
    """Official transformations, preserving 100*log-difference for INDPRO/PAYEMS."""
    log_values = np.log(values.where(values > 0))
    transforms = {
        1: lambda: values.copy(),
        2: lambda: values.diff(),
        3: lambda: values.diff().diff(),
        4: lambda: log_values,
        5: lambda: log_values.diff() * (100.0 if series in {TARGET, "PAYEMS"} else 1.0),
        6: lambda: log_values.diff().diff(),
        7: lambda: values.pct_change(fill_method=None).diff(),
    }
    if code not in transforms:
        raise ValueError(f"Unsupported transformation code {code} for {series}.")
    return transforms[code]().replace([np.inf, -np.inf], np.nan)


def window_indices(values: np.ndarray, dates: pd.DatetimeIndex) -> dict[str, list[int]]:
    """Index by first forecast month; retain the calendar and reject incomplete windows."""
    result = {split: [] for split in SPLITS}
    for first in range(INPUT_LEN, len(values) - OUTPUT_LEN + 1):
        if not np.isfinite(values[first - INPUT_LEN:first + OUTPUT_LEN]).all():
            continue
        first_year = dates[first].year
        last_year = dates[first + OUTPUT_LEN - 1].year
        for split, (start, end) in SPLITS.items():
            if start <= first_year and last_year <= end:
                result[split].append(first)
    return result


def load_sources(root: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Validate frozen inputs and read the official appendix without changing raw files."""
    raw = root / "data" / "raw"
    records = read_json(raw / "sources.json")
    for name, record in records.items():
        if sha256(raw / name) != record["sha256"]:
            raise ValueError(f"Raw source checksum mismatch: {name}")
    frame = pd.read_csv(raw / "fredmd_2026-09.csv")
    codes = frame.iloc[0].drop("sasdate").astype(int)
    frame = frame.iloc[1:].copy()
    dates = pd.to_datetime(frame.pop("sasdate"), format="mixed")
    frame.index = pd.DatetimeIndex(dates).to_period("M").to_timestamp()
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Snapshot dates must be unique and chronologically sorted.")
    calendar = pd.date_range(frame.index[0], frame.index[-1], freq="MS")
    if not frame.index.equals(calendar):
        raise ValueError("The raw snapshot has missing calendar rows; resolve before preparing.")
    frame = frame.apply(pd.to_numeric, errors="raise")
    with zipfile.ZipFile(raw / "fredmd_appendix0324.zip") as archive:
        content = archive.read("FRED-MD Appendix/FRED-MD_updated_appendix.csv")
    appendix = pd.read_csv(StringIO(content.decode("cp1252")))
    appendix["csv_series"] = appendix["fred"].replace(SERIES_ALIASES)
    return frame, codes, appendix


def prepare_data(root: Path) -> dict:
    """Freeze all four groups so teammates can share identical preprocessing and INDPRO."""
    destination = root / "data" / "prepared"
    if (destination / "metadata.json").exists():
        raise FileExistsError("Prepared data already exist. Run verify; do not silently overwrite.")
    frame, codes, appendix = load_sources(root)
    groups = {}
    for group in GROUP_NAMES:
        rows = appendix[appendix["group"] == group]
        names = sorted(name for name in rows["csv_series"] if name != TARGET)
        if len(names) != EXPECTED_COUNTS[group]:
            raise ValueError(f"Unexpected group-{group} membership: {len(names)}")
        missing = set(names) - set(frame.columns)
        if missing:
            raise ValueError(f"Group {group} missing from dated CSV: {sorted(missing)}")
        groups[str(group)] = names
    names = sorted({TARGET, *(s for members in groups.values() for s in members)})
    arrays = {}
    series_metadata = {}
    transformed = {}
    for name in names:
        code = int(codes[name])
        values = transform(frame[name], code, name)
        fit = values.loc["1960-01-01":"1999-12-01"].dropna()
        mean, std = float(fit.mean()), float(fit.std(ddof=0))
        if len(fit) < 2 or not np.isfinite(std) or std <= 0:
            raise ValueError(f"{name}: unusable training scaling statistics; resolve group membership.")
        windows = window_indices(values.to_numpy(), frame.index)
        if not windows["train"]:
            raise ValueError(f"{name}: no valid training windows; resolve before proceeding.")
        arrays[name] = ((values - mean) / std).to_numpy(dtype=np.float64)
        transformed[name] = values
        series_metadata[name] = {
            "code": code, "mean": mean, "std": std, "ddof": 0,
            "training_observations": len(fit), "windows": windows,
            "missing_transformed": int(values.isna().sum()),
            "window_counts": {split: len(indices) for split, indices in windows.items()},
            "log_difference_multiplier": 100 if code == 5 and name in {TARGET, "PAYEMS"} else 1,
        }
    metadata = {
        "snapshot_sha256": sha256(root / "data/raw/fredmd_2026-09.csv"),
        "groups": groups, "series": series_metadata,
        "dates": [date.strftime("%Y-%m-%d") for date in frame.index],
        "scaling_period": ["1960-01-01", "1999-12-01"], "ddof": 0,
        "input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
        "aliases": SERIES_ALIASES, "excluded_series": [],
    }
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "standardized.npz", **arrays)
    write_json(destination / "metadata.json", metadata)
    rows = [{"series": name, **info["window_counts"], "code": info["code"],
             "training_mean": info["mean"], "training_std": info["std"]}
            for name, info in series_metadata.items()]
    pd.DataFrame(rows).to_csv(destination / "window_counts.csv", index=False)
    plot_examples(root, transformed)
    return metadata


def plot_examples(root: Path, transformed: dict[str, pd.Series]) -> None:
    """Show separate axes because transformations and units differ across series."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [TARGET, "PAYEMS", "UNRATE", "RPI", "IPFINAL", "CUMFNS"]
    figure, axes = plt.subplots(len(names), 1, figsize=(11, 12), layout="constrained")
    for axis, name in zip(axes, names):
        transformed[name].loc[:"2019-12-01"].plot(ax=axis, linewidth=0.7)
        axis.set_title(name)
        axis.set_ylabel("Transformed units")
    figure.savefig(root / "reports" / "transformed_examples.png", dpi=140)
    plt.close(figure)
