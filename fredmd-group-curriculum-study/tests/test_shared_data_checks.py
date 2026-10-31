"""Independent re-derivation of the frozen prepared data from the unchanged raw sources.

These checks deliberately avoid fredmd_study.preparation: transformations, scaling
statistics, and window indices are recomputed here and compared with the frozen files.
"""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/raw/fredmd_2026-09.csv"
APPENDIX = ROOT / "data/raw/fredmd_appendix0324.zip"
SNAPSHOT_URL = ("https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/"
                "research/fred-md/monthly/2026-09-md.csv")
SNAPSHOT_SHA256 = "d6e239038afbdbf4667c2a63c13c942817bc196fedab4355cd7e0d353ecdd65d"
TARGET = "INDPRO"
GROUPS = {1: 15, 2: 31, 3: 10, 6: 22}
SPLIT_YEARS = {"train": (1960, 1999), "val": (2000, 2009), "test": (2010, 2019)}
INPUT_LEN, OUTPUT_LEN = 24, 3
PERCENT_GROWTH = {"INDPRO", "PAYEMS"}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def raw() -> tuple[pd.DataFrame, pd.Series]:
    """Raw levels indexed by month, plus the CSV transformation-code row."""
    frame = pd.read_csv(SNAPSHOT)
    codes = frame.iloc[0, 1:].astype(float).astype(int)
    levels = frame.iloc[1:].set_index("sasdate").apply(pd.to_numeric)
    levels.index = pd.to_datetime(levels.index, format="%m/%d/%Y")
    return levels, codes


@pytest.fixture(scope="module")
def appendix() -> pd.DataFrame:
    with zipfile.ZipFile(APPENDIX) as archive:
        content = archive.read("FRED-MD Appendix/FRED-MD_updated_appendix.csv")
    table = pd.read_csv(io.BytesIO(content), encoding="cp1252")
    # The appendix spells one identifier in lower case; the dated CSV uses upper case.
    table["csv_series"] = table["fred"].replace({"IPB51222s": "IPB51222S"})
    return table


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads((ROOT / "data/prepared/metadata.json").read_text())


@pytest.fixture(scope="module")
def standardized() -> dict[str, np.ndarray]:
    return dict(np.load(ROOT / "data/prepared/standardized.npz", allow_pickle=False))


def lag(values: np.ndarray) -> np.ndarray:
    return np.concatenate([[np.nan], values[:-1]])


def independent_transform(levels: np.ndarray, code: int, series: str) -> np.ndarray:
    """FRED-MD transformation codes; non-positive log inputs stay unavailable."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(np.where(levels > 0, levels, np.nan))
        if code == 1:
            result = levels.copy()
        elif code == 2:
            result = levels - lag(levels)
        elif code == 3:
            first = levels - lag(levels)
            result = first - lag(first)
        elif code == 4:
            result = logs
        elif code == 5:
            result = (100.0 if series in PERCENT_GROWTH else 1.0) * (logs - lag(logs))
        elif code == 6:
            first = logs - lag(logs)
            result = first - lag(first)
        elif code == 7:
            growth = levels / lag(levels) - 1.0
            result = growth - lag(growth)
        else:
            raise ValueError(f"{series}: unknown transformation code {code}")
    return np.where(np.isfinite(result), result, np.nan)


def independent_windows(values: np.ndarray, years: np.ndarray) -> dict[str, list[int]]:
    """First-forecast indices whose 27 months are finite, via a cumulative finite count."""
    finite_before = np.concatenate([[0], np.cumsum(np.isfinite(values))])
    firsts = np.arange(INPUT_LEN, len(values) - OUTPUT_LEN + 1)
    complete = finite_before[firsts + OUTPUT_LEN] - finite_before[firsts - INPUT_LEN] == INPUT_LEN + OUTPUT_LEN
    return {
        split: firsts[complete & (years[firsts] >= start) & (years[firsts + OUTPUT_LEN - 1] <= end)].tolist()
        for split, (start, end) in SPLIT_YEARS.items()
    }


def test_snapshot_checksum_matches_provenance_freeze_and_metadata(metadata):
    sources = json.loads((ROOT / "data/raw/sources.json").read_text())
    freeze = json.loads((ROOT / "artifacts/freeze.json").read_text())
    assert sources["fredmd_2026-09.csv"]["url"] == SNAPSHOT_URL
    assert file_sha256(SNAPSHOT) == SNAPSHOT_SHA256
    assert sources["fredmd_2026-09.csv"]["sha256"] == SNAPSHOT_SHA256
    assert freeze["files"]["data/raw/fredmd_2026-09.csv"] == SNAPSHOT_SHA256
    assert metadata["snapshot_sha256"] == SNAPSHOT_SHA256
    assert sources["fredmd_appendix0324.zip"]["sha256"] == file_sha256(APPENDIX)


def test_calendar_is_contiguous_and_matches_metadata(raw, metadata):
    levels, _ = raw
    assert levels.index.equals(pd.date_range("1959-01-01", "2026-08-01", freq="MS"))
    assert metadata["dates"] == [date.strftime("%Y-%m-%d") for date in levels.index]


def test_group_membership_matches_official_appendix(raw, appendix, metadata):
    levels, _ = raw
    for group, expected_count in GROUPS.items():
        official = sorted(set(appendix.loc[appendix["group"] == group, "csv_series"]) - {TARGET})
        assert len(official) == expected_count, group
        assert metadata["groups"][str(group)] == official, group
        assert set(official) <= set(levels.columns), group
    assert {"PAYEMS", "UNRATE"} <= set(metadata["groups"]["2"])
    assert metadata["excluded_series"] == []
    assert set(metadata["series"]) == {TARGET, *(s for g in metadata["groups"].values() for s in g)}


def test_transformation_codes_agree_across_csv_appendix_and_metadata(raw, appendix, metadata):
    _, codes = raw
    official = appendix.set_index("csv_series")["tcode"]
    for name, info in metadata["series"].items():
        assert info["code"] == codes[name] == official[name], name
    assert codes[TARGET] == 5 and codes["PAYEMS"] == 5 and codes["UNRATE"] == 2


def test_transformed_values_reproduce_prepared_arrays(raw, metadata, standardized):
    levels, codes = raw
    for name, info in metadata["series"].items():
        expected = independent_transform(levels[name].to_numpy(dtype=float), int(codes[name]), name)
        restored = standardized[name] * info["std"] + info["mean"]
        assert np.array_equal(np.isnan(expected), np.isnan(restored)), name
        mask = np.isfinite(expected)
        assert np.allclose(restored[mask], expected[mask], rtol=1e-9, atol=1e-12), name


def test_starter_units_for_indpro_payems_and_unrate(raw, metadata, standardized):
    levels, _ = raw
    for name in PERCENT_GROWTH:
        info, level = metadata["series"][name], levels[name].to_numpy(dtype=float)
        growth = standardized[name][1:] * info["std"] + info["mean"]
        assert np.allclose(growth, 100.0 * np.diff(np.log(level)), rtol=1e-9), name
    info, rate = metadata["series"]["UNRATE"], levels["UNRATE"].to_numpy(dtype=float)
    change = standardized["UNRATE"][1:] * info["std"] + info["mean"]
    finite = np.isfinite(np.diff(rate))
    assert np.allclose(change[finite], np.diff(rate)[finite], atol=1e-12)


def test_scaling_uses_finite_1960_1999_observations_with_ddof0(raw, metadata):
    levels, codes = raw
    in_training = (levels.index.year >= 1960) & (levels.index.year <= 1999)
    for name, info in metadata["series"].items():
        values = independent_transform(levels[name].to_numpy(dtype=float), int(codes[name]), name)
        fit = values[in_training & np.isfinite(values)]
        assert info["ddof"] == 0 and info["training_observations"] == len(fit), name
        assert np.isclose(info["mean"], fit.mean(), rtol=1e-12, atol=1e-15), name
        assert np.isclose(info["std"], fit.std(ddof=0), rtol=1e-12), name
        assert info["std"] > 0, name


def test_window_indices_match_independent_derivation(raw, metadata):
    levels, codes = raw
    years = levels.index.year.to_numpy()
    for name, info in metadata["series"].items():
        # Each series is windowed alone, so auxiliary missingness cannot move INDPRO windows.
        values = independent_transform(levels[name].to_numpy(dtype=float), int(codes[name]), name)
        assert info["windows"] == independent_windows(values, years), name
        assert info["windows"]["train"], name
    assert metadata["series"][TARGET]["window_counts"] == {"train": 465, "val": 118, "test": 118}
