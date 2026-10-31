"""Focused tests for leakage, batch exposure, transformations, and architecture."""
from collections import Counter
import numpy as np
import pandas as pd
import torch

from fredmd_study.models import model_spec
from fredmd_study.preparation import transform, window_indices
from fredmd_study.protocol import rng
from fredmd_study.runner import mse, mae
from fredmd_study.schedules import allocate_batches, build_orders


def test_transformations_preserve_starter_units_and_missing_dates():
    dates = pd.date_range("1999-01-01", periods=6, freq="MS")
    values = pd.Series([100.0, 110.0, np.nan, 130.0, 140.0, 150.0], index=dates)
    growth = transform(values, 5, "INDPRO")
    assert np.isclose(growth.iloc[1], 100 * np.log(1.1))
    assert growth.iloc[2:4].isna().all()
    assert growth.index.equals(dates)
    assert np.isclose(transform(values, 2).iloc[1], 10.0)
    assert np.isclose(transform(values, 4).iloc[0], np.log(100.0))
    assert transform(pd.Series([-1.0, 0.0, 1.0]), 4).isna().sum() == 2


def test_forecast_boundaries_and_missing_values():
    dates = pd.date_range("1959-01-01", "2020-12-01", freq="MS")
    values = np.ones(len(dates))
    values[100] = np.nan
    indices = window_indices(values, dates)
    for first in indices["train"]:
        assert dates[first].year >= 1960
        assert dates[first + 2].year <= 1999
    for first in indices["val"]:
        assert dates[first].year >= 2000
        assert dates[first + 2].year <= 2009
    for collection in indices.values():
        assert all(not (first - 24 <= 100 <= first + 2) for first in collection)
    # Validation may use history from 1999, while all labels remain in 2000.
    january_2000 = dates.get_loc("2000-01-01")
    assert january_2000 in indices["val"]


def test_all_groups_receive_equal_auxiliary_budgets():
    for count in (15, 31, 10, 22):
        names = [f"series_{number}" for number in range(count)]
        allocation = allocate_batches(names, group=count)
        assert sum(allocation.values()) == 500
        assert max(allocation.values()) - min(allocation.values()) <= 1
        assert set(allocation) == set(names)


def test_ordering_changes_only_sequence():
    target = [f"T{i}" for i in range(300)]
    auxiliary = [f"A{i}" for i in range(500)]
    orders = build_orders(target, auxiliary, group=1)
    for order in orders.values():
        assert Counter(order) == Counter(target + auxiliary)
    assert orders["auxiliary_first"][-300:] == target
    assert orders["target_first"][:300] == target
    assert orders["interleaved"][:8] == ["T0", "A0", "A1", "T1", "A2", "A3", "T2", "A4"]


def test_named_random_streams_are_stable_and_independent():
    a = rng("window_sampling", series="INDPRO").integers(0, 10000, 100)
    b = rng("window_sampling", series="INDPRO").integers(0, 10000, 100)
    c = rng("random_order", series="INDPRO").integers(0, 10000, 100)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_metrics_include_zero_targets():
    actual = torch.tensor([0.0, 2.0])
    prediction = torch.tensor([2.0, 2.0])
    assert mse(prediction, actual).item() == 2.0
    assert mae(prediction, actual).item() == 1.0


def test_model_shapes_and_patchtst_normalization():
    torch.set_num_threads(1)
    for name in ("linear", "mlp", "patchtst"):
        cls, config = model_spec(name)
        model = cls(config)
        output = model(torch.randn(4, 24, 1))
        assert output.shape == (4, 3, 1)
        assert torch.isfinite(output).all()
        assert all(module.p == 0.0 for module in model.modules() if isinstance(module, torch.nn.Dropout))
        if name == "patchtst":
            assert not model.use_revin and not model.decomp
            assert any(isinstance(module, torch.nn.LayerNorm) for module in model.modules())
            assert not any(isinstance(module, torch.nn.BatchNorm1d) for module in model.modules())
