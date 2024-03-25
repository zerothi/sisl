# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from sisl.viz.processors.xarray import group_reduce

pytestmark = [pytest.mark.viz, pytest.mark.processors]


@pytest.fixture(scope="module")
def dataarray():
    return xr.DataArray(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ],
        coords=[("x", [0, 1, 2, 3]), ("y", [0, 1, 2])],
        name="vals",
    )


@pytest.fixture(scope="module")
def dataset():
    arr = xr.DataArray(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
        ],
        coords=[("x", [0, 1, 2, 3]), ("y", [0, 1, 2])],
        name="vals",
    )

    arr2 = arr * 2
    return xr.Dataset({"vals": arr, "double": arr2})


def test_dataarray(dataarray):
    new = group_reduce(
        dataarray,
        [{"selector": [0, 1]}, {"selector": [2, 3]}],
        reduce_dim="x",
        reduce_func=np.sum,
        groups_dim="selection",
    )

    assert isinstance(new, xr.DataArray)
    assert "x" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == [0, 1]
    assert new.sel(selection=0).sum() == 1 + 2 + 3 + 4 + 5 + 6
    assert new.sel(selection=1).sum() == 7 + 8 + 9 + 10 + 11 + 12


def test_dataarray_multidim(dataarray):
    new = group_reduce(
        dataarray,
        [{"selector": ([0, 1], [0, 1])}, {"selector": ([2, 3], [0, 1])}],
        reduce_dim=("x", "y"),
        reduce_func=np.sum,
        groups_dim="selection",
    )

    assert isinstance(new, xr.DataArray)
    assert "x" not in new.dims
    assert "y" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == [0, 1]
    assert new.sel(selection=0).sum() == 1 + 2 + 4 + 5
    assert new.sel(selection=1).sum() == 7 + 8 + 10 + 11


def test_dataarray_multidim_multireduce(dataarray):
    new = group_reduce(
        dataarray,
        [{"selector": ([0, 1], [0, 1])}, {"selector": ([2, 3], [0, 1])}],
        reduce_dim=("x", "y"),
        reduce_func=(np.sum, np.mean),
        groups_dim="selection",
    )

    assert isinstance(new, xr.DataArray)
    assert "x" not in new.dims
    assert "y" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == [0, 1]
    assert new.sel(selection=0).sum() == (1 + 4) / 2 + (2 + 5) / 2
    assert new.sel(selection=1).sum() == (7 + 10) / 2 + (8 + 11) / 2


def test_dataarray_sangroup(dataarray):
    # We use sanitize group to simply set all selectors to [0,1]
    new = group_reduce(
        dataarray,
        [{"selector": [0, 1]}, {"selector": [2, 3]}],
        reduce_dim="x",
        reduce_func=np.sum,
        groups_dim="selection",
        sanitize_group=lambda group: {**group, "selector": [0, 1]},
    )

    assert isinstance(new, xr.DataArray)
    assert "x" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == [0, 1]
    assert new.sel(selection=0).sum() == 1 + 2 + 3 + 4 + 5 + 6
    assert new.sel(selection=1).sum() == 1 + 2 + 3 + 4 + 5 + 6


def test_dataarray_names(dataarray):
    new = group_reduce(
        dataarray,
        [{"selector": [0, 1], "name": "first"}, {"selector": [2, 3], "name": "second"}],
        reduce_dim="x",
        reduce_func=np.sum,
        groups_dim="selection",
    )

    assert isinstance(new, xr.DataArray)
    assert "x" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == ["first", "second"]
    assert new.sel(selection="first").sum() == 1 + 2 + 3 + 4 + 5 + 6
    assert new.sel(selection="second").sum() == 7 + 8 + 9 + 10 + 11 + 12


def test_dataarray_groupvars(dataarray):
    new = group_reduce(
        dataarray,
        [
            {"selector": [0, 1], "name": "first", "color": "red", "size": 3},
            {"selector": [2, 3], "name": "second", "color": "blue", "size": 4},
        ],
        reduce_dim="x",
        reduce_func=np.sum,
        groups_dim="selection",
        group_vars=["color", "size"],
    )

    assert isinstance(new, xr.Dataset)
    assert "x" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == ["first", "second"]

    assert "vals" in new
    assert new.vals.sel(selection="first").sum() == 1 + 2 + 3 + 4 + 5 + 6
    assert new.vals.sel(selection="second").sum() == 7 + 8 + 9 + 10 + 11 + 12

    for k, vals in {"color": ["red", "blue"], "size": [3, 4]}.items():
        assert k in new

        k_data = getattr(new, k)
        assert "selection" in k_data.dims
        assert list(k_data.coords["selection"]) == ["first", "second"]
        assert list(k_data) == vals


@pytest.mark.parametrize("drop", [True, False])
def test_dataarray_empty_selector(dataarray, drop):
    new = group_reduce(
        dataarray,
        [{"selector": []}, {"selector": [2, 3]}],
        reduce_dim="x",
        reduce_func=np.sum,
        groups_dim="selection",
        drop_empty=drop,
        fill_empty=0.0,
    )

    assert isinstance(new, xr.DataArray)
    assert "x" not in new.dims
    assert "selection" in new.dims
    if drop:
        assert len(new.coords["selection"]) == 1
        assert list(new.coords["selection"]) == [1]
    else:
        assert len(new.coords["selection"]) == 2
        assert list(new.coords["selection"]) == [0, 1]

    if not drop:
        assert new.sel(selection=0).sum() == 0.0
    assert new.sel(selection=1).sum() == 7 + 8 + 9 + 10 + 11 + 12


def test_dataarray_empty_selector_0d():
    """When reducing an array along its only dimension, you get a 0d array.

    This was creating an error when filling empty selections. This test
    ensures that it doesn' happen again
    """

    new = group_reduce(
        xr.DataArray([1, 2, 3], coords=[("x", [0, 1, 2])]),
        [{"selector": []}, {"selector": [1, 2]}],
        reduce_dim="x",
        reduce_func=np.sum,
        groups_dim="selection",
        drop_empty=False,
        fill_empty=0.0,
    )

    assert isinstance(new, xr.DataArray)
    assert "x" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == [0, 1]

    assert new.sel(selection=0).sum() == 0.0
    assert new.sel(selection=1).sum() == 2 + 3


def test_dataset(dataset):
    new = group_reduce(
        dataset,
        [{"selector": [0, 1]}, {"selector": [2, 3]}],
        reduce_dim="x",
        reduce_func=np.sum,
        groups_dim="selection",
    )

    assert isinstance(new, xr.Dataset)
    assert "x" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == [0, 1]

    assert "vals" in new
    assert "double" in new
    assert new.sel(selection=0).sum() == (1 + 2 + 3 + 4 + 5 + 6) * 3
    assert new.sel(selection=1).sum() == (7 + 8 + 9 + 10 + 11 + 12) * 3


def test_dataset_multidim(dataset):
    new = group_reduce(
        dataset,
        [{"selector": ([0, 1], [0, 1])}, {"selector": ([2, 3], [0, 1])}],
        reduce_dim=("x", "y"),
        reduce_func=np.sum,
        groups_dim="selection",
    )

    assert isinstance(new, xr.Dataset)
    assert "x" not in new.dims
    assert "y" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == [0, 1]

    assert "vals" in new
    assert "double" in new
    assert new.sel(selection=0).sum() == (1 + 2 + 4 + 5) * 3
    assert new.sel(selection=1).sum() == (7 + 8 + 10 + 11) * 3


def test_dataset_multidim_multireduce(dataset):
    new = group_reduce(
        dataset,
        [{"selector": ([0, 1], [0, 1])}, {"selector": ([2, 3], [0, 1])}],
        reduce_dim=("x", "y"),
        reduce_func=(np.sum, np.mean),
        groups_dim="selection",
    )

    assert isinstance(new, xr.Dataset)
    assert "x" not in new.dims
    assert "y" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == [0, 1]

    assert "vals" in new
    assert "double" in new
    assert new.sel(selection=0).sum() == ((1 + 4) / 2 + (2 + 5) / 2) * 3
    assert new.sel(selection=1).sum() == ((7 + 10) / 2 + (8 + 11) / 2) * 3


def test_dataarray_multidim_multireduce(dataarray):
    new = group_reduce(
        dataarray,
        [{"selector": ([0, 1], [0, 1])}, {"selector": ([2, 3], [0, 1])}],
        reduce_dim=("x", "y"),
        reduce_func=(np.sum, np.mean),
        groups_dim="selection",
    )

    assert isinstance(new, xr.DataArray)
    assert "x" not in new.dims
    assert "y" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == [0, 1]
    assert new.sel(selection=0).sum() == (1 + 4) / 2 + (2 + 5) / 2
    assert new.sel(selection=1).sum() == (7 + 10) / 2 + (8 + 11) / 2


def test_dataset_names(dataset):
    new = group_reduce(
        dataset,
        [{"selector": [0, 1], "name": "first"}, {"selector": [2, 3], "name": "second"}],
        reduce_dim="x",
        reduce_func=np.sum,
        groups_dim="selection",
    )

    assert isinstance(new, xr.Dataset)
    assert "x" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == ["first", "second"]

    assert "vals" in new
    assert "double" in new
    assert new.sel(selection="first").sum() == (1 + 2 + 3 + 4 + 5 + 6) * 3
    assert new.sel(selection="second").sum() == (7 + 8 + 9 + 10 + 11 + 12) * 3


def test_dataset_groupvars(dataset):
    new = group_reduce(
        dataset,
        [
            {"selector": [0, 1], "name": "first", "color": "red", "size": 3},
            {"selector": [2, 3], "name": "second", "color": "blue", "size": 4},
        ],
        reduce_dim="x",
        reduce_func=np.sum,
        groups_dim="selection",
        group_vars=["color", "size"],
    )

    assert isinstance(new, xr.Dataset)
    assert "x" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == ["first", "second"]

    assert "vals" in new
    assert "double" in new
    assert new.double.sel(selection="first").sum() == (1 + 2 + 3 + 4 + 5 + 6) * 2
    assert new.double.sel(selection="second").sum() == (7 + 8 + 9 + 10 + 11 + 12) * 2

    for k, vals in {"color": ["red", "blue"], "size": [3, 4]}.items():
        assert k in new

        k_data = getattr(new, k)
        assert "selection" in k_data.dims
        assert list(k_data.coords["selection"]) == ["first", "second"]
        assert list(k_data) == vals


@pytest.mark.parametrize("drop", [True, False])
def test_dataset_empty_selector(dataset, drop):
    new = group_reduce(
        dataset,
        [{"selector": []}, {"selector": [2, 3]}],
        reduce_dim="x",
        reduce_func=np.sum,
        groups_dim="selection",
        drop_empty=drop,
        fill_empty=0.0,
    )

    assert isinstance(new, xr.Dataset)
    assert "x" not in new.dims
    assert "selection" in new.dims
    if drop:
        assert len(new.coords["selection"]) == 1
        assert list(new.coords["selection"]) == [1]
    else:
        assert len(new.coords["selection"]) == 2
        assert list(new.coords["selection"]) == [0, 1]

    assert "vals" in new
    assert "double" in new
    if not drop:
        assert new.sel(selection=0).sum() == 0.0
    assert new.sel(selection=1).sum() == (7 + 8 + 9 + 10 + 11 + 12) * 3


def test_dataaset_empty_selector_0d():
    """When reducing an array along its only dimension, you get a 0d array.

    This was creating an error when filling empty selections. This test
    ensures that it doesn' happen again
    """
    dataset = xr.Dataset({"vals": (["x"], [1, 2, 3])})

    new = group_reduce(
        dataset,
        [{"selector": []}, {"selector": [1, 2]}],
        reduce_dim="x",
        reduce_func=np.sum,
        groups_dim="selection",
        drop_empty=False,
        fill_empty=0.0,
    )

    assert isinstance(new, xr.Dataset)
    assert "x" not in new.dims
    assert "selection" in new.dims
    assert len(new.coords["selection"]) == 2
    assert list(new.coords["selection"]) == [0, 1]

    assert "vals" in new
    assert new.sel(selection=0).sum() == 0.0
    assert new.sel(selection=1).sum() == 2 + 3
