# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from sisl import Lattice
from sisl.viz.processors.cell import (
    cell_to_lines,
    gen_cell_dataset,
    infer_cell_axes,
    is_1D_cartesian,
    is_cartesian_unordered,
)

pytestmark = [pytest.mark.viz, pytest.mark.processors]


@pytest.fixture(scope="module", params=["numpy", "lattice"])
def Cell(request):
    if request.param == "numpy":
        return np.array
    elif request.param == "lattice":
        return Lattice


def test_cartesian_unordered(Cell):
    assert is_cartesian_unordered(
        Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )

    assert is_cartesian_unordered(
        Cell([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    )

    assert not is_cartesian_unordered(
        Cell([[0, 2, 1], [1, 0, 0], [0, 1, 0]]),
    )


def test_1D_cartesian(Cell):
    assert is_1D_cartesian(Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), "x")

    assert is_1D_cartesian(Cell([[1, 0, 0], [0, 1, 1], [0, 0, 1]]), "x")

    assert not is_1D_cartesian(Cell([[1, 0, 0], [1, 1, 0], [0, 0, 1]]), "x")


def test_infer_cell_axes(Cell):
    assert infer_cell_axes(
        Cell([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), axes=["x", "y", "z"]
    ) == [1, 0, 2]

    assert infer_cell_axes(
        Cell([[0, 1, 0], [1, 0, 0], [0, 0, 1]]), axes=["b", "y"]
    ) == [1, 0]


def test_gen_cell_dataset():
    lattice = Lattice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    cell_dataset = gen_cell_dataset(lattice)

    assert isinstance(cell_dataset, xr.Dataset)

    assert "lattice" in cell_dataset.attrs
    assert cell_dataset.attrs["lattice"] is lattice

    assert "xyz" in cell_dataset.data_vars
    assert cell_dataset.xyz.shape == (2, 2, 2, 3)
    assert np.all(cell_dataset.xyz.values == lattice.vertices())


@pytest.mark.parametrize("mode", ["box", "axes", "other"])
def test_cell_to_lines(mode):
    lattice = Lattice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    cell_dataset = gen_cell_dataset(lattice)

    if mode == "other":
        with pytest.raises(ValueError):
            cell_to_lines(cell_dataset, mode)
    else:
        lines = cell_to_lines(cell_dataset, mode, cell_style={"color": "red"})

        assert isinstance(lines, xr.Dataset)

        if mode == "box":
            # 19 points are required to draw a box
            assert lines.xyz.shape == (19, 3)
        elif mode == "axes":
            # 9 points are required to draw the axes
            assert lines.xyz.shape == (9, 3)
