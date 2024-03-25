# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import sisl
from sisl import Lattice
from sisl.viz.processors.coords import (
    coords_depth,
    project_to_axes,
    projected_1D_data,
    projected_1Dcoords,
    projected_2D_data,
    projected_2Dcoords,
    projected_3D_data,
    sphere,
)

pytestmark = [pytest.mark.viz, pytest.mark.processors]


@pytest.fixture(scope="module", params=["numpy", "lattice"])
def Cell(request):
    if request.param == "numpy":
        return np.array
    elif request.param == "lattice":
        return Lattice


@pytest.fixture(scope="module")
def coords_dataset():
    geometry = sisl.geom.bcc(2.93, "Au", False)

    return xr.Dataset(
        {"xyz": (("atom", "axis"), geometry.xyz)},
        coords={"axis": [0, 1, 2]},
        attrs={"geometry": geometry},
    )


def test_projected_1D_coords(Cell):
    cell = Cell([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

    x, y, z = 3, -4, 2

    coords = np.array([[x, y, z]])

    # Project to cartesian
    projected = projected_1Dcoords(cell, coords, "x")
    assert np.allclose(projected, [[x]])

    projected = projected_1Dcoords(cell, coords, "-y")
    assert np.allclose(projected, [[-y]])

    # Project to lattice
    projected = projected_1Dcoords(cell, coords, "b")
    assert np.allclose(projected, [[x]])

    # Project to vector
    projected = projected_1Dcoords(cell, coords, [x, 0, z])
    assert np.allclose(projected, [[1]])


def test_projected_2D_coords(Cell):
    cell = Cell([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

    x, y, z = 3, -4, 2

    coords = np.array([[x, y, z]])

    # Project to cartesian
    projected = projected_2Dcoords(cell, coords, "x", "y")
    assert np.allclose(projected, [[x, y]])

    projected = projected_2Dcoords(cell, coords, "-x", "y")
    assert np.allclose(projected, [[-x, y]])

    projected = projected_2Dcoords(cell, coords, "z", "x")
    assert np.allclose(projected, [[z, x]])

    # Project to lattice
    projected = projected_2Dcoords(cell, coords, "a", "b")
    assert np.allclose(projected, [[y, x]])

    projected = projected_2Dcoords(cell, coords, "-b", "a")
    assert np.allclose(projected, [[-x, y]])

    # Project to vectors
    projected = projected_2Dcoords(cell, coords, [x, y, 0], [0, 0, z])
    assert np.allclose(projected, [[1, 1]])


def test_coords_depth(coords_dataset):
    depth = coords_depth(coords_dataset, ["x", "y"])
    assert isinstance(depth, np.ndarray)
    assert np.allclose(depth, coords_dataset.xyz.sel(axis=2).values)

    depth = coords_depth(coords_dataset, ["y", "x"])
    assert np.allclose(depth, -coords_dataset.xyz.sel(axis=2).values)

    depth = coords_depth(coords_dataset, [[1, 0, 0], [0, 0, 1]])
    assert np.allclose(depth, -coords_dataset.xyz.sel(axis=1).values)


@pytest.mark.parametrize("center", [[0, 0, 0], [1, 1, 0]])
def test_sphere(center):
    coords = sphere(center=center, r=3.5, vertices=15)

    assert isinstance(coords, dict)

    assert "x" in coords
    assert "y" in coords
    assert "z" in coords

    assert coords["x"].shape == coords["y"].shape == coords["z"].shape == (15**2,)

    R = np.linalg.norm(
        np.array([coords["x"], coords["y"], coords["z"]]).T - center, axis=1
    )

    assert np.allclose(R, 3.5)


def test_projected_1D_data(coords_dataset):
    # No data
    projected = projected_1D_data(coords_dataset, "y")
    assert isinstance(projected, xr.Dataset)
    assert "x" in projected.data_vars
    assert np.allclose(projected.x, coords_dataset.xyz.sel(axis=1))
    assert "y" in projected.data_vars
    assert np.allclose(projected.y, 0)

    # Data from function
    projected = projected_1D_data(coords_dataset, "-y", dataaxis_1d=np.sin)
    assert isinstance(projected, xr.Dataset)
    assert "x" in projected.data_vars
    assert np.allclose(projected.x, -coords_dataset.xyz.sel(axis=1))
    assert "y" in projected.data_vars
    assert np.allclose(projected.y, np.sin(-coords_dataset.xyz.sel(axis=1)))

    # Data from array
    projected = projected_1D_data(
        coords_dataset, "-y", dataaxis_1d=coords_dataset.xyz.sel(axis=2).values
    )
    assert isinstance(projected, xr.Dataset)
    assert "x" in projected.data_vars
    assert np.allclose(projected.x, -coords_dataset.xyz.sel(axis=1))
    assert "y" in projected.data_vars
    assert np.allclose(projected.y, coords_dataset.xyz.sel(axis=2))


def test_projected_2D_data(coords_dataset):
    projected = projected_2D_data(coords_dataset, "-y", "x")
    assert isinstance(projected, xr.Dataset)
    assert "x" in projected.data_vars
    assert np.allclose(projected.x, -coords_dataset.xyz.sel(axis=1))
    assert "y" in projected.data_vars
    assert np.allclose(projected.y, coords_dataset.xyz.sel(axis=0))

    assert "depth" in projected.data_vars

    projected = projected_2D_data(coords_dataset, "x", "y", sort_by_depth=True)
    assert isinstance(projected, xr.Dataset)
    assert "x" in projected.data_vars
    assert np.allclose(projected.x, coords_dataset.xyz.sel(axis=0))
    assert "y" in projected.data_vars
    assert np.allclose(projected.y, coords_dataset.xyz.sel(axis=1))

    assert "depth" in projected.data_vars
    assert np.allclose(projected.depth.values, coords_dataset.xyz.sel(axis=2))
    # Check that points are sorted by depth.
    assert np.all(np.diff(projected.depth) > 0)


def test_projected_3D_data(coords_dataset):
    projected = projected_3D_data(coords_dataset)
    assert isinstance(projected, xr.Dataset)
    assert "x" in projected.data_vars
    assert np.allclose(projected.x, coords_dataset.xyz.sel(axis=0))
    assert "y" in projected.data_vars
    assert np.allclose(projected.y, coords_dataset.xyz.sel(axis=1))
    assert "z" in projected.data_vars
    assert np.allclose(projected.z, coords_dataset.xyz.sel(axis=2))


def test_project_to_axes(coords_dataset):
    projected = project_to_axes(coords_dataset, ["z"], dataaxis_1d=4)
    assert isinstance(projected, xr.Dataset)
    assert "x" in projected.data_vars
    assert np.allclose(projected.x, coords_dataset.xyz.sel(axis=2))
    assert "y" in projected.data_vars
    assert np.allclose(projected.y, 4)
    assert "z" not in projected.data_vars

    projected = project_to_axes(coords_dataset, ["-y", "x"])
    assert isinstance(projected, xr.Dataset)
    assert "x" in projected.data_vars
    assert np.allclose(projected.x, -coords_dataset.xyz.sel(axis=1))
    assert "y" in projected.data_vars
    assert np.allclose(projected.y, coords_dataset.xyz.sel(axis=0))
    assert "z" not in projected.data_vars

    projected = project_to_axes(coords_dataset, ["x", "y", "z"])
    assert isinstance(projected, xr.Dataset)
    assert "x" in projected.data_vars
    assert np.allclose(projected.x, coords_dataset.xyz.sel(axis=0))
    assert "y" in projected.data_vars
    assert np.allclose(projected.y, coords_dataset.xyz.sel(axis=1))
    assert "z" in projected.data_vars
    assert np.allclose(projected.z, coords_dataset.xyz.sel(axis=2))
