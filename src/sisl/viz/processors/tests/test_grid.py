# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from sisl import Geometry, Grid, Lattice
from sisl.viz.processors.grid import (
    apply_transforms,
    get_ax_vals,
    get_grid_axes,
    get_grid_representation,
    get_isos,
    get_offset,
    grid_geometry,
    grid_to_dataarray,
    interpolate_grid,
    orthogonalize_grid,
    orthogonalize_grid_if_needed,
    reduce_grid,
    should_transform_grid_cell_plotting,
    sub_grid,
    tile_grid,
    transform_grid_cell,
)

pytestmark = [pytest.mark.viz, pytest.mark.processors]


@pytest.fixture(scope="module", params=["orthogonal", "skewed"])
def skewed(request) -> bool:
    return request.param == "skewed"


real_part = np.arange(10 * 10 * 10).reshape(10, 10, 10)
imag_part = np.arange(10 * 10 * 10).reshape(10, 10, 10) + 1


@pytest.fixture(scope="module")
def origin():
    return [1, 2, 3]


@pytest.fixture(scope="module")
def grid(origin, skewed) -> Grid:
    if skewed:
        lattice = Lattice([[3, 0, 0], [1, -1, 0], [0, 0, 3]], origin=origin)
    else:
        lattice = Lattice([[3, 0, 0], [0, 2, 0], [0, 0, 6]], origin=origin)

    geometry = Geometry([[0, 0, 0]], lattice=lattice)
    grid = Grid([10, 10, 10], geometry=geometry, dtype=np.complex128)

    grid.grid[:] = (real_part + imag_part * 1j).reshape(10, 10, 10)

    return grid


def test_get_grid_representation(grid):
    assert np.allclose(get_grid_representation(grid, "real").grid, real_part)
    assert np.allclose(get_grid_representation(grid, "imag").grid, imag_part)
    assert np.allclose(
        get_grid_representation(grid, "mod").grid,
        np.sqrt(real_part**2 + imag_part**2),
    )
    assert np.allclose(
        get_grid_representation(grid, "phase").grid, np.arctan2(imag_part, real_part)
    )
    assert np.allclose(
        get_grid_representation(grid, "rad_phase").grid,
        np.arctan2(imag_part, real_part),
    )
    assert np.allclose(
        get_grid_representation(grid, "deg_phase").grid,
        np.arctan2(imag_part, real_part) * 180 / np.pi,
    )


def test_tile_grid(grid):
    # By default it is not tiled
    tiled = tile_grid(grid)
    assert isinstance(tiled, Grid)
    assert tiled.shape == grid.shape
    assert np.allclose(tiled.grid, grid.grid)
    assert np.allclose(tiled.origin, grid.origin)

    # Now tile it
    tiled = tile_grid(grid, (1, 2, 1))
    assert isinstance(tiled, Grid)
    assert tiled.shape == (grid.shape[0], grid.shape[1] * 2, grid.shape[2])
    assert np.allclose(tiled.grid[:, : grid.shape[1]], grid.grid)
    assert np.allclose(tiled.grid[:, grid.shape[1] :], grid.grid)
    assert np.allclose(tiled.origin, grid.origin)


def test_transform_grid_cell(grid, skewed):
    # Convert to a cartesian cell
    new_grid = transform_grid_cell(grid, cell=np.eye(3), output_shape=(10, 10, 10))

    assert new_grid.shape == (10, 10, 10)
    assert new_grid.lattice.is_cartesian()

    if not skewed:
        assert np.allclose(new_grid.lattice.cell, grid.lattice.cell)

    assert np.allclose(new_grid.grid, grid.grid) == (not skewed)
    assert not np.allclose(new_grid.grid, 0)

    assert np.allclose(new_grid.origin, grid.origin) == (not skewed)

    # Convert to a skewed cell
    directions = np.array([[1, 2, 3], [-1, 2, -4], [2, -1, 1]])
    new_grid = transform_grid_cell(grid, cell=directions, output_shape=(5, 5, 5))

    assert new_grid.shape == (5, 5, 5)
    for i in range(3):
        n = new_grid.lattice.cell[i] / directions[i]
        assert np.allclose(n, n[0])


@pytest.mark.parametrize("interp", [1, 2])
def test_orthogonalize_grid(grid, interp, skewed):
    ort_grid = orthogonalize_grid(grid, interp=(interp, interp, interp))

    assert ort_grid.shape == (10, 10, 10) if interp == 1 else (20, 20, 20)
    assert ort_grid.lattice.is_cartesian()

    if not skewed:
        assert np.allclose(ort_grid.lattice.cell, grid.lattice.cell)
        if interp == 1:
            assert np.allclose(ort_grid.grid, grid.grid)
    else:
        if interp == 1:
            assert not np.allclose(ort_grid.grid, grid.grid)

    assert not np.allclose(ort_grid.grid, 0)
    assert np.allclose(ort_grid.origin, grid.origin) == (not skewed and interp == 1)


def test_should_transform_grid_cell_plotting(grid, skewed):
    assert should_transform_grid_cell_plotting(grid, axes=["x", "y"]) == skewed
    assert should_transform_grid_cell_plotting(grid, axes=["z"]) == False


@pytest.mark.parametrize("interp", [1, 2])
def test_orthogonalize_grid_if_needed(grid, skewed, interp):
    # Orthogonalize the skewed cell, since it is xy skewed.
    ort_grid = orthogonalize_grid_if_needed(
        grid, axes=["x", "y"], interp=(interp, interp, interp)
    )

    assert ort_grid.shape == (10, 10, 10) if interp == 1 else (20, 20, 20)
    assert ort_grid.lattice.is_cartesian()

    if not skewed:
        assert np.allclose(ort_grid.lattice.cell, grid.lattice.cell)
        if interp == 1:
            assert np.allclose(ort_grid.grid, grid.grid)
    else:
        if interp == 1:
            assert not np.allclose(ort_grid.grid, grid.grid)

    assert not np.allclose(ort_grid.grid, 0)
    assert np.allclose(ort_grid.origin, grid.origin) == (not skewed)

    # Do not orthogonalize the skewed cell, since it is not z skewed.
    ort_grid = orthogonalize_grid_if_needed(
        grid, axes=["z"], interp=(interp, interp, interp)
    )

    assert ort_grid.shape == (10, 10, 10) if interp == 1 else (20, 20, 20)

    if skewed:
        assert not ort_grid.lattice.is_cartesian()

    assert np.allclose(ort_grid.lattice.cell, grid.lattice.cell)
    assert np.allclose(ort_grid.grid, grid.grid)


def test_apply_transforms(grid):
    # Apply a function
    transf = apply_transforms(grid, transforms=[np.sqrt])
    assert np.allclose(transf.grid, np.sqrt(grid.grid))
    assert np.allclose(transf.origin, grid.origin)

    # Apply a numpy function specifying a string
    transf = apply_transforms(grid, transforms=["sqrt"])
    assert np.allclose(transf.grid, np.sqrt(grid.grid))
    assert np.allclose(transf.origin, grid.origin)

    # Apply two consecutive functions
    transf = apply_transforms(grid, transforms=[np.angle, "sqrt"])
    assert np.allclose(transf.grid, np.sqrt(np.angle(grid.grid)))
    assert np.allclose(transf.origin, grid.origin)


@pytest.mark.parametrize("reduce_method", ["sum", "mean"])
def test_reduce_grid(grid, reduce_method):
    reduce_func = {"sum": np.sum, "mean": np.mean}[reduce_method]

    reduced = reduce_grid(grid, reduce_method, keep_axes=[0, 1])

    assert reduced.shape == (10, 10, 1)
    assert np.allclose(reduced.grid[:, :, 0], reduce_func(grid.grid, axis=2))
    assert np.allclose(reduced.origin, grid.origin)


@pytest.mark.parametrize("direction", ["x", "y", "z"])
def test_sub_grid(grid, skewed, direction):
    coord_ax = "xyz".index(direction)
    kwargs = {f"{direction}_range": (0.5, 1.5)}

    expected_origin = grid.origin.copy()

    if skewed and direction != "z":
        with pytest.raises(ValueError):
            sub = sub_grid(grid, **kwargs, cart_tol=1e-3)
    else:
        sub = sub_grid(grid, **kwargs, cart_tol=1e-3)

        # Check that the lattice has been reduced to contain the requested range,
        # taking into account that the bounds of the range might not be exactly
        # on the grid points.
        assert (
            1 + sub.dcell[:, coord_ax].sum() * 2
            >= sub.lattice.cell[:, coord_ax].sum()
            >= 1 - sub.dcell[:, coord_ax].sum() * 2
        )

        expected_origin[coord_ax] += 0.5

        assert np.allclose(sub.origin, expected_origin)


def test_interpolate_grid(grid):
    interp = interpolate_grid(grid, (20, 20, 20))

    # Check that the shape has been augmented
    assert np.all(interp.shape == np.array((20, 20, 20)) * grid.shape)

    # The integral over the grid should be the same (or very similar)
    assert (grid.grid.sum() * 20**3 - interp.grid.sum()) < 1e-3


def test_grid_geometry(grid):
    assert grid_geometry(grid) is grid.geometry

    geom_copy = grid.geometry.copy()

    assert grid_geometry(grid, geom_copy) is geom_copy


def test_get_grid_axes(grid, skewed):
    assert get_grid_axes(grid, ["x", "y", "z"]) == [0, 1, 2]
    # This function doesn't care about what the axes are in 3D
    assert get_grid_axes(grid, ["y", "-x", "z"]) == [0, 1, 2]

    if skewed:
        with pytest.raises(ValueError):
            get_grid_axes(grid, ["x", "y"])
    else:
        assert get_grid_axes(grid, ["x", "y"]) == [0, 1]
        assert get_grid_axes(grid, ["y", "x"]) == [1, 0]


def test_get_ax_vals(grid, skewed, origin):
    r = get_ax_vals(grid, "x", nsc=(1, 1, 1))

    assert isinstance(r, np.ndarray)
    assert r.shape == (grid.shape[0],)

    if not skewed:
        assert r[0] == origin[0]
        assert (
            abs(r[-1] - (origin[0] + grid.lattice.cell[0, 0] - grid.dcell[0, 0])) < 1e-3
        )

    r = get_ax_vals(grid, "a", nsc=(2, 1, 1))

    assert isinstance(r, np.ndarray)
    assert r.shape == (grid.shape[0],)

    assert r[0] == 0
    assert abs(r[-1] - 2) < 1e-3


def test_get_offset(grid, origin):
    assert get_offset(grid, "x") == origin[0]
    assert get_offset(grid, "b") == 0
    assert get_offset(grid, 2) == 0


def test_grid_to_dataarray(grid, skewed):
    # Test 1D
    av_grid = grid.average(0).average(1)

    arr = grid_to_dataarray(av_grid, ["z"], [2], nsc=(1, 1, 1))

    assert isinstance(arr, xr.DataArray)
    assert len(arr.coords) == 1
    assert "x" in arr.coords
    assert arr.x.shape == (grid.shape[2],)

    assert np.allclose(arr.values, av_grid.grid[0, 0, :])

    if skewed:
        return

    # Test 2D
    av_grid = grid.average(0)

    arr = grid_to_dataarray(av_grid, ["y", "z"], [1, 2], nsc=(1, 1, 1))

    assert isinstance(arr, xr.DataArray)
    assert len(arr.coords) == 2
    assert "x" in arr.coords
    assert arr.x.shape == (grid.shape[1],)
    assert "y" in arr.coords
    assert arr.y.shape == (grid.shape[2],)

    assert np.allclose(arr.values, av_grid.grid[0, :, :])

    # Test 2D with unordered axes
    av_grid = grid.average(0)

    arr = grid_to_dataarray(av_grid, ["z", "y"], [2, 1], nsc=(1, 1, 1))

    assert isinstance(arr, xr.DataArray)
    assert len(arr.coords) == 2
    assert "x" in arr.coords
    assert arr.x.shape == (grid.shape[2],)
    assert "y" in arr.coords
    assert arr.y.shape == (grid.shape[1],)

    assert np.allclose(arr.values, av_grid.grid[0, :, :].T)

    # Test 3D
    av_grid = grid

    arr = grid_to_dataarray(av_grid, ["x", "y", "z"], [0, 1, 2], nsc=(1, 1, 1))

    assert isinstance(arr, xr.DataArray)
    assert len(arr.coords) == 3
    assert "x" in arr.coords
    assert arr.x.shape == (grid.shape[0],)
    assert "y" in arr.coords
    assert arr.y.shape == (grid.shape[1],)
    assert "z" in arr.coords
    assert arr.z.shape == (grid.shape[2],)

    assert np.allclose(arr.values, av_grid.grid)


def test_get_isos(grid, skewed):
    pytest.importorskip("skimage")

    if skewed:
        return

    # we cast to the real value, to bypass scikit-images problems
    # scikit-image only allows float data-types
    grid.grid = grid.grid.real

    # Test isocontours (2D)
    arr = grid_to_dataarray(grid.average(2), ["x", "y"], [0, 1, 2], nsc=(1, 1, 1))

    assert get_isos(arr, []) == []

    contours = get_isos(arr, [{"frac": 0.5}])

    assert isinstance(contours, list)
    assert len(contours) == 1
    assert isinstance(contours[0], dict)
    assert "x" in contours[0]
    assert isinstance(contours[0]["x"], list)
    assert "y" in contours[0]
    assert isinstance(contours[0]["y"], list)
    assert "z" not in contours[0]

    # Test isosurfaces (3D)
    arr = grid_to_dataarray(grid, ["x", "y", "z"], [0, 1, 2], nsc=(1, 1, 1))

    surfs = get_isos(arr, [])

    assert isinstance(surfs, list)
    assert len(surfs) == 2
    assert isinstance(surfs[0], dict)

    # Sanity checks on the first surface
    assert "color" in surfs[0]
    assert surfs[0]["color"] is None
    assert "opacity" in surfs[0]
    assert surfs[0]["opacity"] is None
    assert "name" in surfs[0]
    assert isinstance(surfs[0]["name"], str)
    assert "vertices" in surfs[0]
    assert isinstance(surfs[0]["vertices"], np.ndarray)
    assert surfs[0]["vertices"].dtype == np.float64
    assert surfs[0]["vertices"].shape[1] == 3
    assert "faces" in surfs[0]
    assert isinstance(surfs[0]["faces"], np.ndarray)
    assert surfs[0]["faces"].dtype == np.int32
    assert surfs[0]["faces"].shape[1] == 3

    surfs = get_isos(arr, [{"val": 3, "color": "red", "opacity": 0.5, "name": "test"}])

    assert isinstance(surfs, list)
    assert len(surfs) == 1
    assert isinstance(surfs[0], dict)
    assert "color" in surfs[0]
    assert surfs[0]["color"] == "red"
    assert "opacity" in surfs[0]
    assert surfs[0]["opacity"] == 0.5
    assert "name" in surfs[0]
    assert surfs[0]["name"] == "test"
    assert "vertices" in surfs[0]
    assert isinstance(surfs[0]["vertices"], np.ndarray)
    assert surfs[0]["vertices"].dtype == np.float64
    assert surfs[0]["vertices"].shape[1] == 3
    assert "faces" in surfs[0]
    assert isinstance(surfs[0]["faces"], np.ndarray)
    assert surfs[0]["faces"].dtype == np.int32
    assert surfs[0]["faces"].shape[1] == 3
