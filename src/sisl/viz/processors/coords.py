# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from xarray import Dataset

from sisl._core._lattice import cell_invert
from sisl._core.lattice import Lattice
from sisl.typing import LatticeLike
from sisl.utils.mathematics import fnorm

from ..types import Axes, Axis
from .axes import axes_cross_product, axis_direction, get_ax_title

CoordsDataset = Dataset


def projected_2Dcoords(
    cell: LatticeLike,
    xyz: npt.NDArray[np.float64],
    xaxis: Axis = "x",
    yaxis: Axis = "y",
) -> npt.NDArray[np.float64]:
    """Moves the 3D positions of the atoms to a 2D supspace.

    In this way, we can plot the structure from the "point of view" that we want.

    NOTE: If xaxis/yaxis is one of {"a", "b", "c", "1", "2", "3"} the function doesn't
    project the coordinates in the direction of the lattice vector. The fractional
    coordinates, taking in consideration the three lattice vectors, are returned
    instead.

    Parameters
    ------------
    geometry: sisl.Geometry
        the geometry for which you want the projected coords
    xyz: array-like of shape (natoms, 3), optional
        the 3D coordinates that we want to project.
        otherwise they are taken from the geometry.
    xaxis: {"x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
        the direction to be displayed along the X axis.
    yaxis: {"x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
        the direction to be displayed along the X axis.

    Returns
    ----------
    np.ndarray of shape (2, natoms)
        the 2D coordinates of the geometry, with all positions projected into the plane
        defined by xaxis and yaxis.
    """
    cell = Lattice.new(cell).cell

    try:
        all_lattice_vecs = len(set([xaxis, yaxis]).intersection(["a", "b", "c"])) == 2
    except:
        # If set fails it is because xaxis/yaxis is unhashable, which means it
        # is a numpy array
        all_lattice_vecs = False

    if all_lattice_vecs:
        coord_indices = ["abc".index(ax) for ax in (xaxis, yaxis)]

        icell = cell_invert(cell.astype(float))
    else:
        # Get the directions that these axes represent
        xaxis = axis_direction(xaxis, cell)
        yaxis = axis_direction(yaxis, cell)

        fake_cell = np.array([xaxis, yaxis, np.cross(xaxis, yaxis)], dtype=np.float64)
        icell = cell_invert(fake_cell)
        coord_indices = [0, 1]

    return np.dot(xyz, icell.T)[..., coord_indices]


def projected_1Dcoords(
    cell: LatticeLike, xyz: npt.NDArray[np.float64], axis: Axis = "x"
):
    """
    Moves the 3D positions of the atoms to a 2D supspace.

    In this way, we can plot the structure from the "point of view" that we want.

    NOTE: If axis is one of {"a", "b", "c", "1", "2", "3"} the function doesn't
    project the coordinates in the direction of the lattice vector. The fractional
    coordinates, taking in consideration the three lattice vectors, are returned
    instead.

    Parameters
    ------------
    geometry: sisl.Geometry
        the geometry for which you want the projected coords
    xyz: array-like of shape (natoms, 3), optional
        the 3D coordinates that we want to project.
        otherwise they are taken from the geometry.
    axis: {"x", "y", "z", "a", "b", "c", "1", "2", "3"} or array-like of shape 3, optional
        the direction to be displayed along the X axis.
    nsc: array-like of shape (3, ), optional
        only used if `axis` is a lattice vector. It is used to rescale everything to the unit
        cell lattice vectors, otherwise `GeometryPlot` doesn't play well with `GridPlot`.

    Returns
    ----------
    np.ndarray of shape (natoms, )
        the 1D coordinates of the geometry, with all positions projected into the line
        defined by axis.
    """
    cell = Lattice.new(cell).cell

    if isinstance(axis, str) and axis in ("a", "b", "c", "0", "1", "2"):
        return projected_2Dcoords(
            cell, xyz, xaxis=axis, yaxis="a" if axis == "c" else "c"
        )[..., 0]

    # Get the direction that the axis represents
    axis = axis_direction(axis, cell)

    return xyz.dot(axis / fnorm(axis)) / fnorm(axis)


def coords_depth(coords_data: CoordsDataset, axes: Axes) -> npt.NDArray[np.float64]:
    """Computes the depth of 3D points as projected in a 2D plane

    Parameters
    ----------
    coords_data: CoordsDataset
        The coordinates for which the depth is to be computed.
    axes: Axes
        The axes that define the plane where the coordinates are projected.
    """
    cell = _get_cell_from_dataset(coords_data=coords_data)

    depth_vector = axes_cross_product(axes[0], axes[1], cell)
    depth = project_to_axes(coords_data, axes=[depth_vector]).x.values

    return depth


def sphere(
    center: npt.ArrayLike = [0, 0, 0], r: float = 1, vertices: int = 10
) -> dict[str, np.ndarray]:
    """Computes a mesh defining a sphere."""
    phi, theta = np.mgrid[
        0.0 : np.pi : 1j * vertices, 0.0 : 2.0 * np.pi : 1j * vertices
    ]
    center = np.array(center)

    phi = np.ravel(phi)
    theta = np.ravel(theta)

    x = center[0] + r * np.sin(phi) * np.cos(theta)
    y = center[1] + r * np.sin(phi) * np.sin(theta)
    z = center[2] + r * np.cos(phi)

    return {"x": x, "y": y, "z": z}


def _get_cell_from_dataset(coords_data: CoordsDataset) -> npt.NDArray[np.float64]:
    cell = coords_data.attrs.get("cell")
    if cell is None:
        if "lattice" in coords_data.attrs:
            cell = coords_data.lattice.cell
        else:
            cell = coords_data.geometry.cell

    return cell


def projected_1D_data(
    coords_data: CoordsDataset,
    axis: Axis = "x",
    dataaxis_1d: Union[Callable, npt.NDArray, None] = None,
) -> CoordsDataset:
    cell = _get_cell_from_dataset(coords_data=coords_data)

    xyz = coords_data.xyz.values

    x = projected_1Dcoords(cell, xyz=xyz, axis=axis)

    dims = coords_data.xyz.dims[:-1]

    if dataaxis_1d is None:
        y = np.zeros_like(x)
    else:
        if callable(dataaxis_1d):
            y = dataaxis_1d(x)
        elif isinstance(dataaxis_1d, (int, float)):
            y = np.full_like(x, dataaxis_1d)
        else:
            y = dataaxis_1d

    coords_data = coords_data.assign(x=(dims, x), y=(dims, y))

    return coords_data


def projected_2D_data(
    coords_data: CoordsDataset,
    xaxis: Axis = "x",
    yaxis: Axis = "y",
    sort_by_depth: bool = False,
) -> CoordsDataset:
    cell = _get_cell_from_dataset(coords_data=coords_data)

    xyz = coords_data.xyz.values

    xy = projected_2Dcoords(cell, xyz, xaxis=xaxis, yaxis=yaxis)

    x, y = xy[..., 0], xy[..., 1]
    dims = coords_data.xyz.dims[:-1]

    coords_data = coords_data.assign(x=(dims, x), y=(dims, y))

    coords_data = coords_data.assign(
        {"depth": (dims, coords_depth(coords_data, [xaxis, yaxis]).data)}
    )
    if sort_by_depth:
        coords_data = coords_data.sortby("depth")

    return coords_data


def projected_3D_data(coords_data: CoordsDataset) -> CoordsDataset:
    x, y, z = np.moveaxis(coords_data.xyz.values, -1, 0)
    dims = coords_data.xyz.dims[:-1]

    coords_data = coords_data.assign(x=(dims, x), y=(dims, y), z=(dims, z))

    return coords_data


def project_to_axes(
    coords_data: CoordsDataset,
    axes: Axes,
    dataaxis_1d: Optional[Union[npt.ArrayLike, Callable]] = None,
    sort_by_depth: bool = False,
    cartesian_units: str = "Ang",
) -> CoordsDataset:
    ndim = len(axes)
    if ndim == 3:
        xaxis, yaxis, zaxis = axes
        coords_data = projected_3D_data(coords_data)
    elif ndim == 2:
        xaxis, yaxis = axes
        coords_data = projected_2D_data(
            coords_data, xaxis=xaxis, yaxis=yaxis, sort_by_depth=sort_by_depth
        )
    elif ndim == 1:
        xaxis = axes[0]
        yaxis = dataaxis_1d
        coords_data = projected_1D_data(
            coords_data, axis=xaxis, dataaxis_1d=dataaxis_1d
        )

    plot_axes = ["x", "y", "z"][:ndim]

    for ax, plot_ax in zip(axes, plot_axes):
        coords_data[plot_ax].attrs["axis"] = {
            "title": get_ax_title(ax, cartesian_units=cartesian_units),
        }

    coords_data.attrs["ndim"] = ndim

    return coords_data
