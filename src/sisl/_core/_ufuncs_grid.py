# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np

import sisl._array as _a
from sisl._ufuncs import register_sisl_dispatch
from sisl.messages import SislError
from sisl.typing import CellAxis, GridLike, SileLike
from sisl.utils import import_attr
from sisl.utils.misc import direction

from .grid import Grid

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(Grid, module="sisl")
def copy(grid: Grid, dtype=None) -> Grid:
    """Copy the object, possibly changing the data-type"""
    d = grid._sc_geometry_dict()
    if dtype is None:
        d["dtype"] = grid.dtype
    else:
        d["dtype"] = dtype
    out = grid.__class__([1] * 3, **d)
    # This also ensures the shape is copied!
    out.grid = grid.grid.astype(dtype=d["dtype"])
    return out


@register_sisl_dispatch(Grid, module="sisl")
def write(grid: Grid, sile: SileLike, *args, **kwargs) -> None:
    """Writes grid to the `sile` using `sile.write_grid`

    Parameters
    ----------
    sile :
        a `Sile` object which will be used to write the grid
        if it is a string it will create a new sile using `get_sile`
    *args, **kwargs:
        Any other args will be passed directly to the
        underlying routine

    See Also
    --------
    Grid.read : reads a `Grid` from a given `Sile`/file
    """
    # this only works because, they *must*
    # have been imported previously
    from sisl.io import BaseSile, get_sile

    if isinstance(sile, BaseSile):
        sile.write_grid(grid, *args, **kwargs)
    else:
        with get_sile(sile, mode="w") as fh:
            fh.write_grid(grid, *args, **kwargs)


@register_sisl_dispatch(Grid, module="sisl")
def apply(grid: Grid, function_, *args, **kwargs):
    """Applies a function to the grid and returns a new grid

    You can also apply a function that does not return a grid (maybe you want to do
    some measurement). In that case, you will get the result instead of a `Grid`.

    Parameters
    ----------
    function_ : str or function
        for a string the full module path to the function should be given.
        The function that will be called should have the grid as the first argument in its
        interface.
    *args and **kwargs :
        arguments that go directly to the function call

    Notes
    -----
    The function argument name `function_` is named so that `function` can be an eligible
    keyword argument for the function.
    """
    if isinstance(function_, str):
        function_ = import_attr(function_)

    result = function_(grid.grid, *args, **kwargs)

    # Maybe the result is not a grid, because there are methods that actually
    # do measurements of the grid
    # TODO what to do about functions that squeeze shape == 1 dimensions?
    if not isinstance(result, np.ndarray) or result.ndim != 3:
        return result

    # If the result is a grid, we will generate a copy of this one with the new grid values
    grid = grid.copy()
    grid.grid = result

    return grid


@register_sisl_dispatch(Grid, module="sisl")
def swapaxes(grid: Grid, axis1: CellAxis, axis2: CellAxis) -> Grid:
    """Swap two axes in the grid (also swaps axes in the lattice)

    If ``swapaxes(0, 1)`` it returns the 0 in the 1 values.

    Parameters
    ----------
    axis1, axis2 :
        axes indices to be swapped
    """
    axis1 = direction(axis1)
    axis2 = direction(axis2)
    # Create index vector
    idx = _a.arangei(3)
    idx[axis2] = axis1
    idx[axis1] = axis2
    s = np.copy(grid.shape)
    d = grid._sc_geometry_dict()
    d["lattice"] = d["lattice"].swapaxes(axis1, axis2)
    d["dtype"] = grid.dtype
    out = grid.__class__(s[idx], **d)
    # We need to force the C-order or we loose the contiguity
    out.grid = np.copy(np.swapaxes(grid.grid, axis1, axis2), order="C")
    return out


@register_sisl_dispatch(Grid, module="sisl")
def sub(grid: Grid, idx: Union[int, Sequence[int]], axis: CellAxis) -> Grid:
    """Retains certain indices from a specified axis.

    Works exactly opposite to `remove`.

    Parameters
    ----------
    idx :
       the indices of the grid axis `axis` to be retained
    axis :
       the axis segment from which we retain the indices `idx`
    """
    axis = direction(axis)
    idx = _a.asarrayi(idx).ravel()
    shift_geometry = False
    if len(idx) > 1:
        if np.allclose(np.diff(idx), 1):
            shift_geometry = not grid.geometry is None

    if shift_geometry:
        out = grid._copy_sub(len(idx), axis)
        min_xyz = out.dcell[axis, :] * idx[0]
        # Now shift the geometry according to what is retained
        geom = out.geometry.translate(-min_xyz)
        geom.set_lattice(out.lattice)
        out.set_geometry(geom)
    else:
        out = grid._copy_sub(len(idx), axis, scale_geometry=True)

    # Remove the indices
    # First create the opposite, index
    out.grid[:, :, :] = np.take(grid.grid, idx, axis)

    return out


@register_sisl_dispatch(Grid, module="sisl")
def remove(grid: Grid, idx: Union[int, Sequence[int]], axis: CellAxis) -> Grid:
    """Removes certain indices from a specified axis.

    Works exactly opposite to `sub`.

    Parameters
    ----------
    idx :
       the indices of the grid axis `axis` to be removed
    axis :
       the axis segment from which we remove all indices `idx`
    """
    axis = direction(axis)
    ret_idx = np.delete(_a.arangei(grid.shape[axis]), _a.asarrayi(idx))
    return grid.sub(ret_idx, axis)


@register_sisl_dispatch(Grid, module="sisl")
def append(grid: Grid, other: GridLike, axis: CellAxis) -> Grid:
    """Appends other `Grid` to this grid along axis"""
    axis = direction(axis)
    shape = list(grid.shape)
    other = grid.new(other)
    shape[axis] += other.shape[axis]
    d = grid._sc_geometry_dict()
    if "geometry" in d:
        if not other.geometry is None:
            d["geometry"] = d["geometry"].append(other.geometry, axis)
    else:
        d["geometry"] = other.geometry
    d["lattice"] = grid.lattice.append(other.lattice, axis)
    d["dtype"] = grid.dtype
    return grid.__class__(shape, **d)


@register_sisl_dispatch(Grid, module="sisl")
def tile(grid: Grid, reps: int, axis: CellAxis) -> Grid:
    """Tile grid to create a bigger one

    The atomic indices for the base Geometry will be retained.

    Parameters
    ----------
    reps :
       number of tiles (repetitions)
    axis :
       direction of tiling, 0, 1, 2 according to the cell-direction

    Raises
    ------
    SislError : when the lattice is not commensurate with the geometry

    See Also
    --------
    Geometry.tile : equivalent method for Geometry class
    """
    if not grid._is_commensurate():
        raise SislError(
            f"{grid.__class__.__name__} cannot tile the grid since the contained"
            " Geometry and Lattice are not commensurate."
        )
    axis = direction(axis)
    out = grid.copy()
    out.grid = None
    reps_all = [1, 1, 1]
    reps_all[axis] = reps
    out.grid = np.tile(grid.grid, reps_all)
    lattice = grid.lattice.tile(reps, axis)
    if grid.geometry is not None:
        out.set_geometry(grid.geometry.tile(reps, axis))
    out.set_lattice(lattice)
    return out
