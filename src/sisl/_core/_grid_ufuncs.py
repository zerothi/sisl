# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from sisl._ufuncs import register_sisl_dispatch
from sisl.messages import SislError

from .grid import Grid

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(module="sisl")
def tile(grid: Grid, reps: int, axis: int):
    """Tile grid to create a bigger one

    The atomic indices for the base Geometry will be retained.

    Parameters
    ----------
    grid : Grid
        the object to act on
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
