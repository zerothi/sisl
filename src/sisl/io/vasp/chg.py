# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from numbers import Integral

import numpy as np

from sisl import Grid
from sisl._internal import set_module

from .._help import grid_reduce_indices
from ..sile import add_sile, sile_fh_open
from .car import carSileVASP
from .sile import SileVASP

__all__ = ["chgSileVASP"]


@set_module("sisl.io.vasp")
class chgSileVASP(carSileVASP):
    """Charge density plus geometry

    This file-object handles the charge-density from VASP
    """

    @sile_fh_open(True)
    def read_grid(self, index=0, dtype=np.float64, **kwargs):
        """Reads the charge density from the file and returns with a grid (plus geometry)

        Parameters
        ----------
        index : int or array_like, optional
           the index of the grid to read. For a spin-polarized VASP calculation 0 and 1 are
           allowed, UP/DOWN. For non-collinear 0, 1, 2 or 3 is allowed which equals,
           TOTAL, x, y, z charge density with the Cartesian directions equal to the charge
           magnetization. For array-like they refer to the fractional
           contributions for each corresponding index.
        dtype : numpy.dtype, optional
           grid stored dtype
        spin : optional
           same as `index` argument. `spin` argument has precedence.

        Returns
        -------
        Grid : charge density grid with associated geometry
        """
        index = kwargs.get("spin", index)
        geom = self.read_geometry()
        V = geom.lattice.volume

        rl = self.readline

        # Now we are past the cell and geometry
        # We can now read the size of CHGCAR
        rl()
        nx, ny, nz = list(map(int, rl().split()))
        n = nx * ny * nz

        is_index = True
        if isinstance(index, Integral):
            max_index = index + 1
        else:
            is_index = False
            max_index = len(index)

        vals = []
        vext = vals.extend

        is_chgcar = True
        i = 0
        while i < n * max_index:
            line = rl().split()
            # CHG: 10 columns, CHGCAR: 5 columns
            if is_chgcar and len(line) > 5:
                # we have a data line with more than 5 columns, must be a CHG file
                is_chgcar = False
            vext(line)
            i = len(vals)

            if i % n == 0 and i < n * max_index:
                if is_chgcar:
                    # Read over augmentation occupancies
                    line = rl()
                    while "augmentation" in line:
                        occ = int(line.split()[-1])
                        j = 0
                        while j < occ:
                            j += len(rl().split())
                        line = rl()
                # one line of nx, ny, nz
                rl()

        # Cut size before proceeding (otherwise it *may* fail)
        vals = np.array(vals).astype(dtype, copy=False)
        if is_index:
            val = vals[n * index : n * (index + 1)].reshape(nz, ny, nx)
        else:
            vals = vals[: n * max_index].reshape(-1, nz, ny, nx)
            val = grid_reduce_indices(vals, index, axis=0)
        del vals

        # Make it C-ordered with nx, ny, nz
        val = np.swapaxes(val, 0, 2) / V

        # Create the grid with data
        # Since we populate the grid data afterwards there
        # is no need to create a bigger grid than necessary.
        grid = Grid([1, 1, 1], dtype=dtype, geometry=geom)
        grid.grid = val

        return grid


# CHG has low-precision, so the user should prefer CHGCAR
add_sile("CHG", chgSileVASP, gzip=True)
add_sile("CHGCAR", chgSileVASP, gzip=True)
