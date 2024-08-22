# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from numbers import Integral

import numpy as np

from sisl import Grid
from sisl._internal import set_module
from sisl.typing import UnitsVar
from sisl.unit import serialize_units_arg, unit_convert

from .._help import grid_reduce_indices
from ..sile import add_sile, sile_fh_open
from .car import carSileVASP

__all__ = ["locpotSileVASP"]


@set_module("sisl.io.vasp")
class locpotSileVASP(carSileVASP):
    """Electrostatic (or total) potential plus geometry

    This file-object handles the electrostatic(total) potential from VASP
    """

    @sile_fh_open(True)
    def read_grid(
        self, index=0, dtype=np.float64, units: UnitsVar = "eV", **kwargs
    ) -> Grid:
        """Reads the potential from the file and returns with a grid (plus geometry)

        Parameters
        ----------
        index : int or array_like, optional
           the index of the potential to read. For a spin-polarized VASP calculation 0 and 1 are
           allowed, UP/DOWN. For non-collinear 0, 1, 2 or 3 is allowed which equals,
           TOTAL, x, y, z total potential with the Cartesian directions equal to the potential
           for the magnetization directions. For array-like they refer to the fractional
           contributions for each corresponding index.
        dtype : numpy.dtype, optional
           grid stored dtype
        units :
           selects units in the returned data
        spin : optional
           same as `index` argument. `spin` argument has precedence.

        Returns
        -------
        Grid
            potential with associated geometry
        """
        units = serialize_units_arg(units)
        eV2unit = unit_convert("eV", units["energy"])

        index = kwargs.get("spin", index)
        geom = self.read_geometry()
        V = geom.lattice.volume

        # Now we are past the cell and geometry
        # We can now read the size of LOCPOT
        self.readline()
        nx, ny, nz = list(map(int, self.readline().split()))
        n = nx * ny * nz

        is_index = True
        if isinstance(index, Integral):
            max_index = index + 1
        else:
            is_index = False
            max_index = len(index)

        rl = self.readline
        vals = []
        vapp = vals.append

        i = 0
        while i < n * max_index:
            dat = [l for l in rl().split()]
            vapp(dat)
            i += len(dat)

            if i % n == 0 and i < n * max_index:
                # Each time a new spin-index is present, we need to read the coordinates
                j = 0
                while j < geom.na:
                    j += len(rl().split())

                # one line of nx, ny, nz
                rl()

        # Cut size before proceeding (otherwise it *may* fail)
        vals = np.array(vals).astype(dtype).ravel()
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
        grid.grid = val * eV2unit

        return grid


add_sile("LOCPOT", locpotSileVASP, gzip=True)
