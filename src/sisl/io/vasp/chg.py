# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from numbers import Integral

import numpy as np

from sisl import Grid
from sisl._internal import set_module

from .._help import grid_reduce_indices
from ..sile import add_sile, sile_fh_open
from .car import carSileVASP

__all__ = ["chgSileVASP"]


@set_module("sisl.io.vasp")
class chgSileVASP(carSileVASP):
    """Charge density plus geometry

    This file-object handles the charge-density from VASP
    """

    @sile_fh_open(True)
    def read_grid(self, index=0, dtype=np.float64, **kwargs) -> Grid:
        r"""Reads the charge density from the file and returns with a grid (plus geometry)

        Parameters
        ----------
        index : int or array_like, optional
           the index of the grid to read.
           For spin-polarized calculations, 0 and 1 refer to the charge (spin-up plus spin-down) and
           magnetitization (spin-up minus spin-down), respectively.
           For non-collinear calculations, 0 refers to the charge while 1, 2 and 3 to
           the magnetization in the :math:`\sigma_1`, :math:`\sigma_2`, and :math:`\sigma_3` directions, respectively.
           The directions are related via the VASP input option ``SAXIS``.
           TOTAL, x, y, z charge density with the Cartesian directions equal to the charge
           magnetization.
           For array-like they refer to the fractional contributions for each corresponding index.
        dtype : numpy.dtype, optional
           grid stored dtype
        spin : optional
           same as `index` argument. `spin` argument has precedence.

        Examples
        --------
        Read the spin polarization from a spin-polarized CHGCAR file

        >>> fh = sisl.get_sile('CHGCAR')
        >>> charge = fh.read_grid()
        >>> spin = fh.read_grid(1)
        >>> up_density = fh.read_grid([0.5, 0.5])
        >>> assert np.allclose((charge + spin).grid / 2, up_density.grid)
        >>> down_density = fh.read_grid([0.5, -0.5])
        >>> assert np.allclose((charge - spin).grid / 2, down_density.grid)

        Returns
        -------
        Grid
            charge density grid with associated geometry
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
            line = rl()
            if line == "":
                raise ValueError(
                    f"{self.__class__.__name__}.read_grid cannot find requested index in {self!r}"
                )
            line = line.split()
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
                    # read over an additional block with geom.na entries???
                    j = len(line.split())
                    while j < geom.na:
                        j += len(rl().split())

                # one line of nx, ny, nz
                assert np.allclose(list(map(int, rl().split())), [nx, ny, nz])

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
