from __future__ import print_function

import numpy as np

from .sile import SileCDFSiesta
from ..sile import *

from sisl import Grid
from sisl.unit.siesta import unit_convert

__all__ = ['tsvncSileSiesta']

Bohr2Ang = unit_convert('Bohr', 'Ang')
eV2Ry = unit_convert('eV', 'Ry')


class tsvncSileSiesta(SileCDFSiesta):
    """ TranSiesta potential input Grid file object

    This potential input file is mainly intended for the Hartree solution
    which complements N-electrode calculations in TranSiesta.

    See Also
    --------
    Grid.topyamg : intrinsic grid conversion to the Poisson equation
    """

    def read_grid(self, *args, **kwargs):
        """ Reads the TranSiesta potential input grid """
        # Create the grid
        na = len(self._dimension('a'))
        nb = len(self._dimension('b'))
        nc = len(self._dimension('c'))

        v = self._variable('V')

        # Create the grid, Siesta uses periodic, always
        grid = Grid([nc, nb, na], bc=Grid.PERIODIC, dtype=v.dtype)

        grid.grid[:, :, :] = v[:, :, :] / eV2Ry

        # Read the grid, we want the z-axis to be the fastest
        # looping direction, hence x,y,z == 0,1,2
        return grid.swapaxes(0, 2)

    def write_grid(self, grid):
        """ Write the Poisson solution to the TSV.nc file """
        sile_raise_write(self)

        self._crt_dim(self, 'one', 1)
        self._crt_dim(self, 'a', grid.shape[0])
        self._crt_dim(self, 'b', grid.shape[1])
        self._crt_dim(self, 'c', grid.shape[2])

        vmin = self._crt_var(self, 'Vmin', 'f8', ('one',))
        vmin.info = 'Minimum value in the Poisson solution (for TranSiesta interpolation)'
        vmin.unit = 'Ry'
        vmax = self._crt_var(self, 'Vmax', 'f8', ('one',))
        vmax.info = 'Maximum value in the Poisson solution (for TranSiesta interpolation)'
        vmax.unit = 'Ry'

        v = self._crt_var(self, 'V', grid.dtype, ('c', 'b', 'a'))
        v.info = 'Poisson solution with custom boundary conditions'
        v.unit = 'Ry'

        vmin[:] = grid.grid.min() * eV2Ry
        vmax[:] = grid.grid.max() * eV2Ry
        v[:, :, :] = np.swapaxes(grid.grid, 0, 2) * eV2Ry


add_sile('TSV.nc', tsvncSileSiesta)
