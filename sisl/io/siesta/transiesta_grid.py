import numpy as np

from .sile import SileCDFSiesta
from ..sile import add_sile, sile_fh_open, sile_raise_write

from sisl._internal import set_module
from sisl import Grid
from .siesta_grid import gridncSileSiesta
from sisl.unit.siesta import unit_convert


__all__ = ['tsvncSileSiesta']


_eV2Ry = unit_convert('eV', 'Ry')
_Ry2eV = 1. / _eV2Ry


@set_module("sisl.io.siesta")
class tsvncSileSiesta(gridncSileSiesta):
    """ TranSiesta potential input Grid file object

    This potential input file is mainly intended for the Hartree solution
    which complements N-electrode calculations in TranSiesta.

    See Also
    --------
    Grid.topyamg : intrinsic grid conversion to the Poisson equation
    """

    def read_grid(self, *args, **kwargs):
        """ Reads the TranSiesta potential input grid """
        sc = self.read_supercell().swapaxes(0, 2)

        # Create the grid
        na = len(self._dimension('a'))
        nb = len(self._dimension('b'))
        nc = len(self._dimension('c'))

        v = self._variable('V')

        # Create the grid, Siesta uses periodic, always
        grid = Grid([nc, nb, na], bc=Grid.PERIODIC, sc=sc, dtype=v.dtype)

        grid.grid[:, :, :] = v[:, :, :] * _Ry2eV

        # Read the grid, we want the z-axis to be the fastest
        # looping direction, hence x,y,z == 0,1,2
        return grid.swapaxes(0, 2)

    def write_grid(self, grid):
        """ Write the Poisson solution to the TSV.nc file """
        sile_raise_write(self)

        self.write_supercell(grid.sc)

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

        vmin[:] = grid.grid.min() * _eV2Ry
        vmax[:] = grid.grid.max() * _eV2Ry
        v[:, :, :] = np.swapaxes(grid.grid, 0, 2) * _eV2Ry


add_sile('TSV.nc', tsvncSileSiesta)
