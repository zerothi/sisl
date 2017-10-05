from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import SileCDFSiesta
from ..sile import *

# Import the geometry object
from sisl import SuperCell, Grid
from sisl.unit.siesta import unit_convert

__all__ = ['gridncSileSiesta']

Bohr2Ang = unit_convert('Bohr', 'Ang')


class gridncSileSiesta(SileCDFSiesta):
    """ Siesta Grid file object """

    def read_supercell(self):
        """ Returns a SuperCell object from a Siesta.grid.nc file
        """
        cell = np.array(self._value('cell'), np.float64)
        # Yes, this is ugly, I really should implement my unit-conversion tool
        cell *= Bohr2Ang
        cell.shape = (3, 3)

        return SuperCell(cell)

    def write_supercell(self, sc):
        """ Write a supercell to the grid.nc file """
        sile_raise_write(self)

        # Create initial dimensions
        self._crt_dim(self, 'xyz', 3)
        self._crt_dim(self, 'abc', 3)

        v = self._crt_var(self, 'cell', 'f8', ('abc', 'xyz'))
        v.info = 'Unit cell'
        v.unit = 'Bohr'
        v[:, :] = sc.cell[:, :] / Bohr2Ang

    def read_grid(self, name='gridfunc', idx=0, *args, **kwargs):
        """ Reads a grid in the current Siesta.grid.nc file

        Enables the reading and processing of the grids created by Siesta
        """
        # Swap as we swap back in the end
        sc = self.read_supercell().swapaxes(0, 2)

        # Create the grid
        nx = len(self._dimension('n1'))
        ny = len(self._dimension('n2'))
        nz = len(self._dimension('n3'))

        if name is None:
            v = self._variable('gridfunc')
        else:
            v = self._variable(name)

        # Create the grid, Siesta uses periodic, always
        grid = Grid([nz, ny, nx], bc=Grid.PERIODIC, sc=sc,
                    dtype=v.dtype)

        if len(v.shape) == 3:
            grid.grid[:, :, :] = v[:, :, :]
        else:
            grid.grid[:, :, :] = v[idx, :, :, :]

        # Read the grid, we want the z-axis to be the fastest
        # looping direction, hence x,y,z == 0,1,2
        return grid.swapaxes(0, 2)

    def write_grid(self, grid, ispin=0, nspin=None):
        """ Write a grid to the grid.nc file """
        sile_raise_write(self)

        self.write_supercell(grid.sc)

        if nspin is not None:
            self._crt_dim(self, 'spin', nspin)

        self._crt_dim(self, 'n1', grid.shape[0])
        self._crt_dim(self, 'n2', grid.shape[1])
        self._crt_dim(self, 'n3', grid.shape[2])

        if nspin is None:
            v = self._crt_var(self, 'gridfunc', 'f4', ('n3', 'n2', 'n1'))
        else:
            v = self._crt_var(self, 'gridfunc', 'f4', ('spin', 'n3', 'n2', 'n1'))
        v.info = 'Grid function'

        if nspin is None:
            v[:, :, :] = np.swapaxes(grid.grid, 0, 2)
        else:
            v[ispin, :, :, :] = np.swapaxes(grid.grid, 0, 2)


add_sile('grid.nc', gridncSileSiesta)
