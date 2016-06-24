"""
Sile object for reading/writing SIESTA Grid files
"""
from __future__ import print_function

import numpy as np

# Import sile objects
from .sile import NCSileSIESTA
from ..sile import *

# Import the geometry object
from sisl import Geometry, SuperCell, Grid
from sisl import Bohr

__all__ = ['SIESTAGridSile']


class SIESTAGridSile(NCSileSIESTA):
    """ SIESTA Grid file object """

    @Sile_fh_open
    def read_sc(self):
        """ Returns a SuperCell object from a SIESTA.grid.nc file
        """
        cell = np.array(self.variables['cell'][:], np.float64)
        # Yes, this is ugly, I really should implement my unit-conversion tool
        cell = cell / Bohr
        cell.shape = (3, 3)

        return SuperCell(cell)

    @Sile_fh_open
    def read_grid(self, name='gridfunc', idx=0, *args, **kwargs):
        """ Reads a grid in the current SIESTA.grid.nc file

        Enables the reading and processing of the grids created by SIESTA
        """
        # Swap as we swap back in the end
        sc = self.read_sc().swapaxes(0, 2)

        # Create the grid
        nx = len(self.dimensions['n1'])
        ny = len(self.dimensions['n2'])
        nz = len(self.dimensions['n3'])

        if name is None:
            v = self.variables['gridfunc']
        else:
            v = self.variables[name]

        # Create the grid, SIESTA uses periodic, always
        grid = Grid([nz, ny, nx], bc=Grid.Periodic, sc=sc,
                    dtype=v.dtype)

        if len(v[:].shape) == 3:
            grid.grid = v[:, :, :]
        else:
            grid.grid = v[idx, :, :, :]

        # Read the grid, we want the z-axis to be the fastest
        # looping direction, hence x,y,z == 0,1,2
        return grid.swapaxes(0, 2)


add_sile('grid.nc', SIESTAGridSile)
