from __future__ import print_function, division

import numpy as np

from .sile import SileVASP
from ..sile import *
from .car import carSileVASP

from sisl import Grid


__all__ = ['locpotSileVASP']


class locpotSileVASP(carSileVASP):
    """ Electrostatic (or total) potential plus geometry

    This file-object handles the electrostatic(total) potential from VASP
    """

    @sile_fh_open(True)
    def read_grid(self, index=0, dtype=np.float64):
        """ Reads the potential (in eV) from the file and returns with a grid (plus geometry)

        Parameters
        ----------
        index : int, optional
           the index of the potential to read. For a spin-polarized VASP calculation 0 and 1 are
           allowed, UP/DOWN. For non-collinear 0, 1, 2 or 3 is allowed which equals,
           TOTAL, x, y, z total potential with the Cartesian directions equal to the potential
           for the magnetization directions.
        dtype : numpy.dtype, optional
           grid stored dtype

        Returns
        -------
        Grid : potential with associated geometry
        """
        geom = self.read_geometry()
        V = geom.sc.volume

        # Now we are past the cell and geometry
        # We can now read the size of CHGCAR
        self.readline()
        nx, ny, nz = list(map(int, self.readline().split()))
        n = nx * ny * nz

        i = 0
        vals = []
        while i < n * (index + 1):
            dat = [float(l) for l in self.readline().split()]
            vals.append(dat)
            i += len(dat)
        vals = np.swapaxes(np.array(vals).reshape(nz, ny, nx), 0, 2)
        vals = vals[index * n:(index + 1) * n] / V

        # Create the grid with data
        # Since we populate the grid data afterwards there
        # is no need to create a bigger grid than necessary.
        grid = Grid([1, 1, 1], dtype=dtype, geometry=geom)
        grid.grid = vals

        return grid


add_sile('LOCPOT', locpotSileVASP, gzip=True)
