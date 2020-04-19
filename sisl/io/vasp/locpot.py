from numbers import Integral
import numpy as np

from .sile import SileVASP
from ..sile import add_sile, sile_fh_open
from .car import carSileVASP

from sisl._internal import set_module
from sisl import Grid


__all__ = ['locpotSileVASP']


@set_module("sisl.io.vasp")
class locpotSileVASP(carSileVASP):
    """ Electrostatic (or total) potential plus geometry

    This file-object handles the electrostatic(total) potential from VASP
    """

    @sile_fh_open(True)
    def read_grid(self, index=0, dtype=np.float64):
        """ Reads the potential (in eV) from the file and returns with a grid (plus geometry)

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

        Returns
        -------
        Grid : potential with associated geometry
        """
        geom = self.read_geometry()
        V = geom.sc.volume

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
            val = vals[n * index:n * (index + 1)].reshape(nz, ny, nx)
        else:
            vals = vals[:n * max_index].reshape(-1, nz, ny, nx)
            val = vals[0] * index[0]
            for i, scale in enumerate(index[1:]):
                val += vals[i + 1] * scale
        del vals

        # Make it C-ordered with nx, ny, nz
        val = np.swapaxes(val, 0, 2) / V

        # Create the grid with data
        # Since we populate the grid data afterwards there
        # is no need to create a bigger grid than necessary.
        grid = Grid([1, 1, 1], dtype=dtype, geometry=geom)
        grid.grid = val

        return grid


add_sile('LOCPOT', locpotSileVASP, gzip=True)
