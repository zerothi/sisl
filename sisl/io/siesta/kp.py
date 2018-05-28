from __future__ import print_function

import numpy as np

# Import sile objects
from ..sile import add_sile, Sile_fh_open, sile_raise_write
from .sile import *


__all__ = ['kpSileSiesta']


class kpSileSiesta(SileSiesta):
    """ k-point Siesta file object """

    @Sile_fh_open
    def read_data(self, sc=None):
        """ Returns K-points from the file (note that these are in reciprocal units)

        Parameters
        ----------
        sc : SuperCellChild
           if supplied the returned k-points will be in reduced coordinates

        Returns
        -------
        k : k-points
        w : weights for k-points
        """
        nk = int(self.readline())

        k = np.empty([nk, 3], np.float64)
        w = np.empty([nk], np.float64)
        for ik in range(nk):
            l = self.readline().split()
            k[ik, :] = float(l[1]), float(l[2]), float(l[3])
            w[ik] = float(l[4])

        if sc is None:
            return k, w
        return np.dot(k, sc.cell.T / (2 * np.pi)), w

    @Sile_fh_open
    def write_data(self, bz, fmt='%.9e'):
        """ Writes K-points to file

        Parameters
        ----------
        bz : BrillouinZone
           object contain all weights and k-points
        """
        sile_raise_write(self)

        nk = len(bz)
        self._write('{}\n'.format(nk))
        _fmt = ('%d' + (' ' + fmt) * 4) + '\n'

        for i, k in enumerate(bz):
            self._write(_fmt % (i + 1, k[0], k[1], k[2], bz.weight[i]))

add_sile('KP', kpSileSiesta, case=False, gzip=True)
