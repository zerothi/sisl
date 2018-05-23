from __future__ import print_function

import numpy as np

# Import sile objects
from ..sile import add_sile, Sile_fh_open, sile_raise_write
from .sile import *


__all__ = ['kpSileSiesta']


class kpSileSiesta(SileSiesta):
    """ k-point Siesta file object """

    @Sile_fh_open
    def read_data(self):
        """ Returns K-points from the file (note that these are in reciprocal units) """
        nk = int(self.readline())

        k = np.empty([nk, 3], np.float64)
        for ik in range(nk):
            k[ik, :] = list(map(float, self.readline().split()[1:]))

        return k

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

        for i, k in enumerate(bz):
            self._write(('%d' + (' ' + fmt) * 4) % (i + 1, k[0], k[1], k[2], bz.weight[i]) + '\n')

add_sile('KP', kpSileSiesta, case=False, gzip=True)
