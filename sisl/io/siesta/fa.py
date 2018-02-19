from __future__ import print_function

import numpy as np

# Import sile objects
from ..sile import add_sile, Sile_fh_open
from .sile import *


__all__ = ['faSileSiesta']


class faSileSiesta(SileSiesta):
    """ Force Siesta file object """

    @Sile_fh_open
    def read_data(self):
        """ Returns forces from the file """
        na = int(self.readline())

        f = np.empty([na, 3], np.float64)
        for ia in range(na):
            f[ia, :] = map(float, self.readline().split()[1:])

        # Units are already eV / Ang
        return f

    # Short-cut
    read_force = read_data

add_sile('FA', faSileSiesta, case=False, gzip=True)
add_sile('FAC', faSileSiesta, case=False, gzip=True)
