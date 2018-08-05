"""
Sile object for reading the force constant matrix written by GULP
"""
from __future__ import print_function, division

import numpy as np
from numpy import abs as np_abs
from scipy.sparse import lil_matrix

from .sile import SileGULP
from ..sile import *


__all__ = ['fcSileGULP']


class fcSileGULP(SileGULP):
    """ GULP output file object """

    @sile_fh_open()
    def read_force_constant(self, **kwargs):
        """ Returns a sparse matrix in coo format which contains the GULP force constant matrix.

        This routine expects the units to be in eV/Ang**2.

        Parameters
        ----------
        cutoff : float, optional
            absolute values below the cutoff are considered 0. Defaults to 1e-4 eV/Ang**2.
        dtype: np.dtype (np.float64)
           default data-type of the matrix
        """
        # Default cutoff
        cutoff = kwargs.get('cutoff', 1e-4)
        dtype = kwargs.get('dtype', np.float64)

        # Read number of atoms in the file...
        na = int(self.readline())
        no = na * 3

        fc = lil_matrix((no, no), dtype=dtype)

        # Reduce overhead...
        rl = self.fh.readline

        i = 0
        for ia in range(na):
            j = 0
            for ja in range(na):

                # read line that should contain:
                #  ia ja
                lsplit = rl().split()
                if int(lsplit[0]) != ia + 1 or int(lsplit[1]) != ja + 1:
                    raise ValueError("Inconsistent 2ND file data")

                # Read 3x3 data
                for o in [0, 1, 2]:
                    ii = i + o
                    lsplit = rl().split()
                    for oo in [0, 1, 2]:
                        fc[ii, j+oo] = float(lsplit[oo])

                j += 3
            i += 3

        # Convert to COO format
        fc = fc.tocoo()
        fc.data[np_abs(fc.data) < cutoff] = 0.
        fc.eliminate_zeros()

        return fc


add_sile('FORCE_CONSTANTS_2ND', fcSileGULP, gzip=True)
