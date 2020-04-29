"""
Sile object for reading the force constant matrix written by GULP
"""

import numpy as np
from numpy import abs as np_abs
from scipy.sparse import lil_matrix

from sisl._internal import set_module
from .sile import SileGULP
from ..sile import *


__all__ = ['fcSileGULP']


@set_module("sisl.io.gulp")
class fcSileGULP(SileGULP):
    """ GULP output file object """

    @sile_fh_open()
    def read_force_constant(self, **kwargs):
        """ Returns a sparse matrix in coo format which contains the GULP force constant matrix.

        This routine expects the units to be in eV/Ang**2.

        Parameters
        ----------
        cutoff : float, optional
            absolute values below the cutoff are considered 0. Defaults to 0 eV/Ang**2.
        dtype: np.dtype (np.float64)
           default data-type of the matrix

        Returns
        -------
        FC : force constant in `scipy.sparse.coo_matrix` format
        """
        # Default cutoff
        cutoff = kwargs.get('cutoff', 0.)
        dtype = kwargs.get('dtype', np.float64)

        # Read number of atoms in the file...
        na = int(self.readline())
        no = na * 3

        fc = lil_matrix((no, no), dtype=dtype)

        # Reduce overhead...
        rl = self.fh.readline

        for ia in range(na):
            for ja in range(na):

                # read line that should contain:
                #  ia ja
                lsplit = rl().split()
                if int(lsplit[0]) != ia + 1 or int(lsplit[1]) != ja + 1:
                    raise ValueError("Inconsistent 2ND file data")

                # Read 3x3 data
                i = ia * 3
                for o in [0, 1, 2]:
                    fc[i+o, ja*3:(ja+1)*3] = list(map(float, rl().split()[:3]))

        # Convert to COO format
        fc = fc.tocoo()
        fc.data[np_abs(fc.data) < cutoff] = 0.
        fc.eliminate_zeros()

        return fc


add_sile('FORCE_CONSTANTS_2ND', fcSileGULP, gzip=True)
