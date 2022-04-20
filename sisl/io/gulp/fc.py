# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Sile object for reading the force constant matrix written by GULP
"""

import numpy as np
from scipy.sparse import lil_matrix

from ..sile import add_sile, sile_fh_open
from sisl._internal import set_module
from .sile import SileGULP


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
        tmp = np.empty([3, na, 3], dtype=dtype)

        # Reduce overhead...
        rl = self.fh.readline

        for ia in range(na):
            i = ia * 3
            for ja in range(na):

                # read line that should contain:
                #  ia ja
                lsplit = rl().split()
                assert int(lsplit[0]) == ia + 1, "Inconsistent 2ND file data"
                assert int(lsplit[1]) == ja + 1, "Inconsistent 2ND file data"

                # Read 3x3 data
                tmp[0, ja, :] = [float(x) for x in rl().split()]
                tmp[1, ja, :] = [float(x) for x in rl().split()]
                tmp[2, ja, :] = [float(x) for x in rl().split()]

            # much faster assignment
            fc[i, :] = tmp[0].ravel()
            fc[i+1, :] = tmp[1].ravel()
            fc[i+2, :] = tmp[2].ravel()

        # Convert to COO format
        fc = fc.tocoo()
        fc.data[np.fabs(fc.data) < cutoff] = 0.
        fc.eliminate_zeros()

        return fc


add_sile('FORCE_CONSTANTS_2ND', fcSileGULP, gzip=True)
