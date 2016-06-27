"""
Sile object for reading the Hessian matrix written by GULP
"""
from __future__ import print_function

# Import sile objects
from .sile import SileGULP
from ..sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl import Ry, Bohr

import numpy as np

__all__ = ['GULPHessianSile']


class GULPHessianSile(SileGULP):
    """ GULP output file object """

    @Sile_fh_open
    def read_dynmat(self, **kwargs):
        """ Returns a sparse matrix in coo format which contains the GULP
        Hessian matrix. """

        dtype = kwargs.get('dtype', np.float64)

        # Read number of atoms in the file...
        na = int(self.readline())
        no = na * 3

        # Easier for creation of the sparsity pattern
        from scipy.sparse import lil_matrix

        dyn = lil_matrix((no, no), dtype=dtype)

        # Temporary container (to not de/alloc all the time)
        dat = np.empty([3], dtype=dtype)

        # Reduce overhead...
        rl = self.readline

        i = 0
        for ia in range(na):
            j = 0
            for ja in range(na):
                
                # read line that should contain:
                #  ia ja
                I, J = map(int, rl().split())
                if I != ia + 1 or J != ja + 1:
                    raise ValueError("Inconsistent 2ND file data")

                # Read data
                for o in [0, 1, 2]:
                    dat[:] = map(float, rl().split())
                    
                    # Assign data...
                    dyn[i+o, j  ] = dat[0]
                    dyn[i+o, j+1] = dat[1]
                    dyn[i+o, j+2] = dat[2]

                j += 3
            i += 3

        # Convert to COO format
        dyn = dyn.tocoo()
        
        # Convert the 2ND data to standard units
        dyn.data[:] /= Ry / Bohr ** 2

        return dyn


add_sile('FORCE_CONSTANTS_2ND', GULPHessianSile, gzip=True)
