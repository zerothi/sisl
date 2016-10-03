"""
Sile object for reading the Hessian matrix written by GULP
"""
from __future__ import print_function

# Import sile objects
from .sile import SileGULP
from ..sile import *

# Import the geometry object
from sisl import Geometry, Atom, SuperCell
from sisl.units import unit_convert

import numpy as np

__all__ = ['HessianSileGULP']


class HessianSileGULP(SileGULP):
    """ GULP output file object """

    @Sile_fh_open
    def read_dynmat(self, **kwargs):
        """ Returns a sparse matrix in coo format which contains the GULP
        Hessian matrix. 

        This routine expects the units to be in eV/Ang**2.

        Parameters
        ----------
        cutoff: float (0.001 eV/Ang**2)
           the cutoff of the force-constant matrix for adding to the matrix
        dtype: np.dtype (np.float64)
           default data-type of the matrix
        """
        # Default cutoff
        cutoff = kwargs.get('cutoff', 0.001)

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
                    dat[:] = [float(x) for x in rl().split()]
                    
                    # Assign data...
                    if dat[0] >= cutoff:
                        dyn[i+o, j  ] = dat[0]
                    if dat[1] >= cutoff:
                        dyn[i+o, j+1] = dat[1]
                    if dat[2] >= cutoff:
                        dyn[i+o, j+2] = dat[2]

                j += 3
            i += 3

        # Convert to COO format
        dyn = dyn.tocoo()

        return dyn


add_sile('FORCE_CONSTANTS_2ND', HessianSileGULP, gzip=True)
