"""
Tight-binding class to create tight-binding models.
"""
from __future__ import print_function, division

from numbers import Integral

import numpy as np


from sids import Atom, Geometry, Quaternion
from .tb import TightBinding


__all__ = ['PhononTightBinding']


class PhononTightBinding(TightBinding):
    """ Phonon tight-binding model with slight modifications """

    Energy = TightBinding.Energy ** 2

    def correct_Newton(self):
        """
        Sometimes the dynamical matrix does not obey Newtons laws.
        
        We correct the dynamical matrix by imposing zero force.
        
        Correcting for Newton forces the matrix to be finalized.
        """
        from scipy.sparse import lil_matrix

        # Create UC dynamical matrix
        d_sc, S_sc = self.tocsr() ; del S_sc
        d_sc = d_sc.tocoo()
        d_uc = lil_matrix( (self.no,self.no) , dtype=d_sc.dtype)

        # Convert SC to UC
        for j, i, d in zip(d_sc.row,d_sc.col,d_sc.data):
            d_uc[j, i % self.no] += d
        del d_sc
        d_uc = d_uc.tocsc()

        # we need to correct the dynamical matrix found in GULP
        # This ensures that Newtons laws are obeyed, (i.e. 
        # action == re-action)
        om = np.sqrt(np.array([a.mass for a in self.atoms],np.float64))
        MM = np.empty([len(om)],np.float64)
        r3 = range(3)

        for ja in range(self.na):

            # Create conversion to force-constant, and revert back
            # after correcting
            MM[:] = om[:] / om[ja]
            jo = ja * 3

            for j in r3:
                for i in r3:
                    D, S = self[jo+j,jo+i]
                    self[jo+j,jo+i] = D - d_uc[jo+j,i::3].multiply(MM).sum(), S

        del d_uc

        
if __name__ == "__main__":
    pass
