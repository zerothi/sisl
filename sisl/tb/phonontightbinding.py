"""
Tight-binding class to create tight-binding models.
"""
from __future__ import print_function, division

from numbers import Integral

import numpy as np


from sisl import Atom, Geometry, Quaternion
from .tightbinding import TightBinding


__all__ = ['PhononTightBinding']


class PhononTightBinding(TightBinding):
    """ Phonon tight-binding model with slight modifications """

    # The order of the Energy
    # I.e. whether energy should be in other units than Ry
    # This conversion is made: [eV] ** _E_order
    _E_order = 2

    def correct_Newton(self):
        """
        Sometimes the dynamical matrix does not obey Newtons laws.

        We correct the dynamical matrix by imposing zero force.

        Correcting for Newton forces the matrix to be finalized.
        """
        from scipy.sparse import lil_matrix

        # Create UC dynamical matrix
        d_sc, S_sc = self.tocsr()
        del S_sc
        d_sc = d_sc.tocoo()
        d_uc = lil_matrix((self.no, self.no), dtype=d_sc.dtype)

        # Convert SC to UC
        for j, i, d in zip(d_sc.row, d_sc.col, d_sc.data):
            d_uc[j, i % self.no] += d
        del d_sc
        d_uc = d_uc.tocsc()

        # we need to correct the dynamical matrix found in GULP
        # This ensures that Newtons laws are obeyed, (i.e.
        # action == re-action)
        om = np.sqrt(np.array([a.mass for a in self.atoms], np.float64))
        MM = np.empty([len(om)], np.float64)

        for ja in self.geom:

            # Create conversion to force-constant, and revert back
            # after correcting
            MM[:] = om[:] / om[ja]
            jo = ja * 3

            # Unroll...
            D, S = self[jo, jo]
            self[jo, jo] = D - d_uc[jo, ::3].multiply(MM).sum(), S
            D, S = self[jo, jo + 1]
            self[jo, jo + 1] = D - d_uc[jo, 1::3].multiply(MM).sum(), S
            D, S = self[jo, jo + 2]
            self[jo, jo + 2] = D - d_uc[jo, 2::3].multiply(MM).sum(), S

            D, S = self[jo + 1, jo]
            self[jo + 1, jo] = D - d_uc[jo + 1, ::3].multiply(MM).sum(), S
            D, S = self[jo + 1, jo + 1]
            self[jo + 1, jo + 1] = D - d_uc[jo + 1, 1::3].multiply(MM).sum(), S
            D, S = self[jo + 1, jo + 2]
            self[jo + 1, jo + 2] = D - d_uc[jo + 1, 2::3].multiply(MM).sum(), S

            D, S = self[jo + 2, jo]
            self[jo + 2, jo] = D - d_uc[jo + 2, ::3].multiply(MM).sum(), S
            D, S = self[jo + 2, jo + 1]
            self[jo + 2, jo + 1] = D - d_uc[jo + 2, 1::3].multiply(MM).sum(), S
            D, S = self[jo + 2, jo + 2]
            self[jo + 2, jo + 2] = D - d_uc[jo + 2, 2::3].multiply(MM).sum(), S

        del d_uc


if __name__ == "__main__":
    pass
