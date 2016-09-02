"""
Dynamical matrix.
"""
from __future__ import print_function, division

import numpy as np

from sisl.quantity import Hamiltonian

__all__ = ['DynamicalMatrix']


class DynamicalMatrix(Hamiltonian):
    """ Dynamical matrix of a geometry """

    # The order of the Energy
    # I.e. whether energy should be in other units than Ry
    # This conversion is made: [eV] ** _E_order
    _E_order = 2

    D = property(Hamiltonian._get_H, Hamiltonian._set_H)
    Dk = Hamiltonian.Hk

    def correct_Newton(self):
        """
        Sometimes the dynamical matrix does not obey Newtons laws.

        We correct the dynamical matrix by imposing zero force.

        Correcting for Newton forces the matrix to be finalized.
        """
        from scipy.sparse import lil_matrix

        # Create UC dynamical matrix
        dyn_sc = self.tocsr(0)
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
        om = np.sqrt(self.mass)
        MM = np.empty([len(om)], np.float64)

        for ja in self.geom:

            # Create conversion to force-constant, and revert back
            # after correcting
            MM[:] = om[:] / om[ja]
            jo = ja * 3

            # Unroll...
            D = self.D[jo, jo]
            self.D[jo, jo] = D - d_uc[jo, ::3].multiply(MM).sum()
            D = self.D[jo, jo + 1]
            self.D[jo, jo + 1] = D - d_uc[jo, 1::3].multiply(MM).sum()
            D = self.D[jo, jo + 2]
            self.D[jo, jo + 2] = D - d_uc[jo, 2::3].multiply(MM).sum()

            D = self.D[jo + 1, jo]
            self.D[jo + 1, jo] = D - d_uc[jo + 1, ::3].multiply(MM).sum()
            D = self.D[jo + 1, jo + 1]
            self.D[jo + 1, jo + 1] = D - d_uc[jo + 1, 1::3].multiply(MM).sum()
            D = self.D[jo + 1, jo + 2]
            self.D[jo + 1, jo + 2] = D - d_uc[jo + 1, 2::3].multiply(MM).sum()

            D = self.D[jo + 2, jo]
            self.D[jo + 2, jo] = D - d_uc[jo + 2, ::3].multiply(MM).sum()
            D = self.D[jo + 2, jo + 1]
            self.D[jo + 2, jo + 1] = D - d_uc[jo + 2, 1::3].multiply(MM).sum()
            D = self.D[jo + 2, jo + 2]
            self.D[jo + 2, jo + 2] = D - d_uc[jo + 2, 2::3].multiply(MM).sum()

        del d_uc
