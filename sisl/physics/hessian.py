"""
Dynamical matrix.
"""
from __future__ import print_function, division

import numpy as np

from .sparse_physics import SparseOrbitalBZ
from sisl._help import _zip as zip

__all__ = ['Hessian', 'DynamicalMatrix']


class Hessian(SparseOrbitalBZ):
    """ Dynamical matrix of a geometry """

    # The order of the Energy
    # I.e. whether energy should be in other units than Ry
    # This conversion is made: [eV] ** _E_order
    _E_order = 2

    def __init__(self, geom, dim=1, dtype=None, nnzpr=None, **kwargs):
        """ Initializes the dynamical matrix from a geometry """
        super(Hessian, self).__init__(geom, dim, dtype, nnzpr, **kwargs)

        self.Dk = self._Pk

    def _get_D(self):
        self._def_dim = 0
        return self

    def _set_D(self, key, value):
        if len(key) == 2:
            self._def_dim = 0
        self[key] = value

    D = property(_get_D, _set_D)

    def correct_Newton(self):
        """
        Sometimes the dynamical matrix does not obey Newtons laws.

        We correct the dynamical matrix by imposing zero force.

        Correcting for Newton forces the matrix to be finalized.
        """
        from scipy.sparse import lil_matrix

        # Create UC dynamical matrix
        dyn_sc = self.tocsr(0)
        no = self.no
        d_uc = lil_matrix((no, no), dtype=dyn_sc.dtype)

        for i, _ in self.sc:
            d_uc[:, :] += dyn_sc[:, i*no: (i+1)*no]

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

DynamicalMatrix = Hessian
