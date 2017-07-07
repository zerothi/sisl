"""
Density matrix class
"""
from __future__ import print_function, division

import warnings
from numbers import Integral
import itertools as itools

import numpy as np
import scipy.linalg as sli
from scipy.sparse import isspmatrix, csr_matrix
import scipy.sparse.linalg as ssli

from sisl._help import get_dtype
from sisl._help import _zip as zip, _range as range
from sisl.sparse import SparseCSR, ispmatrix, ispmatrixd
from sisl.sparse_geometry import SparseOrbital
from .sparse_physics import SparseOrbitalBZSpin
from .spin import Spin
from .brillouinzone import BrillouinZone

__all__ = ['DensityMatrix']


class DensityMatrix(SparseOrbitalBZSpin):
    """ DensityMatrix object containing the density matrix elements

    The object contains information regarding the 
     - geometry
     - density matrix elements between orbitals

    Assigning or changing elements is as easy as with
    standard ``numpy`` assignments:

    >>> DM = DensityMatrix(...)
    >>> DM.D[1,2] = 0.1

    which assigns 0.1 as the density element between orbital 2 and 3.
    (remember that Python is 0-based elements).
    """

    def __init__(self, geom, dim=1, dtype=None, nnzpr=None, **kwargs):
        """Create DensityMatrix model from geometry

        Initializes a DensityMatrix using the ``geom`` object.
        """
        super(DensityMatrix, self).__init__(geom, dim, dtype, nnzpr, **kwargs)

        if self.spin.is_unpolarized:
            self.Dk = self._Pk_unpolarized
        elif self.spin.is_polarized:
            self.Dk = self._Pk_polarized
        elif self.spin.is_noncolinear:
            self.Dk = self._Pk_non_colinear
        elif self.spin.is_spinorbit:
            self.Dk = self._Pk_spin_orbit

    def Dk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the density matrix for a given k-point

        Creation and return of the density matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
          D(k) = D_{ij} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.

        Another possible gauge is the orbital distance which can be written as

        .. math::
          D(k) = D_{ij} e^{i k r}

        where :math:`r` is the distance between the orbitals :math:`i` and :math:`j`.
        Currently the second gauge is not implemented (yet).

        Parameters
        ----------
        k : array_like
           the k-point to setup the density matrix at
        dtype : numpy.dtype , optional 
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is '`numpy.complex128``
        gauge : {'R', 'r'}
           the chosen gauge, `R` for cell vector gauge, and `r` for orbital distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the ``scipy.sparse.csr_matrix``,
           however if one always requires operations on dense matrices, one can always
           return in ``numpy.ndarray`` (`'array'`) or ``numpy.matrix`` (`'dense'`).
        """
        pass

    def _get_D(self):
        self._def_dim = self.UP
        return self

    def _set_D(self, key, value):
        if len(key) == 2:
            self._def_dim = self.UP
        self[key] = value

    D = property(_get_D, _set_D)

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads density matrix from `Sile` using `read_density_matrix`.

        Parameters
        ----------
        sile : `Sile`, str
            a `Sile` object which will be used to read the density matrix
            and the overlap matrix (if any)
            if it is a string it will create a new sile using `get_sile`.
        * : args passed directly to ``read_density_matrix(,**)``
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_density_matrix(*args, **kwargs)
        else:
            return get_sile(sile).read_density_matrix(*args, **kwargs)

    def write(self, sile, *args, **kwargs):
        """ Writes a density matrix to the `Sile` as implemented in the :code:`Sile.write_density_matrix` method """
        self.finalize()

        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_density_matrix(self, *args, **kwargs)
        else:
            get_sile(sile, 'w').write_density_matrix(self, *args, **kwargs)
