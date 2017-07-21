"""
Tight-binding class to create tight-binding models.
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

__all__ = ['Hamiltonian', 'TightBinding']


class Hamiltonian(SparseOrbitalBZSpin):
    """ Hamiltonian object containing the coupling constants between orbitals.

    The Hamiltonian object contains information regarding the 
     - geometry
     - coupling constants between orbitals

    It contains an intrinsic sparse matrix of the Hamiltonian elements.

    Assigning or changing Hamiltonian elements is as easy as with
    standard ``numpy`` assignments:

    >>> ham = Hamiltonian(...)
    >>> ham.H[1,2] = 0.1

    which assigns 0.1 as the coupling constant between orbital 2 and 3.
    (remember that Python is 0-based elements).
    """

    def __init__(self, geom, dim=1, dtype=None, nnzpr=None, **kwargs):
        """Create Hamiltonian model from geometry

        Initializes a Hamiltonian using the ``geom`` object
        as the underlying geometry for the tight-binding parameters.
        """
        super(Hamiltonian, self).__init__(geom, dim, dtype, nnzpr, **kwargs)

        self.Hk = self.Pk

    def Hk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the Hamiltonian for a given k-point

        Creation and return of the Hamiltonian for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
          H(k) = H_{ij} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.

        Another possible gauge is the orbital distance which can be written as

        .. math::
          H(k) = H_{ij} e^{i k r}

        where :math:`r` is the distance between the orbitals :math:`i` and :math:`j`.
        Currently the second gauge is not implemented (yet).

        Parameters
        ----------
        k : array_like
           the k-point to setup the Hamiltonian at
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

        See Also
        --------
        Sk : Overlap matrix at `k`
        """
        pass

    def _get_H(self):
        self._def_dim = self.UP
        return self

    def _set_H(self, key, value):
        if len(key) == 2:
            self._def_dim = self.UP
        self[key] = value

    H = property(_get_H, _set_H)

    def shift(self, E):
        """ Shift the electronic structure by a constant energy

        Parameters
        ----------
        E : float
           the energy (in eV) to shift the electronic structure
        """
        if not self.orthogonal:
            # For non-colinear and SO only the diagonal components
            # should be shifted.
            for i in range(min(self.spin.spin, 2)):
                self._csr._D[:, i] -= self._csr._D[:, self.S_idx] * E
        else:
            for i in range(self.shape[0]):
                for j in range(min(self.spin.spin, 2)):
                    self[i, i, j] = self[i, i, j] - E

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads Hamiltonian from `Sile` using `read_hamiltonian`.

        Parameters
        ----------
        sile : `Sile`, str
            a `Sile` object which will be used to read the Hamiltonian
            and the overlap matrix (if any)
            if it is a string it will create a new sile using `get_sile`.
        * : args passed directly to ``read_hamiltonian(,**)``
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_hamiltonian(*args, **kwargs)
        else:
            return get_sile(sile).read_hamiltonian(*args, **kwargs)

    def write(self, sile, *args, **kwargs):
        """ Writes a tight-binding model to the `Sile` as implemented in the :code:`Sile.write_hamiltonian` method """
        self.finalize()

        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_hamiltonian(self, *args, **kwargs)
        else:
            get_sile(sile, 'w').write_hamiltonian(self, *args, **kwargs)

# For backwards compatibility we also use TightBinding
# NOTE: that this is not sub-classed...
TightBinding = Hamiltonian
