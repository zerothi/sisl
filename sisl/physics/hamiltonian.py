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
from .brillouinzone import BrillouinZone

__all__ = ['Hamiltonian', 'TightBinding']


class Hamiltonian(SparseOrbital):
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

    # The order of the Energy
    # I.e. whether energy should be in other units than Ry
    # This conversion is made: [eV] ** _E_order
    _E_order = 1

    def __init__(self, geom, dim=1, dtype=None, nnzpr=None, **kwargs):
        """Create Hamiltonian model from geometry

        Initializes a Hamiltonian using the ``geom`` object
        as the underlying geometry for the tight-binding parameters.
        """
        self._geom = geom

        # Check that the passed parameters are correct
        dim = kwargs.get('spin', dim)
        self._orthogonal = kwargs.get('orthogonal', True)
        if not self.orthogonal:
            dim = dim + 1

        # Initialize the sparsity pattern
        self.reset(dim, dtype, nnzpr)

        if self.orthogonal:
            # Wrapper for always enabling creating an overlap
            # matrix. For orthogonal cases it is simply the diagonal
            # matrix
            def diagonal_Sk(self, k=(0, 0, 0), dtype=None, **kwargs):
                """ For an orthogonal case we always return the identity matrix """
                if dtype is None:
                    dtype = np.float64
                S = csr_matrix((len(self), len(self)), dtype=dtype)
                S.setdiag(1.)
                return S
            self.Sk = diagonal_Sk
            self.S_idx = -1
        else:
            dim = dim - 1
            self.S_idx = dim

        if dim == 1:
            self.UP = 0
            self.DOWN = 0
            self.Hk = self._Hk_unpolarized
            self.Sk = self._Sk
        elif dim == 2:
            self.UP = 0
            self.DOWN = 1
            self.Hk = self._Hk_polarized
            self.Sk = self._Sk
        elif dim == 4:
            self.Hk = self._Hk_non_collinear
            self.Sk = self._Sk_non_collinear
        elif dim == 8:
            self.Hk = self._Hk_spin_orbit
            # The overlap is the same as non-collinear
            self.Sk = self._Sk_non_collinear

    # Override to enable spin configuration and orthogonality
    def _cls_kwargs(self):
        return {'spin': self.spin, 'orthogonal': self.orthogonal}

    # We define this function _ONLY_ for the docstring
    # it provides.
    def Hk(self, k=(0, 0, 0), dtype=None, gauge='R', *args, **kwargs):
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
        """
        pass

    @property
    def spin(self):
        """ Number of spin-components in Hamiltonian """
        if self.orthogonal:
            return self.dim
        return self.dim - 1

    @property
    def orthogonal(self):
        """ Return whether the Hamiltonian is orthogonal """
        return self._orthogonal

    def __len__(self):
        """ Returns number of rows in the Hamiltonian """
        if self.spin > 2:
            return self.no * 2
        return self.no

    def __repr__(self):
        """ Representation of the tight-binding model """
        s = '{{spin: {0}, non-zero: {1}, orthogonal: {2}\n '.format(self.spin, self.nnz, self.orthogonal)
        s += repr(self.geom).replace('\n', '\n ')
        return s + '\n}'

    def _get_H(self):
        self._def_dim = self.UP
        return self

    def _set_H(self, key, value):
        if len(key) == 2:
            self._def_dim = self.UP
        self[key] = value

    H = property(_get_H, _set_H)

    def _get_S(self):
        if self.orthogonal:
            return None
        self._def_dim = self.S_idx
        return self

    def _set_S(self, key, value):
        if self.orthogonal:
            return
        self._def_dim = self.S_idx
        self[key] = value

    S = property(_get_S, _set_S)

    @classmethod
    def fromsp(cls, geom, H, S=None):
        """ Read and return a Hamiltonian object with possible overlap """
        # Calculate maximum number of connections per row
        nc = 0

        # Ensure list of csr format (to get dimensions)
        if isspmatrix(H):
            H = [H]

        # Number of dimensions
        dim = len(H)
        # Sort all indices for the passed sparse matrices
        for i in range(dim):
            H[i] = H[i].tocsr()
            H[i].sort_indices()

        # Figure out the maximum connections per
        # row to reduce number of re-allocations to 0
        for i in range(H[0].shape[0]):
            nc = max(nc, H[0][i, :].getnnz())

        # Create the sparse object
        h = cls(geom, dim, H[0].dtype, nc, orthogonal=S is None)

        for i in range(dim):
            for jo, io, v in ispmatrixd(H[i]):
                h[jo, io, i] = v

        if not S is None:
            for jo, io, v in ispmatrixd(S):
                h.S[jo, io, ] = v

        return h

    # Create iterations on entire set of orbitals
    def iter(self, local=False):
        """ Iterations of the orbital space in the geometry, two indices from loop

        An iterator returning the current atomic index and the corresponding
        orbital index.

        >>> for ia, io in self:

        In the above case `io` always belongs to atom `ia` and `ia` may be
        repeated according to the number of orbitals associated with
        the atom `ia`.

        Parameters
        ----------
        local : `bool=False`
           whether the orbital index is the global index, or the local index relative to 
           the atom it resides on.
        """
        for ia, io in self.geom.iter_orbitals(local=local):
            yield ia, io

    __iter__ = iter

    def _Hk_unpolarized(self, k=(0, 0, 0), dtype=None, gauge='R'):
        """ Return the Hamiltonian in a ``scipy.sparse.csr_matrix`` at `k`.

        Parameters
        ----------
        k: ``array_like``, `[0,0,0]`
           k-point 
        dtype : ``numpy.dtype``
           default to `numpy.complex128`
        gauge : str, 'R'
           chosen gauge
        """
        return self._Hk_polarized(k, dtype=dtype, gauge=gauge)

    def _Hk_polarized(self, k=(0, 0, 0), spin=0, dtype=None, gauge='R'):
        """ Return the Hamiltonian in a ``scipy.sparse.csr_matrix`` at `k` for a polarized calculation

        Parameters
        ----------
        k: ``array_like``, `[0,0,0]`
           k-point 
        spin: ``int``, `0`
           the spin-index of the Hamiltonian
        dtype : ``numpy.dtype``
           default to `numpy.complex128`
        gauge : str, 'R'
           chosen gauge
        """
        if dtype is None:
            dtype = np.complex128

        if gauge != 'R':
            raise ValueError('Only the cell vector gauge has been implemented')

        k = np.asarray(k, np.float64)
        k.shape = (-1,)

        if not np.allclose(k, 0.):
            if np.dtype(dtype).kind != 'c':
                raise ValueError("Hamiltonian setup at k different from Gamma requires a complex matrix")

        exp = np.exp
        dot = np.dot

        # Setup the Hamiltonian for this k-point
        Hf = self.tocsr(spin)

        # number of orbitals
        no = self.no

        H = csr_matrix((no, no), dtype=dtype)

        # Get the reciprocal lattice vectors dotted with k
        kr = dot(self.rcell, k)
        for si, isc in self.sc:
            phase = exp(-1j * dot(kr, dot(self.cell, isc)))

            H += Hf[:, si * no:(si + 1) * no] * phase

        del Hf

        return H

    def _Hk_non_collinear(self, k=(0, 0, 0), dtype=None, gauge='R'):
        """ Return the Hamiltonian in a ``scipy.sparse.csr_matrix`` at `k` for a non-collinear
        Hamiltonian.

        Parameters
        ----------
        k: ``array_like``, `[0,0,0]`
           k-point 
        dtype : ``numpy.dtype``
           default to `numpy.complex128`
        gauge : str, 'R'
           chosen gauge
        """
        if dtype is None:
            dtype = np.complex128

        if np.dtype(dtype).kind != 'c':
            raise ValueError("Non-collinear Hamiltonian setup requires a complex matrix")

        if gauge != 'R':
            raise ValueError('Only the cell vector gauge has been implemented')

        exp = np.exp
        dot = np.dot

        k = np.asarray(k, np.float64)
        k.shape = (-1,)

        # number of orbitals
        no = self.no

        # sparse matrix dimension (2 * self.no)
        H = csr_matrix((len(self), len(self)), dtype=dtype)

        # Get the reciprocal lattice vectors dotted with k
        kr = dot(self.rcell, k)
        for si, isc in self.sc:
            phase = exp(-1j * dot(kr, dot(self.cell, isc)))
            sl = slice(si*no, (si+1) * no, None)

            # diagonal elements
            H[::2, ::2] += self.tocsr(0)[:, sl] * phase
            H[1::2, 1::2] += self.tocsr(1)[:, sl] * phase

            # off-diagonal elements
            H1 = self.tocsr(2)[:, sl]
            H2 = self.tocsr(3)[:, sl]
            H[1::2, ::2] += (H1 - 1j * H2) * phase
            H[::2, 1::2] += (H1 + 1j * H2) * phase

            del H1, H2

        return H

    def _Hk_spin_orbit(self, k=(0, 0, 0), dtype=None, gauge='R'):
        """ The Hamiltonian in a ``scipy.sparse.csr_matrix`` at `k` for a spin-orbit Hamiltonian.

        Parameters
        ----------
        k: ``array_like``, `[0,0,0]`
           k-point 
        dtype : ``numpy.dtype``
           default to `numpy.complex128`
        gauge : str, 'R'
           chosen gauge
        """
        if dtype is None:
            dtype = np.complex128

        if np.dtype(dtype).kind != 'c':
            raise ValueError("Spin orbit Hamiltonian setup requires a complex matrix")

        if gauge != 'R':
            raise ValueError('Only the cell vector gauge has been implemented')

        exp = np.exp
        dot = np.dot

        k = np.asarray(k, np.float64)
        k.shape = (-1,)

        # number of orbitals
        no = self.no

        # sparse matrix dimension (2 * self.no)
        H = csr_matrix((len(self), len(self)), dtype=dtype)

        # Get the reciprocal lattice vectors dotted with k
        kr = dot(self.rcell, k)
        for si, isc in self.sc:
            phase = exp(-1j * dot(kr, dot(self.cell, isc)))
            sl = slice(si*no, (si+1) * no, None)

            # diagonal elements
            H[::2, ::2] += (self.tocsr(0)[:, sl] +
                            1j * self.tocsr(4)[:, sl]) * phase
            H[1::2, 1::2] += (self.tocsr(1)[:, sl] +
                              1j * self.tocsr(5)[:, sl]) * phase

            # lower off-diagonal elements
            H[1::2, ::2] += (self.tocsr(2)[:, sl] -
                             1j * self.tocsr(3)[:, sl]) * phase

            # upper off-diagonal elements
            H[::2, 1::2] += (self.tocsr(6)[:, sl] +
                             1j * self.tocsr(7)[:, sl]) * phase

        return H

    def _Sk(self, k=(0, 0, 0), dtype=None, gauge='R'):
        """ Return the Hamiltonian in a ``scipy.sparse.csr_matrix`` at `k`.

        Parameters
        ----------
        k: ``array_like``, `[0,0,0]`
           k-point 
        dtype : ``numpy.dtype``
           default to `numpy.complex128`
        gauge : str, 'R'
           chosen gauge
        """
        # we forward it to Hk_polarized (same thing for S)
        return self._Hk_polarized(k, spin=self.S_idx, dtype=dtype, gauge=gauge)

    def _Sk_non_collinear(self, k=(0, 0, 0), dtype=None, gauge='R'):
        """ Return the Hamiltonian in a ``scipy.sparse.csr_matrix`` at `k`.

        Parameters
        ----------
        k: ``array_like``, `[0,0,0]`
           k-point 
        dtype : ``numpy.dtype``
           default to `numpy.complex128`
        gauge : str, 'R'
           chosen gauge
        """
        if dtype is None:
            dtype = np.complex128

        if not np.allclose(k, 0.):
            if np.dtype(dtype).kind != 'c':
                raise ValueError("Hamiltonian setup at k different from Gamma requires a complex matrix")

        if gauge != 'R':
            raise ValueError('Only the cell vector gauge has been implemented')

        exp = np.exp
        dot = np.dot

        k = np.asarray(k, np.float64)
        k.shape = (-1,)

        # Get the overlap matrix
        Sf = self.tocsr(self.S_idx)

        # number of orbitals
        no = self.no

        S = csr_matrix((len(self), len(self)), dtype=dtype)

        # Get the reciprocal lattice vectors dotted with k
        kr = dot(self.rcell, k)
        for si, isc in self.sc:
            phase = exp(-1j * dot(kr, dot(self.cell, isc)))

            sf = Sf[:, si*no:(si+1)*no] * phase

            S[::2, ::2] += sf
            S[1::2, 1::2] += sf

            del sf

        return S

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
            for i in range(min(self.spin, 2)):
                self._data._D[:, i] -= self._data._D[:, self.S_idx] * E
        else:
            for i in range(self.shape[0]):
                for j in range(min(self.spin, 2)):
                    self[i, i, j] = self[i, i, j] - E

    def eigh(self, k=(0, 0, 0),
             atoms=None, gauge='R', eigvals_only=True,
             overwrite_a=True, overwrite_b=True,
             *args, **kwargs):
        """ Returns the eigenvalues of the Hamiltonian

        Setup the Hamiltonian and overlap matrix with respect to
        the given k-point, then reduce the space to the specified atoms
        and calculate the eigenvalues.

        All subsequent arguments gets passed directly to :code:`scipy.linalg.eigh`
        """

        # First we check if the k-point is a BrillouinZone object
        if isinstance(k, BrillouinZone):
            # Pre-allocate the eigenvalue spectrum
            eig = np.empty([len(k), len(self)], np.float64)
            for i, k_ in enumerate(k):
                eig[i, :] = self.eigh(k_, atoms=atoms, gauge=gauge,
                                      eigvals_only=eigvals_only,
                                      overwrite_a=overwrite_a, overwrite_b=overwrite_b,
                                      *args, **kwargs)
            return eig

        if self.spin == 2:
            H = self.Hk(k=k, spin=kwargs.pop('spin', 0), gauge=gauge)
        else:
            H = self.Hk(k=k, gauge=gauge)
        if not self.orthogonal:
            S = self.Sk(k=k, gauge=gauge)
        # Reduce sparsity pattern
        if not atoms is None:
            orbs = self.a2o(atoms)
            # Reduce space
            H = H[orbs, orbs]
            if not self.orthogonal:
                S = S[orbs, orbs]
        if self.orthogonal:
            return sli.eigh(H.todense(),
                *args,
                eigvals_only=eigvals_only,
                overwrite_a=overwrite_a,
                **kwargs)

        return sli.eigh(H.todense(), S.todense(),
            *args,
            eigvals_only=eigvals_only,
            overwrite_a=overwrite_a,
            overwrite_b=overwrite_b,
            **kwargs)

    def eigsh(self, k=(0, 0, 0), n=10,
              atoms=None, gauge='R', eigvals_only=True,
              *args, **kwargs):
        """ Returns a subset of eigenvalues of the Hamiltonian (default 10)

        Setup the Hamiltonian and overlap matrix with respect to
        the given k-point, then reduce the space to the specified atoms
        and calculate a subset of the eigenvalues using the sparse algorithms.

        All subsequent arguments gets passed directly to :code:`scipy.linalg.eigsh`
        """

        # First we check if the k-point is a BrillouinZone object
        if isinstance(k, BrillouinZone):
            # Pre-allocate the eigenvalue spectrum
            eig = np.empty([len(k), n], np.float64)
            for i, k_ in enumerate(k):
                eig[i, :] = self.eigsh(k_, n=n,
                                       atoms=atoms, gauge=gauge,
                                       eigvals_only=eigvals_only,
                                       *args, **kwargs)
            return eig

        # We always request the smallest eigenvalues...
        kwargs.update({'which': kwargs.get('which', 'SM')})

        if self.spin == 2:
            H = self.Hk(k=k, spin=kwargs.pop('spin', 0), gauge=gauge)
        else:
            H = self.Hk(k=k, gauge=gauge)
        if not self.orthogonal:
            raise ValueError("The sparsity pattern is non-orthogonal, you cannot use the Arnoldi procedure with scipy")

        # Reduce sparsity pattern
        if not atoms is None:
            orbs = self.a2o(atoms)
            # Reduce space
            H = H[orbs, orbs]

        return ssli.eigsh(H, k=n,
                          *args,
                          return_eigenvectors=not eigvals_only,
                          **kwargs)

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
