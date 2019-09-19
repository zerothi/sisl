from __future__ import print_function, division

import warnings

import numpy as np
from scipy.sparse import csr_matrix, SparseEfficiencyWarning

import sisl.linalg as lin
from sisl._help import _range as range
from sisl.sparse import isspmatrix
from sisl.sparse_geometry import SparseOrbital
from .spin import Spin
from ._matrix_k import matrix_k, matrix_k_nc, matrix_k_so, matrix_k_nc_diag
from ._matrix_dk import matrix_dk
from ._matrix_ddk import matrix_ddk


__all__ = ['SparseOrbitalBZ', 'SparseOrbitalBZSpin']


# Filter warnings from the sparse library
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


class SparseOrbitalBZ(SparseOrbital):
    r""" Sparse object containing the orbital connections in a Brillouin zone

    It contains an intrinsic sparse matrix of the physical elements.

    Assigning or changing elements is as easy as with
    standard `numpy` assignments:

    >>> S = SparseOrbitalBZ(...)
    >>> S[1,2] = 0.1

    which assigns 0.1 as the element between orbital 2 and 3.
    (remember that Python is 0-based elements).

    Parameters
    ----------
    geometry : Geometry
      parent geometry to create a sparse matrix from. The matrix will
      have size equivalent to the number of orbitals in the geometry
    dim : int, optional
      number of components per element
    dtype : np.dtype, optional
      data type contained in the matrix. See details of `Spin` for default values.
    nnzpr : int, optional
      number of initially allocated memory per orbital in the matrix.
      For increased performance this should be larger than the actual number of entries
      per orbital.
    orthogonal : bool, optional
      whether the matrix corresponds to a non-orthogonal basis. In this case
      the dimensionality of the matrix is one more than `dim`.
      This is a keyword-only argument.
    """

    def __init__(self, geometry, dim=1, dtype=None, nnzpr=None, **kwargs):
        self._geometry = geometry
        self._orthogonal = kwargs.get('orthogonal', True)

        # Get true dimension
        if not self.orthogonal:
            dim = dim + 1

        # Initialize the sparsity pattern
        self.reset(dim, dtype, nnzpr)
        self._reset()

    def _reset(self):
        r""" Reset object according to the options, please refer to `SparseOrbital.reset` for details """
        if self.orthogonal:
            self.Sk = self._Sk_diagonal
            self.S_idx = -100

        else:
            self.S_idx = self.shape[-1] - 1
            self.Sk = self._Sk
            self.dSk = self._dSk
            self.ddSk = self._ddSk

        self.Pk = self._Pk
        self.dPk = self._dPk
        self.ddPk = self._ddPk

    # Override to enable spin configuration and orthogonality
    def _cls_kwargs(self):
        return {'orthogonal': self.orthogonal}

    @property
    def orthogonal(self):
        r""" True if the object is using an orthogonal basis """
        return self._orthogonal

    @property
    def non_orthogonal(self):
        r""" True if the object is using a non-orthogonal basis """
        return not self._orthogonal

    def __len__(self):
        r""" Returns number of rows in the basis (if non-collinear or spin-orbit, twice the number of orbitals) """
        return self.no

    def __str__(self):
        r""" Representation of the model """
        s = self.__class__.__name__ + '{{dim: {0}, non-zero: {1}, orthogonal: {2}\n '.format(self.dim, self.nnz, self.orthogonal)
        s += str(self.geometry).replace('\n', '\n ')
        return s + '\n}'

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

    S = property(_get_S, _set_S, doc="Access elements to the sparse overlap")

    @classmethod
    def fromsp(cls, geometry, P, S=None, **kwargs):
        r""" Create a sparse model from a preset `Geometry` and a list of sparse matrices

        The passed sparse matrices are in one of `scipy.sparse` formats.

        Parameters
        ----------
        geometry : Geometry
           geometry to describe the new sparse geometry
        P : list of scipy.sparse or scipy.sparse
           the new sparse matrices that are to be populated in the sparse
           matrix
        S : scipy.sparse, optional
           if provided this refers to the overlap matrix and will force the
           returned sparse matrix to be non-orthogonal
        **kwargs : optional
           any arguments that are directly passed to the ``__init__`` method
           of the class.

        Returns
        -------
        SparseGeometry
             a new sparse matrix that holds the passed geometry and the elements of `P` and optionally being non-orthogonal if `S` is not none
        """
        # Ensure list of csr format (to get dimensions)
        if isspmatrix(P):
            P = [P]
        if isinstance(P, tuple):
            P = list(P)

        # Number of dimensions
        dim = len(P)
        nnzpr = 1
        # Sort all indices for the passed sparse matrices
        for i in range(dim):
            P[i] = P[i].tocsr()
            P[i].sort_indices()
            P[i].sum_duplicates()

            nnzpr = max(nnzpr, P[i].nnz // P[i].shape[0])

        # Create the sparse object
        p = cls(geometry, dim, P[0].dtype, nnzpr, orthogonal=S is None, **kwargs)

        if p._size != P[0].shape[0]:
            raise ValueError(cls.__name__ + '.fromsp cannot create a new class, the geometry ' + \
                             'and sparse matrices does not have coinciding dimensions size != P[0].shape[0]')

        for i in range(dim):
            ptr = P[i].indptr
            col = P[i].indices
            D = P[i].data

            # loop and add elements
            for r in range(p.shape[0]):
                sl = slice(ptr[r], ptr[r+1], None)
                p[r, col[sl], i] = D[sl]

        if not S is None:
            S = S.tocsr()
            S.sort_indices()
            S.sum_duplicates()
            ptr = S.indptr
            col = S.indices
            D = S.data

            # loop and add elements
            for r in range(p.shape[0]):
                sl = slice(ptr[r], ptr[r+1], None)
                p.S[r, col[sl]] = D[sl]

        return p

    def iter_orbitals(self, atom=None, local=False):
        r""" Iterations of the orbital space in the geometry, two indices from loop

        An iterator returning the current atomic index and the corresponding
        orbital index.

        >>> for ia, io in self.iter_orbitals():

        In the above case `io` always belongs to atom `ia` and `ia` may be
        repeated according to the number of orbitals associated with
        the atom `ia`.

        Parameters
        ----------
        atom : int or array_like, optional
           only loop on the given atoms, default to all atoms
        local : bool, optional
           whether the orbital index is the global index, or the local index relative to
           the atom it resides on.

        Yields
        ------
        ia
           atomic index
        io
           orbital index

        See Also
        --------
        Geometry.iter_orbitals : method used to iterate orbitals
        """
        for ia, io in self.geometry.iter_orbitals(local=local):
            yield ia, io

    def _Pk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', _dim=0):
        r""" Sparse matrix (``scipy.sparse.csr_matrix``) at `k` for a polarized system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        k = np.asarray(k, np.float64).ravel()
        return matrix_k(gauge, self, _dim, self.sc, k, dtype, format)

    def _dPk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', _dim=0):
        r""" Sparse matrix (``scipy.sparse.csr_matrix``) at `k` differentiated with respect to `k` for a polarized system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        k = np.asarray(k, np.float64).ravel()
        return matrix_dk(gauge, self, _dim, self.sc, k, dtype, format)

    def _ddPk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', _dim=0):
        r""" Sparse matrix (``scipy.sparse.csr_matrix``) at `k` double differentiated with respect to `k` for a polarized system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        k = np.asarray(k, np.float64).ravel()
        return matrix_ddk(gauge, self, _dim, self.sc, k, dtype, format)

    def Sk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the overlap matrix for a given k-point

        Creation and return of the overlap matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \mathbf S(k) = \mathbf S_{\nu\mu} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`\nu`, :math:`\mu` are orbital indices.

        Another possible gauge is the orbital distance which can be written as

        .. math::
           \mathbf S(k) = \mathbf S_{\nu\mu} e^{i k r}

        where :math:`r` is the distance between the orbitals.

        Parameters
        ----------
        k : array_like, optional
           the k-point to setup the overlap at (default Gamma point)
        dtype : numpy.dtype, optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge : {'R', 'r'}
           the chosen gauge, `R` for cell vector gauge, and `r` for orbital distance
           gauge.
        format : {'csr', 'array', 'matrix', 'coo', ...}
           the returned format of the matrix, defaulting to the ``scipy.sparse.csr_matrix``,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).

        See Also
        --------
        dSk : Overlap matrix derivative with respect to `k`
        ddSk : Overlap matrix double derivative with respect to `k`

        Returns
        -------
        object : the overlap matrix for the :math:`k`-point, `format` determines the object type.
        """
        pass

    def _Sk_diagonal(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" For an orthogonal case we always return the identity matrix """
        if dtype is None:
            dtype = np.float64
        no = len(self)
        # In the "rare" but could be found situation where
        # the matrix only describes neighbouring couplings it is vital
        # to not return anything
        # TODO
        if format in ['array', 'matrix', 'dense']:
            return np.diag(np.ones(no, dtype=dtype))
        S = csr_matrix((no, no), dtype=dtype)
        S.setdiag(1.)
        return S.asformat(format)

    def _Sk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Overlap matrix in a ``scipy.sparse.csr_matrix`` at `k`.

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        return self._Pk(k, dtype=dtype, gauge=gauge, format=format, _dim=self.S_idx)

    def dSk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the :math:`k`-derivatie of the overlap matrix for a given k-point

        Creation and return of the derivative of the overlap matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_k \mathbf S_\alpha(k) = i R_\alpha \mathbf S_{\nu\mu} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`\nu`, :math:`\mu` are orbital indices.
        And :math:`\alpha` is one of the Cartesian directions.

        Another possible gauge is the orbital distance which can be written as

        .. math::
           \nabla_k \mathbf S_\alpha(k) = i r_\alpha \mathbf S_{ij} e^{i k r}

        where :math:`r` is the distance between the orbitals.

        Parameters
        ----------
        k : array_like, optional
           the k-point to setup the overlap at (default Gamma point)
        dtype : numpy.dtype, optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge : {'R', 'r'}
           the chosen gauge, `R` for cell vector gauge, and `r` for orbital distance
           gauge.
        format : {'csr', 'array', 'matrix', 'coo', ...}
           the returned format of the matrix, defaulting to the ``scipy.sparse.csr_matrix``,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).

        See Also
        --------
        Sk : Overlap matrix at `k`
        ddSk : Overlap matrix double derivative at `k`

        Returns
        -------
        tuple : for each of the Cartesian directions a :math:`\partial \mathbf S(k)/\partial k` is returned.
        """
        pass

    def _dSk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Overlap matrix in a ``scipy.sparse.csr_matrix`` at `k` differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        return self._dPk(k, dtype=dtype, gauge=gauge, format=format, _dim=self.S_idx)

    def ddSk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the double :math:`k`-derivatie of the overlap matrix for a given k-point

        Creation and return of the double derivative of the overlap matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_k^2 \mathbf S_{\alpha\beta}(k) = - R_\alpha R_\beta \mathbf S_{\nu\mu} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`\nu`, :math:`\mu` are orbital indices.
        And :math:`\alpha` and :math:`\beta` are one of the Cartesian directions.

        Another possible gauge is the orbital distance which can be written as

        .. math::
           \nabla_k^2 \mathbf S_{\alpha\beta}(k) = - r_\alpha r_\beta \mathbf S_{ij} e^{i k r}

        where :math:`r` is the distance between the orbitals.

        Parameters
        ----------
        k : array_like, optional
           the k-point to setup the overlap at (default Gamma point)
        dtype : numpy.dtype, optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge : {'R', 'r'}
           the chosen gauge, `R` for cell vector gauge, and `r` for orbital distance
           gauge.
        format : {'csr', 'array', 'matrix', 'coo', ...}
           the returned format of the matrix, defaulting to the ``scipy.sparse.csr_matrix``,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).

        See Also
        --------
        Sk : Overlap matrix at `k`
        dSk : Overlap matrix derivative at `k`

        Returns
        -------
        tuple of tuples : for each of the Cartesian directions
        """
        pass

    def _ddSk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Overlap matrix in a ``scipy.sparse.csr_matrix`` at `k` double differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        return self._ddPk(k, dtype=dtype, gauge=gauge, format=format, _dim=self.S_idx)

    def eig(self, k=(0, 0, 0), gauge='R', eigvals_only=True, **kwargs):
        r""" Returns the eigenvalues of the physical quantity (using the non-Hermitian solver)

        Setup the system and overlap matrix with respect to
        the given k-point and calculate the eigenvalues.

        All subsequent arguments gets passed directly to :code:`scipy.linalg.eig`
        """
        dtype = kwargs.pop('dtype', None)
        P = self.Pk(k=k, dtype=dtype, gauge=gauge, format='array')
        if self.orthogonal:
            if eigvals_only:
                return lin.eigvals_destroy(P, **kwargs)
            return lin.eig_destroy(P, **kwargs)

        S = self.Sk(k=k, dtype=dtype, gauge=gauge, format='array')
        if eigvals_only:
            return lin.eigvals_destroy(P, S, **kwargs)
        return lin.eig_destroy(P, S, **kwargs)

    def eigh(self, k=(0, 0, 0), gauge='R', eigvals_only=True, **kwargs):
        r""" Returns the eigenvalues of the physical quantity

        Setup the system and overlap matrix with respect to
        the given k-point and calculate the eigenvalues.

        All subsequent arguments gets passed directly to :code:`scipy.linalg.eigh`
        """
        dtype = kwargs.pop('dtype', None)
        P = self.Pk(k=k, dtype=dtype, gauge=gauge, format='array')
        if self.orthogonal:
            return lin.eigh_destroy(P, eigvals_only=eigvals_only, **kwargs)

        S = self.Sk(k=k, dtype=dtype, gauge=gauge, format='array')
        return lin.eigh_destroy(P, S, eigvals_only=eigvals_only, **kwargs)

    def eigsh(self, k=(0, 0, 0), n=10, gauge='R', eigvals_only=True, **kwargs):
        r""" Calculates a subset of eigenvalues of the physical quantity  (default 10)

        Setup the quantity and overlap matrix with respect to
        the given k-point and calculate a subset of the eigenvalues using the sparse algorithms.

        All subsequent arguments gets passed directly to :code:`scipy.linalg.eigsh`
        """
        # We always request the smallest eigenvalues...
        kwargs.update({'which': kwargs.get('which', 'SM')})

        dtype = kwargs.pop('dtype', None)

        P = self.Pk(k=k, dtype=dtype, gauge=gauge)
        if not self.orthogonal:
            raise ValueError("The sparsity pattern is non-orthogonal, you cannot use the Arnoldi procedure with scipy")

        return lin.eigsh(P, k=n, return_eigenvectors=not eigvals_only, **kwargs)

    def __getstate__(self):
        return {
            'sparseorbitalbz': super(SparseOrbitalBZ, self).__getstate__(),
            'orthogonal': self._orthogonal
        }

    def __setstate__(self, state):
        self._orthogonal = state['orthogonal']
        super(SparseOrbitalBZ, self).__setstate__(state['sparseorbitalbz'])
        self._reset()


class SparseOrbitalBZSpin(SparseOrbitalBZ):
    r""" Sparse object containing the orbital connections in a Brillouin zone with possible spin-components

    It contains an intrinsic sparse matrix of the physical elements.

    Assigning or changing elements is as easy as with
    standard `numpy` assignments::

    >>> S = SparseOrbitalBZSpin(...)
    >>> S[1,2] = 0.1

    which assigns 0.1 as the element between orbital 2 and 3.
    (remember that Python is 0-based elements).

    Parameters
    ----------
    geometry : Geometry
      parent geometry to create a sparse matrix from. The matrix will
      have size equivalent to the number of orbitals in the geometry
    dim : int or Spin, optional
      number of components per element, may be a `Spin` object
    dtype : np.dtype, optional
      data type contained in the matrix. See details of `Spin` for default values.
    nnzpr : int, optional
      number of initially allocated memory per orbital in the matrix.
      For increased performance this should be larger than the actual number of entries
      per orbital.
    spin : Spin, optional
      equivalent to `dim` argument. This keyword-only argument has precedence over `dim`.
    orthogonal : bool, optional
      whether the matrix corresponds to a non-orthogonal basis. In this case
      the dimensionality of the matrix is one more than `dim`.
      This is a keyword-only argument.
    """

    def __init__(self, geometry, dim=1, dtype=None, nnzpr=None, **kwargs):
        # Check that the passed parameters are correct
        if 'spin' not in kwargs:
            if isinstance(dim, Spin):
                spin = dim
            else:
                spin = {1: Spin.UNPOLARIZED,
                        2: Spin.POLARIZED,
                        4: Spin.NONCOLINEAR,
                        8: Spin.SPINORBIT}.get(dim)
        else:
            spin = kwargs.pop('spin')
        self._spin = Spin(spin, dtype)

        super(SparseOrbitalBZSpin, self).__init__(geometry, len(self.spin), self.spin.dtype, nnzpr, **kwargs)
        self._reset()

    def _reset(self):
        r""" Reset object according to the options, please refer to `SparseOrbital.reset` for details """
        super(SparseOrbitalBZSpin, self)._reset()

        if self.spin.is_unpolarized:
            self.UP = 0
            self.DOWN = 0
            self.Pk = self._Pk_unpolarized
            self.Sk = self._Sk
            self.dPk = self._dPk_unpolarized
            self.dSk = self._dSk

        elif self.spin.is_polarized:
            self.UP = 0
            self.DOWN = 1
            self.Pk = self._Pk_polarized
            self.dPk = self._dPk_polarized
            self.Sk = self._Sk
            self.dSk = self._dSk

        elif self.spin.is_noncolinear:
            if self.spin.dkind == 'f':
                self.M11 = 0
                self.M22 = 1
                self.M12r = 2
                self.M12i = 3
            else:
                self.M11 = 0
                self.M22 = 1
                self.M12 = 2
                raise ValueError('Currently not implemented')
            self.Pk = self._Pk_non_colinear
            self.Sk = self._Sk_non_colinear
            self.dPk = None
            self.dSk = None

        elif self.spin.is_spinorbit:
            if self.spin.dkind == 'f':
                self.M11r = 0
                self.M22r = 1
                self.M12r = 2
                self.M12i = 3
                self.M11i = 4
                self.M22i = 5
                self.M21r = 6
                self.M21i = 7
            else:
                self.M11 = 0
                self.M22 = 1
                self.M12 = 2
                self.M21 = 3
                raise ValueError('Currently not implemented')
            # The overlap is the same as non-collinear
            self.Pk = self._Pk_spin_orbit
            self.Sk = self._Sk_non_colinear
            self.dPk = None
            self.dSk = None

        if self.orthogonal:
            self.Sk = self._Sk_diagonal

    # Override to enable spin configuration and orthogonality
    def _cls_kwargs(self):
        return {'spin': self.spin.kind, 'orthogonal': self.orthogonal}

    @property
    def spin(self):
        r""" Associated spin class """
        return self._spin

    def __len__(self):
        r""" Returns number of rows in the basis (if non-collinear or spin-orbit, twice the number of orbitals) """
        if self.spin.spins > 2:
            return self.no * 2
        return self.no

    def __str__(self):
        r""" Representation of the model """
        s = self.__class__.__name__ + '{{non-zero: {0}, orthogonal: {1},\n '.format(self.nnz, self.orthogonal)
        s += str(self.spin).replace('\n', '\n ') + ',\n '
        s += str(self.geometry).replace('\n', '\n ')
        return s + '\n}'

    def _Pk_unpolarized(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Sparse matrix (``scipy.sparse.csr_matrix``) at `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        return self._Pk(k, dtype=dtype, gauge=gauge, format=format)

    def _Pk_polarized(self, k=(0, 0, 0), spin=0, dtype=None, gauge='R', format='csr'):
        r""" Sparse matrix (``scipy.sparse.csr_matrix``) at `k` for a polarized system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        spin : int, optional
           the spin-index of the quantity
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        return self._Pk(k, dtype=dtype, gauge=gauge, format=format, _dim=spin)

    def _Pk_non_colinear(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Sparse matrix (``scipy.sparse.csr_matrix``) at `k` for a non-collinear system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        k = np.asarray(k, np.float64).ravel()
        return matrix_k_nc(gauge, self, self.sc, k, dtype, format)

    def _Pk_spin_orbit(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Sparse matrix (``scipy.sparse.csr_matrix``) at `k` for a spin-orbit system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        k = np.asarray(k, np.float64).ravel()
        return matrix_k_so(gauge, self, self.sc, k, dtype, format)

    def _dPk_unpolarized(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Tuple of sparse matrix (``scipy.sparse.csr_matrix``) at `k`, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        return self._dPk(k, dtype=dtype, gauge=gauge, format=format)

    def _dPk_polarized(self, k=(0, 0, 0), spin=0, dtype=None, gauge='R', format='csr'):
        r""" Tuple of sparse matrix (``scipy.sparse.csr_matrix``) at `k`, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        spin : int, optional
           the spin-index of the quantity
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        return self._dPk(k, dtype=dtype, gauge=gauge, format=format, _dim=spin)

    def _Sk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Overlap matrix in a ``scipy.sparse.csr_matrix`` at `k`.

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        return self._Pk(k, dtype=dtype, gauge=gauge, format=format, _dim=self.S_idx)

    def _Sk_non_colinear(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Overlap matrix (``scipy.sparse.csr_matrix``) at `k` for a non-collinear system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        k = np.asarray(k, np.float64).ravel()
        return matrix_k_nc_diag(gauge, self, self.S_idx, self.sc, k, dtype, format)

    def eig(self, k=(0, 0, 0), gauge='R', eigvals_only=True, **kwargs):
        r""" Returns the eigenvalues of the physical quantity (using the non-Hermitian solver)

        Setup the system and overlap matrix with respect to
        the given k-point and calculate the eigenvalues.

        All subsequent arguments gets passed directly to :code:`scipy.linalg.eig`

        Parameters
        ----------
        spin : int, optional
           the spin-component to calculate the eigenvalue spectrum of, note that
           this parameter is only valid for `Spin.POLARIZED` matrices.
        """
        spin = kwargs.pop('spin', 0)
        dtype = kwargs.pop('dtype', None)

        if self.spin.kind == Spin.POLARIZED:
            P = self.Pk(k=k, dtype=dtype, gauge=gauge, spin=spin, format='array')
        else:
            P = self.Pk(k=k, dtype=dtype, gauge=gauge, format='array')

        if self.orthogonal:
            if eigvals_only:
                return lin.eigvals_destroy(P, **kwargs)
            return lin.eig_destroy(P, **kwargs)

        S = self.Sk(k=k, dtype=dtype, gauge=gauge, format='array')
        if eigvals_only:
            return lin.eigvals_destroy(P, S, **kwargs)
        return lin.eig_destroy(P, S, **kwargs)

    def eigh(self, k=(0, 0, 0), gauge='R', eigvals_only=True, **kwargs):
        r""" Returns the eigenvalues of the physical quantity

        Setup the system and overlap matrix with respect to
        the given k-point and calculate the eigenvalues.

        All subsequent arguments gets passed directly to :code:`scipy.linalg.eigh`

        Parameters
        ----------
        spin : int, optional
           the spin-component to calculate the eigenvalue spectrum of, note that
           this parameter is only valid for `Spin.POLARIZED` matrices.
        """
        spin = kwargs.pop('spin', 0)
        dtype = kwargs.pop('dtype', None)

        if self.spin.kind == Spin.POLARIZED:
            P = self.Pk(k=k, dtype=dtype, gauge=gauge, spin=spin, format='array')
        else:
            P = self.Pk(k=k, dtype=dtype, gauge=gauge, format='array')

        if self.orthogonal:
            return lin.eigh_destroy(P, eigvals_only=eigvals_only, **kwargs)

        S = self.Sk(k=k, dtype=dtype, gauge=gauge, format='array')
        return lin.eigh_destroy(P, S, eigvals_only=eigvals_only, **kwargs)

    def eigsh(self, k=(0, 0, 0), n=10, gauge='R', eigvals_only=True, **kwargs):
        r""" Calculates a subset of eigenvalues of the physical quantity  (default 10)

        Setup the quantity and overlap matrix with respect to
        the given k-point and calculate a subset of the eigenvalues using the sparse algorithms.

        All subsequent arguments gets passed directly to :code:`scipy.linalg.eigsh`

        Parameters
        ----------
        spin : int, optional
           the spin-component to calculate the eigenvalue spectrum of, note that
           this parameter is only valid for `Spin.POLARIZED` matrices.
        """
        # We always request the smallest eigenvalues...
        spin = kwargs.pop('spin', 0)
        dtype = kwargs.pop('dtype', None)
        kwargs.update({'which': kwargs.get('which', 'SM')})

        if self.spin.kind == Spin.POLARIZED:
            P = self.Pk(k=k, dtype=dtype, spin=spin, gauge=gauge)
        else:
            P = self.Pk(k=k, dtype=dtype, gauge=gauge)
        if not self.orthogonal:
            raise ValueError("The sparsity pattern is non-orthogonal, you cannot use the Arnoldi procedure with scipy")

        return lin.eigsh(P, k=n, return_eigenvectors=not eigvals_only, **kwargs)

    def transpose(self, hermitian=False):
        r""" A transpose copy of this object, possibly apply the Hermitian conjugate as well (default)

        Parameters
        ----------
        hermitian : bool, optional
           if true, also emply a spin-box Hermitian operator to ensure TRS, otherwise
           only return the transpose values.
        """
        new = super(SparseOrbitalBZSpin, self).transpose()
        sp = self.spin
        D = new._csr._D

        if hermitian:
            if sp.is_noncolinear:
                # conjugate the imaginary value
                if sp.dkind == 'f':
                    D[:, 3] = -D[:, 3]
                else:
                    D[:, 2] = np.conj(D[:, 2])
            elif sp.is_spinorbit:
                # conjugate the imaginary value and transpose spin-box
                if sp.dkind == 'f':
                    # imaginary components (including transposing)
                    #    12,11,22,21
                    D[:, [3, 4, 5, 7]] = -D[:, [7, 4, 5, 3]]
                    # M21r -> M12r
                    D[:, [2, 6]] = D[:, [6, 2]]
                else:
                    D[:, [0, 1]] = np.conj(D[:, [0, 1]])
                    D[:, [2, 3]] = np.conj(D[:, [3, 2]])
        elif sp.is_spinorbit:
            # transpose spin-box
            if sp.dkind == 'f':
                #    12 -> 21
                D[:, [2, 3, 6, 7]] = D[:, [6, 7, 2, 3]]
            else:
                D[:, [2, 3]] = D[:, [3, 2]]

        return new

    def trs(self):
        r""" Create a new matrix with applied time-reversal-symmetry

        Time reversal symmetry is applied using the following equality:

        .. math::

            2\mathbf M^{\mathrm{TRS}} = \mathbf M + \boldsymbol\sigma_y \mathbf M^* \boldsymbol\sigma_y

        where :math:`*` is the conjugation operator.
        """
        new = self.copy()
        sp = self.spin
        D = new._csr._D

        # Apply Pauli-Y on the left and right of each spin-box
        if sp.is_spinorbit:
            if sp.dkind == 'f':
                # [R11, R22, R12, I12, I11, I22, R21, I21]
                # [R11, R22] = [R22, R11]
                # [I12, I21] = [I21, I12] (conj + Y @ Y[sign-changes conj])
                D[:, [0, 1, 3, 7]] = D[:, [1, 0, 7, 3]]
                # [I11, I22] = -[I22, I11] (conj + Y @ Y[no sign change])
                # [R12, R21] = -[R21, R12] (Y @ Y)
                D[:, [4, 5, 2, 6]] = -D[:, [5, 4, 6, 2]]
            else:
                raise NotImplementedError
        elif sp.is_noncolinear:
            if sp.dkind == 'f':
                # [R11, R22, R12, I12]
                D[:, 2] = -D[:, 2]
            else:
                raise NotImplementedError

        return new

    def __getstate__(self):
        return {
            'sparseorbitalbzspin': super(SparseOrbitalBZSpin, self).__getstate__(),
            'spin': self._spin.__getstate__(),
            'orthogonal': self._orthogonal,
        }

    def __setstate__(self, state):
        self._orthogonal = state['orthogonal']
        spin = Spin()
        spin.__setstate__(state['spin'])
        self._spin = spin
        super(SparseOrbitalBZSpin, self).__setstate__(state['sparseorbitalbzspin'])
