import warnings

import numpy as np
from scipy.sparse import csr_matrix, SparseEfficiencyWarning

from sisl._internal import set_module
import sisl.linalg as lin
import sisl._array as _a
from sisl.messages import warn
from sisl.sparse import isspmatrix
from sisl.sparse_geometry import SparseOrbital
from .spin import Spin
from ._matrix_k import matrix_k, matrix_k_nc, matrix_k_so, matrix_k_nc_diag
from ._matrix_dk import matrix_dk, matrix_dk_nc, matrix_dk_so, matrix_dk_nc_diag
from ._matrix_ddk import matrix_ddk, matrix_ddk_nc, matrix_ddk_so, matrix_ddk_nc_diag


__all__ = ['SparseOrbitalBZ', 'SparseOrbitalBZSpin']


# Filter warnings from the sparse library
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


@set_module("sisl.physics")
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
        s = self.__class__.__name__ + f'{{dim: {self.dim}, non-zero: {self.nnz}, orthogonal: {self.orthogonal}\n '
        s += str(self.geometry).replace('\n', '\n ')
        return s + '\n}'

    def __repr__(self):
        g = self.geometry
        return f"<{self.__module__}.{self.__class__.__name__} na={g.na}, no={g.no}, nsc={g.nsc}, dim={self.dim}, nnz={self.nnz}>"

    @property
    def S(self):
        r""" Access the overlap elements associated with the sparse matrix """
        if self.orthogonal:
            return None
        self._def_dim = self.S_idx
        return self

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

        # Number of dimensions, before S!
        dim = len(P)
        if not S is None:
            P.append(S)

        p = cls(geometry, dim, P[0].dtype, 1, orthogonal=S is None, **kwargs)
        p._csr = p._csr.fromsp(*P, dtype=kwargs.get("dtype"))

        if p._size != P[0].shape[0]:
            raise ValueError(f"{cls.__name__}.fromsp cannot create a new class, the geometry "
                             "and sparse matrices does not have coinciding dimensions size != P[0].shape[0]")

        return p

    def iter_orbitals(self, atoms=None, local=False):
        r""" Iterations of the orbital space in the geometry, two indices from loop

        An iterator returning the current atomic index and the corresponding
        orbital index.

        >>> for ia, io in self.iter_orbitals():

        In the above case `io` always belongs to atom `ia` and `ia` may be
        repeated according to the number of orbitals associated with
        the atom `ia`.

        Parameters
        ----------
        atoms : int or array_like, optional
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
        yield from self.geometry.iter_orbitals(atoms=atoms, local=local)

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
        k = _a.asarrayd(k).ravel()
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
        k = _a.asarrayd(k).ravel()
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
        k = _a.asarrayd(k).ravel()
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
           Prefixing with 'sc:', or simply 'sc' returns the matrix in supercell format
           with phases. This is useful for e.g. bond-current calculations where individual
           hopping + phases are required.

        See Also
        --------
        dSk : Overlap matrix derivative with respect to `k`
        ddSk : Overlap matrix double derivative with respect to `k`

        Returns
        -------
        matrix : numpy.ndarray or scipy.sparse.*_matrix
            the overlap matrix at :math:`k`. The returned object depends on `format`.
        """
        pass

    def _Sk_diagonal(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" For an orthogonal case we always return the identity matrix """
        if dtype is None:
            dtype = np.float64
        nr = len(self)
        nc = nr
        if 'sc:' in format:
            format = format[3:]
            nc = self.n_s * nr
        elif 'sc' == format:
            format = 'csr'
            nc = self.n_s * nr
        # In the "rare" but could be found situation where
        # the matrix only describes neighbouring couplings it is vital
        # to not return anything
        # TODO
        if format in ['array', 'matrix', 'dense']:
            S = np.zeros([nr, nc], dtype=dtype)
            np.fill_diagonal(S, 1.)
            return S
        S = csr_matrix((nr, nc), dtype=dtype)
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
        tuple
            for each of the Cartesian directions a :math:`\partial \mathbf S(k)/\partial k` is returned.
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

    def _dSk_non_colinear(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Overlap matrix in a ``scipy.sparse.csr_matrix`` at `k` for non-collinear spin, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_dk_nc_diag(gauge, self, self.S_idx, self.sc, k, dtype, format)

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
        list of matrices
            for each of the Cartesian directions (in Voigt representation); xx, yy, zz, zy, xz, xy
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

    def _ddSk_non_colinear(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Overlap matrix in a ``scipy.sparse.csr_matrix`` at `k` for non-collinear spin, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_ddk_nc_diag(gauge, self, self.S_idx, self.sc, k, dtype, format)

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
        if self.orthogonal:
            return lin.eigsh(P, k=n, return_eigenvectors=not eigvals_only, **kwargs)
        S = self.Sk(k=k, dtype=dtype, gauge=gauge)
        return lin.eigsh(P, M=S, k=n, return_eigenvectors=not eigvals_only, **kwargs)

    def __getstate__(self):
        return {
            'sparseorbitalbz': super().__getstate__(),
            'orthogonal': self._orthogonal
        }

    def __setstate__(self, state):
        self._orthogonal = state['orthogonal']
        super().__setstate__(state['sparseorbitalbz'])
        self._reset()


@set_module("sisl.physics")
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

        super().__init__(geometry, len(self.spin), self.spin.dtype, nnzpr, **kwargs)
        self._reset()

    def _reset(self):
        r""" Reset object according to the options, please refer to `SparseOrbital.reset` for details """
        super()._reset()

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
                raise NotImplementedError('Currently not implemented')
            self.Pk = self._Pk_non_colinear
            self.Sk = self._Sk_non_colinear
            self.dPk = self._dPk_non_colinear
            self.dSk = self._dSk_non_colinear
            self.ddPk = self._ddPk_non_colinear
            self.ddSk = self._ddSk_non_colinear

        elif self.spin.is_spinorbit:
            if self.spin.dkind == 'f':
                self.SX = np.array([0, 0, 1, 0, 0, 0, 1, 0], self.dtype)
                self.SY = np.array([0, 0, 0, -1, 0, 0, 0, 1], self.dtype)
                self.SZ = np.array([1, -1, 0, 0, 0, 0, 0, 0], self.dtype)
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
                raise NotImplementedError('Currently not implemented')
            # The overlap is the same as non-collinear
            self.Pk = self._Pk_spin_orbit
            self.Sk = self._Sk_non_colinear
            self.dPk = self._dPk_spin_orbit
            self.dSk = self._dSk_non_colinear
            self.ddPk = self._ddPk_spin_orbit
            self.ddSk = self._ddSk_non_colinear

        if self.orthogonal:
            self.Sk = self._Sk_diagonal

    # Override to enable spin configuration and orthogonality
    def _cls_kwargs(self):
        return {'spin': self.spin.kind, 'orthogonal': self.orthogonal}

    @property
    def spin(self):
        r""" Associated spin class """
        return self._spin

    def create_construct(self, R, param):
        r""" Create a simple function for passing to the `construct` function.

        This is to relieve the creation of simplistic
        functions needed for setting up sparse elements.

        For simple matrices this returns a function:

        >>> def func(self, ia, atoms, atoms_xyz=None):
        ...     idx = self.geometry.close(ia, R=R, atoms=atoms, atoms_xyz=atoms_xyz)
        ...     for ix, p in zip(idx, param):
        ...         self[ia, ix] = p

        In the non-colinear case the matrix element :math:`M_{ij}` will be set
        to input values `param` if :math:`i \le j` and the Hermitian conjugated
        values for :math:`j < i`.

        Notes
        -----
        This function only works for geometry sparse matrices (i.e. one
        element per atom). If you have more than one element per atom
        you have to implement the function your-self.

        This method issues warnings if the on-site terms are not Hermitian
        for spin-orbit systems. Do note that it *still* creates the matrices
        based on the input.

        Parameters
        ----------
        R : array_like
           radii parameters for different shells.
           Must have same length as `param` or one less.
           If one less it will be extended with ``R[0]/100``
        param : array_like
           coupling constants corresponding to the `R`
           ranges. ``param[0,:]`` are the elements
           for the all atoms within ``R[0]`` of each atom.

        See Also
        --------
        construct : routine to create the sparse matrix from a generic function (as returned from `create_construct`)
        """
        if len(R) != len(param):
            raise ValueError(f"{self.__class__.__name__}.create_construct got different lengths of `R` and `param`")
        if self.spin.has_noncolinear:
            is_complex = self.dkind == 'c'
            if self.spin.is_spinorbit:
                if is_complex:
                    nv = 4
                    # Hermitian parameters
                    paramH = [[p[0].conj(), p[1].conj(), p[3].conj(), p[2].conj(), *p[4:]]
                              for p in param]
                else:
                    nv = 8
                    # Hermitian parameters
                    paramH = [[p[0], p[1], p[6], -p[7], -p[4], -p[5], p[2], -p[3], *p[8:]]
                              for p in param]
                if not self.orthogonal:
                    nv += 1

                # ensure we have correct number of values
                assert all(len(p) == nv for p in param)

                if R[0] <= 0.1001: # no atom closer than 0.1001 Ang!
                    # We check that the the parameters here is Hermitian
                    p = param[0]
                    if is_complex:
                        onsite = np.array([[p[0], p[2]],
                                           [p[3], p[1]]], self.dtype)
                    else:
                        onsite = np.array([[p[0] + 1j * p[4], p[2] + 1j * p[3]],
                                           [p[6] + 1j * p[7], p[1] + 1j * p[5]]], np.complex128)
                    if not np.allclose(onsite, onsite.T.conj()):
                        warn(f"{self.__class__.__name__}.create_construct is NOT Hermitian for on-site terms. This is your responsibility!")

            elif self.spin.is_noncolinear:
                if is_complex:
                    nv = 3
                    # Hermitian parameters
                    paramH = [[p[0].conj(), p[1].conj(), p[2], *p[3:]]
                              for p in param]
                else:
                    nv = 4
                    # Hermitian parameters
                    # Note that we don't need to do anything here.
                    # H_ij = [[0, 2 + 1j 3],
                    #         [2 - 1j 3, 1]]
                    # H_ji = [[0, 2 + 1j 3],
                    #         [2 - 1j 3, 1]]
                    # H_ij^H == H_ji^H
                    paramH = param
                if not self.orthogonal:
                    nv += 1

                # we don't need to check hermiticity for NC
                # Since the values are ensured Hermitian in the on-site case anyways.

                # ensure we have correct number of values
                assert all(len(p) == nv for p in param)

            na = self.geometry.na

            # Now create the function that returns the assignment function
            def func(self, ia, atoms, atoms_xyz=None):
                idx = self.geometry.close(ia, R=R, atoms=atoms, atoms_xyz=atoms_xyz)
                for ix, p, pc in zip(idx, param, paramH):
                    ix_ge = (ix % na) >= ia
                    self[ia, ix[ix_ge]] = p
                    self[ia, ix[~ix_ge]] = pc

            return func

        return super().create_construct(R, param)

    def __len__(self):
        r""" Returns number of rows in the basis (if non-collinear or spin-orbit, twice the number of orbitals) """
        if self.spin.has_noncolinear:
            return self.no * 2
        return self.no

    def __str__(self):
        r""" Representation of the model """
        s = self.__class__.__name__ + f'{{non-zero: {self.nnz}, orthogonal: {self.orthogonal},\n '
        s += str(self.spin).replace('\n', '\n ') + ',\n '
        s += str(self.geometry).replace('\n', '\n ')
        return s + '\n}'

    def __repr__(self):
        g = self.geometry
        spin = {
            Spin.UNPOLARIZED: "unpolarized",
            Spin.POLARIZED: "polarized",
            Spin.NONCOLINEAR: "noncolinear",
            Spin.SPINORBIT: "spinorbit"
            }.get(self.spin._kind, f"unkown({self.spin._kind})")
        return f"<{self.__module__}.{self.__class__.__name__} na={g.na}, no={g.no}, nsc={g.nsc}, dim={self.dim}, nnz={self.nnz}, spin={spin}>"

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
        k = _a.asarrayd(k).ravel()
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
        k = _a.asarrayd(k).ravel()
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

    def _dPk_non_colinear(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Tuple of sparse matrix (``scipy.sparse.csr_matrix``) at `k` for a non-collinear system, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_dk_nc(gauge, self, self.sc, k, dtype, format)

    def _dPk_spin_orbit(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Tuple of sparse matrix (``scipy.sparse.csr_matrix``) at `k` for a non-collinear system, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_dk_so(gauge, self, self.sc, k, dtype, format)

    def _ddPk_non_colinear(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Tuple of sparse matrix (``scipy.sparse.csr_matrix``) at `k` for a non-collinear system, differentiated with respect to `k` twice

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_ddk_nc(gauge, self, self.sc, k, dtype, format)

    def _ddPk_spin_orbit(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
        r""" Tuple of sparse matrix (``scipy.sparse.csr_matrix``) at `k` for a non-collinear system, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge : {'R', 'r'}
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_ddk_so(gauge, self, self.sc, k, dtype, format)

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
        k = _a.asarrayd(k).ravel()
        return matrix_k_nc_diag(gauge, self, self.S_idx, self.sc, k, dtype, format)

    def _dSk_non_colinear(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr'):
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
        k = _a.asarrayd(k).ravel()
        return matrix_dk_nc_diag(gauge, self, self.S_idx, self.sc, k, dtype, format)

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
        if self.orthogonal:
            return lin.eigsh(P, k=n, return_eigenvectors=not eigvals_only, **kwargs)
        S = self.Sk(k=k, dtype=dtype, gauge=gauge)
        return lin.eigsh(P, M=S, k=n, return_eigenvectors=not eigvals_only, **kwargs)

    def transpose(self, hermitian=False, spin=True, sort=True):
        r""" A transpose copy of this object, possibly apply the Hermitian conjugate as well

        Parameters
        ----------
        hermitian : bool, optional
           if true, also emply a spin-box Hermitian operator to ensure TRS, otherwise
           only return the transpose values.
        spin : bool, optional
           whether the spin-box is also transposed if this is false, and `hermitian` is true,
           then only imaginary values will change sign.
        sort : bool, optional
           the returned columns for the transposed structure will be sorted
           if this is true, default
        """
        new = super().transpose(sort=sort)
        sp = self.spin
        D = new._csr._D

        if sp.is_spinorbit:
            if hermitian and spin:
                # conjugate the imaginary value and transpose spin-box
                if sp.dkind == 'f':
                    # imaginary components (including transposing)
                    #    12,11,22,21
                    D[:, [3, 4, 5, 7]] = -D[:, [7, 4, 5, 3]]
                    # R12 <-> R21
                    D[:, [2, 6]] = D[:, [6, 2]]
                else:
                    D[:, [0, 1, 2, 3]] = np.conj(D[:, [0, 1, 3, 2]])
            elif hermitian:
                # conjugate the imaginary value
                if sp.dkind == 'f':
                    # imaginary components
                    #    12,11,22,21
                    D[:, [3, 4, 5, 7]] *= -1.
                else:
                    D[:, :] = np.conj(D[:, :])
            elif spin:
                # transpose spin-box, 12 <-> 21
                if sp.dkind == 'f':
                    D[:, [2, 3, 6, 7]] = D[:, [6, 7, 2, 3]]
                else:
                    D[:, [2, 3]] = D[:, [3, 2]]

        elif sp.is_noncolinear:
            if hermitian and spin:
                pass # do nothing, it is already ensured Hermitian
            elif hermitian or spin:
                # conjugate the imaginary value
                # since for transposing D[:, 3] is the same
                # value used for [--, ud]
                #                [du, --]
                #   ud = D[3] == - du
                # So for transposing we should negate the sign
                # to ensure we put the opposite value in the
                # correct place.
                if sp.dkind == 'f':
                    D[:, 3] = -D[:, 3]
                else:
                    D[:, 2] = np.conj(D[:, 2])

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
            'sparseorbitalbzspin': super().__getstate__(),
            'spin': self._spin.__getstate__(),
            'orthogonal': self._orthogonal,
        }

    def __setstate__(self, state):
        self._orthogonal = state['orthogonal']
        spin = Spin()
        spin.__setstate__(state['spin'])
        self._spin = spin
        super().__setstate__(state['sparseorbitalbzspin'])
