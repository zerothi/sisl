# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import warnings
from typing import Literal, Optional, Union

import numpy as np
from scipy.sparse import SparseEfficiencyWarning, csr_matrix

import sisl._array as _a
import sisl.linalg as lin
from sisl import Geometry
from sisl._core.sparse import SparseCSR, issparse
from sisl._core.sparse_geometry import SparseOrbital, _SparseGeometry
from sisl._help import dtype_complex_to_real, dtype_real_to_complex
from sisl._internal import set_module
from sisl.messages import deprecate_argument, warn
from sisl.typing import (
    AtomsIndex,
    GaugeType,
    KPoint,
    OrSequence,
    SparseMatrix,
    SparseMatrixPhysical,
)

from ._matrix_ddk import (
    matrix_ddk,
    matrix_ddk_diag,
    matrix_ddk_nambu,
    matrix_ddk_nc,
    matrix_ddk_so,
)
from ._matrix_dk import (
    matrix_dk,
    matrix_dk_diag,
    matrix_dk_nambu,
    matrix_dk_nc,
    matrix_dk_so,
)
from ._matrix_k import matrix_k, matrix_k_diag, matrix_k_nambu, matrix_k_nc, matrix_k_so
from .spin import Spin

__all__ = ["SparseOrbitalBZ", "SparseOrbitalBZSpin"]


# Filter warnings from the sparse library
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


def _get_spin(M, spin, what: Literal["trace", "box", "vector"] = "box"):
    if what == "trace":
        if spin.spinor == 2:
            # we have both up+down
            # TODO fix spin-orbit with complex values
            return M[..., 0] + M[..., 1]
        return M[..., 0]

    if what == "vector":
        m = np.empty(M.shape[:-1] + (3,), dtype=dtype_complex_to_real(M.dtype))
        if spin.is_unpolarized:
            # no spin-density
            m[...] = 0.0
        else:
            # Same for all spin-configurations
            m[..., 2] = (M[..., 0] - M[..., 1]).real

            # These indices should be reflected in sisl/physics/sparse.py
            # for the Mxy[ri] indices in the reset method
            if spin.is_polarized:
                m[..., :2] = 0.0
            elif spin.is_noncolinear:
                if np.iscomplexobj(M):
                    m[..., 0] = 2 * M[..., 2].real
                    m[..., 1] = -2 * M[..., 2].imag
                else:
                    m[..., 0] = 2 * M[..., 2]
                    m[..., 1] = -2 * M[..., 3]
            else:
                # spin-orbit
                if np.iscomplexobj(M):
                    tmp = M[..., 2].conj() + M[..., 3]
                    m[..., 0] = tmp.real
                    m[..., 1] = tmp.imag
                else:
                    m[..., 0] = M[..., 2] + M[..., 6]
                    m[..., 1] = -M[..., 3] + M[..., 7]
        return m

    if what == "box":
        m = np.empty(M.shape[:-1] + (2, 2), dtype=dtype_real_to_complex(M.dtype))
        if spin.is_unpolarized:
            # no spin-density
            m[...] = 0.0
            m[..., 0, 0] = M[..., 0]
            m[..., 1, 1] = M[..., 0]
        elif spin.is_polarized:
            m[...] = 0.0
            m[..., 0, 0] = M[..., 0]
            m[..., 1, 1] = M[..., 1]
        elif spin.is_noncolinear:
            if np.iscomplexobj(M):
                m[..., 0, 0] = M[..., 0]
                m[..., 1, 1] = M[..., 1]
                m[..., 0, 1] = M[..., 2]
                m[..., 1, 0] = M[..., 2].conj()
            else:
                m[..., 0, 0] = M[..., 0]
                m[..., 1, 1] = M[..., 1]
                m[..., 0, 1] = M[..., 2] + 1j * M[..., 3]
                m[..., 1, 0] = m[..., 0, 1].conj()
        else:
            if np.iscomplexobj(M):
                m[..., 0, 0] = M[..., 0]
                m[..., 1, 1] = M[..., 1]
                m[..., 0, 1] = M[..., 2]
                m[..., 1, 0] = M[..., 3]
            else:
                m[..., 0, 0] = M[..., 0] + 1j * M[..., 4]
                m[..., 1, 1] = M[..., 1] + 1j * M[..., 5]
                m[..., 0, 1] = M[..., 2] + 1j * M[..., 3]
                m[..., 1, 0] = M[..., 6] + 1j * M[..., 7]

        return m

    raise ValueError(f"Wrong 'what' argument got {what}.")


@set_module("sisl.physics")
class SparseOrbitalBZ(SparseOrbital):
    r"""Sparse object containing the orbital connections in a Brillouin zone

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

    def __init__(
        self,
        geometry: Geometry,
        dim: int = 1,
        dtype=None,
        nnzpr: Optional[int] = None,
        **kwargs,
    ):
        self._geometry = geometry
        self._orthogonal = kwargs.get("orthogonal", True)

        # Get true dimension
        if not self.orthogonal:
            dim = dim + 1

        # Initialize the sparsity pattern
        self.reset(dim, dtype, nnzpr)
        self._reset()

    def _reset(self) -> None:
        r"""Reset object according to the options, please refer to `SparseOrbital.reset` for details"""
        # Update the shape
        self._csr._shape = self.shape[:-1] + self._csr._D.shape[-1:]
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
        return {"orthogonal": self.orthogonal}

    @property
    def orthogonal(self) -> bool:
        r"""True if the object is using an orthogonal basis"""
        return self._orthogonal

    @property
    def non_orthogonal(self) -> bool:
        r"""True if the object is using a non-orthogonal basis"""
        return not self._orthogonal

    def __len__(self) -> int:
        r"""Returns number of rows in the basis (if non-collinear or spin-orbit, twice the number of orbitals)"""
        return self.no

    def __str__(self) -> str:
        r"""Representation of the model"""
        s = f"{self.__class__.__name__}{{dim: {self.dim}, non-zero: {self.nnz}, orthogonal: {self.orthogonal}\n "
        return s + str(self.geometry).replace("\n", "\n ") + "\n}"

    def __repr__(self) -> str:
        g = self.geometry
        return f"<{self.__module__}.{self.__class__.__name__} na={g.na}, no={g.no}, nsc={g.nsc}, dim={self.dim}, nnz={self.nnz}>"

    @property
    def S(self):
        r"""Access the overlap elements associated with the sparse matrix"""
        if self.orthogonal:
            return None
        self._def_dim = self.S_idx
        return self

    @classmethod
    def fromsp(
        cls,
        geometry: Geometry,
        P: Union[OrSequence[SparseMatrix], SparseMatrixPhysical],
        S: Optional[SparseMatrix] = None,
        **kwargs,
    ) -> Self:
        r"""Create a sparse model from a preset `Geometry` and a list of sparse matrices

        The passed sparse matrices are in one of `scipy.sparse` formats.

        Parameters
        ----------
        geometry :
           geometry to describe the new sparse geometry
        P :
           the new sparse matrices that are to be populated in the sparse
           matrix
        S :
           if provided this refers to the overlap matrix and will force the
           returned sparse matrix to be non-orthogonal
        **kwargs :
           any arguments that are directly passed to the ``__init__`` method
           of the class.

        Returns
        -------
        SparseGeometry
             a new sparse matrix that holds the passed geometry and the elements of `P` and optionally being non-orthogonal if `S` is not none
        """
        # Ensure list of csr format (to get dimensions)
        if issparse(P) or isinstance(P, (SparseCSR, _SparseGeometry)):
            P = [P]

        # The logic is that we first have to find out whether
        # we should construct an orthogonal resulting matrix.
        # In this case, if the last argument is a SparseGeometry, and it is
        # non-orthogonal, then we will force it to be non-orthogonal.
        # But we cannot convert
        #   [SparseGeometry(orthogonal=False),
        #    SparseGeometry(orthogonal=False)]
        # to a sparse matrix, because then two of the fields will be
        # the overlap. Perhaps later we should remove all but the last
        # overlap component?
        orthogonal = []
        for p in P:
            try:
                orthogonal.append(p.orthogonal)
            except AttributeError:
                orthogonal.append(True)

        # Check that all but the last sparse matrices are orthogonal
        if not all(orthogonal[:-1]):
            raise ValueError(
                f"{cls.__name__}.fromsp can create a non-orthogonal "
                "matrix if the last item is non-orthogonal."
            )

        # Correct input flag if the last sparse matrix is non-orthogonal
        if not orthogonal[-1]:
            if not kwargs.get("orthogonal", True):
                warn(
                    f"{cls.__name__}.fromsp will override your orthogonal argument "
                    "because the last matrix is non-orthogonal!"
                )
            kwargs["orthogonal"] = False

        # Extract all SparseCSR matrices (or csr_matrix)
        def extract_csr(P):
            try:
                return P._csr
            except AttributeError:
                return P

        P = list(map(extract_csr, P))

        # Number of dimensions, before S!
        def get_3rddim(P):
            if isinstance(P, SparseCSR):
                return P.shape[2]
            return 1

        dim = sum(map(get_3rddim, P))

        if S is None:
            if not kwargs.get("orthogonal", True):
                dim -= 1
        else:
            P.append(S)
            if isinstance(S, SparseCSR):
                if S.shape[2] != 1:
                    raise ValueError(
                        f"{cls.__name__}.fromsp requires S to only have 1 dimension when passing a SparseCSR"
                    )
            kwargs["orthogonal"] = False

        p = cls(geometry, dim, P[0].dtype, 1, **kwargs)
        p._csr = p._csr.fromsp(P, dtype=kwargs.get("dtype"))

        if p._size != P[0].shape[0]:
            raise ValueError(
                f"{cls.__name__}.fromsp cannot create a new class, the geometry "
                "and sparse matrices does not have coinciding dimensions size != P[0].shape[0]"
            )

        return p

    def iter_orbitals(self, atoms: AtomsIndex = None, local: bool = False):
        r"""Iterations of the orbital space in the geometry, two indices from loop

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

    def _Pk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
        _dim=0,
    ):
        r"""Sparse matrix (`scipy.sparse.csr_matrix`) at `k` for a polarized system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_k(gauge, self, _dim, self.lattice, k, dtype, format)

    def _dPk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
        _dim=0,
    ):
        r"""Sparse matrix (`scipy.sparse.csr_matrix`) at `k` differentiated with respect to `k` for a polarized system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_dk(gauge, self, _dim, self.lattice, k, dtype, format)

    def _ddPk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
        _dim=0,
    ):
        r"""Sparse matrix (`scipy.sparse.csr_matrix`) at `k` double differentiated with respect to `k` for a polarized system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_ddk(gauge, self, _dim, self.lattice, k, dtype, format)

    def Sk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
        *args,
        **kwargs,
    ):  # pylint: disable=E0202
        r"""Setup the overlap matrix for a given k-point

        Creation and return of the overlap matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the lattice vector gauge:

        .. math::
           \mathbf S(\mathbf k) = \mathbf S_{ij} e^{i\mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \mathbf S(\mathbf k) = \mathbf S_{ij} e^{i\mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is the distance between the orbitals.

        Parameters
        ----------
        k : array_like, optional
           the k-point to setup the overlap at (default Gamma point)
        dtype : numpy.dtype, optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge
        format : {"csr", "array", "matrix", "coo", ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`"array"`/`"dense"`/`"matrix"`).
           Prefixing with "sc:", or simply "sc" returns the matrix in supercell format
           with phases. This is useful for e.g. bond-current calculations where individual
           hopping + phases are required.

        See Also
        --------
        dSk : Overlap matrix derivative with respect to `k`
        ddSk : Overlap matrix double derivative with respect to `k`

        Returns
        -------
        matrix : numpy.ndarray or scipy.sparse.*_matrix
            the overlap matrix at :math:`\mathbf k`. The returned object depends on `format`.
        """
        pass

    def _Sk_diagonal(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
        *args,
        **kwargs,
    ):
        r"""For an orthogonal case we always return the identity matrix"""
        if dtype is None:
            dtype = np.float64
        nr = len(self)
        nc = nr
        if "sc:" in format:
            format = format[3:]
            nc = self.n_s * nr
        elif "sc" == format:
            format = "csr"
            nc = self.n_s * nr
        # In the "rare" but could be found situation where
        # the matrix only describes neighboring couplings it is vital
        # to not return anything
        # TODO
        if format in ("array", "matrix", "dense"):
            S = np.zeros([nr, nc], dtype=dtype)
            np.fill_diagonal(S, 1.0)
            return S
        S = csr_matrix((nr, nc), dtype=dtype)
        S.setdiag(1.0)
        return S.asformat(format)

    def _Sk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Overlap matrix in a `scipy.sparse.csr_matrix` at `k`.

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        return self._Pk(k, dtype=dtype, gauge=gauge, format=format, _dim=self.S_idx)

    def dSk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
        *args,
        **kwargs,
    ):
        r"""Setup the :math:`\mathbf k`-derivatie of the overlap matrix for a given k-point

        Creation and return of the derivative of the overlap matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the lattice vector gauge:

        .. math::
           \nabla_{\mathbf k} \mathbf S_\alpha(\mathbf k) = i \mathbf R_\alpha \mathbf S_{ij} e^{i\mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.
        And :math:`\alpha` is one of the Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \nabla_{\mathbf k} \mathbf S_\alpha(\mathbf k) = i \mathbf r_\alpha \mathbf S_{ij} e^{i\mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is the distance between the orbitals.

        Parameters
        ----------
        k : array_like, optional
           the k-point to setup the overlap at (default Gamma point)
        dtype : numpy.dtype, optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge.
        format : {"csr", "array", "matrix", "coo", ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`"array"`/`"dense"`/`"matrix"`).

        See Also
        --------
        Sk : Overlap matrix at `k`
        ddSk : Overlap matrix double derivative at `k`

        Returns
        -------
        tuple
            for each of the Cartesian directions a :math:`\partial \mathbf S(\mathbf k)/\partial\mathbf k` is returned.
        """
        pass

    def _dSk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Overlap matrix in a `scipy.sparse.csr_matrix` at `k` differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        return self._dPk(k, dtype=dtype, gauge=gauge, format=format, _dim=self.S_idx)

    def _dSk_non_colinear(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Overlap matrix in a `scipy.sparse.csr_matrix` at `k` for non-collinear spin, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_dk_nc_diag(
            gauge, self, self.S_idx, self.lattice, k, dtype, format
        )

    def ddSk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
        *args,
        **kwargs,
    ):
        r"""Setup the double :math:`\mathbf k`-derivatie of the overlap matrix for a given k-point

        Creation and return of the double derivative of the overlap matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_{\mathbf k^2} \mathbf S_{\alpha\beta}(\mathbf k) = - \mathbf R_\alpha \mathbf R_\beta \mathbf S_{ij} e^{i\mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.
        And :math:`\alpha` and :math:`\beta` are one of the Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \nabla_{\mathbf k^2} \mathbf S_{\alpha\beta}(\mathbf k) = - \mathbf r_\alpha \mathbf r_\beta \mathbf S_{ij} e^{i\mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is the distance between the orbitals.

        Parameters
        ----------
        k : array_like, optional
           the k-point to setup the overlap at (default Gamma point)
        dtype : numpy.dtype, optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge, ``cell`` for cell vector gauge, and ``atom`` for atomic distance
           gauge.
        format : {"csr", "array", "matrix", "coo", ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`"array"`/`"dense"`/`"matrix"`).

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

    def _ddSk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Overlap matrix in a `scipy.sparse.csr_matrix` at `k` double differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        return self._ddPk(k, dtype=dtype, gauge=gauge, format=format, _dim=self.S_idx)

    def _ddSk_non_colinear(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Overlap matrix in a `scipy.sparse.csr_matrix` at `k` for non-collinear spin, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_ddk_diag(
            gauge, self, self.S_idx, 2, self.lattice, k, dtype, format
        )

    def _ddSk_nambu(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Overlap matrix in a `scipy.sparse.csr_matrix` at `k` for Nambu spin, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_ddk_diag(
            gauge, self, self.S_idx, 4, self.lattice, k, dtype, format
        )

    def eig(
        self,
        k: KPoint = (0, 0, 0),
        gauge: GaugeType = "lattice",
        eigvals_only: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""Returns the eigenvalues of the physical quantity (using the non-Hermitian solver)

        Setup the system and overlap matrix with respect to
        the given k-point and calculate the eigenvalues.

        All subsequent arguments gets passed directly to `scipy.linalg.eig`
        """
        dtype = kwargs.pop("dtype", None)
        P = self.Pk(k=k, dtype=dtype, gauge=gauge, format="array")
        if self.orthogonal:
            if eigvals_only:
                return lin.eigvals_destroy(P, **kwargs)
            return lin.eig_destroy(P, **kwargs)

        S = self.Sk(k=k, dtype=dtype, gauge=gauge, format="array")
        if eigvals_only:
            return lin.eigvals_destroy(P, S, **kwargs)
        return lin.eig_destroy(P, S, **kwargs)

    def eigh(
        self,
        k: KPoint = (0, 0, 0),
        gauge: GaugeType = "lattice",
        eigvals_only: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""Returns the eigenvalues of the physical quantity

        Setup the system and overlap matrix with respect to
        the given k-point and calculate the eigenvalues.

        All subsequent arguments gets passed directly to `scipy.linalg.eigh`
        """
        dtype = kwargs.pop("dtype", None)
        P = self.Pk(k=k, dtype=dtype, gauge=gauge, format="array")
        if self.orthogonal:
            return lin.eigh_destroy(P, eigvals_only=eigvals_only, **kwargs)

        S = self.Sk(k=k, dtype=dtype, gauge=gauge, format="array")
        return lin.eigh_destroy(P, S, eigvals_only=eigvals_only, **kwargs)

    def eigsh(
        self,
        k: KPoint = (0, 0, 0),
        n: int = 1,
        gauge: GaugeType = "lattice",
        eigvals_only: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""Calculates a subset of eigenvalues of the physical quantity using sparse matrices

        Setup the quantity and overlap matrix with respect to
        the given k-point and calculate a subset of the eigenvalues using the sparse algorithms.

        All subsequent arguments gets passed directly to `scipy.sparse.linalg.eigsh`.

        Parameters
        ----------
        n :
            number of eigenvalues to calculate.
            Defaults to the `n` smallest magnitude eigevalues.
        **kwargs:
            arguments passed directly to `scipy.sparse.linalg.eigsh`.

        Notes
        -----
        The performance and accuracy of this method depends heavily on `kwargs`.
        Playing around with a small test example before doing large scale calculations
        is adviced!
        """
        # We always request the smallest eigenvalues...
        kwargs.update({"which": kwargs.get("which", "SM")})

        dtype = kwargs.pop("dtype", None)

        P = self.Pk(k=k, dtype=dtype, gauge=gauge)
        if self.orthogonal:
            return lin.eigsh(P, k=n, return_eigenvectors=not eigvals_only, **kwargs)
        S = self.Sk(k=k, dtype=dtype, gauge=gauge)
        return lin.eigsh(P, M=S, k=n, return_eigenvectors=not eigvals_only, **kwargs)

    def astype(self, dtype, copy: bool = True) -> Self:
        """Convert the stored data-type to something else

        Parameters
        ----------
        dtype :
            the new dtype for the sparse matrix
        copy :
            copy when needed, or do not copy when not needed.
        """
        old_dtype = np.dtype(self.dtype)
        new_dtype = np.dtype(dtype)

        if old_dtype == new_dtype:
            if copy:
                return self.copy()
            return self

        new = self.copy()
        new._csr = new._csr.astype(dtype, copy=copy)
        new._reset()
        return new

    def __getstate__(self):
        return {
            "sparseorbitalbz": super().__getstate__(),
            "orthogonal": self._orthogonal,
        }

    def __setstate__(self, state):
        self._orthogonal = state["orthogonal"]
        super().__setstate__(state["sparseorbitalbz"])
        self._reset()


@set_module("sisl.physics")
class SparseOrbitalBZSpin(SparseOrbitalBZ):
    r"""Sparse object containing the orbital connections in a Brillouin zone with possible spin-components

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

    def __init__(
        self,
        geometry: Geometry,
        dim: int = 1,
        dtype=None,
        nnzpr: Optional[int] = None,
        **kwargs,
    ):
        # Check that the passed parameters are correct
        if "spin" not in kwargs:
            if isinstance(dim, Spin):
                spin = dim
            else:
                # Back conversion, actually this should depend
                # on dtype
                spin = {
                    1: Spin.UNPOLARIZED,
                    2: Spin.POLARIZED,
                    4: Spin.NONCOLINEAR,
                    8: Spin.SPINORBIT,
                    16: Spin.NAMBU,
                }.get(dim)
        else:
            spin = kwargs.pop("spin")
        self._spin = Spin(spin)

        super().__init__(geometry, self.spin.size(dtype), dtype, nnzpr, **kwargs)
        self._reset()

    def _reset(self) -> None:
        r"""Reset object according to the options, please refer to `SparseOrbital.reset` for details"""
        super()._reset()

        # Update the dtype of the spin
        self._spin = Spin(self.spin)

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
            if self.dkind in ("f", "i"):
                self.M11 = 0
                self.M22 = 1
                self.M12r = 2
                self.M12i = 3
            else:
                self.M11 = 0
                self.M22 = 1
                self.M12 = 2
            self.Pk = self._Pk_non_colinear
            self.Sk = self._Sk_non_colinear
            self.dPk = self._dPk_non_colinear
            self.dSk = self._dSk_non_colinear
            self.ddPk = self._ddPk_non_colinear
            self.ddSk = self._ddSk_non_colinear

        elif self.spin.is_spinorbit:
            if self.dkind in ("f", "i"):
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

            # The overlap is the same as non-collinear
            self.Pk = self._Pk_spin_orbit
            self.Sk = self._Sk_non_colinear
            self.dPk = self._dPk_spin_orbit
            self.dSk = self._dSk_non_colinear
            self.ddPk = self._ddPk_spin_orbit
            self.ddSk = self._ddSk_non_colinear

        elif self.spin.is_nambu:
            if self.dkind in ("f", "i"):
                self.M11r = 0
                self.M22r = 1
                self.M12r = 2
                self.M12i = 3
                self.M11i = 4
                self.M22i = 5
                self.M21r = 6
                self.M21i = 7
                self.MSr = 8
                self.MSi = 9
                self.MT11r = 10
                self.MT11i = 11
                self.MT22r = 12
                self.MT22i = 13
                self.MT0r = 14
                self.MT0i = 15
            else:
                self.M11 = 0
                self.M22 = 1
                self.M12 = 2
                self.M21 = 3
                self.MS = 4
                self.MT11 = 5
                self.MT22 = 6
                self.MT0 = 7

            self.Pk = self._Pk_nambu
            self.Sk = self._Sk_nambu
            self.dPk = self._dPk_nambu
            self.dSk = self._dSk_nambu
            self.ddPk = self._ddPk_nambu
            self.ddSk = self._ddSk_nambu

        if self.orthogonal:
            self.Sk = self._Sk_diagonal

    # Override to enable spin configuration and orthogonality
    def _cls_kwargs(self):
        return {"spin": self.spin.kind, "orthogonal": self.orthogonal}

    @property
    def spin(self) -> Spin:
        r"""Associated spin class"""
        return self._spin

    def create_construct(self, R, params):
        r"""Create a simple function for passing to the `construct` function.

        This is to relieve the creation of simplistic
        functions needed for setting up sparse elements.

        For simple matrices this returns a function:

        >>> def func(self, ia, atoms, atoms_xyz=None):
        ...     idx = self.geometry.close(ia, R=R, atoms=atoms, atoms_xyz=atoms_xyz)
        ...     for ix, p in zip(idx, params):
        ...         self[ia, ix] = p

        In the non-colinear case the matrix element :math:`\mathbf M_{ij}` will be set
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
        R :
           radii parameters for different shells.
           Must have same length as `params` or one less.
           If one less it will be extended with ``R[0]/100``
        params :
           coupling constants corresponding to the `R`
           ranges. ``params[0,:]`` are the elements
           for the all atoms within ``R[0]`` of each atom.

        See Also
        --------
        construct : routine to create the sparse matrix from a generic function (as returned from `create_construct`)
        """
        if len(R) != len(params):
            raise ValueError(
                f"{self.__class__.__name__}.create_construct got different lengths of 'R' and 'params'"
            )
        if not self.spin.is_diagonal:
            # This portion of code splits the construct into doing Hermitian
            # assignments. This probably needs rigorous testing.

            dtype_cplx = dtype_real_to_complex(self.dtype)

            is_complex = self.dkind == "c"
            if self.spin.is_nambu:
                if is_complex:
                    nv = 8
                    # Hermitian parameters
                    # The input order is [uu, dd, ud, du]
                    paramsH = [
                        [
                            # H^ee
                            p[0].conjugate(),
                            p[1].conjugate(),
                            p[3].conjugate(),
                            p[2].conjugate(),
                            # delta, note the singlet
                            p[4],
                            -p[5],
                            -p[6],
                            -p[7],
                            # because it is already off-diagonal
                            *p[8:],
                        ]
                        for p in params
                    ]
                else:
                    nv = 16
                    # Hermitian parameters
                    # The input order is [Ruu, Rdd, Rud, Iud, Iuu, Idd, Rdu, idu]
                    #                    [ RS,  IS, RTu, ITu, RTd, ITd, RT0, IT0]
                    # delta, note the singlet!
                    paramsH = [
                        [
                            p[0],
                            p[1],
                            p[6],
                            -p[7],
                            -p[4],
                            -p[5],
                            p[2],
                            -p[3],
                            p[8],
                            p[9],
                            -p[10],
                            -p[11],
                            -p[12],
                            -p[13],
                            -p[14],
                            -p[15],
                            *p[16:],
                        ]
                        for p in params
                    ]
                if not self.orthogonal:
                    nv += 1

                # ensure we have correct number of values
                assert all(len(p) == nv for p in params)

                if R[0] <= 0.1001:  # no atom closer than 0.1001 Ang!
                    # We check that the parameters here is Hermitian
                    p = params[0]
                    if is_complex:
                        Me = np.array([[p[0], p[2]], [p[3], p[1]]], dtype_cplx)
                        # do Delta
                        Md = np.array(
                            [[p[5], p[4] + p[7]], [-p[4] + p[7], p[6]]], dtype_cplx
                        )
                    else:
                        Me = np.array(
                            [
                                [p[0] + 1j * p[4], p[2] + 1j * p[3]],
                                [p[6] + 1j * p[7], p[1] + 1j * p[5]],
                            ],
                            dtype_cplx,
                        )
                        Md = np.array(
                            [
                                [
                                    p[10] + 1j * p[11],
                                    p[8] + p[14] + 1j * (p[9] + p[15]),
                                ],
                                [
                                    -p[8] + p[14] + 1j * (-p[9] + p[15]),
                                    p[12] + 1j * p[13],
                                ],
                            ],
                            dtype_cplx,
                        )
                    d = Me - Me.T.conjugate()
                    if not np.allclose(d, 0):
                        warn(
                            f"{self.__class__.__name__}.create_construct got a "
                            f"non-Hermitian on-site term for the M^e elements ({d.ravel()}). "
                            "The code will continue like nothing happened..."
                        )
                    # The sub-diagonal Delta is equivalent to -D^*.
                    # This means that to compare one should do:
                    #   (-D.conjugate()).T.conjugate()
                    # which can be reduced to the following:
                    d = Md + Md.T
                    if not np.allclose(d, 0):
                        warn(
                            f"{self.__class__.__name__}.create_construct got a "
                            f"non-Hermitian on-site term for the M^d elements ({d.ravel()}). "
                            "The code will continue like nothing happened..."
                        )
            elif self.spin.is_spinorbit:
                if is_complex:
                    nv = 4
                    # Hermitian parameters
                    # The input order is [uu, dd, ud, du]
                    paramsH = [
                        [
                            p[0].conjugate(),
                            p[1].conjugate(),
                            p[3].conjugate(),
                            p[2].conjugate(),
                            *p[4:],
                        ]
                        for p in params
                    ]
                else:
                    nv = 8
                    # Hermitian parameters
                    # The input order is [Ruu, Rdd, Rud, Iud, Iuu, Idd, Rdu, idu]
                    paramsH = [
                        [p[0], p[1], p[6], -p[7], -p[4], -p[5], p[2], -p[3], *p[8:]]
                        for p in params
                    ]
                if not self.orthogonal:
                    nv += 1

                # ensure we have correct number of values
                assert all(len(p) == nv for p in params)

                if R[0] <= 0.1001:  # no atom closer than 0.1001 Ang!
                    # We check that the the parameters here is Hermitian
                    p = params[0]
                    if is_complex:
                        onsite = np.array([[p[0], p[2]], [p[3], p[1]]], dtype_cplx)
                    else:
                        onsite = np.array(
                            [
                                [p[0] + 1j * p[4], p[2] + 1j * p[3]],
                                [p[6] + 1j * p[7], p[1] + 1j * p[5]],
                            ],
                            dtype_cplx,
                        )
                    d = onsite - onsite.T.conjugate()
                    if not np.allclose(d, 0):
                        warn(
                            f"{self.__class__.__name__}.create_construct got a "
                            f"non-Hermitian on-site term elements ({d.ravel()}). "
                            "The code will continue like nothing happened..."
                        )

            elif self.spin.is_noncolinear:
                if is_complex:
                    nv = 3
                    # Hermitian parameters
                    paramsH = [
                        [p[0].conjugate(), p[1].conjugate(), p[2], *p[3:]]
                        for p in params
                    ]
                else:
                    nv = 4
                    # Hermitian parameters
                    # Note that we don't need to do anything here.
                    # H_ij = [[0, 2 + 1j 3],
                    #         [2 - 1j 3, 1]]
                    # H_ji = [[0, 2 + 1j 3],
                    #         [2 - 1j 3, 1]]
                    # H_ij^H == H_ji^H
                    paramsH = params
                if not self.orthogonal:
                    nv += 1

                # we don"t need to check hermiticity for NC
                # Since the values are ensured Hermitian in the on-site case anyways.

                # ensure we have correct number of values
                assert all(len(p) == nv for p in params)

            na = self.geometry.na

            # Now create the function that returns the assignment function
            def func(self, ia, atoms, atoms_xyz=None):
                idx = self.geometry.close(ia, R=R, atoms=atoms, atoms_xyz=atoms_xyz)
                for ix, p, pc in zip(idx, params, paramsH):
                    ix_ge = (ix % na) >= ia
                    self[ia, ix[ix_ge]] = p
                    self[ia, ix[~ix_ge]] = pc

            func.R = R
            func.params = params
            func.paramsH = paramsH

            return func

        return super().create_construct(R, params)

    def __len__(self) -> int:
        r"""Returns number of rows in the basis (accounts for the non-collinear cases)"""
        if self.spin.is_diagonal:
            return self.no
        # This will correctly multiply with the spinor size
        # The spinors depends on NC/SOC/Nambu/Polarized
        return self.no * self.spin.spinor

    def __str__(self) -> str:
        r"""Representation of the model"""
        s = (
            self.__class__.__name__
            + f"{{non-zero: {self.nnz}, orthogonal: {self.orthogonal},\n "
        )
        s += str(self.spin).replace("\n", "\n ") + ",\n "
        s += str(self.geometry).replace("\n", "\n ")
        return s + "\n}"

    def __repr__(self) -> str:
        g = self.geometry
        spin = {
            Spin.UNPOLARIZED: "unpolarized",
            Spin.POLARIZED: "polarized",
            Spin.NONCOLINEAR: "noncolinear",
            Spin.SPINORBIT: "spinorbit",
            Spin.NAMBU: "nambu",
        }.get(self.spin.kind, f"unkown({self.spin.kind})")
        return f"<{self.__module__}.{self.__class__.__name__} na={g.na}, no={g.no}, nsc={g.nsc}, dim={self.dim}, nnz={self.nnz}, spin={spin}>"

    def _Pk_unpolarized(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Sparse matrix (`scipy.sparse.csr_matrix`) at `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        return self._Pk(k, dtype=dtype, gauge=gauge, format=format)

    def _Pk_polarized(
        self,
        k: KPoint = (0, 0, 0),
        spin=0,
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Sparse matrix (`scipy.sparse.csr_matrix`) at `k` for a polarized system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        spin : int, optional
           the spin-index of the quantity
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        return self._Pk(k, dtype=dtype, gauge=gauge, format=format, _dim=spin)

    def _Pk_non_colinear(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Sparse matrix (`scipy.sparse.csr_matrix`) at `k` for a non-collinear system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_k_nc(gauge, self, self.lattice, k, dtype, format)

    def _Pk_spin_orbit(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Sparse matrix (`scipy.sparse.csr_matrix`) at `k` for a spin-orbit system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_k_so(gauge, self, self.lattice, k, dtype, format)

    def _Pk_nambu(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Sparse matrix (`scipy.sparse.csr_matrix`) at `k` for a Nambu system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_k_nambu(gauge, self, self.lattice, k, dtype, format)

    def _dPk_unpolarized(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Tuple of sparse matrix (`scipy.sparse.csr_matrix`) at `k`, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        return self._dPk(k, dtype=dtype, gauge=gauge, format=format)

    def _dPk_polarized(
        self,
        k: KPoint = (0, 0, 0),
        spin=0,
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Tuple of sparse matrix (`scipy.sparse.csr_matrix`) at `k`, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        spin : int, optional
           the spin-index of the quantity
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        return self._dPk(k, dtype=dtype, gauge=gauge, format=format, _dim=spin)

    def _dPk_non_colinear(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Tuple of sparse matrix (`scipy.sparse.csr_matrix`) at `k` for a non-collinear system, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_dk_nc(gauge, self, self.lattice, k, dtype, format)

    def _dPk_spin_orbit(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Tuple of sparse matrix (`scipy.sparse.csr_matrix`) at `k` for a spin-orbit system, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_dk_so(gauge, self, self.lattice, k, dtype, format)

    def _dPk_nambu(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Tuple of sparse matrix (`scipy.sparse.csr_matrix`) at `k` for a Nambu spin system, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_dk_nambu(gauge, self, self.lattice, k, dtype, format)

    def _ddPk_non_colinear(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Tuple of sparse matrix (`scipy.sparse.csr_matrix`) at `k` for a non-collinear system, differentiated with respect to `k` twice

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_ddk_nc(gauge, self, self.lattice, k, dtype, format)

    def _ddPk_spin_orbit(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Tuple of sparse matrix (`scipy.sparse.csr_matrix`) at `k` for a spin-orbit system, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_ddk_so(gauge, self, self.lattice, k, dtype, format)

    def _ddPk_nambu(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Tuple of sparse matrix (`scipy.sparse.csr_matrix`) at `k` for a Nambu system, differentiated with respect to `k`

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_ddk_nambu(gauge, self, self.lattice, k, dtype, format)

    def _Sk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Overlap matrix in a `scipy.sparse.csr_matrix` at `k`.

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        return self._Pk(k, dtype=dtype, gauge=gauge, format=format, _dim=self.S_idx)

    def _Sk_non_colinear(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Overlap matrix (`scipy.sparse.csr_matrix`) at `k` for a non-collinear system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_k_diag(gauge, self, self.S_idx, 2, self.lattice, k, dtype, format)

    def _Sk_nambu(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Overlap matrix (`scipy.sparse.csr_matrix`) at `k` for a Nambu system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_k_diag(gauge, self, self.S_idx, 4, self.lattice, k, dtype, format)

    def _dSk_non_colinear(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Overlap matrix (`scipy.sparse.csr_matrix`) at `k` for a non-collinear system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_dk_diag(
            gauge, self, self.S_idx, 2, self.lattice, k, dtype, format
        )

    def _dSk_nambu(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format: str = "csr",
    ):
        r"""Overlap matrix (`scipy.sparse.csr_matrix`) at `k` for a Nambu system

        Parameters
        ----------
        k : array_like, optional
           k-point (default is Gamma point)
        dtype : numpy.dtype, optional
           default to `numpy.complex128`
        gauge :
           chosen gauge
        """
        k = _a.asarrayd(k).ravel()
        return matrix_dk_diag(
            gauge, self, self.S_idx, 4, self.lattice, k, dtype, format
        )

    def eig(
        self,
        k: KPoint = (0, 0, 0),
        gauge: GaugeType = "lattice",
        eigvals_only: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""Returns the eigenvalues of the physical quantity (using the non-Hermitian solver)

        Setup the system and overlap matrix with respect to
        the given k-point and calculate the eigenvalues.

        All subsequent arguments gets passed directly to `scipy.linalg.eig`

        Parameters
        ----------
        spin : int, optional
           the spin-component to calculate the eigenvalue spectrum of, note that
           this parameter is only valid for `Spin.POLARIZED` matrices.
        """
        spin = kwargs.pop("spin", 0)
        dtype = kwargs.pop("dtype", None)

        if self.spin.kind == Spin.POLARIZED:
            P = self.Pk(k=k, dtype=dtype, gauge=gauge, spin=spin, format="array")
        else:
            P = self.Pk(k=k, dtype=dtype, gauge=gauge, format="array")

        if self.orthogonal:
            if eigvals_only:
                return lin.eigvals_destroy(P, **kwargs)
            return lin.eig_destroy(P, **kwargs)

        S = self.Sk(k=k, dtype=dtype, gauge=gauge, format="array")
        if eigvals_only:
            return lin.eigvals_destroy(P, S, **kwargs)
        return lin.eig_destroy(P, S, **kwargs)

    def eigh(
        self,
        k: KPoint = (0, 0, 0),
        gauge: GaugeType = "lattice",
        eigvals_only: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""Returns the eigenvalues of the physical quantity

        Setup the system and overlap matrix with respect to
        the given k-point and calculate the eigenvalues.

        All subsequent arguments gets passed directly to `scipy.linalg.eigh`

        Parameters
        ----------
        spin : int, optional
           the spin-component to calculate the eigenvalue spectrum of, note that
           this parameter is only valid for `Spin.POLARIZED` matrices.
        """
        spin = kwargs.pop("spin", 0)
        dtype = kwargs.pop("dtype", None)

        if self.spin.kind == Spin.POLARIZED:
            P = self.Pk(k=k, dtype=dtype, gauge=gauge, spin=spin, format="array")
        else:
            P = self.Pk(k=k, dtype=dtype, gauge=gauge, format="array")

        if self.orthogonal:
            return lin.eigh_destroy(P, eigvals_only=eigvals_only, **kwargs)

        S = self.Sk(k=k, dtype=dtype, gauge=gauge, format="array")
        return lin.eigh_destroy(P, S, eigvals_only=eigvals_only, **kwargs)

    def eigsh(
        self,
        k: KPoint = (0, 0, 0),
        n: int = 1,
        gauge: GaugeType = "lattice",
        eigvals_only: bool = True,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        r"""Calculates a subset of eigenvalues of the physical quantity using sparse matrices

        Setup the quantity and overlap matrix with respect to
        the given k-point and calculate a subset of the eigenvalues using the sparse algorithms.

        All subsequent arguments gets passed directly to `scipy.sparse.linalg.eigsh`.

        Parameters
        ----------
        n :
           number of eigenvalues to calculate
           Defaults to the `n` smallest magnitude eigevalues.
        spin : int, optional
           the spin-component to calculate the eigenvalue spectrum of, note that
           this parameter is only valid for `Spin.POLARIZED` matrices.
        **kwargs:
            arguments passed directly to `scipy.sparse.linalg.eigsh`.

        Notes
        -----
        The performance and accuracy of this method depends heavily on `kwargs`.
        Playing around with a small test example before doing large scale calculations
        is adviced!
        """
        # We always request the smallest eigenvalues...
        spin = kwargs.pop("spin", 0)
        dtype = kwargs.pop("dtype", None)
        kwargs.update({"which": kwargs.get("which", "SM")})

        if self.spin.kind == Spin.POLARIZED:
            P = self.Pk(k=k, dtype=dtype, spin=spin, gauge=gauge)
        else:
            P = self.Pk(k=k, dtype=dtype, gauge=gauge)
        if self.orthogonal:
            return lin.eigsh(P, k=n, return_eigenvectors=not eigvals_only, **kwargs)
        S = self.Sk(k=k, dtype=dtype, gauge=gauge)
        return lin.eigsh(P, M=S, k=n, return_eigenvectors=not eigvals_only, **kwargs)

    @deprecate_argument(
        "hermitian",
        "conjugate",
        "hermitian argument has changed to conjugate, please update " "your code",
        "0.15.3",
        "0.16",
    )
    def transpose(
        self, *, conjugate: bool = False, spin: bool = True, sort: bool = True
    ) -> Self:
        r"""A transpose copy of this object with options for spin-box and conjugations

        Notes
        -----
        The overlap elements won't be conjugated, in case asked for.

        Parameters
        ----------
        conjugate :
           if true, also apply a conjugation of the values.
           Together with ``spin=True``, this will result in the adjoint operator.
        spin :
           whether the spin-box is also transposed if this is false, and `hermitian` is true,
           then only imaginary values will change sign.
        sort :
           the returned columns for the transposed structure will be sorted
           if this is true, default
        """
        new = super().transpose(sort=sort)
        sp = self.spin
        D = new._csr._D

        if sp.is_nambu:
            if conjugate and spin:
                # conjugate the imaginary value and transpose spin-box
                # For Nambu things are a bit different.
                # This is because we have an extra Delta sub-matrix
                # which is already *off-diagonal*.
                # And so when you do a ^H, you'll here do a i-j ^H
                # but also an internal -Delta^* that needs accounting.
                # I.e. the Hermitian property says:
                #  Delta_eihj = Delta_hjei^H
                #             = (-Delta_ejhi^*)^H
                # Which means:
                #  Delta_ueiuhj = -Delta_uejuhi
                #  Delta_ueidhj = -Delta_dejuhi
                #  Delta_deiuhj = -Delta_uejdhi
                #  Delta_deidhj = -Delta_dejdhi
                # I.e. anti-Hermitian *only* in the electron-hole indices.
                if self.dkind in ("f", "i"):
                    # imaginary components (including transposing)
                    #    12,11,22,21
                    D[:, [3, 4, 5, 7]] = -D[:, [7, 4, 5, 3]]
                    # R12 <-> R21
                    D[:, [2, 6]] = D[:, [6, 2]]
                    # Delta values
                    D[:, 10:16] = -D[:, 10:16]
                else:
                    D[:, [0, 1, 2, 3]] = np.conj(D[:, [0, 1, 3, 2]])
                    # Delta values
                    D[:, 5:8] = -D[:, 5:8]

            elif conjugate:
                # conjugate the imaginary value
                if self.dkind in ("f", "i"):
                    # imaginary components
                    #    12,11,22,21
                    D[:, [3, 4, 5, 7, 9, 11, 13, 15]] = -D[
                        :, [3, 4, 5, 7, 9, 11, 13, 15]
                    ]
                else:
                    D[:, :8] = np.conj(D[:, :8])
            elif spin:
                # transpose spin-box, 12 <-> 21
                if self.dkind in ("f", "i"):
                    D[:, [2, 3, 6, 7]] = D[:, [6, 7, 2, 3]]
                    D[:, [9, 10, 12, 14]] = -D[:, [9, 10, 12, 14]]
                else:
                    D[:, [2, 3]] = D[:, [3, 2]]
                    D[:, 4] = np.conj(D[:, 4])
                    D[:, 5:8] = -np.conj(D[:, 5:8])

        elif sp.is_spinorbit:
            if conjugate and spin:
                # conjugate the imaginary value and transpose spin-box
                if self.dkind in ("f", "i"):
                    # imaginary components (including transposing)
                    #    12,11,22,21
                    D[:, [3, 4, 5, 7]] = -D[:, [7, 4, 5, 3]]
                    # R12 <-> R21
                    D[:, [2, 6]] = D[:, [6, 2]]
                else:
                    D[:, [0, 1, 2, 3]] = np.conj(D[:, [0, 1, 3, 2]])
            elif conjugate:
                # conjugate the imaginary value
                if self.dkind in ("f", "i"):
                    # imaginary components
                    #    12,11,22,21
                    D[:, [3, 4, 5, 7]] = -D[:, [3, 4, 5, 7]]
                else:
                    D[:, :4] = np.conj(D[:, :4])
            elif spin:
                # transpose spin-box, 12 <-> 21
                if self.dkind in ("f", "i"):
                    D[:, [2, 3, 6, 7]] = D[:, [6, 7, 2, 3]]
                else:
                    D[:, [2, 3]] = D[:, [3, 2]]

        elif sp.is_noncolinear:
            if conjugate and spin:
                if self.dkind in ("f", "i"):
                    pass
                else:
                    # While strictly not necessary, this is vital
                    # if the user has wrong specification
                    # of the on-site terms
                    D[:, [0, 1]] = np.conj(D[:, [0, 1]])
            elif conjugate:
                # conjugate the imaginary value
                # since for transposing D[:, 3] is the same
                if self.dkind in ("f", "i"):
                    D[:, 3] = -D[:, 3]
                else:
                    D[:, :3] = np.conj(D[:, :3])
            elif spin:
                # conjugate the imaginary value
                # since for transposing D[:, 3] is the same
                # value used for [--, ud]
                #                [du, --]
                #   ud = D[3] == - du
                # So for transposing we should negate the sign
                # to ensure we put the opposite value in the
                # correct place.
                if self.dkind in ("f", "i"):
                    D[:, 3] = -D[:, 3]
                else:
                    D[:, 2] = np.conj(D[:, 2])

        elif conjugate:
            ns = sp.size(D.dtype)
            D[:, :ns] = np.conj(D[:, :ns])

        if self.dkind not in ("f", "i") and conjugate and not self.orthogonal:
            D[:, -1] = np.conj(D[:, -1])

        return new

    def trs(self) -> Self:
        r"""Return a matrix with applied time-reversal operator

        For a Hamiltonian to obey time reversal symmetry, it must hold this
        equality:

        .. math::

            \mathbf M = \boldsymbol\sigma_y \mathbf M^* \boldsymbol\sigma_y

        This method returns the RHS of the above equation.

        If you want to ensure that your matrix fulfills TRS, simply do:

        .. code::

            M = (M + M.trs()) / 2

        Notes
        -----
        This method will be obsoleted at some point when :issue:`816` is
        completed.
        """
        new = self.copy()
        sp = self.spin
        D = new._csr._D

        # Apply Pauli-Y on the left and right of each spin-box
        if sp.is_nambu:
            if self.dkind in ("f", "i"):
                D[:, [0, 1, 3, 7]] = D[:, [1, 0, 7, 3]]  # diag real, off imag
                D[:, [2, 4, 5, 6]] = -D[:, [6, 5, 4, 2]]  # diag imag, off real

                # Re: S,Tu,Td
                D[:, [8, 10, 12]] = D[:, [8, 12, 10]]
                # Im: S,Tu,Td
                D[:, [9, 11, 13]] = -D[:, [9, 13, 11]]
                # Re: T0
                D[:, 14] = -D[:, 14]
                # nothing for Im T0
            else:
                D[:, [0, 1]] = np.conj(D[:, [1, 0]])
                D[:, [2, 3]] = -np.conj(D[:, [3, 2]])

                # S,Tu,Td
                D[:, [4, 5, 6]] = np.conj(D[:, [4, 6, 5]])
                # T0
                D[:, 7] = -np.conj(D[:, 7])

        elif sp.is_spinorbit:
            if self.dkind in ("f", "i"):
                D[:, [0, 1, 3, 7]] = D[:, [1, 0, 7, 3]]  # diag real, off imag
                D[:, [4, 5, 2, 6]] = -D[:, [5, 4, 6, 2]]  # diag imag, off real
            else:
                D[:, [0, 1]] = np.conj(D[:, [1, 0]])
                D[:, [2, 3]] = -np.conj(D[:, [3, 2]])

        elif sp.is_noncolinear:
            if self.dkind in ("f", "i"):
                D[:, [0, 1]] = D[:, [1, 0]]
                D[:, 2:4] = -D[:, 2:4]
            else:
                D[:, [0, 1]] = np.conj(D[:, [1, 0]])
                D[:, 2] = -D[:, 2]

        elif sp.is_polarized:
            if self.dkind in ("f", "i"):
                D[:, [0, 1]] = D[:, [1, 0]]
            else:
                D[:, [0, 1]] = np.conj(D[:, [1, 0]])

        elif sp.is_unpolarized:
            if self.dkind not in ("f", "i"):
                D[:, 0] = np.conj(D[:, 0])
        else:
            raise NotImplementedError(f"Unknown spin-configuration: {sp!s}")

        if self.dkind not in ("f", "i") and not self.orthogonal:
            # The overlap component is diagonal (like a polarized)
            # so we will take its conjugate
            # This means that the overlap matrix *after*
            #   (M + M.trs()) / 2
            # will have 0 imaginary component.
            D[:, -1] = np.conj(D[:, -1])

        return new

    def transform(self, matrix=None, dtype=None, spin=None, orthogonal=None) -> Self:
        r"""Transform the matrix by either a matrix or new spin configuration

        1. General transformation:
        * If `matrix` is provided, a linear transformation :math:`\mathbf R^n \rightarrow \mathbf R^m` is applied
        to the :math:`n`-dimensional elements of the original sparse matrix.
        The `spin` and `orthogonal` flags are optional but need to be consistent with the creation of an
        `m`-dimensional matrix.

        This method will copy over the overlap matrix in case the `matrix` argument
        only acts on the non-overlap matrix elements and both input and output
        matrices are non-orthogonal.

        2. Spin conversion:
        If `spin` is provided (without `matrix`), the spin class
        is changed according to the following conversions:

        Upscaling
        * unpolarized -> (polarized, non-colinear, spinorbit, nambu): Copy unpolarized value to both up and down components
        * polarized -> (non-colinear, spinorbit, nambu): Copy up and down components
        * non-colinear -> (spinorbit, nambu): Copy first four spin components
        * spinorbit -> nambu: Copy first four spin components
        * all other new spin components are set to zero

        Downscaling
        * (polarized, non-colinear, spinorbit, nambu) -> unpolarized: Set unpolarized value to a mix 0.5*up + 0.5*down
        * (non-colinear, spinorbit, nambu) -> polarized: Keep up and down spin components
        * (spinorbit, nambu) -> non-colinear: Keep first four spin components
        * nambu -> spinorbit: Keep first four spin components
        * all other spin components are dropped

        3. Orthogonality:
        If the `orthogonal` flag is provided, the overlap matrix is either dropped
        or explicitly introduced as the identity matrix.

        Notes
        -----
        The transformation matrix does *not* act on the rows and columns, only on the
        final dimension of the matrix.

        The matrix transformation is done like this:

        >>> out = in @ matrix.T

        Meaning that ``matrix[0, :]`` will be the factors of the input matrix elements.

        Parameters
        ----------
        matrix : array_like, optional
            transformation matrix of shape :math:`m \times n`. Default is no transformation.
        dtype : numpy.dtype, optional
            data type contained in the matrix. Defaults to the input type.
        spin : str, sisl.Spin, optional
            spin class of created matrix. Defaults to the input type.
        orthogonal : bool, optional
            flag to control if the new matrix includes overlaps. Defaults to the input type.
        """
        if dtype is None:
            dtype = self.dtype
        else:
            warn(
                f"{self.__class__.__name__}.transform(dtype=...) is deprecated in favor "
                f"of {self.__class__.__name__}.astype(...), please adapt your code."
            )

        if spin is None:
            spin = self.spin
        else:
            spin = Spin(spin)

        if orthogonal is None:
            orthogonal = self.orthogonal

        # get dimensions to check
        N = n = self.spin.size(self.dtype)
        if not self.orthogonal:
            N += 1
        M = m = spin.size(dtype)
        if not orthogonal:
            M += 1

        if matrix is None:
            if spin == self.spin and orthogonal == self.orthogonal:
                # no transformations needed
                return self.copy(dtype)

            # construct transformation matrix
            matrix = np.zeros([M, N], dtype=dtype)
            matrix[:m, :n] = np.eye(m, n, dtype=dtype)
            if not self.orthogonal and not orthogonal:
                # ensure the overlap matrix is carried over
                matrix[-1, -1] = 1.0

            if spin.is_unpolarized and self.spin.size(self.dtype) > 1:
                if np.dtype(matrix.dtype).kind == "i":
                    matrix = matrix.astype(np.float64)
                # average up and down components
                matrix[0, [0, 1]] = 0.5
            elif spin.size(dtype) > 1 and self.spin.is_unpolarized:
                # set up and down components to unpolarized value
                matrix[[0, 1], 0] = 1.0

        else:
            # convert to numpy array
            matrix = np.asarray(matrix)

            if M != m and matrix.shape[0] == m and N != n and matrix.shape[1] == n:
                # this means that the user wants to preserve the overlap
                dtype = np.promote_types(dtype, matrix.dtype)
                matrix_full = np.zeros([M, N], dtype=dtype)
                matrix_full[:m, :n] = matrix
                matrix_full[-1, -1] = 1.0
                matrix = matrix_full

        if matrix.shape[0] != M or matrix.shape[1] != N:
            # while this check also occurs in the SparseCSR.transform
            # code, but the error message is better placed here.
            raise ValueError(
                f"{self.__class__.__name__}.transform incompatible "
                f"transformation matrix and spin dimensions: "
                f"matrix.shape={matrix.shape} and self.spin={N} ; out.spin={M}"
            )

        # Define the dtype based on matrix
        dtype = np.promote_types(dtype, matrix.dtype)

        new = self.__class__(
            self.geometry.copy(), spin=spin, dtype=dtype, nnzpr=1, orthogonal=orthogonal
        )
        new._csr = self._csr.transform(matrix, dtype=dtype)

        if self.orthogonal and not orthogonal:
            # set identity overlap matrix, loop over rows
            for i in range(new._csr.shape[0]):
                new._csr[i, i, -1] = 1.0

        return new

    def astype(self, dtype, *, copy: bool = True) -> Self:
        """Convert the sparse matrix to a specific `dtype`

        The data-conversion depends on the spin configuration of the system.
        In practice this means that real valued arrays in non-colinear calculations
        can be packed to complex valued arrays, which might be more intuitive.

        Notes
        -----
        Historically, `sisl` was built with large inspiration from SIESTA.
        In SIESTA the matrices are always stored in real valued arrays, meaning
        that spin-orbit systems has 2x2x2 = 8 values to represent the spin-box.
        However, this might as well be stored in 2x2 = 4 complex valued arrays.

        Later versions of `sisl` might force non-colinear matrices to be stored
        in complex arrays for consistency, but for now, both the real and the
        complex valued arrays are allowed.

        Parameters
        ----------
        dtype :
            the resulting data-type of the returned new sparse matrix
        copy:
            whether the data should be copied (i.e. if `dtype` is not changed)
        """
        # Change details
        old_dtype = np.dtype(self.dtype)
        new_dtype = np.dtype(dtype)

        if old_dtype == new_dtype:
            if copy:
                return self.copy()
            return self

        # Lets do the actual conversion.
        # It all depends on the spin-configuration.
        # The data-type is new, so we *have* to copy!
        M = self.copy()
        spin = M.spin
        csr = M._csr
        shape = csr._D.shape

        def r2c(D, re, im):
            return (D[..., re] + 1j * D[..., im]).astype(dtype, copy=False)

        if old_dtype.kind in ("f", "i"):
            if new_dtype.kind in ("f", "i"):
                # this is just simple casting
                csr._D = csr._D.astype(dtype)
            elif new_dtype.kind == "c":
                # we need to *collect* it
                if spin.is_diagonal:
                    # this is just simple casting,
                    # each diagonal component has its own index
                    csr._D = csr._D.astype(dtype)
                elif spin.is_noncolinear:
                    D = np.empty(shape[:-1] + (shape[-1] - 1,), dtype=dtype)
                    # These should be real only anyways!
                    D[..., [0, 1]] = csr._D[..., [0, 1]].real.astype(dtype)
                    D[..., 2] = r2c(csr._D, 2, 3)
                    if D.shape[-1] > 4:
                        D[..., 3:] = csr._D[..., 4:].astype(dtype)
                    csr._D = D
                elif spin.is_spinorbit:
                    D = np.empty(shape[:-1] + (shape[-1] - 4,), dtype=dtype)
                    D[..., 0] = r2c(csr._D, 0, 4)
                    D[..., 1] = r2c(csr._D, 1, 5)
                    D[..., 2] = r2c(csr._D, 2, 3)
                    D[..., 3] = r2c(csr._D, 6, 7)
                    if D.shape[-1] > 4:
                        D[..., 4:] = csr._D[..., 8:].astype(dtype)
                    csr._D = D
                elif spin.is_nambu:
                    D = np.empty(shape[:-1] + (shape[-1] - 8,), dtype=dtype)
                    D[..., 0] = r2c(csr._D, 0, 4)
                    D[..., 1] = r2c(csr._D, 1, 5)
                    D[..., 2] = r2c(csr._D, 2, 3)
                    D[..., 3] = r2c(csr._D, 6, 7)
                    D[..., 4] = r2c(csr._D, 8, 9)  # S
                    D[..., 5] = r2c(csr._D, 10, 11)  # Tuu
                    D[..., 6] = r2c(csr._D, 12, 13)  # Tdd
                    D[..., 7] = r2c(csr._D, 14, 15)  # T0
                    if D.shape[-1] > 8:
                        D[..., 8:] = csr._D[..., 16:].astype(dtype)
                    csr._D = D
                else:
                    raise NotImplementedError("Unknown spin-type")
            else:
                raise NotImplementedError("Unknown datatype")

        elif old_dtype.kind == "c":
            if new_dtype.kind == "c":
                # this is just simple casting
                csr._D = csr._D.astype(dtype)
            elif new_dtype.kind in ("f", "i"):
                # we need to *collect it
                if spin.is_diagonal:
                    # this is just simple casting,
                    # each diagonal component has its own index
                    csr._D = csr._D.astype(dtype)
                elif spin.is_noncolinear:
                    D = np.empty(shape[:-1] + (shape[-1] + 1,), dtype=dtype)
                    # These should be real only anyways!
                    D[..., [0, 1]] = csr._D[..., [0, 1]].real.astype(dtype)
                    D[..., 2] = csr._D[..., 2].real.astype(dtype)
                    D[..., 3] = csr._D[..., 2].imag.astype(dtype)
                    if D.shape[-1] > 4:
                        D[..., 4:] = csr._D[..., 3:].real.astype(dtype)
                    csr._D = D
                elif spin.is_spinorbit:
                    D = np.empty(shape[:-1] + (shape[-1] + 4,), dtype=dtype)
                    D[..., 0] = csr._D[..., 0].real.astype(dtype)
                    D[..., 1] = csr._D[..., 1].real.astype(dtype)
                    D[..., 2] = csr._D[..., 2].real.astype(dtype)
                    D[..., 3] = csr._D[..., 2].imag.astype(dtype)
                    D[..., 4] = csr._D[..., 0].imag.astype(dtype)
                    D[..., 5] = csr._D[..., 1].imag.astype(dtype)
                    D[..., 6] = csr._D[..., 3].real.astype(dtype)
                    D[..., 7] = csr._D[..., 3].imag.astype(dtype)
                    if D.shape[-1] > 8:
                        D[..., 8:] = csr._D[..., 4:].real.astype(dtype)
                    csr._D = D
                elif spin.is_nambu:
                    D = np.empty(shape[:-1] + (shape[-1] + 8,), dtype=dtype)
                    D[..., 0] = csr._D[..., 0].real.astype(dtype)
                    D[..., 1] = csr._D[..., 1].real.astype(dtype)
                    D[..., 2] = csr._D[..., 2].real.astype(dtype)
                    D[..., 3] = csr._D[..., 2].imag.astype(dtype)
                    D[..., 4] = csr._D[..., 0].imag.astype(dtype)
                    D[..., 5] = csr._D[..., 1].imag.astype(dtype)
                    D[..., 6] = csr._D[..., 3].real.astype(dtype)
                    D[..., 7] = csr._D[..., 3].imag.astype(dtype)
                    D[..., 8] = csr._D[..., 4].real.astype(dtype)  # S
                    D[..., 9] = csr._D[..., 4].imag.astype(dtype)
                    D[..., 10] = csr._D[..., 5].real.astype(dtype)  # Tuu
                    D[..., 11] = csr._D[..., 5].imag.astype(dtype)
                    D[..., 12] = csr._D[..., 6].real.astype(dtype)  # Tdd
                    D[..., 13] = csr._D[..., 6].imag.astype(dtype)
                    D[..., 14] = csr._D[..., 7].real.astype(dtype)  # T0
                    D[..., 15] = csr._D[..., 7].imag.astype(dtype)
                    if D.shape[-1] > 16:
                        D[..., 16:] = csr._D[..., 8:].real.astype(dtype)
                    csr._D = D
                else:
                    raise NotImplementedError("Unknown spin-type")
            else:
                raise NotImplementedError("Unknown datatype")

        # Resetting M is necessary to reflect the new shape
        M._reset()
        return M

    def __getstate__(self):
        return {
            "sparseorbitalbzspin": super().__getstate__(),
            "spin": self._spin.__getstate__(),
            "orthogonal": self._orthogonal,
        }

    def __setstate__(self, state):
        self._orthogonal = state["orthogonal"]
        spin = Spin()
        spin.__setstate__(state["spin"])
        self._spin = spin
        super().__setstate__(state["sparseorbitalbzspin"])
