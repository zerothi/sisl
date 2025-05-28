# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from functools import singledispatchmethod
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
from numpy import bool_, einsum, exp, ndarray

import sisl._array as _a
from sisl._core import Geometry
from sisl._help import dtype_float_to_complex
from sisl._internal import set_module
from sisl.linalg import eigh_destroy
from sisl.messages import deprecate_argument, warn
from sisl.typing import CartesianAxes, GaugeType, ProjectionType
from sisl.typing._physics import ProjectionTypeHadamard, ProjectionTypeHadamardAtoms

from ._common import comply_gauge, comply_projection

__all__ = ["degenerate_decouple", "Coefficient", "State", "StateC"]

_pi = np.pi
_pi2 = np.pi * 2


def _inner(v1, v2):
    return np.dot(np.conj(v1), v2)


@set_module("sisl.physics")
def degenerate_decouple(state, M):
    r""" Return `vec` decoupled via matrix `M`

    The decoupling algorithm is this recursive algorithm starting from :math:`i=0`:

    .. math::

       \mathbf p &= \mathbf V^\dagger \mathbf M_i \mathbf V
       \\
       \mathbf p \mathbf u &= \boldsymbol \lambda \mathbf u
       \\
       \mathbf V &= \mathbf u^T \mathbf V

    Parameters
    ----------
    state : numpy.ndarray or State
       states to be decoupled on matrices `M`
       The states must have C-ordering, i.e. ``[0, ...]`` is the first
       state.
    M : numpy.ndarray
       matrix to project to before disentangling the states
    """
    if isinstance(state, State):
        state.state = degenerate_decouple(state.state, M)
    else:
        # since M may be a sparse matrix, we cannot use __matmul__
        p = np.conj(state) @ (M @ state.T)
        state = eigh_destroy(p)[1].T @ state
    return state


"""
        degenerate : float or list of array_like, optional
           If a float is passed it is regarded as the degeneracy tolerance used to calculate the degeneracy
           levels. Defaults to 1e-5 eV.
           If a list, it contains the indices of degenerate states. In that case a prior diagonalization
           is required to decouple them. See `degenerate_dir` for the sum of directions.
        degenerate_dir : (3,), optional
           a direction used for degenerate decoupling. The decoupling based on the velocity along this direction

        # Now parse the degeneracy handling
        if degenerate is not None:
            if isinstance(degenerate, Real):
                degenerate = self.degenerate(degenerate)

            # normalize direction
            degenerate_dir = _a.asarrayd(degenerate_dir)
            degenerate_dir /= (degenerate_dir @ degenerate_dir) ** 0.5

            # de-coupling is only done for the 1st derivative

            # create the degeneracy decoupling projector
            deg_dPk = sum(d * dh for d, dh in zip(degenerate_dir, dPk))

            if is_orthogonal:
                for deg in degenerate:
                    # Set the average energy
                    energy[deg] = np.average(energy[deg])
                    # Now diagonalize to find the contributions from individual states
                    # then re-construct the seperated degenerate states
                    # Since we do this for all directions we should decouple them all
                    state[deg] = degenerate_decouple(state[deg], deg_dPk)
            else:
                for deg in degenerate:
                    e = np.average(energy[deg])
                    energy[deg] = e
                    deg_dSk = sum((d * e) * ds for d, ds in zip(degenerate_dir, dSk))
                    state[deg] = degenerate_decouple(state[deg], deg_dPk - deg_dSk)
                    del deg_dSk

            del deg_dPk
"""


class _FakeMatrix:
    """Replacement object which superseedes a matrix"""

    __slots__ = ("n", "m")
    ndim = 2

    def __init__(self, n, m=None):
        self.n = n
        if m is None:
            m = n
        self.m = m

    @property
    def shape(self):
        return (self.n, self.m)

    @staticmethod
    def dot(v):
        return v

    @staticmethod
    def __matmul__(v):
        return v

    def multiply(self, v):
        try:
            if v.shape == self.shape:
                diag = np.diagonal(v)
                out = np.zeros_like(v)
                np.fill_diagonal(out, diag)
                return out
        except Exception:
            pass
        return v

    def __getitem__(self, key):
        n, m = self.n, self.m
        if isinstance(key, tuple):
            if key[0] == slice(None, None, 2):
                n = n // 2
            else:
                raise RuntimeError
            if key[1] == slice(None, None, 2):
                m = m // 2
            else:
                raise RuntimeError
        else:
            if key == slice(None, None, 2):
                n = n // 2

        return self.__class__(n, m)

    @property
    def T(self):
        return self


@set_module("sisl.physics")
class ParentContainer:
    """A container for parent and information"""

    __slots__ = ["parent", "info"]

    def __init__(self, parent, **info):
        self.parent = parent
        """ object
the object from where the contained information comes from
"""
        self.info = info
        """ dict
information regarding the creation of the object
"""

    @singledispatchmethod
    def _sanitize_index(self, index) -> np.ndarray:
        r"""Ensure indices are transferred to acceptable integers"""
        if index is None:
            # in case __len__ is not defined, this will fail...
            return np.arange(len(self))
        index = _a.asarray(index)
        if index.size == 0:
            return _a.asarrayl([])
        elif index.dtype == bool_:
            return index.nonzero()[0]
        return index

    @_sanitize_index.register
    def _(self, index: np.ndarray) -> np.ndarray:
        if index.dtype == bool_:
            return np.flatnonzero(index)
        return index

    @_sanitize_index.register
    def _(self, index: slice) -> ndarray:
        index = index.indices(len(self))
        return np.arange(*index)

    def _geometry(self):
        """Return the parent geometry if one is found"""
        if isinstance(self.parent, Geometry):
            return self.parent
        return self.parent.geometry


@set_module("sisl.physics")
class Coefficient(ParentContainer):
    """An object holding coefficients for a parent with info

    Parameters
    ----------
    c : array_like
       coefficients
    parent : obj, optional
       a parent object that defines the origin of the coefficient
    **info : dict, optional
       an info dictionary that turns into an attribute on the object.
       This `info` may contain anything that may be relevant for the coefficient.
    """

    __slots__ = ["c"]

    def __init__(self, c, parent=None, **info):
        super().__init__(parent, **info)
        self.c = np.atleast_1d(c)
        """ c : numpy.ndarray
coefficients retained in this object
        """

    def __str__(self):
        """The string representation of this object"""
        s = f"{self.__class__.__name__}{{coefficients: {len(self)}, kind: {self.dkind}"
        if self.parent is None:
            s += "}}"
        else:
            s += ",\n {}\n}}".format(str(self.parent).replace("\n", "\n "))
        return s

    def __len__(self):
        """Number of coefficients"""
        return self.shape[0]

    @property
    def dtype(self):
        """Data-type for the coefficients"""
        return self.c.dtype

    @property
    def dkind(self):
        """The data-type of the coefficient (in str)"""
        return np.dtype(self.c.dtype).kind

    @property
    def shape(self):
        """Returns the shape of the coefficients"""
        return self.c.shape

    @deprecate_argument(
        "eps",
        "atol",
        "argument eps has been deprecated in favor of atol",
        "0.15",
        "0.17",
    )
    def degenerate(self, atol: float):
        """Find degenerate coefficients with a specified precision

        Parameters
        ----------
        atol :
           the precision above which coefficients are not considered degenerate

        Returns
        -------
        list of numpy.ndarray
            a list of indices
        """
        deg = list()
        sidx = np.argsort(self.c)
        dc = np.diff(self.c[sidx])

        # Degenerate indices
        idx = (np.absolute(dc) <= atol).nonzero()[0]
        if len(idx) == 0:
            # There are no degenerate coefficients
            return deg

        # These are the points were we split the degeneracies
        seps = (np.diff(idx) > 1).nonzero()[0]
        IDX = np.array_split(idx, seps + 1)
        for idx in IDX:
            deg.append(np.append(sidx[idx], sidx[idx[-1] + 1]))
        return deg

    @abstractmethod
    def sub(self, *args, **kwargs):
        """Return a subset of this instance"""
        # defined in _ufuncs_*.py

    @abstractmethod
    def remove(self, *args, **kwargs):
        """Return a subset of this instance, by removing some elements"""
        # defined in _ufuncs_*.py

    def __getitem__(self, key):
        """Return a new coefficient object with only one associated coefficient

        Parameters
        ----------
        key : int or array_like
           the indices for the returned coefficients

        Returns
        -------
        Coefficient
            a new coefficient *only* with the indexed values
        """
        return self.sub(key)

    def iter(self, asarray=False):
        """An iterator looping over the coefficients in this system

        Parameters
        ----------
        asarray : bool, optional
           if true the yielded values are the coefficient vectors, i.e. a numpy array.
           Otherwise an equivalent object is yielded.

        Yields
        ------
        coeff : Coefficent
           the current coefficent as an object, only returned if `asarray` is false.
        coeff : numpy.ndarray
           the current the coefficient as an array, only returned if `asarray` is true.
        """
        if asarray:
            for i in range(len(self)):
                yield self.c[i]
        else:
            for i in range(len(self)):
                yield self.sub(i)

    __iter__ = iter


@set_module("sisl.physics")
class State(ParentContainer):
    """An object handling a set of vectors describing a given *state*

    Parameters
    ----------
    state : array_like
       state vectors ``state[i, :]`` containing the i'th state vector
    parent : obj, optional
       a parent object that defines the origin of the state.
    **info : dict, optional
       an info dictionary that turns into an attribute on the object.
       This `info` may contain anything that may be relevant for the state.

    Notes
    -----
    This class should be subclassed!
    """

    __slots__ = ["state"]

    def __init__(self, state, parent=None, **info):
        """Define a state container with a given set of states"""
        super().__init__(parent, **info)
        self.state = np.atleast_2d(state)
        """ numpy.ndarray
state coefficients
        """

    def __str__(self):
        """The string representation of this object"""
        s = f"{self.__class__.__name__}{{states: {len(self)}, kind: {self.dkind}"
        if self.parent is None:
            s += "}"
        else:
            s += ",\n {}\n}}".format(str(self.parent).replace("\n", "\n "))
        return s

    def __len__(self):
        """Number of states"""
        return self.shape[0]

    @property
    def dtype(self):
        """Data-type for the state"""
        return self.state.dtype

    @property
    def dkind(self):
        """The data-type of the state (in str)"""
        return np.dtype(self.state.dtype).kind

    @property
    def shape(self):
        """Returns the shape of the state"""
        return self.state.shape

    @abstractmethod
    def sub(self, *args, **kwargs):
        """Return a subset of this instance"""

    @abstractmethod
    def remove(self, *args, **kwargs):
        """Return a subset of this instance, by removing some elements"""

    def translate(self, isc):
        r"""Translate the vectors to a new unit-cell position

        The method is thoroughly explained in `tile` while this one only
        selects the corresponding state vector

        Parameters
        ----------
        isc : (3,)
           number of offsets for the statevector

        See Also
        --------
        tile : equivalent method for generating more cells simultaneously
        """
        # the k-point gets reduced
        k = _a.asarrayd(self.info.get("k", [0] * 3))
        assert len(isc) == 3

        s = self.copy()
        # translate the bloch coefficients with:
        #   exp(i k.T)
        # with T being
        #   i * a_0 + j * a_1 + k * a_2
        if not np.allclose(k, 0):
            # there will only be a phase if k != 0
            s.state *= exp(2j * _pi * k @ isc)
        return s

    def __getitem__(self, key):
        """Return a new state with only one associated state

        Parameters
        ----------
        key : int or array_like
           the indices for the returned states

        Returns
        -------
        State
            a new state *only* with the indexed values
        """
        return self.sub(key)

    def iter(self, asarray: bool = False):
        """An iterator looping over the states in this system

        Parameters
        ----------
        asarray : bool, optional
           if true the yielded values are the state vectors, i.e. a numpy array.
           Otherwise an equivalent object is yielded.

        Yields
        ------
        state : State
           a state *only* containing individual elements, if `asarray` is false
        state : numpy.ndarray
           a state *only* containing individual elements, if `asarray` is true
        """
        if asarray:
            for i in range(len(self)):
                yield self.state[i]
        else:
            for i in range(len(self)):
                yield self.sub(i)

    __iter__ = iter

    def norm(self):
        r"""Return a vector with the Euclidean norm of each state :math:`\sqrt{\langle\psi|\psi\rangle}`

        Returns
        -------
        numpy.ndarray
            the Euclidean norm for each state
        """
        return self.norm2() ** 0.5

    @deprecate_argument(
        "sum",
        "projection",
        "argument sum has been deprecated in favor of projection",
        "0.15",
        "0.17",
    )
    def norm2(
        self,
        projection: Union[
            ProjectionType, ProjectionTypeHadamard, ProjectionTypeHadamardAtoms
        ] = "diagonal",
    ):
        r"""Return a vector with the norm of each state :math:`\langle\psi|\psi\rangle`

        Parameters
        ----------
        projection :
           whether to compute the norm per state as a single number, atom-resolved or per
           basis dimension.

        See Also
        --------
        inner: used method for calculating the squared norm.

        Returns
        -------
        numpy.ndarray
            the squared norm for each state
        """
        return self.inner(projection=projection)

    def ipr(self, q: int = 2):
        r""" Calculate the inverse participation ratio (IPR) for arbitrary `q` values

        The inverse participation ratio is defined as

        .. math::
            I_{q,\alpha} = \frac{\sum_i |\psi_{\alpha,i}|^{2q}}{
               \big[\sum_i |\psi_{\alpha,i}|^2\big]^q}

        where :math:`\alpha` is the band index and :math:`i` is the orbital.
        The order of the IPR is defaulted to :math:`q=2`, see following equation for details.
        The IPR may be used to distinguish Anderson localization and extended
        states:

        .. math::
           :nowrap:

            \begin{align}
             \lim_{L\to\infty} I_{2,\alpha} = \left\{
               \begin{aligned}
                1/L^d &\quad \text{extended state}
                \\
                \text{const.} &\quad \text{localized state}
               \end{aligned}\right.
            \end{align}

        For further details see :cite:`Murphy2011`. Note that for eigenstates the IPR reduces to:

        .. math::
            I_{q,\alpha} = \sum_i |\psi_{\alpha,i}|^{2q}

        since the denominator is :math:`1^{q} = 1`.

        Parameters
        ----------
        q :
          order parameter for the IPR
        """
        # This *has* to be a real value C * C^* == real
        state_abs2 = self.norm2(projection="hadamard").real
        assert q >= 2, f"{self.__class__.__name__}.ipr requires q>=2"
        # abs2 is already having the exponent 2
        return (state_abs2**q).sum(-1) / state_abs2.sum(-1) ** q

    def normalize(self):
        r"""Return a normalized state where each state has :math:`|\psi|^2=1`

        This is roughly equivalent to:

        >>> state = State(np.arange(10))
        >>> n = state.norm()
        >>> norm_state = State(state.state / n.reshape(-1, 1))

        Notes
        -----
        This does *not* take into account a possible overlap matrix when non-orthogonal basis sets are used.

        Returns
        -------
        State
            a new state with all states normalized, otherwise equal to this
        """
        n = self.norm()
        s = self.__class__(self.state / n.reshape(-1, 1), parent=self.parent)
        s.info = self.info
        return s

    def outer(self, ket=None, matrix=None):
        r"""Return the outer product by :math:`\sum_\alpha|\psi_\alpha\rangle\langle\psi'_\alpha|`

        Parameters
        ----------
        ket : State, optional
           the ket object to calculate the outer product of, if not passed it will do the outer
           product with itself. The object itself will always be the bra :math:`|\psi_\alpha\rangle`
        matrix : array_like, optional
           whether a matrix is sandwiched between the ket and bra, defaults to the identity matrix.
           1D arrays will be treated as a diagonal matrix.

        Notes
        -----
        This does *not* take into account a possible overlap matrix when non-orthogonal basis sets are used.

        Returns
        -------
        numpy.ndarray
            a matrix with the sum of outer state products
        """
        if matrix is None:
            M = _FakeMatrix(self.shape[-1])
        else:
            M = matrix
        ndim = M.ndim
        if ndim not in (0, 1, 2):
            raise ValueError(
                f"{self.__class__.__name__}.outer only accepts matrices up to 2 dimensions."
            )

        bra = self.state
        # decide on the ket
        if ket is None:
            ket = self.state
        elif isinstance(ket, State):
            # check whether this, and ket are both originating from
            # non-orthogonal basis. That would be non-ideal
            ket = ket.state
        if len(ket.shape) == 1:
            ket.shape = (1, -1)

        # check that the shapes matches (ket should be transposed)
        #  ket M bra
        if ndim == 0:
            # M,N @ N, L
            if ket.shape[1] != bra.shape[1]:
                raise ValueError(
                    f"{self.__class__.__name__}.outer requires the objects to have matching shapes bra @ ket bra={self.shape}, ket={ket.shape[::-1]}"
                )
        elif ndim == 1:
            # M,N @ N @ N, L
            if ket.shape[0] != M.shape[0] or M.shape[0] != bra.shape[0]:
                raise ValueError(
                    f"{self.__class__.__name__}.outer requires the objects to have matching shapes ket @ M @ bra ket={ket.shape[::-1]}, M={M.shape}, bra={self.shape}"
                )
        elif ndim == 2:
            # M,N @ N,K @ K,L
            if ket.shape[0] != M.shape[0] or M.shape[1] != bra.shape[0]:
                raise ValueError(
                    f"{self.__class__.__name__}.outer requires the objects to have matching shapes ket @ M @ bra ket={ket.shape[::-1]}, M={M.shape}, bra={self.shape}"
                )

        if ndim == 2:
            Aij = ket.T @ (M @ np.conj(bra))
        elif ndim == 1:
            Aij = einsum("ij,i,ik->jk", ket, M, np.conj(bra))
        elif ndim == 0:
            Aij = einsum("ij,ik->jk", ket * M, np.conj(bra))
        return Aij

    @deprecate_argument(
        "diag",
        "projection",
        "argument diag has been deprecated in favor of projection",
        "0.15",
        "0.17",
    )
    def inner(
        self,
        ket=None,
        matrix=None,
        projection: Union[
            ProjectionType, ProjectionTypeHadamard, ProjectionTypeHadamardAtoms
        ] = "diagonal",
    ):
        r"""Calculate the inner product as :math:`\mathbf A_{ij} = \langle\psi_i|\mathbf M|\psi'_j\rangle`

        Inner product calculation allows for a variety of things.

        * for ``matrix`` it will compute off-diagonal elements as well

            .. math::
                \mathbf A_{\alpha\beta} = \langle\psi_\alpha|\mathbf M|\psi'_\beta\rangle

        * for ``diag`` only the diagonal components will be returned

            .. math::
                \mathbf a_\alpha = \langle\psi_\alpha|\mathbf M|\psi_\alpha\rangle

        * for ``basis``, only do inner products for individual states, but return them basis-resolved

            .. math::
                \mathbf A_{\alpha\beta} = \psi^*_{\alpha,\beta} \mathbf M|\psi_\alpha\rangle_\beta

        * for ``atoms``, only do inner products for individual states, but return them atom-resolved


        Parameters
        ----------
        ket : State, optional
           the ket object to calculate the inner product with, if not passed it will do the inner
           product with itself. The object itself will always be the bra :math:`\langle\psi_i|`
        matrix : array_like, optional
           whether a matrix is sandwiched between the bra and ket, defaults to the identity matrix.
           1D arrays will be treated as a diagonal matrix.
        projection:
            how to perform the final projection.
            This can be used to sum specific sub-elements, return the diagonal, or the
            full matrix.

            * ``diagonal`` only return the diagonal of the inner product ('ii' elements)
            * ``matrix`` a matrix with diagonals and the off-diagonals ('ij' elements)
            * ``hadamard`` only do element wise products for the states (equivalent to
              basis resolved inner-products)
            * ``atoms`` only do inner products for individual states, but return them atom-resolved


        Notes
        -----
        This does *not* take into account a possible overlap matrix when
        non-orthogonal basis sets are used.
        One have to add the overlap matrix in the `matrix` argument, if needed.

        Raises
        ------
        ValueError
            if the number of state coefficients are different for the bra and ket
        RuntimeError
            if the matrix shapes are incompatible with an atomic resolution conversion

        Returns
        -------
        numpy.ndarray
            a matrix with the sum of inner state products
        """
        if matrix is None:
            M = _FakeMatrix(self.shape[-1])
        else:
            M = matrix
        ndim = M.ndim
        if ndim not in (0, 1, 2):
            raise ValueError(
                f"{self.__class__.__name__}.inner only accepts matrices up to 2 dimensions."
            )

        bra = self.state
        # decide on the ket
        if ket is None:
            ket = self.state
        elif isinstance(ket, State):
            # check whether this, and ket are both originating from
            # non-orthogonal basis. That would be non-ideal
            ket = ket.state
        if ket.ndim == 1:
            ket = ket.reshape(1, -1)

        # check that the shapes matches (ket should be transposed)
        #  bra M ket
        if ndim == 0:
            # M, N @ N, L
            if bra.shape[1] != ket.shape[1]:
                raise ValueError(
                    f"{self.__class__.__name__}.inner requires the objects to have matching shapes bra @ ket bra={self.shape}, ket={ket.shape[::-1]}"
                )
        elif ndim == 1:
            # M,N @ N @ N, L
            if bra.shape[1] != M.shape[0] or M.shape[0] != ket.shape[1]:
                raise ValueError(
                    f"{self.__class__.__name__}.inner requires the objects to have matching shapes bra @ M @ ket bra={self.shape}, M={M.shape}, ket={ket.shape[::-1]}"
                )
        elif ndim == 2:
            # M,N @ N,K @ K,L
            if bra.shape[1] != M.shape[0] or M.shape[1] != ket.shape[1]:
                raise ValueError(
                    f"{self.__class__.__name__}.inner requires the objects to have matching shapes bra @ M @ ket bra={self.shape}, M={M.shape}, ket={ket.shape[::-1]}"
                )

        if isinstance(projection, bool):
            projection = "diagonal" if projection else "matrix"
        projection = comply_projection(projection)

        if projection == "diagonal":
            if bra.shape[0] != ket.shape[0]:
                raise ValueError(
                    f"{self.__class__.__name__}.inner diagonal matrix product is "
                    "non-square, please use projection!=diagonal or reduce number of vectors."
                )
            if ndim == 2:
                Aij = einsum("ij,ji->i", np.conj(bra), M @ ket.T)
            elif ndim == 1:
                Aij = einsum("ij,j,ij->i", np.conj(bra), M, ket)
            elif ndim == 0:
                Aij = einsum("ij,ij->i", np.conj(bra), ket) * M

        elif projection == "matrix":
            if ndim == 2:
                Aij = np.conj(bra) @ (M @ ket.T)
            elif ndim == 1:
                Aij = einsum("ij,j,kj->ik", np.conj(bra), M, ket)
            elif ndim == 0:
                Aij = einsum("ij,kj->ik", np.conj(bra), ket) * M

        elif projection == "hadamard":
            if ndim == 2:
                Aij = np.conj(bra) * (M @ ket.T).T
            else:
                Aij = np.conj(bra) * ket * M

        elif projection == "hadamard:atoms":
            if ndim == 2:
                Aij = np.conj(bra) * (M @ ket.T).T
            else:
                Aij = np.conj(bra) * ket * M

            # Now we need to convert it
            geom = self._geometry()
            if Aij.shape[1] == geom.no * 2:
                # We have some kind of spin-configuration (hidden)
                def mapper(atom):
                    return np.arange(geom.firsto[atom] * 2, geom.firsto[atom + 1] * 2)

            elif Aij.shape[1] == geom.no:

                def mapper(atom):
                    return np.arange(geom.firsto[atom], geom.firsto[atom + 1])

            else:
                raise RuntimeError(
                    f"{self.__class__.__name__}.inner could not determine "
                    "the correct atom conversions."
                )
            Aij = geom.apply(Aij, np.sum, mapper, axis=1)

        elif projection == "trace":
            if bra.shape[0] != ket.shape[0]:
                raise ValueError(
                    f"{self.__class__.__name__}.inner diagonal matrix product is "
                    "non-square, cannot do the trace."
                )
            if ndim == 2:
                Aij = einsum("ij,ji->i", np.conj(bra), M @ ket.T).sum()
            elif ndim == 1:
                Aij = einsum("ij,j,ij->i", np.conj(bra), M, ket).sum()
            elif ndim == 0:
                Aij = (einsum("ij,ij->i", np.conj(bra), ket) * M).sum()

        else:
            raise ValueError(
                f"{self.__class__.__name__}.inner got unknown argument 'projection'={projection}"
            )

        return Aij

    def phase(self, method: Literal["max", "all"] = "max", ret_index: bool = False):
        r"""Calculate the Euler angle (phase) for the elements of the state, in the range :math:`]-\pi;\pi]`

        Parameters
        ----------
        method : {'max', 'all'}
           for max, the phase for the element which has the largest absolute magnitude is returned,
           for all, all phases are calculated
        ret_index :
           return indices for the elements used when ``method=='max'``
        """
        if method == "max":
            idx = np.argmax(np.absolute(self.state), 1)
            if ret_index:
                return np.angle(self.state[:, idx]), idx
            return np.angle(self.state[:, idx])
        elif method == "all":
            return np.angle(self.state)
        raise ValueError(
            f"{self.__class__.__name__}.phase only accepts method in [max, all]"
        )

    def align_phase(self, other: State, ret_index: bool = False, inplace: bool = False):
        r"""Align `self` with the phases for `other`, a copy may be returned

        States will be rotated by :math:`\pi` provided the phase difference between the states are above :math:`|\Delta\theta| > \pi/2`.

        Parameters
        ----------
        other : State
           the other state to align onto this state
        ret_index :
           return which indices got swapped
        inplace :
           rotate the states in-place

        See Also
        --------
        align_norm : re-order states such that site-norms have a smaller residual
        """
        other_phase, idx = other.phase(ret_index=True)
        phase = np.angle(self.state[:, idx])

        # Calculate absolute phase difference
        abs_phase = np.absolute((phase - other_phase + _pi) % _pi2 - _pi)

        idx = (abs_phase > _pi / 2).nonzero()[0]

        ret = None
        if inplace:
            if ret_index:
                ret = ret_index
            self.state[idx] *= -1
            return ret

        out = self.copy()
        out.state[idx] *= -1
        if ret_index:
            return out, idx
        return out

    def align_norm(self, other: State, ret_index: bool = False, inplace: bool = False):
        r"""Align `self` with the site-norms of `other`, a copy may optionally be returned

        To determine the new ordering of `self` first calculate the residual norm of the site-norms.

        .. math::
           \delta N_{\alpha\beta} = \sum_i \big(\langle \psi^\alpha_i | \psi^\alpha_i\rangle - \langle \psi^\beta_i | \psi^\beta_i\rangle\big)^2

        where :math:`\alpha` and :math:`\beta` correspond to state indices in `self` and `other`, respectively.
        The new states (from `self`) returned is then ordered such that the index
        :math:`\alpha \equiv \beta'` where :math:`\delta N_{\alpha\beta}` is smallest.

        Parameters
        ----------
        other : State
           the other state to align against
        ret_index :
           also return indices for the swapped indices
        inplace :
           swap states in-place

        Returns
        -------
        self_swap : State
            A swapped instance of `self`, only if `inplace` is False
        index : array of int
            the indices that swaps `self` to be ``self_swap``, i.e. ``self_swap = self.sub(index)``
            Only if `inplace` is False and `ret_index` is True
        Notes
        -----
        The input state and output state have the same number of states, but their ordering is not necessarily the same.

        See Also
        --------
        align_phase : rotate states such that their phases align
        """
        snorm = self.norm2(projection="hadamard").real
        onorm = other.norm2(projection="hadamard").real

        # Now find new orderings
        show_warn = False

        sidx = _a.fulli(len(self), -1)
        oidx = _a.emptyi(len(self))
        for i in range(len(self)):
            R = snorm[i] - onorm
            R = einsum("ij,ij->i", R, R)

            # Figure out which band it should correspond to
            # find closest largest one
            for j in np.argsort(R):
                if j not in sidx[:i]:
                    sidx[i] = j
                    oidx[j] = i
                    break
                show_warn = True

        if show_warn:
            warn(
                f"{self.__class__.__name__}.align_norm found multiple possible candidates with minimal residue, swapping not unique"
            )

        if inplace:
            self.sub(oidx, inplace=True)
            if ret_index:
                return oidx
        elif ret_index:
            return self.sub(oidx), oidx
        else:
            return self.sub(oidx)

    def change_gauge(self, gauge: GaugeType, offset=(0, 0, 0)):
        r"""In-place change of the gauge of the state coefficients

        The two gauges are related through:

        .. math::

            \tilde C_\alpha = e^{i\mathbf k\mathbf r_\alpha} C_\alpha

        where :math:`C_\alpha` and :math:`\tilde C_\alpha` belongs to the ``atomic`` and
        ``lattice`` gauge, respectively.

        Parameters
        ----------
        gauge :
            specify the new gauge for the mode coefficients
        offset : array_like, optional
            whether the coordinates should be offset by another phase-factor
        """
        gauge = comply_gauge(gauge)

        # These calls will fail if the gauge is not specified.
        # In that case it will not do anything
        if self.info.get("gauge", gauge) == gauge:
            # Quick return
            return

        # Update gauge value
        self.info["gauge"] = gauge

        # Check that we can do a gauge transformation
        k = _a.asarrayd(self.info.get("k", [0.0, 0.0, 0.0]))
        if k.dot(k) <= 0.000000001:
            return

        # Try and bypass whether the parent is a geometry, or not
        g = self._geometry()
        xyz = g.xyz + offset
        phase = (xyz @ (k @ g.rcell))[g.o2a(_a.arange(g.no))]

        try:
            if not self.parent.spin.is_diagonal:
                # for NC/SOC we have a 2x2 spin-box per orbital
                phase = np.repeat(phase, 2)
        except Exception:
            # This should enter in case where spin is not part of the parent
            # So lets just check for sizes of the arrays
            if self.shape[1] == g.no * 2:
                phase = np.repeat(phase, 2)

        if gauge == "atomic":
            # R -> r gauge tranformation.
            self.state *= exp(-1j * phase).reshape(1, -1)
        elif gauge == "lattice":
            # r -> R gauge tranformation.
            self.state *= exp(1j * phase).reshape(1, -1)
        else:
            raise ValueError("change_gauge: gauge must be in [lattice, atomic]")

    # def toStateC(self, norm=1.):
    #     r""" Transforms the states into normalized values equal to `norm` and specifies the coefficients in `StateC` as the norm

    #     This is an easy method to renormalize the state vectors to a common (or state dependent) normalization constant.

    #     .. math::
    #         c_i &= \sqrt{\langle \psi_i | \psi_i\rangle} / \mathrm{norm} \\
    #           |\psi_i\rangle &= | \psi_i\rangle / c_i

    #     Parameters
    #     ----------
    #     norm : value, array_like
    #         the resulting norm of all (or individual states)

    #     Returns
    #     -------
    #     StateC
    #        a new coefficient state object with associated coefficients
    #     """
    #     n = len(self)
    #     norm = _a.asarray(norm).ravel()
    #     if norm.size == 1 and n > 1:
    #         norm = np.tile(norm, n)
    #     elif norm.size != n:
    #         raise ValueError(self.__class__.__name__ + '.toStateC requires the input norm to be a single float or having equal length as this state!')

    #     # Correct data-type
    #     if norm.dtype in [np.complex64, np.complex128]:
    #         dtype = norm.dtype
    #     else:
    #         dtype = dtype_complex_to_float(self.dtype)

    #     # TODO check datatype if norm is complex but state is real
    #     c = np.empty(n, dtype=dtype)
    #     state = np.empty(self.shape, dtype=self.dtype)

    #     for i in range(n):
    #         c[i] = (_inner1(self.state[i].ravel()).astype(dtype) / norm[i]) ** 0.5
    #         state[i, ...] = self.state[i, ...] / c[i]

    #     cs = StateC(state, c, parent=self.parent)
    #     cs.info = self.info
    #     return cs


_dM_Operator = Callable[
    [
        npt.ArrayLike,
        Optional[Literal["x", "y", "z", "xx", "yy", "zz", "yz", "xz", "xy"]],
    ],
    npt.ArrayLike,
]


# Although the StateC could inherit from both Coefficient and State
# there are problems with __slots__ and multiple inheritance schemes.
# I.e. we are forced to do *one* inheritance, which we choose to be State.
@set_module("sisl.physics")
class StateC(State):
    """An object handling a set of vectors describing a given *state* with associated coefficients `c`

    Parameters
    ----------
    state : array_like
       state vectors ``state[i, :]`` containing the i'th state vector
    c : array_like
       coefficients for the states ``c[i]`` containing the i'th coefficient
    parent : obj, optional
       a parent object that defines the origin of the state.
    **info : dict, optional
       an info dictionary that turns into an attribute on the object.
       This `info` may contain anything that may be relevant for the state.

    Notes
    -----
    This class should be subclassed!
    """

    __slots__ = ["c"]

    def __init__(self, state, c, parent=None, **info):
        """Define a state container with a given set of states and coefficients for the states"""
        super().__init__(state, parent, **info)
        self.c = np.atleast_1d(c)
        """ numpy.ndarray
coefficients assigned to each state
        """

        if len(self.c) != len(self.state):
            raise ValueError(
                f"{self.__class__.__name__} could not be created with coefficients and states "
                "having unequal length."
            )

    def normalize(self):
        r"""Return a normalized state where each state has :math:`|\psi|^2=1`

        This is roughly equivalent to:

        >>> state = StateC(np.arange(10), 1)
        >>> n = state.norm()
        >>> norm_state = StateC(state.state / n.reshape(-1, 1), state.c.copy())
        >>> norm_state.c[0] == 1

        Returns
        -------
        State
            a new state with all states normalized, otherwise equal to this
        """
        n = self.norm()
        s = self.__class__(
            self.state / n.reshape(-1, 1), self.c.copy(), parent=self.parent
        )
        s.info = self.info
        return s

    def sort(self, ascending: bool = True):
        """Sort and return a new `StateC` by sorting the coefficients (default to ascending)

        Parameters
        ----------
        ascending :
            sort the contained elements ascending, else they will be sorted descending
        """
        if ascending:
            idx = np.argsort(self.c)
        else:
            idx = np.argsort(-self.c)
        return self.sub(idx)

    def derivative(
        self,
        order: Literal[1, 2] = 1,
        matrix: bool = False,
        axes: CartesianAxes = "xyz",
        operator: _dM_Operator = lambda M, d=None: M,
    ):
        r"""Calculate the derivative with respect to :math:`\mathbf k` for a set of states up to a given order

        These are calculated using the analytic expression (:math:`\alpha` corresponding to the Cartesian directions),
        here only shown for the 1st order derivative:

        .. math::

           \mathbf{d}_{\alpha ij} = \langle \psi_i |
                    \frac{\partial}{\partial\mathbf k_\alpha} \mathbf H(\mathbf k) | \psi_j \rangle

        In case of non-orthogonal basis the equations substitutes :math:`\mathbf H(\mathbf k)` by
        :math:`\mathbf H(\mathbf k) - \epsilon_i\mathbf S(\mathbf k)`.

        The 2nd order derivatives are calculated with the Berry curvature correction:

        .. math::

           \mathbf d^2_{\alpha \beta ij} = \langle\psi_i|
               \frac{\partial^2}{\partial\mathbf k_\alpha\partial\mathbf k_\beta} \mathbf H(\mathbf k) | \psi_j\rangle
               - \frac12\frac{\mathbf{d}_{\alpha ij}\mathbf{d}_{\beta ij}}
                     {\epsilon_i - \epsilon_j}

        Notes
        -----

        When requesting 2nd derivatives it will not be advisable to use a `sub` before
        calculating the derivatives since the 1st order perturbation uses the energy
        differences (Berry contribution) and the 1st derivative matrix for correcting the curvature.

        For states at the :math:`\Gamma` point you may get warnings about casting complex numbers
        to reals. In these cases you should force the state at the :math:`\Gamma` point to be calculated
        in complex numbers to enable the correct decoupling.

        Parameters
        ----------
        order :
           an integer specifying which order of the derivative is being calculated.
        matrix :
           whether the full matrix or only the diagonal components are returned
        axes:
            NOTE: this argument may change in future versions.
            only calculate the derivative(s) along specified Cartesian directions.
            The axes argument will be sorted internally, so the order will always
            be xyz. For the higher order derivatives all those involving only the provided axes will be used.
        operator :
            an operator that translates the :math:`\delta` matrices to another operator.
            The same operator will be applied to both ``P`` and ``S`` matrices.

        See Also
        --------
        SparseOrbitalBZ.dPk : function for generating the matrix derivatives
        SparseOrbitalBZ.dSk : function for generating the matrix derivatives in non-orthogonal basis

        Returns
        -------
        dv
            the 1st derivative, has shape ``(3, state.shape[0])`` for ``matrix=False``, else
            has shape ``(3, state.shape[0], state.shape[0])``
            Also returned for ``order >= 2`` since it is used in the higher order derivatives
        ddv
            the 2nd derivative, has shape ``(6, state.shape[0])`` for ``matrix=False``, else
            has shape ``(6, state.shape[0], state.shape[0])``, the first dimension is in the Voigt representation
            Only returned for ``order >= 2``
        """
        parent = self.parent

        # Figure out arguments
        opt = {
            "k": self.info.get("k", np.zeros(3)),
            "dtype": dtype_float_to_complex(self.dtype),
        }

        axes_d_all = "xyz"
        axes_dd_all = ("xx", "yy", "zz", "yz", "xz", "xy")

        # Determine what to calculate
        if isinstance(axes, int):
            axes = [axes]
        elif isinstance(axes, str):
            axes = list(axes)
        axes = tuple(sorted(axes))

        CONV = {
            ("x",): ((0,), (0,)),
            (0,): ((0,), (0,)),
            ("y",): ((1,), (1,)),
            (1,): ((1,), (1,)),
            ("z",): ((2,), (2,)),
            (2,): ((2,), (2,)),
            ("x", "y"): ((0, 1), (0, 1, 5)),
            (0, 1): ((0, 1), (0, 1, 5)),
            ("x", "z"): ((0, 2), (0, 2, 4)),
            (0, 2): ((0, 1), (0, 1, 5)),
            ("y", "z"): ((1, 2), (1, 2, 3)),
            (1, 2): ((0, 1), (0, 1, 5)),
            ("x", "y", "z"): ((0, 1, 2), (0, 1, 2, 3, 4, 5)),
            (0, 1, 2): ((0, 1), (0, 1, 5)),
        }
        axes_d, axes_dd = CONV[axes]

        """
        axes_d =
        if isinstance(axes[0], str):
            axes_d = [axes_d_all.index(ax) for ax in set(axes)]
        else:
            axes_d = list(set(axes))
        axes_d.sort()

        if len(axes_d) == 1:
            axes_dd = [axes_d[0]]
        elif len(axes_d) == 2:
            if 0 in axes_d:
                if 1 in axes_d:
                    axes_dd = [0, 1, 5]
                elif 2 in axes_d:
                    axes_dd = [0, 2, 4]
            else:
                # it must be 1 and 2
                axes_dd = [1, 2, 3]
        else:
            axes_dd = [0, 1, 2, 3, 4, 5]
        """

        axes_d_str = "".join((axes_d_all[i] for i in axes_d))
        axes_dd_str = [axes_dd_all[i] for i in axes_dd]

        def reduce_d(ds):
            return [ds[ax] for ax in axes_d]

        def reduce_dd(dds):
            return [dds[ax] for ax in axes_dd]

        if order not in (1, 2):
            raise NotImplementedError(
                f"{self.__class__.__name__}.derivative required order to be in this list: [1, 2], higher order derivatives are not implemented"
            )

        def add_keys(opt, *keys):
            for key in keys:
                # Store gauge, if present
                val = self.info.get(key)
                if val is not None:
                    # this must mean the methods accepts this as an argument
                    opt[key] = val

        add_keys(opt, "gauge", "format")

        # Initialize variables
        ddPk = dSk = ddSk = None

        # Figure out if we are dealing with a non-orthogonal basis set
        is_orthogonal = True
        try:
            if not parent.orthogonal:
                is_orthogonal = False
                dSk = [
                    operator(dS, d)
                    for d, dS in zip(axes_d_str, reduce_d(parent.dSk(**opt)))
                ]
                if order > 1:
                    ddSk = [
                        operator(dS, d)
                        for d, dS in zip(axes_dd_str, reduce_dd(parent.ddSk(**opt)))
                    ]
        except Exception:
            pass

        # Now figure out if spin is a thing
        add_keys(opt, "spin")

        dPk = [
            operator(dP, d) for d, dP in zip(axes_d_str, reduce_d(parent.dPk(**opt)))
        ]
        nd = len(dPk)
        if order > 1:
            ddPk = [
                operator(dP, d)
                for d, dP in zip(axes_dd_str, reduce_dd(parent.ddPk(**opt)))
            ]
            ndd = len(ddPk)

        # short-hand, everything will now be done *in-place* of this object
        state = self.state
        # in case the state is not an Eigenstate*
        energy = self.c

        # States have been decoupled and we can calculate things now
        # number of states
        nstate, lstate = state.shape
        # we calculate the first derivative matrix
        cstate = np.conj(state)
        stateT = state.T

        # We split everything up into orthogonal and non-orthogonal
        # This reduces if-checks
        if is_orthogonal:

            if matrix or order > 1:
                # calculate the full matrix
                v = np.empty([nd, nstate, nstate], dtype=opt["dtype"])
                for i in range(nd):
                    v[i] = cstate @ dPk[i] @ state.T

                if matrix:
                    ret = (v,)
                else:
                    ret = (np.diagonal(v, axis1=1, axis2=2).copy(),)

            else:
                # calculate projections on states
                v = np.empty([nd, nstate], dtype=opt["dtype"])

                for i in range(nd):
                    v[i] = einsum("ij,ji->i", cstate, dPk[i] @ state.T)

                ret = (v,)

            if order > 1:
                # Now calculate the 2nd order corrections
                # loop energies

                if matrix:
                    vv = np.empty([ndd, nstate, nstate], dtype=opt["dtype"])
                    for s, e in enumerate(energy):
                        de = e - energy
                        # add factor 2 here
                        np.divide(2, de, where=(de != 0), out=de)

                        # we will use this multiple times
                        absv = np.absolute(v[:, s])

                        # calculate 2nd derivative
                        for i in range(nd):
                            ## xx, for instance
                            vv[i, s] = cstate[s] @ ddPk[i] @ stateT - de * absv[i] ** 2

                        for i in range(nd, ndd):
                            # this will be 3, 4, 5
                            # or 2
                            # or []
                            # yz
                            i0 = (i + 1) % nd
                            i1 = (i + 2) % nd
                            vv[i, s] = (
                                cstate[s] @ ddPk[i] @ stateT - de * absv[i0] * absv[i1]
                            )

                else:
                    vv = np.empty([ndd, nstate], dtype=opt["dtype"])
                    for s, e in enumerate(energy):
                        de = e - energy
                        # add factor 2 here
                        np.divide(2, de, where=(de != 0), out=de)

                        # we will use this multiple times
                        absv = np.absolute(v[:, s])

                        # calculate 2nd derivative
                        for i in range(nd):
                            ## xx, for instance
                            vv[i, s] = (
                                cstate[s] @ ddPk[i] @ state[s] - de @ absv[i] ** 2
                            )

                        for i in range(nd, ndd):
                            # this will be 3, 4, 5
                            # or 2
                            # or []
                            # yz
                            i0 = (i + 1) % nd
                            i1 = (i + 2) % nd
                            vv[i, s] = cstate[s] @ ddPk[i] @ state[s] - de @ (
                                absv[i0] * absv[i1]
                            )

                ret += (vv,)
        else:
            # non-orthogonal basis set

            if matrix or order > 1:
                # calculate the full matrix
                v = np.empty([nd, nstate, nstate], dtype=opt["dtype"])
                for i in range(nd):
                    for s, e in enumerate(energy):
                        v[i, s] = cstate[s] @ (dPk[i] - e * dSk[i]) @ stateT

                if matrix:
                    ret = (v,)
                else:
                    # diagonal of nd removes axis0 and axis1 and appends a new
                    # axis to the end, this is not what we want
                    ret = (np.diagonal(v, axis1=1, axis2=2).copy(),)

            else:
                # calculate diagonal components on states
                v = np.empty([nd, nstate], dtype=opt["dtype"])
                for i in range(nd):
                    for s, e in enumerate(energy):
                        v[i, s] = cstate[s] @ (dPk[i] - e * dSk[i]) @ state[s]

                ret = (v,)

            if order > 1:
                # Now calculate the 2nd order corrections
                # loop energies

                if matrix:
                    vv = np.empty([ndd, nstate, nstate], dtype=opt["dtype"])
                    for s, e in enumerate(energy):
                        de = e - energy
                        # add factor 2 here
                        np.divide(2, de, where=(de != 0), out=de)

                        # we will use this multiple times
                        absv = np.absolute(v[:, s])

                        # calculate 2nd derivative
                        for i in range(nd):
                            ## xx, for instance
                            vv[i, s] = (
                                cstate[s] @ (ddPk[i] - e * ddSk[i]) @ stateT
                                - de * absv[i] ** 2
                            )

                        for i in range(nd, ndd):
                            # this will be 3, 4, 5
                            # or 2
                            # or []
                            # yz
                            i0 = (i + 1) % nd
                            i1 = (i + 2) % nd
                            vv[i, s] = (
                                cstate[s] @ (ddPk[i] - e * ddSk[i]) @ stateT
                                - de * absv[i0] * absv[i1]
                            )

                else:
                    vv = np.empty([ndd, nstate], dtype=opt["dtype"])
                    for s, e in enumerate(energy):
                        de = e - energy
                        # add factor 2 here
                        np.divide(2, de, where=(de != 0), out=de)

                        # we will use this multiple times
                        absv = np.absolute(v[:, s])

                        # calculate 2nd derivative
                        for i in range(nd):
                            ## xx, for instance
                            vv[i, s] = (
                                cstate[s] @ (ddPk[i] - e * ddSk[i]) @ state[s]
                                - de @ absv[i] ** 2
                            )

                        for i in range(nd, ndd):
                            # this will be 3, 4, 5
                            # or 2
                            # or []
                            # yz
                            i0 = (i + 1) % nd
                            i1 = (i + 2) % nd
                            vv[i, s] = cstate[s] @ (ddPk[i] - e * ddSk[i]) @ state[
                                s
                            ] - de @ (absv[i0] * absv[i1])

                ret += (vv,)

        if len(ret) == 1:
            return ret[0]
        return ret

    @deprecate_argument(
        "eps",
        "atol",
        "argument eps has been deprecated in favor of atol",
        "0.15",
        "0.17",
    )
    def degenerate(self, atol: float):
        """Find degenerate coefficients with a specified precision

        Parameters
        ----------
        atol :
           the precision above which coefficients are not considered degenerate

        Returns
        -------
        list of numpy.ndarray
            a list of indices
        """
        deg = list()

        # Sort them in ascending order
        sidx = np.argsort(self.c)
        dc = np.diff(self.c[sidx])

        # Degenerate indices
        idx = (dc < atol).nonzero()[0]
        if len(idx) == 0:
            # There are no degenerate coefficients
            return deg

        # These are the points were we split the degeneracies
        seps = (np.diff(idx) > 1).nonzero()[0]
        IDX = np.array_split(idx, seps + 1)
        for idx in IDX:
            deg.append(np.append(sidx[idx], sidx[idx[-1] + 1]))
        return deg

    def asState(self):
        s = State(self.state.copy(), self.parent)
        s.info = self.info
        return s

    def asCoefficient(self):
        c = Coefficient(self.c.copy(), self.parent)
        c.info = self.info
        return c
