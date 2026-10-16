# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import heapq
import itertools
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.linalg.blas import get_blas_funcs

import sisl._array as _a
from sisl._core.oplist import oplist
from sisl._internal import set_module
from sisl.messages import SislError, warn
from sisl.typing import SeqOrScalarInt

from .brillouinzone import BrillouinZone, BrillouinZoneDispatcher
from .spin import Spin

__all__ = ["AdaptiveBrillouinZone", "GaussKronrodGrid"]


@set_module("sisl.physics")
class AdaptiveBrillouinZone(BrillouinZone, metaclass=ABCMeta):
    r"""Base class for Brillouin zone integrations with adaptive k-point grids

    An adaptive grid selects its k-points from the values of the integrand.
    Before an integration, there are no k-points. Therefore `k`, `weight` and
    ``len(bz)`` raise a `TypeError`.

    The integral is the Brillouin zone average (the weights sum to 1), as for
    `MonkhorstPack`.

    ``bz.apply.average`` integrates the methods of the parent, as for
    `BrillouinZone` (``wrap`` and ``eta`` work the same way, ``wrap`` gets
    ``weight=None``). The other ``apply`` modes need fixed k-points and raise a
    `TypeError`, use them on `last_bz`:

    >>> G = bz.apply.average.green(E)  # doctest: +SKIP
    >>> eigs = bz.last_bz().apply.array.eigh()  # doctest: +SKIP

    After an integration, `last_bz` returns a `BrillouinZone` with the k-points
    and weights that the integration used. The object only stores what is necessary
    to create these again, so the `BrillouinZone` is only created on request.

    The base class only stores the `parent`. Each sub-class defines its own
    parameters (directions, limits, tolerances).

    Parameters
    ----------
    parent : object or array-like
       An object with associated ``parent.cell`` and ``parent.rcell`` or
       an array of floats which may be turned into a `Lattice`

    See Also
    --------
    GaussKronrodGrid : adaptive Gauss-Kronrod cubature
    """

    # own dispatches, the BrillouinZone dispatches need fixed k-points
    apply = BrillouinZoneDispatcher(
        "apply",
        type_dispatcher=None,
        obj_getattr=lambda obj, key: getattr(obj.parent, key),
    )

    def __init__(self, parent):
        self.set_parent(parent)
        self._bz_attr = ("", None)

    def _undefined(self, what: str) -> TypeError:
        """The error for `what` (k-points, weights, length) that has no value before an integration"""
        return TypeError(
            f"{self.__class__.__name__} has no {what} before an integration; "
            "use last_bz() after integrate(...) to get the used k-points."
        )

    @property
    def k(self):
        """Not defined for adaptive grids, raises `TypeError`"""
        raise self._undefined("k-points")

    @property
    def weight(self):
        """Not defined for adaptive grids, raises `TypeError`"""
        raise self._undefined("weights")

    def __len__(self) -> int:
        raise self._undefined("length")

    def __bool__(self) -> bool:
        return True

    def _not_converged(self, err: float, tol: float, nk: int, limit: str) -> None:
        """Warn that the integration stopped at `limit` before reaching `tol`"""
        warn(
            f"{self.__class__.__name__}.integrate did not converge: "
            f"error estimate {err:.3e} > tolerance {tol:.3e} using {nk} k-points "
            f"({limit})"
        )

    @abstractmethod
    def integrate(self, func: Callable) -> Union[np.ndarray, oplist]:
        """Integrate `func` over the Brillouin zone

        The k-points that the integration used are available from `last_bz`.

        Parameters
        ----------
        func :
           the integrand, called as ``func(k)`` with ``k`` a reduced k-point with
           shape ``(3,)``. It must return a scalar, an array with a fixed shape,
           or a sequence (`list`, `tuple` or `~sisl.oplist`, not `numpy.ndarray`)
           of these. A sequence is integrated element by element, so its elements
           can have different shapes (as for ``bz.apply.average`` with `~sisl.oplist`).
           The convergence includes all elements.

        Returns
        -------
        numpy.ndarray or oplist
            the integral (equal to the Brillouin zone average). An `~sisl.oplist`
            when `func` returns a sequence.
        """

    @abstractmethod
    def last_bz(self) -> BrillouinZone:
        """The k-points and weights of the last `integrate` call

        The weights sum to 1, and
        ``sum(w * func(k) for k, w in last_bz().iter(True))`` is the integral
        that `integrate` returned.
        A grid can evaluate the integrand at more k-points than it uses in the
        final result (e.g. `GaussKronrodGrid` does not re-use the k-points of a
        halved box). These k-points are not in the returned object.

        Raises
        ------
        SislError
            if `integrate` was not called before
        """

    def _str_attrs(self) -> str:
        """Attributes shown by `__str__`, sub-classes extend this"""
        return ""

    def __str__(self) -> str:
        parent = str(self._parent_lattice()).replace("\n", "\n ")
        return f"{self.__class__.__name__}{{{self._str_attrs()}\n {parent}\n}}"

    def __getstate__(self):
        state = self.__dict__.copy()
        state["parent_class"] = self.parent.__class__
        state["parent"] = self.parent.__getstate__()
        return state

    def __setstate__(self, state):
        state = state.copy()
        parent_class = state.pop("parent_class")
        parent = parent_class.__new__(parent_class)
        parent.__setstate__(state.pop("parent"))
        self.__dict__.update(state)
        self.set_parent(parent)


# Gauss-Kronrod G7-K15 rule on [-1, 1] (QUADPACK qk15)
_GK15_XK = np.array(
    [
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144845693013,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
        0.000000000000000000000000000000000,
    ]
)
_GK15_WK = np.array(
    [
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714,
    ]
)
_GK15_WG = np.array(
    [
        0.129484966168869693270611432679082,
        0.279705391489276667901467771423780,
        0.381830050505118944950369775488975,
        0.417959183673469387755102040816327,
    ]
)


def _gk15_unit():
    """Nodes on [0, 1] and Kronrod and Gauss weights (Gauss is 0 at Kronrod-only nodes)"""
    j = np.concatenate([np.arange(7), np.arange(7, -1, -1)])
    sign = np.concatenate([-np.ones(7), np.ones(8)])
    x = sign * _GK15_XK[j]
    wk = _GK15_WK[j]
    wg = np.where(j % 2 == 1, _GK15_WG[(j - 1) // 2], 0.0)
    # map [-1, 1] -> [0, 1]
    return (x + 1) / 2, wk / 2, wg / 2


_GK_X, _GK_WK, _GK_WG = _gk15_unit()


def _tensor_weights(ws: list) -> np.ndarray:
    """Weights of the tensor product rule (flattened) from the 1D weights `ws`"""
    w = ws[0]
    for wa in ws[1:]:
        w = np.multiply.outer(w, wa)
    return w.ravel()


def _is_oplist(a) -> bool:
    """Whether `a` is a sequence that is integrated element by element (as `oplist`)"""
    return isinstance(a, (list, tuple)) and not isinstance(a, np.ndarray)


def _maxabs(a) -> float:
    """Largest absolute element of an array, or of all elements of an `oplist`"""
    if _is_oplist(a):
        return max((_maxabs(x) for x in a), default=0.0)
    return float(np.max(np.abs(a)))


def _box_w1d(w: np.ndarray, h: np.ndarray, scale: float) -> list:
    """1D weights `w` along each direction of a box, `scale` is in the first direction"""
    ws = [w * h[a] for a in range(len(h))]
    ws[0] = ws[0] * scale
    return ws


def _structure_error() -> ValueError:
    return ValueError(
        "GaussKronrodGrid requires the integrand to return the same structure "
        "(sequence lengths and array shapes) for all k-points"
    )


def _axpy_numpy(x: np.ndarray, y: np.ndarray, a) -> np.ndarray:
    y += a * x
    return y


def _axpy(dtype):
    """``axpy(x, y, a)`` adds ``a * x`` to the 1D array `y`, in place"""
    if dtype in (np.float32, np.float64, np.complex64, np.complex128):
        return get_blas_funcs("axpy", dtype=dtype)
    return _axpy_numpy


class _WeightedSums:
    """Weighted sums of one array element of the integrand

    Sum 0 is a compensated sum (``sums[0] + comp``): each line of `_GK_X` values
    is summed in `line`, and the line sum is added with Knuth TwoSum.
    The other sums are plain. `t` and `z` are scratch buffers.
    All arrays are 1D, `shape` is the shape of the integrand value.
    """

    __slots__ = ("shape", "dtype", "axpy", "sums", "comp", "line", "nline", "t", "z")

    def __init__(self, r: np.ndarray, n: int):
        self.shape = r.shape
        self._allocate(np.result_type(r.dtype, np.float64), n)

    def _allocate(self, dtype, n: int) -> None:
        self.dtype = dtype
        self.axpy = _axpy(dtype)
        size = int(np.prod(self.shape))
        self.sums = [np.zeros(size, dtype) for _ in range(n)]
        self.comp = np.zeros(size, dtype)
        self.line = np.zeros(size, dtype)
        self.nline = 0
        self.t = np.empty(size, dtype)
        self.z = np.empty(size, dtype)

    def _upcast(self, dtype) -> None:
        old = (self.sums, self.comp, self.line, self.nline)
        self._allocate(dtype, len(old[0]))
        for s, o in zip(self.sums, old[0]):
            s[:] = o
        self.comp[:] = old[1]
        self.line[:] = old[2]
        self.nline = old[3]

    def add(self, r, ws) -> None:
        """Add ``ws[i] * r`` to sum `i`, in place"""
        r = np.asarray(r)
        if r.shape != self.shape:
            raise _structure_error()
        dtype = np.result_type(self.dtype, r.dtype)
        if dtype != self.dtype:
            self._upcast(dtype)
        x = np.ascontiguousarray(r, dtype=self.dtype).reshape(-1)
        for s, w in zip(self.sums[1:], ws[1:]):
            self.axpy(x, s, a=w)
        self.axpy(x, self.line, a=ws[0])
        self.nline += 1
        if self.nline == len(_GK_X):
            self._flush()

    def _flush(self) -> None:
        """Add `line` to the compensated sum (Knuth TwoSum), and zero it"""
        s, x, t, z = self.sums[0], self.line, self.t, self.z
        np.add(s, x, out=t)
        np.subtract(t, s, out=z)
        # rounding error (s - (t - z)) + (x - z) goes to comp
        x -= z
        np.subtract(t, z, out=z)
        s -= z
        s += x
        self.comp += s
        # t is the new sum, s the next scratch buffer
        self.sums[0], self.t = t, s
        x[:] = 0
        self.nline = 0

    def value(self, i: int) -> np.ndarray:
        """Weighted sum `i`"""
        if i == 0:
            if self.nline > 0:
                self._flush()
            v = self.sums[0] + self.comp
        else:
            v = self.sums[i]
        return v.reshape(self.shape)


def _sums_new(r, n: int):
    """`n` zero weighted sums with the structure of the integrand value `r`

    A sequence (not `numpy.ndarray`) is integrated element by element.
    """
    if _is_oplist(r):
        return [_sums_new(x, n) for x in r]
    return _WeightedSums(np.asarray(r), n)


def _sums_add(sums, r, ws) -> None:
    """Add ``ws[i] * r`` to the weighted sums `i` of `sums`, in place"""
    if isinstance(sums, _WeightedSums):
        if _is_oplist(r):
            raise _structure_error()
        sums.add(r, ws)
        return
    if not _is_oplist(r) or len(r) != len(sums):
        raise _structure_error()
    for s, x in zip(sums, r):
        _sums_add(s, x, ws)


def _sums_get(sums, i: int):
    """Weighted sum `i`, an `oplist` for a sequence integrand"""
    if isinstance(sums, _WeightedSums):
        return sums.value(i)
    return oplist(_sums_get(s, i) for s in sums)


def _copy(a):
    """Copy of an array or a (nested) `oplist`"""
    if isinstance(a, oplist):
        return oplist(_copy(x) for x in a)
    return np.array(a)


def _zeros_like(a):
    """Zeros with the structure of an array or a (nested) `oplist`"""
    if isinstance(a, oplist):
        return oplist(_zeros_like(x) for x in a)
    return np.zeros_like(a)


def _two_sum(s, c, x, sign: int = 1):
    """Add ``sign * x`` to the compensated sum ``s + c`` (Knuth TwoSum)

    The rounding error of each addition is added to `c`, so ``s + c`` keeps
    full precision when large terms cancel. Returns the new ``(s, c)``.
    """
    if isinstance(s, oplist):
        for i, (si, ci, xi) in enumerate(zip(s, c, x)):
            s[i], c[i] = _two_sum(si, ci, xi, sign)
        return s, c
    dtype = np.result_type(s.dtype, x.dtype)
    if dtype != s.dtype:
        s, c = s.astype(dtype), c.astype(dtype)
    if sign < 0:
        x = -x
    t = s + x
    z = t - s
    # rounding error: (s - (t - z)) + (x - z)
    s -= t - z
    z *= -1
    z += x
    s += z
    c += s
    return t, c


def _sum_value(s, c):
    """The value of the compensated sum ``s + c``"""
    if isinstance(s, oplist):
        return oplist(_sum_value(x, y) for x, y in zip(s, c))
    return s + c


def _fsum_add(s: float, c: float, x: float) -> tuple[float, float]:
    """Add `x` to the compensated (Neumaier) sum ``s + c``"""
    t = s + x
    if abs(s) >= abs(x):
        c += (s - t) + x
    else:
        c += (x - t) + s
    return t, c


@set_module("sisl.physics")
class GaussKronrodGrid(AdaptiveBrillouinZone):
    r"""Adaptive Gauss-Kronrod cubature of the Brillouin zone

    The integrated Brillouin zone is split in boxes. Each box is integrated with
    a tensor product of the 15-point Kronrod rule (7-point Gauss embedded).
    A box uses :math:`15^d` k-points for :math:`d` integrated directions.
    The error of a box is the largest absolute element of the difference between
    the Kronrod and the Gauss results. The box with the largest error is
    halved along the direction with the largest error, until the summed error
    is below the tolerance.

    This is efficient for integrands with few sharp features (e.g. small
    broadening), mainly in 1 or 2 dimensions.

    The integration domain is the reduced Brillouin zone :math:`]-1/2 ; 1/2]` along
    the integrated directions. The k-point component along the other directions is 0.

    With time-reversal symmetry (`trs`), only :math:`[0 ; 1/2]` along the first
    integrated direction is integrated, and the weights are doubled. As for
    `MonkhorstPack`, the result is only correct when :math:`f(\mathbf k)` and
    :math:`f(-\mathbf k)` have the same integral, e.g. a Green function symmetrized
    as :math:`(\mathbf G + \mathbf G^T)/2` after the integration.

    Parameters
    ----------
    parent : object or array-like
       An object with associated ``parent.cell`` and ``parent.rcell`` or
       an array of floats which may be turned into a `Lattice`
    nkpt :
       initial number of k-points along each lattice direction, per reciprocal
       lattice vector (as for `MonkhorstPack`). A direction with 0 k-points is not
       integrated. A scalar is the same value for the periodic directions of the
       parent lattice (``pbc``); the other directions are not integrated.
       The value is rounded up to a multiple of 15 (one box has 15 k-points along a
       direction). With `trs`, the integrated half of the direction gets half the k-points.
       The initial k-points are always integrated, also when they exceed `max_k`.
    max_k :
       a scalar is the maximum number of integrand evaluations for one integration.
       A sequence of 3 values is the maximum k-point density along each lattice
       direction, in k-points per reciprocal lattice vector: a box of width :math:`h`
       (in reduced units) is not halved along a direction when :math:`15/(h/2)` is
       larger than the value of that direction. The box is then halved along the
       direction with the next largest error. The total number is then not limited.
       Values for directions that are not integrated are not used.
    trs :
       whether time-reversal symmetry is used. If ``None``, it is true for
       (un)-polarized calculations, and false for non-colinear and higher spin
       configurations (as for `MonkhorstPack`).
    atol :
       absolute tolerance of the integral (largest absolute element error)
    rtol :
       relative tolerance of the integral, relative to the largest absolute element.
       For a sequence (`~sisl.oplist`) result, the errors and the largest element
       are taken over all elements. Elements with very different magnitudes should
       then be scaled, or use `atol`.

    Examples
    --------
    The Green function of a 1D chain, with a small broadening, integrated along the
    first lattice direction only:

    >>> H = Hamiltonian(geom.sc(1.0, Atom(1, R=1.1)))
    >>> H.set_nsc([3, 1, 1])
    >>> H.construct([[0.1, 1.1], [0.0, -1.0]])
    >>> gk = GaussKronrodGrid(H, [15, 0, 0], atol=1e-8)
    >>> z = 0.5 + 1e-3j
    >>> def G(k):
    ...     return np.linalg.inv(z * H.Sk(k, format="array") - H.Hk(k, format="array"))
    >>> Gavg = gk.integrate(G)
    >>> print(f"{Gavg[0, 0]:.6f}")
    0.000069-0.516398j
    >>> len(gk.last_bz())  # the k-points that were used
    255

    Integrate the first two directions, and limit the k-point density to that of a
    1000 x 200 grid:

    >>> gk = GaussKronrodGrid(H, [15, 15, 0], max_k=[1000, 200, 1])
    """

    def __init__(
        self,
        parent,
        nkpt: SeqOrScalarInt = 15,
        max_k: SeqOrScalarInt = 10000,
        trs: Optional[bool] = None,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ):
        super().__init__(parent)
        cls = self.__class__.__name__

        nkpt3 = _a.arrayi(nkpt)
        if nkpt3.ndim == 0:
            if nkpt3 < 1:
                raise ValueError(f"{cls} requires a scalar nkpt >= 1, got {nkpt}")
            try:
                pbc = self._parent_lattice().pbc
            except (AttributeError, ValueError):
                pbc = np.ones(3, dtype=bool)
            if not np.any(pbc):
                raise ValueError(
                    f"{cls} found no periodic directions for a scalar nkpt; "
                    "give nkpt for each lattice direction"
                )
            nkpt3 = np.where(pbc, nkpt3, 0)
        if nkpt3.shape != (3,) or np.any(nkpt3 < 0) or not np.any(nkpt3 > 0):
            raise ValueError(
                f"{cls} requires nkpt to be a scalar or 3 values >= 0 "
                f"(at least one > 0), got {nkpt}"
            )
        self._nkpt_in = nkpt3
        self._axes = np.flatnonzero(nkpt3 > 0)

        nk = _a.arrayi(max_k)
        if nk.ndim == 0:
            if nk < 1:
                raise ValueError(f"{cls} requires max_k >= 1, got {max_k}")
            self._max_k = int(nk)
            self._max_k_axes = None
        elif nk.shape == (3,):
            if np.any(nk[self._axes] < 1):
                raise ValueError(
                    f"{cls} requires max_k >= 1 along the integrated directions, "
                    f"got {max_k}"
                )
            self._max_k = None
            self._max_k_axes = nk[self._axes]
        else:
            raise ValueError(
                f"{cls} requires max_k to be a scalar or 3 values, got {max_k}"
            )
        self._max_k_in = nk

        if trs is None:
            trs = True
            try:
                trs = parent.spin <= Spin("polarized")
            except AttributeError:
                pass
        self._trs = int(self._axes[0]) if trs else -1

        # initial number of boxes, from the k-points in the integrated domain
        _, width = self._domain()
        nbox = np.ceil(nkpt3[self._axes] * width / len(_GK_X) - 1e-12)
        self._n = np.maximum(nbox, 1).astype(int)

        if atol < 0 or rtol < 0:
            raise ValueError(f"{cls} requires atol, rtol >= 0")
        self._atol = atol
        self._rtol = rtol
        self._last = None

    @property
    def trs(self) -> bool:
        """Whether time-reversal symmetry is used"""
        return self._trs >= 0

    def _str_attrs(self) -> str:
        """Settings shown by `__str__`"""
        return (
            f"nkpt: {self._nkpt_in.tolist()}, max_k: {self._max_k_in.tolist()}, "
            f"trs: {self.trs}, "
            f"atol: {self._atol}, rtol: {self._rtol},"
        )

    def _domain(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower bound and width of the integrated domain along the integrated directions"""
        lo = np.full(len(self._axes), -0.5)
        width = np.ones(len(self._axes))
        if self._trs >= 0:
            i = np.flatnonzero(self._axes == self._trs)[0]
            lo[i] = 0.0
            width[i] = 0.5
        return lo, width

    def _tolerance(self, value: npt.ArrayLike) -> float:
        """Requested absolute tolerance for the integral `value`"""
        return max(self._atol, self._rtol * _maxabs(value))

    def _k_full(self, k: np.ndarray) -> np.ndarray:
        """Convert k-points along the integrated directions to 3D reduced k-points"""
        kfull = _a.zerosd([len(k), 3])
        kfull[:, self._axes] = k
        return kfull

    def _box_kw(self, lo: np.ndarray, h: np.ndarray, scale: float):
        """k-points and (Kronrod) weights of one box, the weights are multiplied by `scale`"""
        d = len(lo)
        nodes = [lo[a] + h[a] * _GK_X for a in range(d)]
        kd = np.stack([g.ravel() for g in np.meshgrid(*nodes, indexing="ij")], axis=1)
        return self._k_full(kd), _tensor_weights(_box_w1d(_GK_WK, h, scale))

    def _box(self, func, lo: np.ndarray, h: np.ndarray, scale: float):
        """Integrate one box

        Returns the Kronrod value, the error and the axes sorted by decreasing error.
        Each integrand value is added to the weighted sums at once, it is not stored.
        """
        d = len(lo)
        k, wk = self._box_kw(lo, h, scale)
        wk1 = _box_w1d(_GK_WK, h, scale)
        wg1 = _box_w1d(_GK_WG, h, scale)
        # weights of K, of K - G and of K - Q_a for each direction a
        W = [wk, wk - _tensor_weights(wg1)]
        if d > 1:
            W.extend(
                wk - _tensor_weights(wk1[:a] + [wg1[a]] + wk1[a + 1 :])
                for a in range(d)
            )
        W = np.stack(W, axis=1)

        sums = None
        for kk, w in zip(k, W):
            r = func(kk)
            if sums is None:
                sums = _sums_new(r, len(w))
            _sums_add(sums, r, w)

        K = _sums_get(sums, 0)
        err = _maxabs(_sums_get(sums, 1))
        if d == 1:
            dir_err = np.array([err])
        else:
            dir_err = np.array([_maxabs(_sums_get(sums, 2 + a)) for a in range(d)])
        order = np.argsort(dir_err, kind="stable")[::-1].tolist()
        return K, err, order

    def _can_split(self, h: np.ndarray, a: int) -> bool:
        """Whether halving `h` along integrated direction `a` keeps the density below `max_k`"""
        if self._max_k_axes is None:
            return True
        return len(_GK_X) / (h[a] / 2) <= self._max_k_axes[a] * (1 + 1e-12)

    def integrate(self, func: Callable) -> Union[np.ndarray, oplist]:
        """Integrate `func` over the Brillouin zone with adaptive Gauss-Kronrod boxes

        The integration starts with the boxes from `nkpt`. Then it halves the box
        with the largest error, until the summed error of all boxes is below
        ``max(atol, rtol * max|I|)``, or until `max_k` stops it.

        Parameters
        ----------
        func :
           the integrand, called as ``func(k)`` with ``k`` a reduced k-point with
           shape ``(3,)``. It must return a scalar, an array with a fixed shape,
           or a sequence (`list`, `tuple` or `~sisl.oplist`, not `numpy.ndarray`)
           of these. A sequence is integrated element by element, and its elements
           can have different shapes. The errors include all elements.
           Each returned value is used at once and not stored, so `func` can
           return the same array (changed in place) at each call.

        Returns
        -------
        numpy.ndarray or oplist
            the Brillouin zone average of `func`. It is an `~sisl.oplist` when
            `func` returns a sequence.

        Raises
        ------
        ValueError
            if `func` returns sequences with different lengths, or arrays with
            different shapes

        Notes
        -----
        The integration keeps one integrated value for each box, and a few values
        for the box that it evaluates (6 values for :math:`d=1`, :math:`6+d` for
        :math:`d>1` integrated directions). The memory thus scales with the number
        of boxes times the size of a value of `func`.

        Warns
        -----
        SislWarning
            when `max_k` stops the integration before the error is below the tolerance.
            The returned value is then the last value.
            Also when `func` returns a value that is not finite (NaN or inf); the
            integration then stops at once.

        See Also
        --------
        last_bz : the k-points and weights that the integration used

        Examples
        --------
        >>> gk = GaussKronrodGrid(Lattice(1), [15, 0, 0], atol=1e-10, trs=False)
        >>> def f(k):
        ...     return 1 / (1.5 - np.cos(2 * np.pi * k[0]))
        >>> print(f"{gk.integrate(f):.10f}")
        0.8944271910
        >>> print(f"{1 / np.sqrt(1.5**2 - 1):.10f}")
        0.8944271910

        Values with different shapes in one integration:

        >>> v = gk.integrate(lambda k: [f(k), np.ones((2, 2)) * f(k)])
        >>> type(v).__name__, v[1].shape
        ('oplist', (2, 2))
        """
        d = len(self._axes)
        nper = len(_GK_X) ** d
        lo0, width = self._domain()
        scale = 1 / float(np.prod(width))
        h0 = width / self._n

        # boxes that may be halved (heap on the error), and boxes that may not
        boxes = []
        frozen = []
        counter = itertools.count()

        def push(lo, h):
            K, err, order = self._box(func, lo, h, scale)
            heapq.heappush(boxes, (-err, next(counter), lo, h, K, order))
            return K, err

        # running integral (value + value_c) and error (err + err_c) of all boxes,
        # both compensated sums: a split adds the children and subtracts the parent
        init = [
            push(lo0 + _a.arrayd(idx) * h0, h0.copy())
            for idx in itertools.product(*[range(n) for n in self._n])
        ]
        value = _copy(init[0][0])
        value_c = _zeros_like(value)
        err, err_c = init[0][1], 0.0
        for K, e in init[1:]:
            value, value_c = _two_sum(value, value_c, K)
            err, err_c = _fsum_add(err, err_c, e)
        del init
        nk = len(boxes) * nper

        tol = self._tolerance(value)
        while not (err + err_c <= tol and np.isfinite(err + err_c)):
            if not np.isfinite(err + err_c):
                self._not_converged(err + err_c, tol, nk, "the integrand is not finite")
                break
            if len(boxes) == 0 or (
                self._max_k is not None and nk + 2 * nper > self._max_k
            ):
                self._not_converged(
                    err + err_c, tol, nk, f"max_k={self._max_k_in.tolist()}"
                )
                break
            box = heapq.heappop(boxes)
            neg_err, _, lo, h, K, order = box
            split = next((a for a in order if self._can_split(h, a)), None)
            if split is None:
                frozen.append(box)
                continue
            del box
            h = h.copy()
            h[split] /= 2
            hi = lo.copy()
            hi[split] += h[split]
            K_lo, e_lo = push(lo, h)
            K_hi, e_hi = push(hi, h)
            nk += 2 * nper

            value, value_c = _two_sum(value, value_c, K_lo)
            value, value_c = _two_sum(value, value_c, K_hi)
            value, value_c = _two_sum(value, value_c, K, -1)
            del K, K_lo, K_hi
            err, err_c = _fsum_add(err, err_c, e_lo)
            err, err_c = _fsum_add(err, err_c, e_hi)
            err, err_c = _fsum_add(err, err_c, neg_err)
            tol = self._tolerance(value)

        all_boxes = boxes + frozen
        self._last = {
            "lo": np.array([b[2] for b in all_boxes]),
            "h": np.array([b[3] for b in all_boxes]),
            "scale": scale,
        }
        value = _sum_value(value, value_c)
        if isinstance(value, oplist):
            return value
        return np.asarray(value)[()]

    def last_bz(self) -> BrillouinZone:
        """The k-points and weights of the last `integrate` call

        The k-points are the Gauss-Kronrod nodes of the final boxes. They are
        created from the stored box bounds when this method is called.
        The k-points of halved boxes were also evaluated, but they are not in the
        final result, and thus not in the returned object.

        Returns
        -------
        BrillouinZone
            the k-points and weights (the weights sum to 1). The weighted sum of
            the integrand at these k-points is the value that `integrate` returned.

        Raises
        ------
        SislError
            if `integrate` was not called before

        Examples
        --------
        >>> gk = GaussKronrodGrid(Lattice(1), [15, 0, 0], trs=False)
        >>> v = gk.integrate(lambda k: k[0] ** 2)
        >>> bz = gk.last_bz()
        >>> len(bz)
        15
        >>> print(f"{sum(w * k[0] ** 2 for k, w in bz.iter(True)):.10f}")
        0.0833333333
        """
        if self._last is None:
            raise SislError(
                f"{self.__class__.__name__}.last_bz requires a call to integrate first"
            )
        last = self._last
        kw = [
            self._box_kw(lo, h, last["scale"]) for lo, h in zip(last["lo"], last["h"])
        ]
        k = np.concatenate([k for k, _ in kw])
        w = np.concatenate([w for _, w in kw])
        return BrillouinZone(self.parent, k, w)
