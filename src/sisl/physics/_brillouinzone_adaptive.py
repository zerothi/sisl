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

import sisl._array as _a
from sisl._core.oplist import oplist
from sisl._internal import set_module
from sisl.messages import SislError, warn
from sisl.typing import SeqOrScalarInt

from .brillouinzone import BrillouinZone
from .spin import Spin

__all__ = ["AdaptiveBrillouinZone", "GaussKronrodGrid"]


@set_module("sisl.physics")
class AdaptiveBrillouinZone(BrillouinZone, metaclass=ABCMeta):
    r"""Base class for Brillouin zone integrations with adaptive k-point grids

    An adaptive grid selects its k-points from the values of the integrand.
    Before an integration, there are no k-points. Therefore `k`, `weight` and
    ``len(bz)`` raise a `TypeError`. This also makes ``bz.apply`` fail, use
    `integrate` instead.

    The integral is the Brillouin zone average (the weights sum to 1), as for
    `MonkhorstPack`.

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


def _contract(F: np.ndarray, ws) -> np.ndarray:
    """Contract the leading axes of `F` with the weights in `ws`, one by one"""
    for w in ws:
        F = np.tensordot(w, F, axes=(0, 0))
    return F


def _is_oplist(a) -> bool:
    """Whether `a` is a sequence that is integrated element by element (as `oplist`)"""
    return isinstance(a, (list, tuple)) and not isinstance(a, np.ndarray)


def _maxabs(a) -> float:
    """Largest absolute element of an array, or of all elements of an `oplist`"""
    if _is_oplist(a):
        return max((_maxabs(x) for x in a), default=0.0)
    return float(np.max(np.abs(a)))


def _gk_rule(results: list, wk: list, wg: list):
    """Kronrod value, error and error for each direction of the integrand values `results`

    `results` has one integrand value for each node of the box.
    A sequence (not `numpy.ndarray`) is integrated element by element,
    and the Kronrod value is then an `oplist`.
    The errors are the largest absolute errors of all elements.
    """
    d = len(wk)
    first = results[0]
    if _is_oplist(first):
        n = len(first)
        if any(not _is_oplist(r) or len(r) != n for r in results):
            raise ValueError(
                "GaussKronrodGrid requires the integrand to return sequences "
                "with the same length for all k-points"
            )
        parts = [_gk_rule([r[i] for r in results], wk, wg) for i in range(n)]
        K = oplist(p[0] for p in parts)
        err = max((p[1] for p in parts), default=0.0)
        dir_err = np.max([p[2] for p in parts], axis=0) if n > 0 else np.zeros(d)
        return K, err, dir_err

    F = np.stack([np.asarray(r) for r in results])
    F = F.reshape((len(_GK_X),) * d + F.shape[1:])
    K = _contract(F, wk)
    err = _maxabs(K - _contract(F, wg))
    if d == 1:
        dir_err = np.array([err])
    else:
        dir_err = np.array(
            [
                _maxabs(K - _contract(F, wk[:a] + [wg[a]] + wk[a + 1 :]))
                for a in range(d)
            ]
        )
    return K, err, dir_err


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
       integrated. A scalar is the same value for all 3 directions.
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
            nkpt3 = np.repeat(nkpt3, 3)
            if nkpt3[0] < 1:
                raise ValueError(f"{cls} requires a scalar nkpt >= 1, got {nkpt}")
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
        w = _GK_WK * h[0] * scale
        for a in range(1, d):
            w = np.multiply.outer(w, _GK_WK * h[a])
        return self._k_full(kd), w.ravel()

    def _box(self, func, lo: np.ndarray, h: np.ndarray, scale: float):
        """Integrate one box

        Returns the Kronrod value, the error and the axes sorted by decreasing error.
        """
        d = len(lo)
        k, _ = self._box_kw(lo, h, scale)
        results = [func(kk) for kk in k]

        wk = [_GK_WK * h[a] for a in range(d)]
        wg = [_GK_WG * h[a] for a in range(d)]
        wk[0] = wk[0] * scale
        wg[0] = wg[0] * scale
        K, err, dir_err = _gk_rule(results, wk, wg)
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

        Returns
        -------
        numpy.ndarray or oplist
            the Brillouin zone average of `func`. It is an `~sisl.oplist` when
            `func` returns a sequence.

        Raises
        ------
        ValueError
            if `func` returns sequences with different lengths

        Warns
        -----
        SislWarning
            when `max_k` stops the integration before the error is below the tolerance.
            The returned value is then the last value.

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

        for idx in itertools.product(*[range(n) for n in self._n]):
            push(lo0 + _a.arrayd(idx) * h0, h0.copy())
        nk = len(boxes) * nper

        def total():
            all_boxes = boxes + frozen
            return sum(b[4] for b in all_boxes), sum(-b[0] for b in all_boxes)

        value, err = total()
        tol = self._tolerance(value)
        while err > tol:
            if len(boxes) == 0 or (
                self._max_k is not None and nk + 2 * nper > self._max_k
            ):
                self._not_converged(err, tol, nk, f"max_k={self._max_k_in.tolist()}")
                break
            box = heapq.heappop(boxes)
            _, _, lo, h, _, order = box
            split = next((a for a in order if self._can_split(h, a)), None)
            if split is None:
                frozen.append(box)
                continue
            h = h.copy()
            h[split] /= 2
            hi = lo.copy()
            hi[split] += h[split]
            push(lo, h)
            push(hi, h)
            nk += 2 * nper
            value, err = total()
            tol = self._tolerance(value)

        all_boxes = boxes + frozen
        self._last = {
            "lo": np.array([b[2] for b in all_boxes]),
            "h": np.array([b[3] for b in all_boxes]),
            "scale": scale,
        }
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
