# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import heapq
import itertools
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import Optional

import numpy as np
import numpy.typing as npt

import sisl._array as _a
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

    def _undefined(self, what: str):
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
    def integrate(self, func: Callable[[np.ndarray], npt.ArrayLike]) -> np.ndarray:
        """Integrate `func` over the Brillouin zone

        The k-points that the integration used are available from `last_bz`.

        Parameters
        ----------
        func :
           the integrand, called as ``func(k)`` with ``k`` a reduced k-point with
           shape ``(3,)``. It must return a scalar or an array with a fixed shape.

        Returns
        -------
        numpy.ndarray
            the integral (equal to the Brillouin zone average)
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


def _maxabs(a) -> float:
    return float(np.max(np.abs(a)))


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
    n :
       initial number of boxes along each lattice direction (as the ``nkpt``
       argument of `MonkhorstPack`). A direction with 0 boxes is not integrated.
       A scalar is the same value for all 3 directions.
       The initial boxes are always integrated, also when they exceed `max_k`.
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
       relative tolerance of the integral, relative to the largest absolute element

    Examples
    --------
    Integrate along the first lattice direction only:

    >>> H = si.Hamiltonian(...)
    >>> gk = GaussKronrodGrid(H, [1, 0, 0], atol=1e-8)
    >>> def G(k):
    ...     return np.linalg.inv((E + 1j * eta) * H.Sk(k, format="array") - H.Hk(k, format="array"))
    >>> Gavg = gk.integrate(G)
    >>> bz = gk.last_bz()  # the k-points and weights that were used
    >>> len(bz)

    Integrate the first two directions, and limit the k-point density to that of a
    1000 x 200 grid:

    >>> gk = GaussKronrodGrid(H, [1, 1, 0], max_k=[1000, 200, 1])
    """

    def __init__(
        self,
        parent,
        n: SeqOrScalarInt = 1,
        max_k: SeqOrScalarInt = 10000,
        trs: Optional[bool] = None,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ):
        super().__init__(parent)
        cls = self.__class__.__name__

        n3 = _a.arrayi(n)
        if n3.ndim == 0:
            n3 = np.repeat(n3, 3)
            if n3[0] < 1:
                raise ValueError(f"{cls} requires a scalar n >= 1, got {n}")
        if n3.shape != (3,) or np.any(n3 < 0) or not np.any(n3 > 0):
            raise ValueError(
                f"{cls} requires n to be a scalar or 3 values >= 0 "
                f"(at least one > 0), got {n}"
            )
        self._axes = np.flatnonzero(n3 > 0)
        self._n = n3[self._axes]

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
        n3 = np.zeros(3, dtype=int)
        n3[self._axes] = self._n
        return (
            f"n: {n3.tolist()}, max_k: {self._max_k_in.tolist()}, trs: {self.trs}, "
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
        return max(self._atol, self._rtol * float(np.max(np.abs(value))))

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
        F = np.stack([np.asarray(func(kk)) for kk in k])
        F = F.reshape((len(_GK_X),) * d + F.shape[1:])

        wk = [_GK_WK * h[a] for a in range(d)]
        wg = [_GK_WG * h[a] for a in range(d)]
        wk[0] = wk[0] * scale
        wg[0] = wg[0] * scale
        K = _contract(F, wk)
        err = _maxabs(K - _contract(F, wg))
        if d == 1:
            order = [0]
        else:
            dir_err = [
                _maxabs(K - _contract(F, wk[:a] + [wg[a]] + wk[a + 1 :]))
                for a in range(d)
            ]
            order = np.argsort(dir_err, kind="stable")[::-1].tolist()
        return K, err, order

    def _can_split(self, h: np.ndarray, a: int) -> bool:
        """Whether halving `h` along integrated direction `a` keeps the density below `max_k`"""
        if self._max_k_axes is None:
            return True
        return len(_GK_X) / (h[a] / 2) <= self._max_k_axes[a] * (1 + 1e-12)

    def integrate(self, func):
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
        return np.asarray(value)[()]

    def last_bz(self) -> BrillouinZone:
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
