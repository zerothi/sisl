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
from sisl._internal import set_module
from sisl.messages import warn
from sisl.typing import CellAxes, SeqOrScalarInt
from sisl.utils.misc import direction, listify

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

    The integration domain is the reduced Brillouin zone :math:`]-1/2 ; 1/2]` along
    the sampled `axes`. The weights sum to 1, so the integral is equal to the
    Brillouin zone average, as for `MonkhorstPack`.

    With time-reversal symmetry (`trs`), only :math:`[0 ; 1/2]` along the first of the
    `axes` is integrated, and the weights are doubled. As for `MonkhorstPack`, the
    result is only correct when :math:`f(\mathbf k)` and :math:`f(-\mathbf k)` have the
    same integral, e.g. a Green function symmetrized as
    :math:`(\mathbf G + \mathbf G^T)/2` after the integration.

    Parameters
    ----------
    parent : object or array-like
       An object with associated ``parent.cell`` and ``parent.rcell`` or
       an array of floats which may be turned into a `Lattice`
    axes :
       the lattice directions that are integrated. The k-point component
       along the other directions is 0.
    max_k :
       a scalar is the maximum number of integrand evaluations for one integration.
       A sequence (one value for each of the `axes`) is the maximum k-point density
       along each axis, in k-points per reciprocal lattice vector (as the
       ``nkpt`` argument of `MonkhorstPack`). The total number is then not limited.
    trs :
       whether time-reversal symmetry is used. If ``None``, it is true for
       (un)-polarized calculations, and false for non-colinear and higher spin
       configurations (as for `MonkhorstPack`).
    atol :
       absolute tolerance of the integral (largest absolute element error)
    rtol :
       relative tolerance of the integral, relative to the largest absolute element

    See Also
    --------
    GaussKronrodGrid : adaptive Gauss-Kronrod cubature
    """

    def __init__(
        self,
        parent,
        axes: CellAxes = (0, 1, 2),
        max_k: SeqOrScalarInt = 10000,
        trs: Optional[bool] = None,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ):
        self.set_parent(parent)
        self._bz_attr = ("", None)
        iaxes = [direction(axis) for axis in listify(axes)]
        if len(iaxes) == 0 or len(set(iaxes)) != len(iaxes):
            raise ValueError(
                f"{self.__class__.__name__} requires unique axes, got {iaxes}"
            )
        self._axes = _a.arrayi(sorted(iaxes))
        if atol < 0 or rtol < 0:
            raise ValueError(f"{self.__class__.__name__} requires atol, rtol >= 0")
        self._atol = atol
        self._rtol = rtol

        nk = _a.arrayi(max_k)
        if np.any(nk < 1):
            raise ValueError(f"{self.__class__.__name__} requires max_k >= 1")
        if nk.ndim == 0:
            self._max_k = int(nk)
            self._max_k_axes = None
        elif nk.shape == self._axes.shape:
            self._max_k = None
            self._max_k_axes = nk
        else:
            raise ValueError(
                f"{self.__class__.__name__} requires max_k to be a scalar or have "
                f"one value per axes ({len(self._axes)}), got {nk.tolist()}"
            )

        if trs is None:
            trs = True
            try:
                trs = parent.spin <= Spin("polarized")
            except AttributeError:
                pass
        self._trs = int(self._axes[0]) if trs else -1

    def _undefined(self, what: str):
        return TypeError(
            f"{self.__class__.__name__} has no {what} before an integration; "
            "use integrate(..., ret_bz=True) to get the realized k-points."
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

    @property
    def axes(self) -> np.ndarray:
        """Lattice directions that are integrated"""
        return self._axes

    @property
    def trs(self) -> bool:
        """Whether time-reversal symmetry is used"""
        return self._trs >= 0

    def _domain(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower bound and width of the integrated domain along `axes`"""
        lo = np.full(len(self._axes), -0.5)
        width = np.ones(len(self._axes))
        if self._trs >= 0:
            i = np.flatnonzero(self._axes == self._trs)[0]
            lo[i] = 0.0
            width[i] = 0.5
        return lo, width

    def _max_k_str(self) -> str:
        if self._max_k_axes is None:
            return str(self._max_k)
        return str(self._max_k_axes.tolist())

    def _tolerance(self, value: npt.ArrayLike) -> float:
        """Requested absolute tolerance for the integral `value`"""
        return max(self._atol, self._rtol * float(np.max(np.abs(value))))

    def _k_full(self, k: np.ndarray) -> np.ndarray:
        """Convert k-points along `axes` to 3D reduced k-points"""
        kfull = _a.zerosd([len(k), 3])
        kfull[:, self._axes] = k
        return kfull

    def _not_converged(self, err: float, tol: float, nk: int) -> None:
        warn(
            f"{self.__class__.__name__}.integrate did not converge: "
            f"error estimate {err:.3e} > tolerance {tol:.3e} using {nk} k-points "
            f"(max_k={self._max_k_str()})"
        )

    @abstractmethod
    def integrate(
        self,
        func: Callable[[np.ndarray], npt.ArrayLike],
        ret_bz: bool = False,
    ) -> Union[np.ndarray, tuple[np.ndarray, BrillouinZone]]:
        """Integrate `func` over the Brillouin zone

        Parameters
        ----------
        func :
           the integrand, called as ``func(k)`` with ``k`` a reduced k-point with
           shape ``(3,)``. It must return a scalar or an array with a fixed shape.
        ret_bz :
           also return a `BrillouinZone` with the k-points and weights used.

        Returns
        -------
        value : numpy.ndarray
            the integral (equal to the Brillouin zone average)
        bz : BrillouinZone
            the realized k-points and weights, only if `ret_bz` is true
        """

    def __str__(self) -> str:
        parent = str(self._parent_lattice()).replace("\n", "\n ")
        return (
            f"{self.__class__.__name__}{{axes: {self._axes.tolist()}, "
            f"max_k: {self._max_k_str()}, trs: {self.trs}, "
            f"atol: {self._atol}, rtol: {self._rtol},\n "
            f"{parent}\n}}"
        )

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

    The sampled Brillouin zone is split in boxes. Each box is integrated with
    a tensor product of the 15-point Kronrod rule (7-point Gauss embedded).
    A box uses :math:`15^d` k-points for :math:`d` sampled `axes`.
    The error of a box is the largest absolute element of the difference between
    the Kronrod and the Gauss results. The box with the largest error is
    halved along the direction with the largest error, until the summed error
    is below the tolerance.

    This is efficient for integrands with few sharp features (e.g. small
    broadening), mainly in 1 or 2 dimensions.

    Parameters
    ----------
    parent : object or array-like
       An object with associated ``parent.cell`` and ``parent.rcell`` or
       an array of floats which may be turned into a `Lattice`
    axes :
       the lattice directions that are integrated
    n :
       initial number of boxes along each of the sampled `axes`.
       The initial boxes are always integrated, also when they exceed `max_k`.
    max_k :
       a scalar is the maximum number of integrand evaluations.
       A sequence (one value for each of the `axes`) is the maximum k-point density
       along each axis: a box of width :math:`h` (in reduced units) is not halved along
       an axis when :math:`15/(h/2)` is larger than the value of that axis.
       The box is then halved along the axis with the next largest error.
    trs :
       whether time-reversal symmetry is used, see `AdaptiveBrillouinZone`.
    atol :
       absolute tolerance of the integral
    rtol :
       relative tolerance of the integral

    Examples
    --------
    >>> H = si.Hamiltonian(...)
    >>> gk = GaussKronrodGrid(H, axes=0, atol=1e-8)
    >>> def G(k):
    ...     return np.linalg.inv((E + 1j * eta) * H.Sk(k, format="array") - H.Hk(k, format="array"))
    >>> Gavg, bz = gk.integrate(G, ret_bz=True)
    >>> len(bz)  # the number of k-points used

    Limit the k-point density to that of a 1000 x 200 grid:

    >>> gk = GaussKronrodGrid(H, axes=[0, 1], max_k=[1000, 200])
    """

    def __init__(
        self,
        parent,
        axes: CellAxes = (0, 1, 2),
        n: SeqOrScalarInt = 1,
        max_k: SeqOrScalarInt = 10000,
        trs: Optional[bool] = None,
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ):
        super().__init__(parent, axes=axes, max_k=max_k, trs=trs, atol=atol, rtol=rtol)
        nbox = _a.arrayi(n).ravel()
        if nbox.size == 1:
            nbox = np.repeat(nbox, len(self._axes))
        if nbox.size != len(self._axes) or np.any(nbox < 1):
            raise ValueError(
                f"{self.__class__.__name__} requires n >= 1 for each of the axes"
            )
        self._n = nbox

    def __str__(self) -> str:
        s = super().__str__()
        return s.replace(", max_k:", f", n: {self._n.tolist()}, max_k:", 1)

    def _box(self, func, lo: np.ndarray, h: np.ndarray, scale: float):
        """Integrate one box

        Returns k, weights, Kronrod value, error and the axes sorted by decreasing error.
        The weights are multiplied by `scale`.
        """
        d = len(lo)
        nodes = [lo[a] + h[a] * _GK_X for a in range(d)]
        kd = np.stack([g.ravel() for g in np.meshgrid(*nodes, indexing="ij")], axis=1)
        k = self._k_full(kd)
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

        w = wk[0]
        for wa in wk[1:]:
            w = np.multiply.outer(w, wa)
        return k, w.ravel(), K, err, order

    def _can_split(self, h: np.ndarray, a: int) -> bool:
        """Whether halving `h` along axis `a` keeps the density below `max_k`"""
        if self._max_k_axes is None:
            return True
        return len(_GK_X) / (h[a] / 2) <= self._max_k_axes[a] * (1 + 1e-12)

    def integrate(self, func, ret_bz: bool = False):
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
            k, w, K, err, order = self._box(func, lo, h, scale)
            heapq.heappush(boxes, (-err, next(counter), lo, h, k, w, K, order))

        for idx in itertools.product(*[range(n) for n in self._n]):
            push(lo0 + _a.arrayd(idx) * h0, h0.copy())
        nk = len(boxes) * nper

        def total():
            all_boxes = boxes + frozen
            return sum(b[6] for b in all_boxes), sum(-b[0] for b in all_boxes)

        value, err = total()
        tol = self._tolerance(value)
        while err > tol:
            if len(boxes) == 0 or (
                self._max_k is not None and nk + 2 * nper > self._max_k
            ):
                self._not_converged(err, tol, nk)
                break
            box = heapq.heappop(boxes)
            _, _, lo, h, *_, order = box
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

        value = np.asarray(value)[()]
        if ret_bz:
            k = np.concatenate([b[4] for b in boxes + frozen])
            w = np.concatenate([b[5] for b in boxes + frozen])
            return value, BrillouinZone(self.parent, k, w)
        return value
