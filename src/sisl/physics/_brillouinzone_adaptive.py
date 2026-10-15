# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import heapq
import itertools
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from typing import Union

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull, Voronoi

import sisl._array as _a
from sisl._internal import set_module
from sisl.messages import SislError, warn
from sisl.typing import CellAxes, SeqOrScalarInt
from sisl.utils.misc import direction, listify

from .brillouinzone import BrillouinZone

__all__ = ["AdaptiveBrillouinZone", "GaussKronrod", "Farey"]


@set_module("sisl.physics")
class AdaptiveBrillouinZone(BrillouinZone, metaclass=ABCMeta):
    r"""Base class for Brillouin zone integrations with adaptive k-point grids

    An adaptive grid selects its k-points from the values of the integrand.
    Before an integration, there are no k-points. Therefore `k`, `weight` and
    ``len(bz)`` raise a `TypeError`. This also makes ``bz.apply`` fail, use
    `integrate` instead.

    The integration domain is the reduced Brillouin zone :math:`]-1/2 ; 1/2]` along
    the sampled `axes`. The domain has unit volume, so the integral is equal to the
    Brillouin zone average, as for `MonkhorstPack`.

    Parameters
    ----------
    parent : object or array-like
       An object with associated ``parent.cell`` and ``parent.rcell`` or
       an array of floats which may be turned into a `Lattice`
    axes :
       the lattice directions that are integrated. The k-point component
       along the other directions is 0.
    atol :
       absolute tolerance of the integral (largest absolute element error)
    rtol :
       relative tolerance of the integral, relative to the largest absolute element
    max_k :
       maximum number of integrand evaluations for one integration

    See Also
    --------
    GaussKronrod : adaptive Gauss-Kronrod cubature
    Farey : nested Farey grids
    """

    def __init__(
        self,
        parent,
        axes: CellAxes = (0, 1, 2),
        atol: float = 1e-6,
        rtol: float = 1e-6,
        max_k: int = 10000,
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
        self._max_k = max_k

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
            f"(max_k={self._max_k})"
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
            f"atol: {self._atol}, rtol: {self._rtol}, max_k: {self._max_k},\n "
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
class GaussKronrod(AdaptiveBrillouinZone):
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
       initial number of boxes along each of the sampled `axes`
    atol :
       absolute tolerance of the integral
    rtol :
       relative tolerance of the integral
    max_k :
       maximum number of integrand evaluations

    Examples
    --------
    >>> H = si.Hamiltonian(...)
    >>> gk = GaussKronrod(H, axes=0, atol=1e-8)
    >>> def G(k):
    ...     return np.linalg.inv((E + 1j * eta) * H.Sk(k, format="array") - H.Hk(k, format="array"))
    >>> Gavg, bz = gk.integrate(G, ret_bz=True)
    >>> len(bz)  # the number of k-points used
    """

    def __init__(
        self,
        parent,
        axes: CellAxes = (0, 1, 2),
        n: SeqOrScalarInt = 1,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        max_k: int = 10000,
    ):
        super().__init__(parent, axes=axes, atol=atol, rtol=rtol, max_k=max_k)
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
        return s.replace("{", f"{{n: {self._n.tolist()}, ", 1)

    def _box(self, func, lo: np.ndarray, h: np.ndarray):
        """Integrate one box, returns k, weights, Kronrod value, error and split direction"""
        d = len(lo)
        nodes = [lo[a] + h[a] * _GK_X for a in range(d)]
        kd = np.stack([g.ravel() for g in np.meshgrid(*nodes, indexing="ij")], axis=1)
        k = self._k_full(kd)
        F = np.stack([np.asarray(func(kk)) for kk in k])
        F = F.reshape((len(_GK_X),) * d + F.shape[1:])

        wk = [_GK_WK * h[a] for a in range(d)]
        wg = [_GK_WG * h[a] for a in range(d)]
        K = _contract(F, wk)
        err = _maxabs(K - _contract(F, wg))
        if d == 1:
            split = 0
        else:
            split = int(
                np.argmax(
                    [
                        _maxabs(K - _contract(F, wk[:a] + [wg[a]] + wk[a + 1 :]))
                        for a in range(d)
                    ]
                )
            )

        w = wk[0]
        for wa in wk[1:]:
            w = np.multiply.outer(w, wa)
        return k, w.ravel(), K, err, split

    def integrate(self, func, ret_bz: bool = False):
        d = len(self._axes)
        nper = len(_GK_X) ** d
        h0 = 1.0 / self._n

        boxes = []
        counter = itertools.count()

        def push(lo, h):
            k, w, K, err, split = self._box(func, lo, h)
            heapq.heappush(boxes, (-err, next(counter), lo, h, k, w, K, split))

        for idx in itertools.product(*[range(n) for n in self._n]):
            push(-0.5 + _a.arrayd(idx) * h0, h0.copy())
        nk = len(boxes) * nper

        def total():
            return sum(b[6] for b in boxes), sum(-b[0] for b in boxes)

        value, err = total()
        tol = self._tolerance(value)
        while err > tol:
            if nk + 2 * nper > self._max_k:
                self._not_converged(err, tol, nk)
                break
            _, _, lo, h, *_, split = heapq.heappop(boxes)
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
            k = np.concatenate([b[4] for b in boxes])
            w = np.concatenate([b[5] for b in boxes])
            return value, BrillouinZone(self.parent, k, w)
        return value


@lru_cache(maxsize=16)
def _farey_1d(order: int):
    r"""Farey points :math:`p/q`, :math:`q\le` `order`, :math:`-q/2 < p \le q/2`, gcd(p, q) = 1"""
    p, q = [], []
    for qq in range(1, order + 1):
        pp = np.arange(-((qq - 1) // 2), qq // 2 + 1)
        pp = pp[np.gcd(pp, qq) == 1]
        p.append(pp)
        q.append(np.full_like(pp, qq))
    return np.concatenate(p), np.concatenate(q)


def _farey_points(order: int, ndim: int):
    """Integer numerators and denominators of the `ndim` Farey grid of `order`

    Only points with lcm of the denominators ``<= order`` are kept.
    """
    p1, q1 = _farey_1d(order)
    p = np.zeros((1, 0), dtype=np.int64)
    q = np.zeros((1, 0), dtype=np.int64)
    lcm = np.ones(1, dtype=np.int64)
    for _ in range(ndim):
        L = np.lcm.outer(lcm, q1)
        i, j = np.nonzero(L <= order)
        p = np.column_stack([p[i], p1[j]])
        q = np.column_stack([q[i], q1[j]])
        lcm = L[i, j]
    return p, q


def _voronoi_weights(k: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """Periodic Voronoi weights of reduced points `k` (in ]-1/2; 1/2]) for the `metric`

    The returned weights sum to 1.
    """
    N, d = k.shape
    if d == 1:
        idx = np.argsort(k[:, 0])
        ks = k[idx, 0]
        nxt = np.roll(ks, -1)
        nxt[-1] += 1
        prv = np.roll(ks, 1)
        prv[0] -= 1
        w = np.empty(N)
        w[idx] = (nxt - prv) / 2
        return w

    L = np.linalg.cholesky(metric)
    total = abs(np.linalg.det(L))
    vol = np.empty(N)
    for nimg in range(1, 4):
        shift = range(-nimg, nimg + 1)
        offsets = np.array(list(itertools.product(shift, repeat=d)))
        offsets = offsets[np.argsort(np.abs(offsets).sum(1), kind="stable")]
        x = (k[None, :, :] + offsets[:, None, :]).reshape(-1, d) @ L
        vor = Voronoi(x)
        vol = np.empty(N)
        for i in range(N):
            region = vor.regions[vor.point_region[i]]
            if len(region) == 0 or -1 in region:
                vol[i] = np.inf
            else:
                vol[i] = ConvexHull(vor.vertices[region]).volume
        if np.isclose(vol.sum(), total, rtol=1e-8, atol=0):
            return vol / vol.sum()
    raise SislError(
        f"Farey Voronoi volumes ({vol.sum()}) do not sum to the "
        f"Brillouin zone volume ({total})"
    )


@set_module("sisl.physics")
class Farey(AdaptiveBrillouinZone):
    r"""Nested Farey grids of the Brillouin zone [Chen2022]_

    The Farey grid of order :math:`l` contains the reduced k-points with components
    :math:`p_i/q_i` (:math:`\gcd(p_i, q_i)=1`, :math:`-q_i/2<p_i\le q_i/2`) where the
    smallest commensurate supercell is not larger than :math:`l`, i.e.
    :math:`\mathrm{lcm}(q_1,\dots,q_d)\le l`.
    The grid of order :math:`l` contains all points of the lower orders.
    The integration increments the order, and re-uses all earlier integrand
    values. The error estimate of order :math:`l` is the difference to the
    integral of order :math:`\lfloor l/2\rfloor`. The integration stops when two
    consecutive orders have an error estimate below the tolerance.

    The Voronoi weights give a low-order rule. For integrands that are cheap at all
    k-points, `GaussKronrod` or `MonkhorstPack` often need fewer k-points.

    The weight of a k-point is the volume of its Voronoi cell in the
    (Cartesian) periodic Brillouin zone. Crystal symmetries are not used.

    Parameters
    ----------
    parent : object or array-like
       An object with associated ``parent.cell`` and ``parent.rcell`` or
       an array of floats which may be turned into a `Lattice`
    axes :
       the lattice directions that are integrated
    order :
       the first Farey order that is integrated (1 is the :math:`\Gamma`-point)
    atol :
       absolute tolerance of the integral
    rtol :
       relative tolerance of the integral
    max_k :
       maximum number of integrand evaluations

    References
    ----------
    .. [Chen2022] S. Chen, P. T. Salzbrenner, B. Monserrat, "Nonuniform grids for
       Brillouin zone integration and interpolation", Phys. Rev. B **106**, 155102 (2022)
    """

    def __init__(
        self,
        parent,
        axes: CellAxes = (0, 1, 2),
        order: int = 1,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        max_k: int = 10000,
    ):
        super().__init__(parent, axes=axes, atol=atol, rtol=rtol, max_k=max_k)
        if order < 1:
            raise ValueError(f"{self.__class__.__name__} requires order >= 1")
        self._order = int(order)

    def __str__(self) -> str:
        s = super().__str__()
        return s.replace("{", f"{{order: {self._order}, ", 1)

    def integrate(self, func, ret_bz: bool = False):
        d = len(self._axes)
        rcell = self.rcell[self._axes]
        metric = rcell @ rcell.T
        cache = {}

        def evaluate(order):
            p, q = _farey_points(order, d)
            keys = list(map(tuple, np.concatenate([p, q], axis=1)))
            k = self._k_full(p / q)
            for key, kk in zip(keys, k):
                if key not in cache:
                    cache[key] = np.asarray(func(kk))
            w = _voronoi_weights(p / q, metric)
            F = np.stack([cache[key] for key in keys])
            return k, w, np.tensordot(w, F, axes=(0, 0))

        values = {}
        order = self._order
        k, w, values[order] = evaluate(order)
        err = np.inf
        nconverged = 0
        while nconverged < 2:
            if len(_farey_points(order + 1, d)[0]) > self._max_k:
                self._not_converged(err, self._tolerance(values[order]), len(cache))
                break
            order += 1
            k, w, values[order] = evaluate(order)
            half = max(order // 2, 1)
            if half not in values:
                values[half] = evaluate(half)[2]
            err = _maxabs(values[order] - values[half])
            if err <= self._tolerance(values[order]):
                nconverged += 1
            else:
                nconverged = 0
        value = values[order]

        value = np.asarray(value)[()]
        if ret_bz:
            return value, BrillouinZone(self.parent, k, w)
        return value
