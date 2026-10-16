# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import math
import pickle
import tracemalloc

import numpy as np
import pytest

from sisl import (
    AdaptiveBrillouinZone,
    Atom,
    BrillouinZone,
    GaussKronrodGrid,
    Hamiltonian,
    Lattice,
    MonkhorstPack,
    SislError,
    SislWarning,
    Spin,
    geom,
    oplist,
)

pytestmark = [pytest.mark.physics, pytest.mark.brillouinzone, pytest.mark.bz]


def peak(a):
    r""":math:`1/(a - \cos 2\pi k)`, its BZ average is :math:`1/\sqrt{a^2-1}`"""

    def func(k):
        return 1 / (a - np.cos(2 * np.pi * k[0]))

    return func, 1 / np.sqrt(a**2 - 1)


@pytest.mark.parametrize("cls", [GaussKronrodGrid])
def test_adaptive_undefined_grid(cls):
    H = Hamiltonian(geom.sc(1.0, Atom(1, R=1.1)))
    bz = cls(H)
    assert isinstance(bz, BrillouinZone)
    assert isinstance(bz, AdaptiveBrillouinZone)
    assert bz
    with pytest.raises(TypeError):
        len(bz)
    with pytest.raises(TypeError):
        bz.k
    with pytest.raises(TypeError):
        bz.weight
    with pytest.raises(TypeError):
        bz.apply.array.eigh()
    str(bz)


_FIXED_K_MODES = sorted(set(BrillouinZone.apply._dispatchs) - {"average"})


@pytest.mark.parametrize("cls", [GaussKronrodGrid])
@pytest.mark.parametrize("mode", _FIXED_K_MODES)
def test_adaptive_apply_fixed_k_modes(cls, mode):
    # all apply modes except average need fixed k-points
    H = Hamiltonian(geom.sc(1.0, Atom(1, R=1.1)))
    bz = cls(H)
    with pytest.raises(TypeError, match="last_bz"):
        bz.apply[mode].eigh()
    with pytest.raises(TypeError, match="last_bz"):
        bz.apply[mode](lambda k: k)
    # the default dispatch (iter) also raises
    with pytest.raises(TypeError, match="last_bz"):
        bz.apply.eigh()


@pytest.mark.parametrize("cls", [GaussKronrodGrid])
def test_adaptive_apply_average(cls):
    H = Hamiltonian(geom.graphene())
    H.construct([(0.1, 1.44), (0, -2.7)])
    bz = cls(H, [15, 15, 0], atol=1e-8, rtol=0, trs=False, max_k=50000)

    D = bz.apply.average.Pk(format="array")
    nk = len(bz.last_bz())
    assert D == pytest.approx(bz.integrate(lambda k: H.Pk(k, format="array")))
    assert len(bz.last_bz()) == nk
    mp = MonkhorstPack(H, [30, 30, 1], trs=False)
    assert np.allclose(D, mp.apply.average.Pk(format="array"), atol=1e-12, rtol=0)

    # wrap is applied before the integration, and gets weight=None
    def wrap(v, parent, k, weight):
        assert parent is H
        assert weight is None
        return [v, 2 * v]

    v = bz.apply.average.Pk(format="array", wrap=wrap)
    assert isinstance(v, oplist)
    assert np.allclose(v[0], D) and np.allclose(v[1], 2 * D)

    # a function instead of a parent method
    v = bz.apply.average(lambda k: H.Pk(k, format="array"))()
    assert np.allclose(v, D)

    with pytest.warns(SislWarning, match="pool"):
        v = bz.apply.renew(pool=2).average.Pk(format="array")
    assert np.allclose(v, D)


@pytest.mark.parametrize("cls", [GaussKronrodGrid])
def test_adaptive_last_bz(cls):
    bz = cls(Lattice(1))
    with pytest.raises(SislError):
        bz.last_bz()

    def check(f):
        v = bz.integrate(f)
        last = bz.last_bz()
        assert type(last) is BrillouinZone
        assert last.weight.sum() == pytest.approx(1)
        assert sum(w * f(k) for k, w in last.iter(True)) == pytest.approx(v, abs=1e-14)
        return last

    last = check(lambda k: k[0] ** 2 + k[1] ** 4)
    assert len(pickle.loads(pickle.dumps(bz)).last_bz()) == len(last)
    # a new integration replaces the k-points
    check(lambda k: np.array([k[2] ** 2, 1.0]))


def test_gk_last_bz_used_k():
    used = []

    def f(k):
        used.append(k.copy())
        return 1 / (1.05 - np.cos(2 * np.pi * k[0]))

    gk = GaussKronrodGrid(Lattice(1), [15, 0, 0], atol=1e-9, rtol=0, trs=False)
    gk.integrate(f)
    last = gk.last_bz()
    # the final boxes, the halved boxes were also evaluated
    assert len(last) < len(used)
    assert set(map(tuple, last.k)) <= set(map(tuple, used))


def test_adaptive_abstract():
    with pytest.raises(TypeError):
        AdaptiveBrillouinZone(Lattice(1))


@pytest.mark.parametrize("cls", [GaussKronrodGrid])
def test_adaptive_pickle(cls):
    bz = cls(Lattice(1))
    bz2 = pickle.loads(pickle.dumps(bz))
    assert str(bz2) == str(bz)
    assert np.allclose(bz2.rcell, bz.rcell)


def test_gk_nkpt():
    gk = GaussKronrodGrid(Lattice(1), trs=False)
    assert np.all(gk._axes == [0, 1, 2]) and np.all(gk._n == 1)
    # rounded up to a multiple of 15
    gk = GaussKronrodGrid(Lattice(1), [0, 16, 15], trs=False)
    assert np.all(gk._axes == [1, 2]) and np.all(gk._n == [2, 1])
    assert "nkpt: [0, 16, 15]" in str(gk)
    gk = GaussKronrodGrid(Lattice(1), [0, 1, 45], trs=False)
    assert np.all(gk._n == [1, 3])
    # TRS integrates half of the first direction with half the k-points
    gk = GaussKronrodGrid(Lattice(1), [0, 30, 30], trs=True)
    assert np.all(gk._n == [1, 2])
    gk = GaussKronrodGrid(Lattice(1), [0, 31, 30], trs=True)
    assert np.all(gk._n == [2, 2])
    gk.integrate(lambda k: 1.0)
    assert len(gk.last_bz()) == (2 * 15) * (2 * 15)
    for nkpt in (0, [0, 0, 0], [1, 1], [1, -1, 0]):
        with pytest.raises(ValueError):
            GaussKronrodGrid(Lattice(1), nkpt)


def test_gk_nkpt_periodic():
    # a scalar nkpt only integrates the periodic directions
    H = Hamiltonian(geom.graphene())
    assert GaussKronrodGrid(H)._axes.tolist() == [0, 1]
    assert GaussKronrodGrid(H, 30, trs=False)._n.tolist() == [2, 2]
    lattice = Lattice(1)
    lattice.pbc = [False, True, False]
    assert GaussKronrodGrid(lattice)._axes.tolist() == [1]
    # an explicit nkpt is used as given
    assert GaussKronrodGrid(lattice, [15, 15, 0])._axes.tolist() == [0, 1]
    lattice.pbc = [False] * 3
    with pytest.raises(ValueError, match="no periodic"):
        GaussKronrodGrid(lattice)
    GaussKronrodGrid(lattice, [15, 0, 0])


@pytest.mark.parametrize("gauss", [3, 7, 10])
def test_gk_rule_constants(gauss):
    from sisl.physics._brillouinzone_adaptive import _GK_RULES

    x, wk, wg = _GK_RULES[gauss]
    n = 2 * gauss + 1
    assert len(x) == n and np.all(np.diff(x) > 0) and 0 < x[0] and x[-1] < 1
    # the Gauss nodes are the Gauss-Legendre nodes
    gx, gw = np.polynomial.legendre.leggauss(gauss)
    assert np.allclose(x[wg != 0], (gx + 1) / 2, rtol=0, atol=1e-15)
    assert np.allclose(wg[wg != 0], gw / 2, rtol=0, atol=1e-15)
    # K is exact up to degree 3g+1 (g even) or 3g+2 (g odd), G up to 2g-1
    deg_k = 3 * gauss + 1 + gauss % 2
    deg_g = 2 * gauss - 1
    t = 2 * x - 1

    def err(w, p):
        return abs(w @ t**p - (1 / (p + 1) if p % 2 == 0 else 0))

    assert max(err(wk, p) for p in range(deg_k + 1)) < 1e-15
    assert max(err(wg, p) for p in range(deg_g + 1)) < 1e-15
    # and not exact at the next (even) degree
    assert err(wk, deg_k + 1) > 1e-13
    assert err(wg, deg_g + 1) > 1e-13


def test_gk_gauss():
    for gauss in (3, 7, 10):
        n = 2 * gauss + 1
        # one box along each periodic direction by default
        gk = GaussKronrodGrid(Lattice(1), trs=False, gauss=gauss)
        assert np.all(gk._n == 1)
        assert f"gauss: {gauss}," in str(gk)
        assert f"nkpt: {[n] * 3}" in str(gk)
        # rounded up to a multiple of n
        gk = GaussKronrodGrid(Lattice(1), [n + 1, 0, 0], trs=False, gauss=gauss)
        assert gk._n.tolist() == [2]
        gk.integrate(lambda k: 1.0)
        assert len(gk.last_bz()) == 2 * n
        gk2 = pickle.loads(pickle.dumps(gk))
        assert str(gk2) == str(gk)
    gk = GaussKronrodGrid(Lattice(1), [1, 1, 1], trs=False, gauss=3)
    gk.integrate(lambda k: 1.0)
    assert len(gk.last_bz()) == 7**3
    for gauss in (1, 5, 15, "7"):
        with pytest.raises(ValueError, match="gauss"):
            GaussKronrodGrid(Lattice(1), gauss=gauss)


def test_gk_gauss_order():
    # a smooth integrand needs fewer evaluations with a higher order
    func, ref = peak(1.5)
    nk = {}
    for gauss in (3, 7, 10):
        gk = GaussKronrodGrid(
            Lattice(1), [1, 0, 0], atol=1e-13, rtol=0, trs=False, gauss=gauss
        )
        assert gk.integrate(func) == pytest.approx(ref, abs=1e-13)
        nk[gauss] = len(gk.last_bz())
    assert nk[3] > nk[7] > nk[10]


def test_gk_polynomial_exact():
    # G7 is exact up to degree 13, so one box is enough
    gk = GaussKronrodGrid(Lattice(1), nkpt=[15, 0, 0], atol=1e-14, rtol=0)
    v = gk.integrate(lambda k: k[0] ** 12)
    bz = gk.last_bz()
    assert v == pytest.approx(0.5**12 / 13, abs=1e-15)
    assert len(bz) == 15
    assert bz.weight.sum() == pytest.approx(1)
    assert np.all(bz.k[:, 1:] == 0)


def test_gk_peak_local_refinement():
    func, ref = peak(1 + 1e-3)
    gk = GaussKronrodGrid(Lattice(1), nkpt=[15, 0, 0], atol=0, rtol=1e-10, max_k=5000)
    v = gk.integrate(func)
    bz = gk.last_bz()
    assert v == pytest.approx(ref, rel=1e-9)
    assert bz.weight.sum() == pytest.approx(1)
    # points concentrate around the peak at k=0
    assert np.sum(np.abs(bz.k[:, 0]) < 0.05) > len(bz) / 2


def test_gk_oplist():
    func, ref = peak(1.2)
    gk = GaussKronrodGrid(Lattice(1), [15, 0, 0], atol=1e-12, rtol=0)
    M = np.arange(6).reshape(2, 3)

    for ret in (list, tuple, oplist):
        v = gk.integrate(lambda k: ret([func(k), M * func(k)]))
        assert isinstance(v, oplist)
        assert len(v) == 2
        assert v[0] == pytest.approx(ref, abs=1e-10)
        assert v[1].shape == (2, 3)
        assert np.allclose(v[1], M * ref, atol=1e-10, rtol=0)

    # nested oplist, and arithmetic on the result
    v = gk.integrate(lambda k: oplist([func(k), oplist([2 * func(k), M])]))
    assert isinstance(v[1], oplist)
    assert v[1][0] == pytest.approx(2 * ref, abs=1e-10)
    assert np.allclose(v[1][1], M)
    assert (v * 2)[0] == pytest.approx(2 * ref, abs=1e-10)

    with pytest.raises(ValueError):
        gk.integrate(lambda k: [1.0] * (1 + int(k[0] > 0.25)))


def test_gk_oplist_convergence_all_elements():
    sharp, _ = peak(1 + 1e-3)

    def smooth(k):
        return k[0] ** 2

    def nk(func):
        gk = GaussKronrodGrid(Lattice(1), [15, 0, 0], atol=1e-9, rtol=0, trs=False)
        gk.integrate(func)
        return len(gk.last_bz())

    n_sharp = nk(sharp)
    assert nk(smooth) == 15 < n_sharp
    # the sharp element decides the refinement, in any position
    assert nk(lambda k: [smooth(k), sharp(k)]) == n_sharp
    assert nk(lambda k: (sharp(k), np.full(3, smooth(k)))) == n_sharp


def test_gk_array_and_complex():
    func, ref = peak(1.2)
    gk = GaussKronrodGrid(Lattice(1), nkpt=[15, 0, 0], atol=1e-12, rtol=0)
    v = gk.integrate(lambda k: np.array([[func(k), 2j * func(k)]]))
    assert v.shape == (1, 2)
    assert np.allclose(v, [[ref, 2j * ref]], atol=1e-10, rtol=0)


def test_gk_mixed_dtype():
    # real values first, then complex values
    gk = GaussKronrodGrid(Lattice(1), [15, 0, 0], atol=1e-10, rtol=0, trs=False)
    v = gk.integrate(lambda k: np.array([1.0 if k[0] < 0 else 1j]))
    assert np.allclose(v, [0.5 + 0.5j], atol=1e-14, rtol=0)

    # float32 is summed in float64, longdouble without BLAS
    gk32 = GaussKronrodGrid(Lattice(1), [15, 0, 0], atol=1e-6, rtol=0, trs=False)
    v = gk32.integrate(lambda k: np.float32(np.cos(2 * np.pi * k[0]) ** 2))
    assert v.dtype == np.float64
    assert v == pytest.approx(0.5, abs=1e-7)
    v = gk.integrate(lambda k: np.longdouble(np.cos(2 * np.pi * k[0]) ** 2))
    assert v.dtype == np.longdouble
    assert v == pytest.approx(0.5, abs=1e-14)


def test_gk_shape_mismatch():
    gk = GaussKronrodGrid(Lattice(1), [15, 0, 0], trs=False)
    with pytest.raises(ValueError):
        gk.integrate(lambda k: np.ones(2 + int(k[0] > 0.2)))
    with pytest.raises(ValueError):
        gk.integrate(lambda k: np.ones(2) if k[0] > 0.2 else 1.0)


def test_gk_reused_output_buffer():
    # the values are used at once, so func can return the same array
    buf = np.zeros(3)

    def func(k):
        buf[:] = [np.cos(2 * np.pi * k[0]) ** 2, k[0] ** 2, 1.0]
        return buf

    gk = GaussKronrodGrid(Lattice(1), [15, 0, 0], atol=1e-12, rtol=0, trs=False)
    assert np.allclose(gk.integrate(func), [0.5, 1 / 12, 1], atol=1e-12, rtol=0)


@pytest.mark.parametrize("gauss", [7, 10])
def test_gk_compensated_sums(gauss):
    # the coarse boxes overestimate the peak by orders of magnitude,
    # the running sums must still equal the sum over the final k-points
    def func(k):
        return 1 / (np.sin(np.pi * k[0]) ** 2 + np.sin(np.pi * k[1]) ** 2 / 3 + 1e-6)

    gk = GaussKronrodGrid(
        Lattice(1), [1, 1, 0], atol=1e-14, rtol=0, max_k=3000, trs=False, gauss=gauss
    )
    with pytest.warns(SislWarning):
        v = gk.integrate(func)
    bz = gk.last_bz()
    ref = math.fsum(w * func(k) for k, w in zip(bz.k, bz.weight))
    assert v == pytest.approx(ref, rel=4 * np.finfo(np.float64).eps, abs=0)


@pytest.mark.parametrize("nkpt", [[15, 0, 0], [15, 15, 0], [15, 15, 15]])
def test_gk_memory(nkpt):
    # the integrand values of a box are not stored (np.stack needed 2 * 15**d)
    n = 10000
    base = np.linspace(0, 1, n)

    def func(k):
        return base * (2 + np.cos(2 * np.pi * k[0])) + 0j

    gk = GaussKronrodGrid(Lattice(1), nkpt, max_k=15 ** np.count_nonzero(nkpt))
    tracemalloc.start()
    try:
        gk.integrate(func)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 16 * base.nbytes * 2


def test_gk_2d_split_direction():
    # the integrand does not depend on k_b, so boxes only split along a
    func, ref = peak(1.05)
    gk = GaussKronrodGrid(Lattice(1), nkpt=[15, 15, 0], atol=1e-9, rtol=0, max_k=50000)
    v = gk.integrate(func)
    bz = gk.last_bz()
    assert v == pytest.approx(ref, abs=1e-9)
    assert len(np.unique(bz.k[:, 1])) == 15
    assert len(np.unique(bz.k[:, 0])) > 15


def test_gk_2d_separable():
    f, ref = peak(1.1)
    gk = GaussKronrodGrid(Lattice(1), nkpt=[30, 0, 15], atol=1e-9, rtol=0, max_k=50000)
    v = gk.integrate(lambda k: f(k) * f(k[::-1]))
    assert v == pytest.approx(ref**2, abs=1e-8)


def test_gk_max_k_warns():
    func, _ = peak(1 + 1e-6)
    gk = GaussKronrodGrid(Lattice(1), nkpt=[15, 0, 0], atol=1e-12, rtol=0, max_k=100)
    with pytest.warns(SislWarning, match="did not converge"):
        gk.integrate(func)


def test_gk_trs_default():
    cls = GaussKronrodGrid
    g = geom.sc(1.0, Atom(1, R=1.1))
    assert cls(Lattice(1)).trs
    assert cls(Hamiltonian(g)).trs
    assert cls(Hamiltonian(g, spin=Spin("polarized"))).trs
    assert not cls(Hamiltonian(g, spin=Spin("non-colinear"))).trs
    assert not cls(Lattice(1), trs=False).trs
    assert "trs: True" in str(cls(Lattice(1)))


def test_gk_max_k_input():
    n = [15, 15, 0]
    assert GaussKronrodGrid(Lattice(1), n, max_k=100)._max_k == 100
    gk = GaussKronrodGrid(Lattice(1), n, max_k=[100, 20, 0])
    assert gk._max_k is None
    assert np.all(gk._max_k_axes == [100, 20])
    with pytest.raises(ValueError):
        GaussKronrodGrid(Lattice(1), n, max_k=[100, 20])
    with pytest.raises(ValueError):
        GaussKronrodGrid(Lattice(1), n, max_k=[100, 0, 1])
    with pytest.raises(ValueError):
        GaussKronrodGrid(Lattice(1), n, max_k=0)


def test_gk_trs_half_domain():
    # a peak away from Gamma, so f(k) != f(-k)
    f, ref = peak(1.05)

    def g(k):
        return f(k - 0.1)

    gk = GaussKronrodGrid(Lattice(1), nkpt=[15, 0, 0], atol=1e-10, rtol=0, trs=True)
    v = gk.integrate(lambda k: (g(k) + g(-k)) / 2)
    bz = gk.last_bz()
    assert v == pytest.approx(ref, abs=1e-10)
    assert np.all(bz.k[:, 0] >= 0) and np.all(bz.k[:, 0] <= 0.5)
    assert bz.weight.sum() == pytest.approx(1)

    # an odd integrand: TRS integrates only [0, 1/2] and doubles it
    def odd(k):
        return np.sin(2 * np.pi * k[0])

    assert gk.integrate(odd) == pytest.approx(2 / np.pi, abs=1e-10)
    gk = GaussKronrodGrid(Lattice(1), nkpt=[15, 0, 0], atol=1e-10, rtol=0, trs=False)
    assert gk.integrate(odd) == pytest.approx(0, abs=1e-10)
    assert gk.integrate(g) == pytest.approx(ref, abs=1e-10)


@pytest.mark.parametrize("trs", [True, False])
def test_gk_trs_2d(trs):
    f, ref = peak(1.1)
    gk = GaussKronrodGrid(
        Lattice(1), nkpt=[0, 15, 15], atol=1e-9, rtol=0, max_k=50000, trs=trs
    )
    v = gk.integrate(lambda k: f(k[[1, 0, 2]]) * f(k[[2, 0, 1]]))
    bz = gk.last_bz()
    assert v == pytest.approx(ref**2, abs=1e-8)
    assert bz.weight.sum() == pytest.approx(1)
    assert np.all(bz.k[:, 1] >= 0) == trs


def test_gk_max_k_axes_density():
    func, _ = peak(1 + 1e-6)
    gk = GaussKronrodGrid(
        Lattice(1), nkpt=[15, 0, 0], atol=1e-12, rtol=0, max_k=[150, 1, 1], trs=False
    )
    with pytest.warns(SislWarning, match="did not converge"):
        gk.integrate(func)
        bz = gk.last_bz()
    # boxes are at least 15 / 150 wide, so at most 10 boxes
    assert len(bz) <= 150
    assert bz.weight.sum() == pytest.approx(1)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
@pytest.mark.parametrize("nkpt", [[15, 0, 0], [15, 15, 0]])
def test_gk_not_finite(bad, nkpt):
    gk = GaussKronrodGrid(Lattice(1), nkpt=nkpt, atol=1e-10, trs=False)

    def func(k):
        return np.array([bad if abs(k[0] - 0.2) < 0.05 else 1.0, 1.0])

    with pytest.warns(SislWarning, match="not finite"):
        v = gk.integrate(func)
    assert not np.isfinite(v[0])
    # no splits, only the initial boxes
    assert len(gk.last_bz()) == 15 ** len(gk._axes)


def test_gk_max_k_axes_next_direction():
    # the integrand only varies along a; when a is blocked, b is halved instead
    # until b is also blocked, then the integration stops with a warning
    func, _ = peak(1 + 1e-6)
    gk = GaussKronrodGrid(
        Lattice(1), nkpt=[15, 15, 0], atol=1e-12, rtol=0, max_k=[60, 120, 1], trs=False
    )
    with pytest.warns(SislWarning, match="did not converge"):
        gk.integrate(func)
        bz = gk.last_bz()
    ka = np.unique(bz.k[:, 0])
    kb = np.unique(bz.k[:, 1])
    assert len(ka) <= 60
    assert len(kb) <= 120
    assert len(kb) > 15
    assert bz.weight.sum() == pytest.approx(1)


@pytest.mark.filterwarnings("error::sisl.SislWarning")
@pytest.mark.parametrize("cls, atol", [(GaussKronrodGrid, 1e-8)])
def test_adaptive_green_chain(cls, atol):
    # local Green function of a 1D chain against a dense periodic sum
    H = Hamiltonian(geom.sc(1.0, Atom(1, R=1.1)))
    H.set_nsc([3, 1, 1])
    H.construct(([0.1, 1.1], [0.0, -1.0]))
    z = 0.5 + 0.05j

    def G(k):
        return np.linalg.inv(z * np.eye(1) - H.Hk(k, format="array"))

    k = np.linspace(-0.5, 0.5, 20000, endpoint=False)
    ref = np.mean(1 / (z + 2 * np.cos(2 * np.pi * k)))
    bz = cls(H, nkpt=[15, 0, 0], atol=atol, rtol=0, max_k=20000)
    assert bz.integrate(G)[0, 0] == pytest.approx(ref, abs=atol)
