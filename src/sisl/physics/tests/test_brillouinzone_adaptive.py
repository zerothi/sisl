# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pickle

import numpy as np
import pytest

from sisl import (
    AdaptiveBrillouinZone,
    Atom,
    BrillouinZone,
    GaussKronrodGrid,
    Hamiltonian,
    Lattice,
    SislError,
    SislWarning,
    Spin,
    geom,
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
        bz.apply.average.eigh()
    str(bz)


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


def test_gk_array_and_complex():
    func, ref = peak(1.2)
    gk = GaussKronrodGrid(Lattice(1), nkpt=[15, 0, 0], atol=1e-12, rtol=0)
    v = gk.integrate(lambda k: np.array([[func(k), 2j * func(k)]]))
    assert v.shape == (1, 2)
    assert np.allclose(v, [[ref, 2j * ref]], atol=1e-10, rtol=0)


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
