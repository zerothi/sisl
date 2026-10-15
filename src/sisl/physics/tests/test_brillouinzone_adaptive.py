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
    Farey,
    GaussKronrod,
    Hamiltonian,
    Lattice,
    SislWarning,
    geom,
)
from sisl.physics._brillouinzone_adaptive import _farey_points, _voronoi_weights

pytestmark = [pytest.mark.physics, pytest.mark.brillouinzone, pytest.mark.bz]


def peak(a):
    r""":math:`1/(a - \cos 2\pi k)`, its BZ average is :math:`1/\sqrt{a^2-1}`"""

    def func(k):
        return 1 / (a - np.cos(2 * np.pi * k[0]))

    return func, 1 / np.sqrt(a**2 - 1)


def totient(n):
    return sum(1 for i in range(1, n + 1) if np.gcd(i, n) == 1)


@pytest.mark.parametrize("cls", [GaussKronrod, Farey])
def test_adaptive_undefined_grid(cls):
    H = Hamiltonian(geom.sc(1.0, Atom(1, R=1.1)))
    bz = cls(H, axes=0)
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


def test_adaptive_abstract():
    with pytest.raises(TypeError):
        AdaptiveBrillouinZone(Lattice(1))


@pytest.mark.parametrize("cls", [GaussKronrod, Farey])
def test_adaptive_axes(cls):
    assert np.all(cls(Lattice(1), axes="b").axes == [1])
    assert np.all(cls(Lattice(1), axes=[2, 0]).axes == [0, 2])
    with pytest.raises(ValueError):
        cls(Lattice(1), axes=[0, 0])


@pytest.mark.parametrize("cls", [GaussKronrod, Farey])
def test_adaptive_pickle(cls):
    bz = cls(Lattice(1), axes=[0, 1], atol=1e-3)
    bz2 = pickle.loads(pickle.dumps(bz))
    assert np.all(bz2.axes == bz.axes)
    assert bz2._atol == bz._atol
    assert np.allclose(bz2.rcell, bz.rcell)


def test_gk_polynomial_exact():
    # G7 is exact up to degree 13, so one box is enough
    gk = GaussKronrod(Lattice(1), axes=0, atol=1e-14, rtol=0)
    v, bz = gk.integrate(lambda k: k[0] ** 12, ret_bz=True)
    assert v == pytest.approx(0.5**12 / 13, abs=1e-15)
    assert len(bz) == 15
    assert bz.weight.sum() == pytest.approx(1)
    assert np.all(bz.k[:, 1:] == 0)


def test_gk_peak_local_refinement():
    func, ref = peak(1 + 1e-3)
    gk = GaussKronrod(Lattice(1), axes=0, atol=0, rtol=1e-10, max_k=5000)
    v, bz = gk.integrate(func, ret_bz=True)
    assert v == pytest.approx(ref, rel=1e-9)
    assert bz.weight.sum() == pytest.approx(1)
    # points concentrate around the peak at k=0
    assert np.sum(np.abs(bz.k[:, 0]) < 0.05) > len(bz) / 2


def test_gk_array_and_complex():
    func, ref = peak(1.2)
    gk = GaussKronrod(Lattice(1), axes=0, atol=1e-12, rtol=0)
    v = gk.integrate(lambda k: np.array([[func(k), 2j * func(k)]]))
    assert v.shape == (1, 2)
    assert np.allclose(v, [[ref, 2j * ref]], atol=1e-10, rtol=0)


def test_gk_2d_split_direction():
    # the integrand does not depend on k_b, so boxes only split along a
    func, ref = peak(1.05)
    gk = GaussKronrod(Lattice(1), axes=[0, 1], atol=1e-9, rtol=0, max_k=50000)
    v, bz = gk.integrate(func, ret_bz=True)
    assert v == pytest.approx(ref, abs=1e-9)
    assert len(np.unique(bz.k[:, 1])) == 15
    assert len(np.unique(bz.k[:, 0])) > 15


def test_gk_2d_separable():
    f, ref = peak(1.1)
    gk = GaussKronrod(Lattice(1), axes=[0, 2], n=[2, 1], atol=1e-9, rtol=0, max_k=50000)
    v = gk.integrate(lambda k: f(k) * f(k[::-1]))
    assert v == pytest.approx(ref**2, abs=1e-8)


def test_gk_max_k_warns():
    func, _ = peak(1 + 1e-6)
    gk = GaussKronrod(Lattice(1), axes=0, atol=1e-12, rtol=0, max_k=100)
    with pytest.warns(SislWarning, match="did not converge"):
        gk.integrate(func)


@pytest.mark.parametrize("order", [1, 2, 5, 9])
def test_farey_points_1d(order):
    p, q = _farey_points(order, 1)
    assert len(p) == sum(totient(n) for n in range(1, order + 1))
    k = (p / q).ravel()
    assert np.all(k > -0.5) and np.all(k <= 0.5)
    assert len(np.unique(k)) == len(k)
    assert np.all(np.gcd(p, q) == 1)


@pytest.mark.parametrize("ndim", [2, 3])
def test_farey_points_nested(ndim):
    for order in range(2, 7):
        p, q = _farey_points(order, ndim)
        assert np.all(np.lcm.reduce(q, axis=1) <= order)
        new = set(map(tuple, np.concatenate([p, q], 1)))
        p0, q0 = _farey_points(order - 1, ndim)
        old = set(map(tuple, np.concatenate([p0, q0], 1)))
        assert old < new
    # order 2 in 2D: Gamma, and 1/2 along a, b and a+b
    assert len(_farey_points(2, 2)[0]) == 4


def test_farey_weights_1d():
    p, q = _farey_points(5, 1)
    w = _voronoi_weights(p / q, np.eye(1))
    assert w.sum() == pytest.approx(1)
    # Gamma lies between -1/5 and 1/5
    assert w[np.argmin(np.abs(p[:, 0]))] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "lattice",
    [Lattice(1), Lattice([2, 2, 10, 90, 90, 60]), Lattice([1, 2, 3, 70, 80, 60])],
)
def test_farey_weights_voronoi(lattice):
    rcell = lattice.rcell
    for ndim in (2, 3):
        metric = rcell[:ndim] @ rcell[:ndim].T
        p, q = _farey_points(1, ndim)
        assert np.allclose(_voronoi_weights(p / q, metric), 1)
        # a regular grid (all denominators 4) has equal weights
        k = np.array(np.meshgrid(*[np.arange(-1, 3) / 4] * ndim)).reshape(ndim, -1).T
        w = _voronoi_weights(k, metric)
        assert np.allclose(w, 1 / len(k))
        p, q = _farey_points(6, ndim)
        w = _voronoi_weights(p / q, metric)
        assert w.sum() == pytest.approx(1)
        assert np.all(w > 0)


@pytest.mark.filterwarnings("error::sisl.SislWarning")
def test_farey_integrate_1d():
    func, ref = peak(1.5)
    fa = Farey(Lattice(1), axes=0, atol=1e-4, rtol=0, max_k=5000)
    v, bz = fa.integrate(func, ret_bz=True)
    assert v == pytest.approx(ref, abs=1e-4)
    assert bz.weight.sum() == pytest.approx(1)


@pytest.mark.filterwarnings("error::sisl.SislWarning")
def test_farey_integrate_2d():
    f, ref = peak(2)
    fa = Farey(
        Lattice([2, 2, 10, 90, 90, 60]), axes=[0, 1], atol=1e-2, rtol=0, max_k=5000
    )
    v = fa.integrate(lambda k: f(k) * f(k[[1, 0, 2]]))
    assert v == pytest.approx(ref**2, abs=1e-2)


def test_farey_max_k_warns():
    func, _ = peak(1 + 1e-6)
    fa = Farey(Lattice(1), axes=0, atol=1e-12, rtol=0, max_k=50)
    with pytest.warns(SislWarning, match="did not converge"):
        fa.integrate(func)


@pytest.mark.filterwarnings("error::sisl.SislWarning")
@pytest.mark.parametrize("cls, atol", [(GaussKronrod, 1e-8), (Farey, 1e-3)])
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
    bz = cls(H, axes=0, atol=atol, rtol=0, max_k=20000)
    assert bz.integrate(G)[0, 0] == pytest.approx(ref, abs=atol)
