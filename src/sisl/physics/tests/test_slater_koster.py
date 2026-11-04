# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl
from sisl._core.orbital import _rspherical_harm
from sisl.physics.slater_koster import _sk_factors

pytestmark = [pytest.mark.physics]


def _rotation_matrix(l, Q):
    """D[m', m] = <Y_lm' | Y_lm(Q^T r)> by quadrature of the sisl harmonics"""
    x, w = np.polynomial.legendre.leggauss(6)
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    cos_phi, theta = (a.ravel() for a in np.meshgrid(x, theta, indexing="ij"))
    w = np.repeat(w, 12) * 2 * np.pi / 12
    sin_phi = np.sqrt(1 - cos_phi**2)
    r = np.stack([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi], axis=1)
    rr = r @ Q

    def Y(r):
        t = np.arctan2(r[:, 1], r[:, 0])
        return np.array([_rspherical_harm(l, m, t, r[:, 2]) for m in range(-l, l + 1)])

    return (Y(r) * w) @ Y(rr).T


@pytest.mark.parametrize("l1", [0, 1, 2])
@pytest.mark.parametrize("l2", [0, 1, 2])
def test_sk_block_rotation(l1, l2):
    rng = np.random.default_rng(l1 * 3 + l2)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q *= np.linalg.det(Q)
    V = rng.normal(size=min(l1, l2) + 1)

    # bond along z: diag(V_|m|) in the sisl basis
    h = np.zeros((2 * l1 + 1, 2 * l2 + 1))
    for m in range(-min(l1, l2), min(l1, l2) + 1):
        h[m + l1, m + l2] = V[abs(m)]
    ref = _rotation_matrix(l1, Q) @ h @ _rotation_matrix(l2, Q).T

    block = _sk_factors(l1, l2, Q[:, 2][None]) @ V
    assert np.allclose(block[0], ref, atol=1e-12)


def _atom(tag, labels):
    orbs = []
    for label in labels:
        n, l = int(label[0]), "spd".index(label[1])
        orbs.extend(sisl.AtomicOrbital(n=n, l=l, m=m, R=3.0) for m in range(-l, l + 1))
    return sisl.Atom(1, orbs, tag=tag)


def test_sk_hermitian_rotation():
    A = _atom("A", ["2s", "2p"])
    B = _atom("B", ["3s", "3p", "3d"])
    geom = sisl.Geometry(
        [[0, 0, 0], [1.4, 0.3, -0.2], [0.2, 1.5, 0.4], [-1.1, 0.4, 1.0]],
        [A, B, A, B],
        lattice=[20, 20, 20],
    )
    hopping = {
        ("A", "A"): {
            ("2s", "2p"): {"sigma": 1.1},
            ("2p", "2p"): {"sigma": 2.0, "pi": -0.6},
        },
        ("A", "B"): {("2s", "3d"): {"sigma": -0.8}, ("2p", "3s"): {"sigma": 0.7}},
        ("B", "A"): {("3p", "2p"): {"sigma": 1.3, "pi": ((1.5, 2.0), (-0.5, -0.1))}},
        ("B", "B"): {
            ("3p", "3d"): {"sigma": -1.2, "pi": lambda r: 0.4 * np.exp(-r)},
            ("3d", "3d"): {"sigma": -1.0, "pi": 0.5, "delta": -0.2},
        },
    }
    overlap = {("A", "B"): {("2p", "3p"): {"sigma": 0.1, "pi": -0.05}}}
    onsite = {"A": {"2s": -5.0, "2p": 0.5}, "B": {"3d": -1.0}}
    sk = sisl.SlaterKoster(hopping, onsite=onsite, overlap=overlap, R=2.5)
    assert not sk.orthogonal

    def eig(g):
        H = sisl.Hamiltonian(g, orthogonal=False)
        H.construct(sk)
        Hd, Sd = H.Hk(format="array"), H.Sk(format="array")
        assert np.allclose(Hd, Hd.T) and np.allclose(Sd, Sd.T)
        assert not np.allclose(Hd - np.diag(np.diag(Hd)), 0)
        return H.eigh()

    rotated = geom.rotate([37, [1, 2, 3]], what="xyz").rotate([-71, "x"], what="xyz")
    assert np.allclose(eig(geom), eig(rotated), atol=1e-10)


@pytest.mark.parametrize("spin", ["unpolarized", "polarized", "spin-orbit"])
def test_sk_chain(spin):
    s = sisl.Atom(1, sisl.AtomicOrbital("1s", R=3.0), tag="H")
    geom = sisl.Geometry([0, 0, 0], s, lattice=[1.5, 10, 10])
    geom.set_nsc([7, 1, 1])
    sk = sisl.SlaterKoster(
        {("H", "H"): {("1s", "1s"): {"sigma": ((1.6, 3.1), (-1.0, -0.2))}}},
        onsite={"H": {"1s": 0.3}},
    )
    H = sisl.Hamiltonian(geom, spin=spin)
    H.construct(sk)
    assert H.nnz == 5
    k = 0.15
    E = 0.3 - 2.0 * np.cos(2 * np.pi * k) - 0.4 * np.cos(4 * np.pi * k)
    assert np.allclose(H.eigh(k=[k, 0, 0]), E)

    sk = sisl.SlaterKoster(
        {("H", "H"): {("1s", "1s"): {"sigma": lambda r: -np.exp(-r)}}}, R=3.1
    )
    H = sisl.Hamiltonian(geom)
    H.construct(sk)
    cols = geom.lattice.sc_index([[1, 0, 0], [-2, 0, 0]])
    assert np.allclose(H[0, cols], -np.exp(-np.array([1.5, 3.0])))
    assert H.nnz == 4


def test_sk_open_chain():
    # chain along a non-periodic axis, longer than the cell and starting below 0
    s = sisl.Atom(1, sisl.AtomicOrbital("1s", R=1.1), tag="H")
    N = 25
    geom = sisl.Geometry([[0, 0, z] for z in range(-5, N - 5)], s, lattice=[10, 10, 10])
    geom.set_nsc([3, 3, 1])
    geom.lattice.set_boundary_condition(c=sisl.Lattice.BC.UNKNOWN)
    sk = sisl.SlaterKoster({("H", "H"): {("1s", "1s"): {"sigma": -1.0}}}, R=1.1)
    H = sisl.Hamiltonian(geom)
    H.construct(sk)
    E = -2 * np.cos(np.arange(1, N + 1) * np.pi / (N + 1))
    assert np.allclose(H.eigh(), np.sort(E))


def test_sk_errors():
    s = sisl.Atom(1, sisl.AtomicOrbital("1s", R=1.0), tag="H")
    H = sisl.Hamiltonian(sisl.Geometry([0, 0, 0], s, lattice=[1, 10, 10]))
    hop = {("H", "H"): {("1s", "2p"): {"sigma": 1.0}, ("2p", "1s"): {"sigma": 1.0}}}
    with pytest.raises(ValueError, match="both directions"):
        sisl.SlaterKoster(hop, R=1.0)
    with pytest.raises(ValueError, match="delta"):
        sisl.SlaterKoster({("H", "H"): {("1s", "2p"): {"delta": 1.0}}}, R=1.0)
    with pytest.raises(ValueError, match="needs R"):
        sisl.SlaterKoster({("H", "H"): {("1s", "1s"): {"sigma": 1.0}}})
    with pytest.raises(NotImplementedError):
        sisl.SlaterKoster({("H", "H"): {("4f", "4f"): {"sigma": 1.0}}}, R=1.0)
    with pytest.raises(ValueError, match="'Pi'"):
        sisl.SlaterKoster({("H", "H"): {("1s", "1s"): {"Pi": 1.0}}}, R=1.0)
    plain = sisl.Geometry([0, 0, 0], sisl.Atom(1, R=1.0, tag="H"), lattice=[1, 10, 10])
    with pytest.raises(TypeError, match="AtomicOrbital"):
        sisl.Hamiltonian(plain).construct(
            sisl.SlaterKoster({("H", "H"): {("1s", "1s"): {"sigma": 1.0}}}, R=1.0)
        )

    sk = sisl.SlaterKoster(
        {("H", "H"): {("1s", "1s"): {"sigma": 1.0}}},
        overlap={("H", "H"): {("1s", "1s"): {"sigma": 0.1}}},
        R=1.0,
    )
    with pytest.raises(ValueError, match="orthogonal"):
        H.construct(sk)
    with pytest.raises(NotImplementedError):
        sisl.Hamiltonian(H.geometry, spin="nambu", orthogonal=False).construct(sk)

    X = sisl.Atom(2, sisl.AtomicOrbital("1s", R=1.0), tag="X")
    geom = sisl.Geometry([[0, 0, 0], [0.5, 0, 0]], [s, X], lattice=[10, 10, 10])
    H = sisl.Hamiltonian(geom, orthogonal=False)
    with pytest.warns(sisl.SislWarning, match="'X'"):
        H.construct(sk)
    assert np.allclose(H.Sk(format="array").diagonal(), 1)

    sp = [sisl.AtomicOrbital("1s", R=1.0), sisl.AtomicOrbital("2pzZ1", R=1.0)]
    geom = sisl.Geometry([0, 0, 0], sisl.Atom(1, sp, tag="H"), lattice=[1, 10, 10])
    geom.set_nsc([3, 1, 1])
    sk = sisl.SlaterKoster(
        {("H", "H"): {("1s", "1s"): {"sigma": -1.0}, ("2p", "2p"): {"pi": 0}}},
        onsite={"H": {"1s": 0.0}},
        R=1.0,
    )
    with pytest.warns(sisl.SislWarning) as record:
        sisl.Hamiltonian(geom).construct(sk)
    msg = [str(w.message) for w in record]
    assert len(msg) == 2
    assert "(H, H): (1sZ1, 2pZ1)" in msg[1] and "H: 2pZ1" in msg[0]


def test_sk_spin():
    s = sisl.Atom(1, sisl.AtomicOrbital("1s", R=3.0), tag="H")
    geom = sisl.Geometry([0, 0, 0], s, lattice=[1.5, 10, 10])
    geom.set_nsc([5, 1, 1])
    k = 0.15
    Ek = 0.3 - 2.0 * np.cos(2 * np.pi * k)

    # polarized: [up, down] values
    sk = sisl.SlaterKoster(
        {("H", "H"): {("1s", "1s"): {"sigma": (1.6, [-1.0, -0.5])}}},
        onsite={"H": {"1s": [0.3, -0.3]}},
    )
    H = sisl.Hamiltonian(geom, spin="polarized")
    H.construct(sk)
    E = [0.3 - 2.0 * np.cos(2 * np.pi * k), -0.3 - np.cos(2 * np.pi * k)]
    assert np.allclose([H.eigh(k=[k, 0, 0], spin=i)[0] for i in (0, 1)], E)

    # exchange field b sigma_y, stored as Im(ud)
    b = 0.4
    sk = sisl.SlaterKoster(
        {("H", "H"): {("1s", "1s"): {"sigma": -1.0}}},
        onsite={"H": {"1s": 0.3 * np.eye(2) + b * sisl.Spin.Y}},
        R=1.6,
    )
    for spin in ("nc", "so"):
        for dtype in (np.float64, np.complex128):
            H = sisl.Hamiltonian(geom, spin=spin, dtype=dtype)
            H.construct(sk)
            Hk = H.Hk(k=[k, 0, 0], format="array")
            assert np.allclose(Hk, Ek * np.eye(2) + b * sisl.Spin.Y)

    # dimer with the hopping t + i lam sigma_z: the reverse gets the conjugate
    A, B = (sisl.Atom(1, sisl.AtomicOrbital("1s", R=2.0), tag=t) for t in "AB")
    dimer = sisl.Geometry([[0, 0, 0], [1, 0, 0]], [A, B], lattice=[10, 10, 10])
    t, lam = -1.0, 0.3
    sk = sisl.SlaterKoster(
        {("A", "B"): {("1s", "1s"): {"sigma": t * np.eye(2) + 1j * lam * sisl.Spin.Z}}},
        onsite={"A": {"1s": 0}, "B": {"1s": 0}},
        R=2.0,
    )
    M = t * np.eye(2) + 1j * lam * sisl.Spin.Z
    ref = np.block([[np.zeros((2, 2)), M], [M.conj().T, np.zeros((2, 2))]])
    for spin, dtype in (
        ("so", np.float64),
        ("so", np.complex128),
        ("nc", np.complex128),
    ):
        H = sisl.Hamiltonian(dimer, spin=spin, dtype=dtype)
        H.construct(sk)
        assert np.allclose(H.Hk(format="array"), ref)
    with pytest.raises(ValueError, match="complex dtype"):
        sisl.Hamiltonian(dimer, spin="nc").construct(sk)
    with pytest.raises(ValueError, match="fewer spin"):
        sisl.Hamiltonian(dimer).construct(sk)

    with pytest.raises(ValueError, match="Hermitian"):
        sisl.SlaterKoster({}, onsite={"H": {"1s": [[0, 1], [0, 0]]}})
    sk = sisl.SlaterKoster(
        {("H", "H"): {("1s", "1s"): {"sigma": [[0, 1], [0, 0]]}}}, R=1.6
    )
    with pytest.raises(ValueError, match="Hermitian"):
        sisl.Hamiltonian(geom, spin="so").construct(sk)
