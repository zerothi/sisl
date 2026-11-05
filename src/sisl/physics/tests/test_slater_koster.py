# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import warnings

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

    # polarized: [up, down] values, the second range the same for both spins
    sk = sisl.SlaterKoster(
        {("H", "H"): {("1s", "1s"): {"sigma": ([1.6, 3.1], ([-1.0, -0.5], -0.2))}}},
        onsite={"H": {"1s": [0.3, -0.3]}},
    )
    H = sisl.Hamiltonian(geom, spin="polarized")
    H.construct(sk)
    c1, c2 = np.cos(2 * np.pi * k), np.cos(4 * np.pi * k)
    E = [0.3 - 2.0 * c1 - 0.4 * c2, -0.3 - c1 - 0.4 * c2]
    assert np.allclose([H.eigh(k=[k, 0, 0], spin=i)[0] for i in (0, 1)], E)

    # exchange field b sigma_y, stored as Im(ud)
    b = 0.4
    sk = sisl.SlaterKoster(
        {("H", "H"): {("1s", "1s"): {"sigma": (1.6, -1.0)}}},
        onsite={"H": {"1s": 0.3 * np.eye(2) + b * sisl.Spin.Y}},
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


@pytest.mark.parametrize(
    "spin, dtype",
    [
        ("unpolarized", np.float64),
        ("polarized", np.float64),
        ("nc", np.complex128),
        ("so", np.float64),
    ],
)
def test_sk_from_hamiltonian(spin, dtype):
    # bonds of different lengths along random directions: H comes back
    rng = np.random.default_rng(3)
    labels = {"A": ["2s", "2p"], "B": ["3s", "3d"]}
    A, B = (_atom(t, l) for t, l in labels.items())
    xyz = rng.uniform(0, 2.5, (3, 3))
    geom = sisl.Geometry(xyz, [A, B, A], lattice=[20, 20, 20])
    shape = {"unpolarized": (), "polarized": (2,)}.get(spin, (2, 2))

    def value(herm):
        v = rng.normal(size=shape)
        if len(shape) == 2:
            v = v + 1j * rng.normal(size=shape)
            if herm or spin == "nc":
                v = (v + v.conj().T) / 2
        return v

    hopping, overlap = {}, {}
    for P in [("A", "A"), ("A", "B"), ("B", "B")]:
        for i, a in enumerate(labels[P[0]]):
            for b in labels[P[1]][i if P[0] == P[1] else 0 :]:
                mus = ("sigma", "pi", "delta")[
                    : min("spd".index(a[1]), "spd".index(b[1])) + 1
                ]
                herm = P[0] == P[1] and a == b
                hopping.setdefault(P, {})[a, b] = {mu: value(herm) for mu in mus}
                overlap.setdefault(P, {})[a, b] = {mu: rng.normal() for mu in mus}
    onsite = {t: {a: value(True) for a in l} for t, l in labels.items()}
    sk = sisl.SlaterKoster(hopping, onsite, overlap, R=6.0)
    H = sisl.Hamiltonian(geom, spin=spin, dtype=dtype, orthogonal=False)
    H.construct(sk)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sk = sisl.SlaterKoster.from_hamiltonian(H)
    H2 = sisl.Hamiltonian(geom, spin=spin, dtype=dtype, orthogonal=False)
    H2.construct(sk)
    assert H2.nnz == H.nnz
    for f in ("Hk", "Sk"):
        M, M2 = getattr(H, f)(format="array"), getattr(H2, f)(format="array")
        assert np.allclose(M2, M, rtol=0, atol=1e-12)


def test_sk_from_hamiltonian_pz():
    # p_z in a plane: V_pp_sigma has no effect, thus H does not determine it
    C = sisl.Atom(6, sisl.AtomicOrbital("2pzZ1", R=1.5), tag="C")
    sk = sisl.SlaterKoster(
        {("C", "C"): {("2p", "2p"): {"pi": (1.5, -2.7)}}}, onsite={"C": {"2p": 0.1}}
    )
    H = sisl.Hamiltonian(sisl.geom.graphene(atoms=C))
    H.construct(sk)
    with pytest.warns(sisl.SislWarning, match="does not determine"):
        sk = sisl.SlaterKoster.from_hamiltonian(H)
    H2 = sisl.Hamiltonian(H.geometry)
    H2.construct(sk)
    k = [0.1, 0.2, 0]
    assert np.allclose(H2.Hk(k, format="array"), H.Hk(k, format="array"), atol=1e-12)


def test_sk_merge():
    A, B = _atom("A", ["1s"]), _atom("B", ["1s", "2p"])
    geom = sisl.Geometry([[0, 0, 0], [0.6, 0.7, 0.8]], [A, B], lattice=[2.5, 10, 10])
    geom.set_nsc([3, 1, 1])
    AA = {("A", "A"): {("1s", "1s"): {"sigma": (2.5, -0.5)}}}
    AB = {("A", "B"): {("1s", "2p"): {"sigma": (1.5, 0.7)}}}
    BB = {("B", "B"): {("2p", "2p"): {"sigma": (2.6, 0.4), "pi": (2.6, -0.1)}}}
    S = {("A", "B"): {("1s", "1s"): {"sigma": (1.5, 0.1)}}}
    on = {"A": {"1s": -1.0}, "B": {"1s": 0.5, "2p": 2.0}}
    sk = sisl.SlaterKoster(AA | AB | BB, onsite=on, overlap=S)
    sk1 = sisl.SlaterKoster(AA, onsite={"A": on["A"]}, overlap=S)
    sk2 = sisl.SlaterKoster(AB | BB, onsite={"B": on["B"]}, overlap={})
    merged = sisl.SlaterKoster.merge(sk1, sk2)
    assert merged.R == sk.R == 2.6
    H, H2 = (sisl.Hamiltonian(geom, orthogonal=False) for _ in range(2))
    H.construct(sk)
    H2.construct(merged)
    k = [0.1, 0.2, 0.3]
    assert np.allclose(H.Hk(k).toarray(), H2.Hk(k).toarray())
    assert np.allclose(H.Sk(k).toarray(), H2.Sk(k).toarray())

    rev = {("B", "A"): {("2p", "1s"): {"sigma": (1.5, -0.7)}}}
    for dup, match in [
        (dict(hopping=AB), r"hopping \(A, B\): \(1sZ1, 2pZ1\)"),
        (dict(hopping=rev), r"hopping \(A, B\): \(1sZ1, 2pZ1\)"),
        (dict(hopping={}, overlap=S), r"overlap \(A, B\): \(1sZ1, 1sZ1\)"),
        (dict(hopping={}, onsite={"B": {"2p": 0}}), "on-site B: 2pZ1"),
    ]:
        dup.setdefault("overlap", {})
        with pytest.raises(ValueError, match=match):
            sisl.SlaterKoster.merge(sk, sisl.SlaterKoster(**dup))
    with pytest.raises(ValueError, match="orthogonal"):
        sisl.SlaterKoster.merge(sk, sisl.SlaterKoster(BB))


def test_sk_atoms():
    sk = sisl.SlaterKoster(
        {
            ("C", "C"): {
                ("2p", "2p"): {"sigma": (2.1, 0.4), "pi": (1.1, -0.2)},
                ("2s", "2s"): {"sigma": (1.0, 0)},
                ("2s", "2p"): {"sigma": (1.0, 0)},
            },
            ("H_end", "C"): {("1s", "2p"): {"sigma": (1.2, -0.5)}},
        },
        onsite={"C": {"2s": -1.0, "2p": 1.0}, "H_end": {"1s": 0.0}},
    )
    atoms = sk.atoms
    assert [(a.tag, a.Z) for a in atoms.atom] == [("C", 6), ("H_end", 1)]
    C, H = atoms["C"], atoms["H_end"]
    assert [(o.n, o.l, o.m) for o in C] == [(2, 0, 0)] + [(2, 1, m) for m in (-1, 0, 1)]
    assert np.allclose(C.R, 2.1) and np.allclose(H.R, 1.2)
    S = {("H_end", "H_end"): {("1s", "1s"): {"sigma": (1.3, 0.1)}}}
    assert np.allclose(sisl.SlaterKoster({}, overlap=S).atoms["H_end"].R, 1.3)

    # find_nsc gives the C-C hopping at 2 lattice vectors
    geom = sisl.geom.sc(1.0, C)
    assert np.all(geom.nsc == 5)
    ref = sisl.Geometry(geom.xyz, _atom("C", ["2s", "2p"]), lattice=geom.lattice)
    H, Href = sisl.Hamiltonian(geom), sisl.Hamiltonian(ref)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        H.construct(sk)
        Href.construct(sk)
    k = [0.1, 0.2, 0.3]
    assert np.allclose(H.Hk(k).toarray(), Href.Hk(k).toarray())

    # tags without integrals: the element is the tag, its leading name or symbol
    tags = ["Carbon", "6", "C1", "CH3", "Co_x", "Cl", "Carbon_edge", "Nitrogen1"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        atoms = sisl.SlaterKoster({}, onsite={t: {"1s": 0} for t in tags}).atoms
    assert [atoms[t].symbol for t in tags] == ["C", "C", "C", "C", "Co", "Cl", "C", "N"]
    assert all(np.all(atoms[t].R == 0) for t in tags)
    for tag in ["X1", "12C", 6]:
        with pytest.raises(ValueError, match="element symbol"):
            sisl.SlaterKoster({}, onsite={tag: {"1s": 0}}).atoms


def test_sk_atoms_siesta(sisl_files):
    H = sisl.get_sile(
        sisl_files("siesta", "Si_pdos_k", "Si_pdos.fdf")
    ).read_hamiltonian()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sisl.SislWarning)
        sk = sisl.SlaterKoster.from_hamiltonian(H)
    key = lambda atom: [(o.n, o.l, o.m, o.zeta, o.P) for o in atom]
    ref = H.geometry.atoms.atom[0]
    assert key(sk.atoms[ref.tag]) == key(ref)
