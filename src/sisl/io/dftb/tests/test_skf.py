# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl
from sisl.io.dftb.skf import _interpolate
from sisl.physics.slater_koster import _shell

pytestmark = [pytest.mark.io, pytest.mark.dftb]

# CODATA 2022, independent of sisl.unit
_Bohr = 0.529177210544
_Ha = 27.211386245981
_DIST, _N = 0.1, 29
_SHELLS = ["1s", "2p", "3d"]
# the specification: Hdd0 Hdd1 Hdd2 Hpd0 Hpd1 Hpp0 Hpp1 Hsd0 Hsp0 Hss0, then S
_SPEC = {
    ("3d", "3d"): (0, 1, 2),
    ("2p", "3d"): (3, 4),
    ("2p", "2p"): (5, 6),
    ("1s", "3d"): (7,),
    ("1s", "2p"): (8,),
    ("1s", "1s"): (9,),
}


def _poly(r):
    return 1 + 0.5 * r - 0.1 * r**2 + 0.01 * r**3


def _scale(sign=1):
    """Column c is (c + 1) p(r), except Sdd1 = Sdd0; the columns of unequal l get `sign`"""
    scale = np.arange(1.0, 21.0)
    scale[11] = scale[10]
    scale[[3, 4, 7, 8, 13, 14, 17, 18]] *= sign
    return scale


def _write(path, scale, homo=False, elements=(), basis=()):
    T = scale * _poly(_DIST * np.arange(1, _N + 1))[:, None]
    fmt = lambda v: " ".join(f"{x:.17e}" for x in v)
    lines = [f"{_DIST}, {_N + 1},2"]
    if homo:
        lines.append("-0.1 -0.2 -0.3, 0.0 0.4 0.4 0.4 0.0 4.0 2.0 T 1 0.3")
    lines.append("12.0, 19*0.0,")
    # equal Sdd0 and Sdd1 test the short form n*value
    lines += [f"{fmt(row[:10])}, 2*{row[10]:.17e} {fmt(row[12:])}" for row in T]
    lines += ["20*0.0", "Spline", "1 4.0", "1 2 3", "1.0 4.0 0 0 0 0 0 0"]
    if elements:
        tags = "".join(
            f"<Element{i}>{e}</Element{i}>" for i, e in enumerate(elements, 1)
        )
        basis = "".join(
            f'<Basis atom="{i}"><Shells>{b} </Shells></Basis>'
            for i, b in enumerate(basis, 1)
        )
        lines.append(
            f"<Documentation><General>{tags}</General>"
            f"<SK_table>{basis}</SK_table></Documentation>"
        )
    path.write_text("\n".join(lines) + "\n")


def _dimer(d, tags):
    """Two atoms with s, p, d (9 orbitals each), the second at `d`"""
    orbs = [sisl.AtomicOrbital("1s", R=3)]
    orbs += [sisl.AtomicOrbital(n=2, l=1, m=m, R=3) for m in range(-1, 2)]
    orbs += [sisl.AtomicOrbital(n=3, l=2, m=m, R=3) for m in range(-2, 3)]
    atoms = [sisl.Atom(1, orbs, tag=tag) for tag in tags]
    return sisl.Geometry([[5, 5, 5], np.add(5, d)], atoms, lattice=[20, 20, 20])


def _harmonics(d):
    """sp and sd factors of DFTB+ (sk.F90) with the signs of the sisl harmonics"""
    l, m, n = d / np.linalg.norm(d)
    p = np.array([m, n, l]) * [-1, 1, -1]
    s3 = np.sqrt(3)
    D = [s3 * l * m, s3 * m * n, (3 * n**2 - 1) / 2, s3 * l * n, s3 / 2 * (l**2 - m**2)]
    return p, np.array(D) * [1, -1, 1, -1, 1]


def test_skf_interpolate():
    r = _DIST * np.arange(1, _N + 1)
    p7 = lambda r: 0.3 - r + 0.2 * r**3 - 0.01 * r**7
    f, R = _interpolate(p7(r)[:, None], _DIST)
    assert R == pytest.approx((_N * _DIST + 1) * _Bohr, rel=1e-12)
    # a polynomial of order 7 is exact on the grid, also before the first point
    x = np.linspace(0.05, _N * _DIST, 41)
    assert np.allclose(f(x * _Bohr)[:, 0], p7(x), rtol=0, atol=1e-11)
    # the tail is the 5th order polynomial with value and two derivatives of
    # p7 at the last point, and 0 with two derivatives 0 at R
    r0, r1 = _N * _DIST, _N * _DIST + 1
    rows = [[r**k for k in range(6)] for r in (r0, r1)]
    A = np.array(
        rows
        + [[k * r ** max(k - 1, 0) for k in range(6)] for r in (r0, r1)]
        + [[k * (k - 1) * r ** max(k - 2, 0) for k in range(6)] for r in (r0, r1)]
    )
    dp7 = -1 + 0.6 * r0**2 - 0.07 * r0**6
    ddp7 = 1.2 * r0 - 0.42 * r0**5
    c = np.linalg.solve(A, [p7(r0), 0, dp7, 0, ddp7, 0])
    x = np.linspace(r0, r1 + 0.5, 11)
    tail = np.where(x < r1, np.polynomial.polynomial.polyval(x, c), 0)
    assert np.allclose(f(x * _Bohr)[:, 0], tail, rtol=0, atol=1e-9)


def test_skf_stencil():
    # a coarse grid of sin: each choice of the 8 grid points gives another value
    dist, n = 0.5, 20
    f, _ = _interpolate(np.sin(2 * dist * np.arange(1, n + 1))[:, None], dist)
    x = np.array([0.3, 1.5, 3.7, 9.2, 15.5, 17.9, 19.6])
    # slakoeqgrid.F90: the last of the 8 points is clip(floor(x) + 4, 8, n)
    last = np.clip(np.floor(x).astype(int) + 4, 8, n)
    ref = [
        np.polynomial.polynomial.polyval(
            xi - k,
            np.polynomial.polynomial.polyfit(
                np.arange(-7, 1.0), np.sin(2 * dist * np.arange(k - 7, k + 1)), 7
            ),
        )
        for xi, k in zip(x, last)
    ]
    assert np.allclose(f(x * dist * _Bohr)[:, 0], ref, rtol=0, atol=1e-10)


def test_skf_columns(tmp_path):
    _write(tmp_path / "X-X.skf", _scale(), homo=True)
    sk = sisl.get_sile(tmp_path / "X-X.skf").read_slater_koster({"X": _SHELLS})
    r = np.array([0.77, 2.1]) * _Bohr
    p = _poly(r / _Bohr)
    scale = _scale()
    for (a, b), cols in _SPEC.items():
        key = (_shell(a), _shell(b))
        for ints, unit, h in ((sk._hopping, _Ha, 0), (sk._overlap, 1, 10)):
            values = {imu: f(r) for imu, f in ints[("X", "X")][key]}
            assert sorted(values) == list(range(len(cols)))
            for imu, c in enumerate(cols):
                assert np.allclose(values[imu], scale[c + h] * p * unit)


def test_skf_homonuclear(tmp_path):
    _write(tmp_path / "X-X.skf", _scale(), homo=True)
    sk = sisl.get_sile(tmp_path / "X-X.skf").read_slater_koster({"X": _SHELLS})
    assert not sk.orthogonal
    d = np.array([0.3, -0.5, 0.8])
    d *= 1.234 * _Bohr / np.linalg.norm(d)
    H = sisl.Hamiltonian(_dimer(d, "XX"), orthogonal=False)
    H.construct(sk)
    Hk, Sk = H.Hk().toarray(), H.Sk().toarray()
    assert np.allclose(np.diag(Hk)[:9], np.repeat([-0.3, -0.2, -0.1], [1, 3, 5]) * _Ha)
    p = _poly(1.234)
    # DFTB+: <p_2|H|s_1> = (y, z, x) V_sp along d from atom 1 to 2
    ep, ed = _harmonics(d)
    assert np.allclose(Hk[0, 9], 10 * p * _Ha)
    assert np.allclose(Hk[0, 10:13], 9 * p * _Ha * ep)
    assert np.allclose(Sk[0, 10:13], 19 * p * ep)
    assert np.allclose(Hk[1:4, 9], -9 * p * _Ha * ep)
    assert np.allclose(Hk[0, 13:18], 8 * p * _Ha * ed)
    assert np.allclose(Hk[4:9, 9], 8 * p * _Ha * ed)


def test_skf_heteronuclear(tmp_path):
    _write(tmp_path / "X-Y.skf", _scale())
    _write(tmp_path / "Y-X.skf", _scale(-2))
    shells = {"X": _SHELLS, "Y": _SHELLS}
    sk = sisl.get_sile(tmp_path / "X-Y.skf").read_slater_koster(shells)
    d = np.array([-0.4, 0.2, 0.7])
    d *= 2.5 * _Bohr / np.linalg.norm(d)
    H = sisl.Hamiltonian(_dimer(d, "XY"), orthogonal=False)
    with pytest.warns(sisl.SislWarning, match="on-site"):
        H.construct(sk)
    Hk = H.Hk().toarray()
    p = _poly(2.5) * _Ha
    ep, ed = _harmonics(d)
    # the lower l on X from X-Y, the lower l on Y from Y-X (along -d)
    assert np.allclose(Hk[0, 9], 10 * p)
    assert np.allclose(Hk[0, 10:13], 9 * p * ep)
    assert np.allclose(Hk[9, 1:4], -2 * 9 * p * -ep)
    assert np.allclose(Hk[0, 13:18], 8 * p * ed)
    assert np.allclose(Hk[9, 4:9], -2 * 8 * p * ed)

    # the same model from Y-X.skf
    sk2 = sisl.get_sile(tmp_path / "Y-X.skf").read_slater_koster(shells)
    H2 = H.copy()
    with pytest.warns(sisl.SislWarning, match="on-site"):
        H2.construct(sk2)
    assert np.allclose(Hk, H2.Hk().toarray())


def test_skf_errors(tmp_path):
    shells = {"X": ["1s", "2p"], "Y": ["1s"]}
    _write(tmp_path / "X-Y.skf", _scale())
    with pytest.warns(sisl.SislWarning, match="Y-X.skf"):
        sisl.get_sile(tmp_path / "X-Y.skf").read_slater_koster(shells)
    scale = _scale()
    scale[9] += 1e-6
    _write(tmp_path / "Y-X.skf", scale)
    with pytest.raises(ValueError, match="equal l"):
        sisl.get_sile(tmp_path / "X-Y.skf").read_slater_koster(shells)
    with pytest.raises(ValueError, match="one shell for each l"):
        sisl.get_sile(tmp_path / "X-Y.skf").read_slater_koster(
            {"X": ["1s", "2s"], "Y": ["1s"]}
        )
    with pytest.raises(ValueError, match="shells of Y"):
        sisl.get_sile(tmp_path / "X-Y.skf").read_slater_koster({"X": ["1s"]})
    (tmp_path / "XY.skf").write_text("")
    with pytest.raises(ValueError, match="A-B.skf"):
        sisl.get_sile(tmp_path / "XY.skf").read_slater_koster(shells)
    (tmp_path / "X-X.skf").write_text("@ extended\n0.1 30\n")
    with pytest.raises(NotImplementedError):
        sisl.get_sile(tmp_path / "X-X.skf").read_slater_koster(shells)
    (tmp_path / "Y-Y.skf").write_text("0.1 30\n" + "0 " * 10 + "\n" + "20*0\n" * 10)
    with pytest.raises(ValueError, match="rows of 20"):
        sisl.get_sile(tmp_path / "Y-Y.skf").read_slater_koster(shells)
    (tmp_path / "Y-Y.skf").write_text("0.1 9\n" + "0 " * 10 + "\n" + "20*0\n" * 10)
    with pytest.raises(ValueError, match="more than 8 rows"):
        sisl.get_sile(tmp_path / "Y-Y.skf").read_slater_koster(shells)


@pytest.mark.parametrize("name", ["{}-{}", "{}{}", "{}-{}_short"])
def test_skf_names(tmp_path, name):
    shells = {"X": _SHELLS, "Y": _SHELLS}
    _write(tmp_path / f"{name.format('X', 'Y')}.skf", _scale(), elements="XY")
    # the documentation of the partner has the order of the other file
    _write(tmp_path / f"{name.format('Y', 'X')}.skf", _scale(-2), elements="XY")
    sile = sisl.get_sile(tmp_path / f"{name.format('X', 'Y')}.skf")
    assert "<Element2>Y</Element2>" in sile.read_documentation()
    sk = sile.read_slater_koster(shells)
    assert sk._tags == {"X", "Y"}
    ref = sisl.get_sile(tmp_path / f"{name.format('Y', 'X')}.skf").read_slater_koster(
        shells
    )
    r = np.array([1.1, 2.3])
    for key in [(("X", "Y"), ("1s", "2p")), (("Y", "X"), ("1s", "2p"))]:
        AB, (a, b) = key
        f = dict(sk._hopping[AB][_shell(a), _shell(b)])[0]
        g = dict(ref._hopping[AB][_shell(a), _shell(b)])[0]
        assert np.allclose(f(r), g(r))

    _write(tmp_path / "X-X.skf", _scale(), homo=True, elements="Z")
    with pytest.warns(sisl.SislWarning, match="Z"):
        sk = sisl.get_sile(tmp_path / "X-X.skf").read_slater_koster(shells)
    assert sk._tags == {"X"}


def test_skf_default_shells(tmp_path):
    _write(tmp_path / "X-X.skf", _scale(), homo=True, elements="X", basis=["1s 2p"])
    _write(tmp_path / "Y-Y.skf", _scale(), homo=True, elements="Y", basis=["1s"])
    # the basis of the heteronuclear files is not used
    for name in ["X-Y_short", "Y-X_short"]:
        _write(tmp_path / f"{name}.skf", _scale(), elements="XY", basis=["4f", "4f"])
    sk = sisl.get_sile(tmp_path / "X-Y_short.skf").read_slater_koster()
    assert set(sk._hopping["X", "Y"]) == {
        (_shell("1s"), _shell("1s")),
        (_shell("2p"), _shell("1s")),
    }
    # given shells win
    sk = sisl.get_sile(tmp_path / "X-Y_short.skf").read_slater_koster({"Y": ["2s"]})
    assert (_shell("1s"), _shell("2s")) in sk._hopping["X", "Y"]
    _write(tmp_path / "X-Z.skf", _scale(), elements="XZ")
    with pytest.raises(ValueError, match="no Z-Z.skf"):
        sisl.get_sile(tmp_path / "X-Z.skf").read_slater_koster({"X": ["1s"]})
