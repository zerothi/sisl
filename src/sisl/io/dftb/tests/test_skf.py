# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl
from sisl.io.dftb.skf import _interpolate

pytestmark = [pytest.mark.io, pytest.mark.dftb]

# CODATA 2022, independent of sisl.unit
_Bohr = 0.529177210544
_Ha = 27.211386245981
_DIST, _N = 0.1, 29


def _poly(r):
    return 1 + 0.5 * r - 0.1 * r**2 + 0.01 * r**3


def _table(sign=1):
    """Column c is (c + 1) p(r); the columns of unequal l get `sign`"""
    r = _DIST * np.arange(1, _N + 1)
    scale = np.tile(np.arange(1.0, 11.0), 2)
    scale[[3, 4, 7, 8, 13, 14, 17, 18]] *= sign
    T = scale * _poly(r)[:, None]
    # 3 equal dd columns test the short form n*value
    T[:, :3] = 0.25
    return T


def _write(path, T, homo=False):
    lines = [f"{_DIST}, {_N + 1},2"]
    if homo:
        lines.append("-0.1 -0.2 -0.3, 0.0 0.4 0.4 0.4 0.0 4.0 2.0 T 1 0.3")
    lines.append("12.0, 19*0.0,")
    lines += ["3*0.25 " + " ".join(f"{v:.17e}" for v in row[3:]) for row in T]
    lines += [
        "20*0.0",
        "Spline",
        "1 4.0",
        "1 2 3",
        "1.0 4.0 0 0 0 0 0 0",
        "<Documentation>",
    ]
    path.write_text("\n".join(lines) + "\n")


def _dimer(d, tags):
    atoms = [
        sisl.Atom(
            1,
            [sisl.AtomicOrbital("1s", R=3)]
            + [sisl.AtomicOrbital(n=2, l=1, m=m, R=3) for m in (-1, 0, 1)],
            tag=tag,
        )
        for tag in tags
    ]
    return sisl.Geometry([[5, 5, 5], np.add(5, d)], atoms, lattice=[20, 20, 20])


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


def test_skf_homonuclear(tmp_path):
    T = _table()
    _write(tmp_path / "X-X.skf", T, homo=True)
    sk = sisl.get_sile(tmp_path / "X-X.skf").read_slater_koster({"X": ["1s", "2p"]})
    assert not sk.orthogonal
    d = np.array([0.3, -0.5, 0.8])
    d *= 1.234 * _Bohr / np.linalg.norm(d)
    H = sisl.Hamiltonian(_dimer(d, "XX"), orthogonal=False)
    H.construct(sk)
    Hk, Sk = H.Hk().toarray(), H.Sk().toarray()
    assert np.allclose(np.diag(Hk)[:4], np.array([-0.3, -0.2, -0.2, -0.2]) * _Ha)
    p = _poly(1.234)
    # DFTB+ <p_2|H|s_1> = (y, z, x) V_sp along d from 1 to 2; sisl p is (-y, z, -x)
    e = np.array([-d[1], d[2], -d[0]]) / np.linalg.norm(d)
    assert np.allclose(Hk[0, 4], 10 * p * _Ha)
    assert np.allclose(Hk[0, 5:8], 9 * p * _Ha * e)
    assert np.allclose(Sk[0, 5:8], 9 * p * e)
    assert np.allclose(Hk[1:4, 4], -9 * p * _Ha * e)


def test_skf_heteronuclear(tmp_path):
    _write(tmp_path / "X-Y.skf", _table())
    _write(tmp_path / "Y-X.skf", _table(-2))
    shells = {"X": ["1s", "2p"], "Y": ["1s", "2p"]}
    sk = sisl.get_sile(tmp_path / "X-Y.skf").read_slater_koster(shells)
    d = np.array([-0.4, 0.2, 0.7])
    d *= 2.5 * _Bohr / np.linalg.norm(d)
    H = sisl.Hamiltonian(_dimer(d, "XY"), orthogonal=False)
    with pytest.warns(sisl.SislWarning, match="on-site"):
        H.construct(sk)
    Hk = H.Hk().toarray()
    p = _poly(2.5) * _Ha
    e = np.array([-d[1], d[2], -d[0]]) / np.linalg.norm(d)
    # s of X and p of Y from X-Y, s of Y and p of X from Y-X
    assert np.allclose(Hk[0, 5:8], 9 * p * e)
    assert np.allclose(Hk[4, 1:4], -2 * 9 * p * -e)
    assert np.allclose(Hk[0, 4], 10 * p)

    # the same model from Y-X.skf
    sk2 = sisl.get_sile(tmp_path / "Y-X.skf").read_slater_koster(shells)
    H2 = H.copy()
    with pytest.warns(sisl.SislWarning, match="on-site"):
        H2.construct(sk2)
    assert np.allclose(Hk, H2.Hk().toarray())


def test_skf_errors(tmp_path):
    shells = {"X": ["1s", "2p"], "Y": ["1s"]}
    T = _table()
    _write(tmp_path / "X-Y.skf", T)
    with pytest.warns(sisl.SislWarning, match="Y-X.skf"):
        sisl.get_sile(tmp_path / "X-Y.skf").read_slater_koster(shells)
    T[:, 9] += 1e-6
    _write(tmp_path / "Y-X.skf", T)
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
