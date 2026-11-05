# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl
from sisl_toolbox.models import slater_koster
from sisl_toolbox.models._slater_koster import _VOGL1983, _xu1992_scaling


def _vogl_bloch(material, kc):
    """Bloch matrix of Table A of Vogl 1983 from the cubic parameters"""
    E, V, d = _VOGL1983[material]
    Es_a, Ep_a, Es_c, Ep_c, Ess_a, Ess_c = E
    Vss, Vxx, Vxy, Vsa_pc, Vsc_pa, Vssa_pc, Vpa_ssc = V
    # cation neighbors of the anion, in units of a_L / 4
    sgn = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]])
    phase = np.exp(1j * sgn @ kc * d / 3**0.5) / 4
    g0, g = phase.sum(), sgn.T @ phase
    # basis: s_a, p_a (x, y, z), s*_a, s_c, p_c (x, y, z), s*_c
    Hac = np.zeros((5, 5), dtype=complex)
    Hac[0, 0] = Vss * g0
    Hac[0, 1:4] = Vsa_pc * g
    Hac[1:4, 0] = -Vsc_pa * g
    Hac[4, 1:4] = Vssa_pc * g
    Hac[1:4, 4] = -Vpa_ssc * g
    for i in range(3):
        for j in range(3):
            Hac[1 + i, 1 + j] = Vxx * g0 if i == j else Vxy * g[3 - i - j]
    H = np.zeros((10, 10), dtype=complex)
    H[:5, :5] = np.diag([Es_a, Ep_a, Ep_a, Ep_a, Ess_a])
    H[5:, 5:] = np.diag([Es_c, Ep_c, Ep_c, Ep_c, Ess_c])
    H[:5, 5:] = Hac
    H[5:, :5] = Hac.conj().T
    return H


@pytest.mark.parametrize("material", list(_VOGL1983))
def test_vogl1983_table_a(material):
    H = slater_koster.H.ref.Vogl1983(material)
    rng = np.random.default_rng(1)
    for k in [[0.5, 0, 0.5], rng.random(3)]:
        kc = k @ H.geometry.rcell
        assert np.allclose(
            H.eigh(k=k), np.linalg.eigvalsh(_vogl_bloch(material, kc)), atol=1e-10
        )


def test_vogl1983_known_values():
    # Table 2 of Vogl 1983 (GaAs is fitted at these points)
    H = slater_koster.H.ref.Vogl1983("GaAs")
    G = H.eigh()
    assert np.allclose(G[[0, 1, 2, 5]], [-12.55, 0, 0, 4.71], atol=5e-3)
    assert np.allclose(
        H.eigh(k=[0.5, 0, 0.5])[[3, 4, 5]], [-2.89, 2.03, 2.38], atol=5e-3
    )
    # the integrals extracted from H (McMahan 1997) give the Bloch matrix of Table A
    H2 = sisl.Hamiltonian(H.geometry)
    H2.construct(sisl.SlaterKoster.from_hamiltonian(H))
    kc = [0.3, -0.2, 0.7] @ H.geometry.rcell
    assert np.allclose(
        H2.eigh(k=[0.3, -0.2, 0.7]),
        np.linalg.eigvalsh(_vogl_bloch("GaAs", kc)),
        atol=1e-10,
    )
    # Klimeck et al. 2000, Table 1: Si conduction band minimum 1.171 eV at 0.731 (2 pi / a) along Delta
    H = slater_koster.H.ref.Vogl1983("Si")
    a = 4 * 2.35 / 3**0.5
    k = H.geometry.cell @ [0, 0.731 / a, 0]
    assert H.eigh(k=k)[4] == pytest.approx(1.171, abs=5e-4)


def test_vogl1983_geometry():
    geom = sisl.geom.diamond(5.6, [sisl.Atom("As"), sisl.Atom("Ga")]).tile(2, 0)
    H = slater_koster.H.ref.Vogl1983("GaAs", geometry=geom)
    assert H.geometry.no == 4 * 5
    Hk = H.Hk(k=[0.1, 0.2, 0.3], format="array")
    assert np.allclose(Hk, Hk.T.conj())
    H.geometry.set_nsc([5, 5, 5])
    assert np.all(geom.nsc == 3)
    with pytest.raises(ValueError, match="atomic numbers"):
        slater_koster.H.ref.Vogl1983("Si", geometry=geom)


def test_xu1992():
    # the tail continues the scaling with its slope at r1 and is 0 at rm
    eps = 1e-5
    s = _xu1992_scaling(np.array([2.45 - eps, 2.45, 2.6]))
    assert s[0] == pytest.approx(s[1] + 8.1885359517898e-2 * eps, abs=1e-9)
    assert s[2] == pytest.approx(0, abs=1e-12)

    H = slater_koster.H.ref.Xu1992()
    # diamond at r0: the first neighbor s-s element is V_sssigma
    assert H[0, 4] == pytest.approx(-5.0)
    assert H[0, 0] == -2.99 and H[1, 1] == 3.71
    geom = sisl.geom.graphene(1.42, sisl.Atom(6))
    H = slater_koster.H.ref.Xu1992(geometry=geom)
    assert H.geometry.no == 8
