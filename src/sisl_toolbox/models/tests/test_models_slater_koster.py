# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl
from sisl_toolbox.models import slater_koster
from sisl_toolbox.models._slater_koster import _VOGL1983


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
    with pytest.raises(ValueError, match="atomic numbers"):
        slater_koster.H.ref.Vogl1983("Si", geometry=geom)
