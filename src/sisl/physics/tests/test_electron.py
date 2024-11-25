# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom, Geometry, Hamiltonian, HydrogenicOrbital

pytestmark = [pytest.mark.physics]


def test_EigenstateElectron_norm2():
    pz = HydrogenicOrbital(2, 1, 0, 3.2)
    C1 = Atom(6, (pz))
    C2 = Atom(6, (pz, pz))
    na = 4
    xyz = np.zeros((na, 3))
    xyz[:, 0] = 1.42 * np.arange(na)
    g = Geometry(xyz, atoms=(C1, C2))
    H = Hamiltonian(g)
    for i in range(H.no - 1):
        H[i, i + 1] = -2.7
        H[i + 1, i] = -2.7

    state = H.eigenstate()
    assert len(state) == H.no
    assert state.norm2()[0] == pytest.approx(1)
    assert state.norm2().shape == (H.no,)
    for p in ("diagonal", "orbital", "atom"):
        assert state.norm2(projection=p).sum() == pytest.approx(H.no)

    ns = 3
    state3 = state.sub(range(ns))
    assert state3.norm2(projection="trace").ndim == 0
    assert state3.norm2(projection="diagonal").shape == (ns,)
    assert state3.norm2(projection="orbital").shape == (ns, H.no)
    assert state3.norm2(projection="atom").shape == (ns, H.na)
    assert state3.norm2(projection="atom").sum() == pytest.approx(ns)
