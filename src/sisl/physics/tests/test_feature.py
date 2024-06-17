# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom, Geometry, Hamiltonian, Lattice
from sisl.physics import yield_manifolds

pytestmark = [pytest.mark.physics, pytest.mark.physics_feature]


def test_yield_manifolds_eigenvalues():
    g = Geometry(
        [[i, 0, 0] for i in range(10)],
        Atom(6, R=1.01),
        lattice=Lattice([10, 1, 5.0], nsc=[3, 3, 1]),
    )
    H = Hamiltonian(g, dtype=np.float64)
    H.construct([(0.1, 1.5), (1.0, 0.1)])

    all_manifolds = []
    for manifold in yield_manifolds(H.eigh()):
        all_manifolds.extend(manifold)

    assert np.allclose(all_manifolds, np.arange(len(H)))

    all_manifolds = []
    for manifold in yield_manifolds(H.tile(2, 0).eigh()):
        all_manifolds.extend(manifold)

    assert np.allclose(all_manifolds, np.arange(len(H) * 2))
