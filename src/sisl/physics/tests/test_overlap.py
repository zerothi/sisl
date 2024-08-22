# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Atom, Geometry, Lattice, SphericalOrbital
from sisl.physics.overlap import Overlap

pytestmark = [pytest.mark.physics, pytest.mark.overlap]


@pytest.fixture
def setup():
    class t:
        def __init__(self):
            bond = 1.42
            sq3h = 3.0**0.5 * 0.5
            self.lattice = Lattice(
                np.array(
                    [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
                )
                * bond,
                nsc=[3, 3, 1],
            )

            n = 60
            rf = np.linspace(0, bond * 1.01, n)
            rf = (rf, rf)
            orb = SphericalOrbital(1, rf, 2.0)
            C = Atom(6, orb.toAtomicOrbital())
            self.g = Geometry(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
                atoms=C,
                lattice=self.lattice,
            )
            self.S = Overlap(self.g)

    return t()


class TestOverlap:
    def test_objects(self, setup):
        assert len(setup.S.xyz) == 2
        assert setup.g.no == len(setup.S)

    def test_dtype(self, setup):
        assert setup.S.dtype == np.float64

    def test_ortho(self, setup):
        assert setup.S.orthogonal

    def test_set1(self, setup):
        S = setup.S.copy()
        S.S[0, 0] = 1.0
        assert S[0, 0] == 1.0
        assert S[1, 0] == 0.0

    def test_fromsp(self, setup):
        S = setup.S.copy()
        csr = S.tocsr()
        S2 = Overlap.fromsp(S.geometry, csr)
        assert np.allclose(csr.data, S2.tocsr().data)
