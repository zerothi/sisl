# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

""" pytest test configures """


import numpy as np
import pytest

import sisl

pytestmark = [pytest.mark.io, pytest.mark.gulp]


def test_zz_dynamical_matrix(sisl_files):
    si = sisl.get_sile(sisl_files("gulp", "graphene_zz", "zz.gout"))
    D1 = si.read_dynamical_matrix(order=["got"], cutoff=1.0e-4)
    D2 = si.read_dynamical_matrix(order=["FC"], cutoff=1.0e-4)

    assert D1._csr.spsame(D2._csr)
    D1.finalize()
    D2.finalize()
    assert np.allclose(D1._csr._D, D2._csr._D, atol=1e-5)


def test_zz_sc_geom(sisl_files):
    si = sisl.get_sile(sisl_files("gulp", "graphene_zz", "zz.gout"))
    lattice = si.read_lattice()
    geom = si.read_geometry()
    assert lattice == geom.lattice


def test_graphene_8x8_untiling(sisl_files):
    # Test untiling the graphene example 8 times
    # thanks to Xabier de Cerio for this example
    si = sisl.get_sile(sisl_files("gulp", "ancient", "graphene_8x8.gout"))
    dyn = si.read_dynamical_matrix()

    # Now untile it for different segments
    segs_y = [dyn.untile(8, 1, segment=seg) for seg in range(8)]
    segs_x = [dyn.untile(8, 0, segment=seg) for dyn, seg in zip(segs_y, range(8))]
    for dyn in segs_y:
        dyn.finalize()
    for dyn in segs_x:
        dyn.finalize()

    seg_y = segs_y.pop()
    for seg in segs_y:
        d = seg_y - seg
        assert np.allclose(d._csr._D, 0, atol=1e-6)
        # we can't assert that the sparsity patterns are the same
        # The differences are tiny, 1e-7 but they are there
        # assert seg_y.spsame(seg)

    seg_x = segs_x.pop()
    for seg in segs_x:
        d = seg_x - seg
        assert np.allclose(d._csr._D, 0, atol=1e-6)
