# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import os.path as osp

import numpy as np
import pytest

from sisl.io.siesta.fa import *

pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join("sisl", "io", "siesta")


def test_si_pdos_kgrid_fa(sisl_files):
    f = sisl_files(_dir, "si_pdos_kgrid.FA")
    fa = faSileSiesta(f).read_data()

    assert len(fa) == 2
    fa1 = faSileSiesta(f).read_force()
    assert np.allclose(fa, fa1)


def test_read_write_fa(sisl_tmp):
    f = sisl_tmp("test.FA", _dir)

    fa = np.random.rand(10, 3)
    faSileSiesta(f, "w").write_force(fa)
    fa2 = faSileSiesta(f).read_force()

    assert len(fa) == len(fa2)
    assert np.allclose(fa, fa2)
