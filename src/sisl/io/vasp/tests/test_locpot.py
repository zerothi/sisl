# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.io.vasp.locpot import *

pytestmark = [pytest.mark.io, pytest.mark.vasp]


def test_graphene_locpot(sisl_files):
    f = sisl_files("vasp", "graphene", "LOCPOT")
    gridf64 = locpotSileVASP(f).read_grid()
    gridf32 = locpotSileVASP(f).read_grid(dtype=np.float32)
    geom = locpotSileVASP(f).read_geometry()

    assert gridf64.dtype == np.float64
    assert gridf32.dtype == np.float32
    assert geom == gridf32.geometry

    gridHa = locpotSileVASP(f).read_grid(units="Ha")
    assert not np.allclose(gridf64.grid, gridHa.grid)


def test_graphene_locpot_index_float(sisl_files):
    f = sisl_files("vasp", "graphene", "LOCPOT")
    grid = locpotSileVASP(f).read_grid()
    gridh = locpotSileVASP(f).read_grid(index=[0.5])

    assert grid.grid.sum() / 2 == pytest.approx(gridh.grid.sum())
