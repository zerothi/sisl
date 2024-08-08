# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.io.vasp.chg import *

pytestmark = [pytest.mark.io, pytest.mark.vasp]


@pytest.fixture(scope="module", params=["CHG", "CHGCAR"])
def chg_type(request):
    return request.param


@pytest.fixture(scope="module", params=[np.float64, np.float32])
def np_dtype(request):
    return request.param


def test_graphene_chg(sisl_files, chg_type, np_dtype):
    f = sisl_files("vasp", "graphene", chg_type)
    grid = chgSileVASP(f).read_grid(dtype=np_dtype)
    geom = chgSileVASP(f).read_geometry()

    assert grid.grid.sum() * grid.dvolume == pytest.approx(8)
    assert geom == grid.geometry


def test_graphene_chgcar_index_float(sisl_files):
    f = sisl_files("vasp", "graphene", "CHGCAR")
    grid = chgSileVASP(f).read_grid()
    gridh = chgSileVASP(f).read_grid(index=[0.5])

    assert grid.grid.sum() / 2 == pytest.approx(gridh.grid.sum())


@pytest.fixture(scope="module", params=["unpolarized", "polarized", "spin-orbit"])
def spin_conf(request):
    return request.param


def test_nitric_oxide_chg(sisl_files, spin_conf, chg_type, np_dtype):
    f = sisl_files("vasp", "nitric_oxide", spin_conf, chg_type)
    grid = chgSileVASP(f).read_grid(dtype=np_dtype)
    geom = chgSileVASP(f).read_geometry()

    assert grid.grid.sum() * grid.dvolume == pytest.approx(11)
    assert geom == grid.geometry


def test_nitric_oxide_pol(sisl_files, chg_type, np_dtype):
    f = sisl_files("vasp", "nitric_oxide", "polarized", chg_type)
    grid = chgSileVASP(f).read_grid(1, dtype=np_dtype)
    geom = chgSileVASP(f).read_geometry()

    assert grid.grid.sum() * grid.dvolume == pytest.approx(1, rel=1e-3)
    assert geom == grid.geometry
    # check against first raw datapoint on grid
    if chg_type == "CHG":
        assert grid.grid[0, 0, 0] * grid.volume == pytest.approx(0.00045343)
    elif chg_type == "CHGCAR":
        assert grid.grid[0, 0, 0] * grid.volume == pytest.approx(-0.00080296862896)

    # construct up-spin density
    chg = chgSileVASP(f).read_grid(0, dtype=np_dtype)
    grid.grid += chg.grid
    grid.grid /= 2
    up_spin = chgSileVASP(f).read_grid([0.5, 0.5], dtype=np_dtype)
    assert np.allclose(grid.grid, up_spin.grid)


def test_nitric_oxide_soi(sisl_files, chg_type, np_dtype):
    f = sisl_files("vasp", "nitric_oxide", "spin-orbit", chg_type)
    grids = []
    if chg_type == "CHG":
        val = [0.126, 0.00013571, 0.00014291, 0.00015183]
    elif chg_type == "CHGCAR":
        val = [0.13805890712, -0.00058291925321, -0.00059448387648, -0.00060228376205]

    for i in range(4):
        grid = chgSileVASP(f).read_grid(i, dtype=np_dtype)
        grids.append(grid)

        # check against first raw datapoint on grid
        assert grid.grid[0, 0, 0] * grid.volume == pytest.approx(val[i])

    s = grids[0].grid.sum() * grid.dvolume
    assert s == pytest.approx(11, rel=1e-3)

    s = 0
    for i in range(1, 4):
        grid = grids[i]
        s += (grid.grid.sum() * grid.dvolume) ** 2
    assert s == pytest.approx(1, rel=1e-3)

    # construct up-spin density
    grid.grid += grids[0].grid
    grid.grid /= 2
    up_spin = chgSileVASP(f).read_grid([0.5, 0, 0, 0.5], dtype=np_dtype)
    assert np.allclose(grid.grid, up_spin.grid)
