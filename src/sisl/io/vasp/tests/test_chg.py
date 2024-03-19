# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import os.path as osp

import numpy as np
import pytest

from sisl.io.vasp.chg import *

pytestmark = [pytest.mark.io, pytest.mark.vasp]
_dir = osp.join("sisl", "io", "vasp")


@pytest.fixture(scope="module", params=["CHG", "CHGCAR"])
def chg_type(request):
    return request.param


@pytest.fixture(scope="module", params=[np.float64, np.float32])
def np_dtype(request):
    return request.param


def test_graphene_chg(sisl_files, chg_type, np_dtype):
    f = sisl_files(_dir, "graphene", chg_type)
    grid = chgSileVASP(f).read_grid(dtype=np_dtype)
    geom = chgSileVASP(f).read_geometry()

    assert grid.grid.sum() * grid.dvolume == pytest.approx(8)
    assert geom == grid.geometry


def test_graphene_chgcar_index_float(sisl_files):
    f = sisl_files(_dir, "graphene", "CHGCAR")
    grid = chgSileVASP(f).read_grid()
    gridh = chgSileVASP(f).read_grid(index=[0.5])

    assert grid.grid.sum() / 2 == pytest.approx(gridh.grid.sum())


@pytest.fixture(scope="module", params=["unpol", "pol", "soi"])
def spin_conf(request):
    return request.param


def test_nitric_oxide_chg(sisl_files, spin_conf, chg_type, np_dtype):
    f = sisl_files(_dir, f"nitric_oxide/{spin_conf}", chg_type + ".gz")
    grid = chgSileVASP(f).read_grid(dtype=np_dtype)
    geom = chgSileVASP(f).read_geometry()

    assert grid.grid.sum() * grid.dvolume == pytest.approx(11)
    assert geom == grid.geometry


def test_nitric_oxide_pol(sisl_files, chg_type, np_dtype):
    f = sisl_files(_dir, "nitric_oxide/pol", chg_type + ".gz")
    grid = chgSileVASP(f).read_grid(1, dtype=np_dtype)
    geom = chgSileVASP(f).read_geometry()

    assert grid.grid.sum() * grid.dvolume == pytest.approx(1, rel=1e-3)
    assert geom == grid.geometry


def test_nitric_oxide_soi(sisl_files, chg_type, np_dtype):
    f = sisl_files(_dir, "nitric_oxide/soi", chg_type + ".gz")
    s = 0
    for i in range(1, 4):
        grid = chgSileVASP(f).read_grid(i, dtype=np_dtype)
        s += (grid.grid.sum() * grid.dvolume) ** 2

    assert s == pytest.approx(1, rel=1e-3)
