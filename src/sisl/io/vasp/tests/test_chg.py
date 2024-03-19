# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import os.path as osp

import numpy as np
import pytest

from sisl.io.vasp.chg import *

pytestmark = [pytest.mark.io, pytest.mark.vasp]
_dir = osp.join("sisl", "io", "vasp")


def test_graphene_chg(sisl_files):
    f = sisl_files(_dir, "graphene", "CHG")
    grid = chgSileVASP(f).read_grid()
    gridf32 = chgSileVASP(f).read_grid(dtype=np.float32)
    geom = chgSileVASP(f).read_geometry()

    assert grid.grid.sum() * grid.dvolume == pytest.approx(8)
    assert gridf32.grid.sum() * gridf32.dvolume == pytest.approx(8)
    assert geom == grid.geometry


def test_graphene_chgcar(sisl_files):
    f = sisl_files(_dir, "graphene", "CHGCAR")
    grid = chgSileVASP(f).read_grid()
    gridf32 = chgSileVASP(f).read_grid(index=0, dtype=np.float32)
    geom = chgSileVASP(f).read_geometry()

    assert grid.grid.sum() * grid.dvolume == pytest.approx(8)
    assert gridf32.grid.sum() * gridf32.dvolume == pytest.approx(8)
    assert geom == grid.geometry


def test_graphene_chgcar_index_float(sisl_files):
    f = sisl_files(_dir, "graphene", "CHGCAR")
    grid = chgSileVASP(f).read_grid()
    gridh = chgSileVASP(f).read_grid(index=[0.5])

    assert grid.grid.sum() / 2 == pytest.approx(gridh.grid.sum())


def test_nitric_oxide_chg(sisl_files):
    for spin in "unpol", "pol", "soi":
        f = sisl_files(_dir, f"nitric_oxide/{spin}", "CHG.gz")
        grid = chgSileVASP(f).read_grid()
        gridf32 = chgSileVASP(f).read_grid(dtype=np.float32)
        geom = chgSileVASP(f).read_geometry()

        assert grid.grid.sum() * grid.dvolume == pytest.approx(11)
        assert gridf32.grid.sum() * gridf32.dvolume == pytest.approx(11)
        assert geom == grid.geometry


def test_nitric_oxide_chgcar(sisl_files):
    for spin in "unpol", "pol", "soi":
        f = sisl_files(_dir, f"nitric_oxide/{spin}", "CHGCAR.gz")
        grid = chgSileVASP(f).read_grid()
        gridf32 = chgSileVASP(f).read_grid(dtype=np.float32)
        geom = chgSileVASP(f).read_geometry()

        assert grid.grid.sum() * grid.dvolume == pytest.approx(11)
        assert gridf32.grid.sum() * gridf32.dvolume == pytest.approx(11)
        assert geom == grid.geometry


def test_nitric_oxide_pol(sisl_files):
    for fn in "CHG.gz", "CHGCAR.gz":
        f = sisl_files(_dir, f"nitric_oxide/pol", fn)
        grid = chgSileVASP(f).read_grid(1)
        gridf32 = chgSileVASP(f).read_grid(1, dtype=np.float32)
        geom = chgSileVASP(f).read_geometry()

        assert grid.grid.sum() * grid.dvolume == pytest.approx(1, rel=1e-3)
        assert gridf32.grid.sum() * gridf32.dvolume == pytest.approx(1, rel=1e-3)
        assert geom == grid.geometry


def test_nitric_oxide_soi(sisl_files):
    for fn in "CHG.gz", "CHGCAR.gz":
        f = sisl_files(_dir, f"nitric_oxide/soi", fn)
        s, sf32 = 0, 0
        for i in range(1, 4):
            grid = chgSileVASP(f).read_grid(i)
            gridf32 = chgSileVASP(f).read_grid(i, dtype=np.float32)
            s += (grid.grid.sum() * grid.dvolume) ** 2
            sf32 += (gridf32.grid.sum() * gridf32.dvolume) ** 2

        assert s == pytest.approx(1, rel=1e-3)
        assert sf32 == pytest.approx(1, rel=1e-3)
