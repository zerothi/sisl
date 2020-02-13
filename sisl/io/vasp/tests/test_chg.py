import pytest
import os.path as osp
from sisl.io.vasp.chg import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.vasp]
_dir = osp.join('sisl', 'io', 'vasp')


def test_graphene_chg(sisl_files):
    f = sisl_files(_dir, 'graphene', 'CHG')
    grid = chgSileVASP(f).read_grid()
    gridf32 = chgSileVASP(f).read_grid(dtype=np.float32)
    geom = chgSileVASP(f).read_geometry()

    assert grid.grid.sum() * grid.dvolume == pytest.approx(8)
    assert gridf32.grid.sum() * gridf32.dvolume == pytest.approx(8)
    assert geom == grid.geometry


def test_graphene_chgcar(sisl_files):
    f = sisl_files(_dir, 'graphene', 'CHGCAR')
    grid = chgSileVASP(f).read_grid()
    gridf32 = chgSileVASP(f).read_grid(index=0, dtype=np.float32)
    geom = chgSileVASP(f).read_geometry()

    assert grid.grid.sum() * grid.dvolume == pytest.approx(8)
    assert gridf32.grid.sum() * gridf32.dvolume == pytest.approx(8)
    assert geom == grid.geometry


def test_graphene_chgcar_index_float(sisl_files):
    f = sisl_files(_dir, 'graphene', 'CHGCAR')
    grid = chgSileVASP(f).read_grid()
    gridh = chgSileVASP(f).read_grid(index=[0.5])

    assert grid.grid.sum() / 2 == pytest.approx(gridh.grid.sum())
