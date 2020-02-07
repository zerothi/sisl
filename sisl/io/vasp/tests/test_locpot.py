import pytest
import os.path as osp
from sisl.io.vasp.locpot import *
import numpy as np

pytestmark = [pytest.mark.io, pytest.mark.vasp]
_dir = osp.join('sisl', 'io', 'vasp')


def test_graphene_locpot(sisl_files):
    f = sisl_files(_dir, 'graphene', 'LOCPOT')
    gridf64 = locpotSileVASP(f).read_grid()
    gridf32 = locpotSileVASP(f).read_grid(dtype=np.float32)
    geom = locpotSileVASP(f).read_geometry()

    assert gridf64.dtype == np.float64
    assert gridf32.dtype == np.float32
    assert geom == gridf32.geometry


def test_graphene_locpot_index_float(sisl_files):
    f = sisl_files(_dir, 'graphene', 'LOCPOT')
    grid = locpotSileVASP(f).read_grid()
    gridh = locpotSileVASP(f).read_grid(index=[0.5])

    assert grid.grid.sum() / 2 == pytest.approx(gridh.grid.sum())
