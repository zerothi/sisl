from __future__ import print_function, division

import pytest

from sisl.io.vasp.chg import *

import numpy as np

pytestmark = [pytest.mark.io, pytest.mark.vasp]
_dir = 'sisl/io/vasp'


def test_graphene_chg(sisl_files):
    f = sisl_files(_dir, 'graphene/CHG')
    grid = chgSileVASP(f).read_grid()
    geom = chgSileVASP(f).read_geometry()

    assert grid.grid.sum() * grid.dvolume == pytest.approx(8)
    assert geom == grid.geometry


def test_graphene_chgcar(sisl_files):
    f = sisl_files(_dir, 'graphene/CHGCAR')
    grid = chgSileVASP(f).read_grid()
    geom = chgSileVASP(f).read_geometry()

    assert grid.grid.sum() * grid.dvolume == pytest.approx(8)
    assert geom == grid.geometry
