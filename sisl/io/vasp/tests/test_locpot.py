from __future__ import print_function, division

import pytest

from sisl.io.vasp.locpot import *

import numpy as np

pytestmark = [pytest.mark.io, pytest.mark.vasp]
_dir = 'sisl/io/vasp'


def test_graphene_locpot(sisl_files):
    f = sisl_files(_dir, 'graphene/LOCPOT')
    grid = locpotSileVASP(f).read_grid()
    geom = locpotSileVASP(f).read_geometry()

    assert geom == grid.geometry
