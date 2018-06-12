from __future__ import print_function, division

import pytest

from sisl import geom
from sisl.physics import MonkhorstPack
from sisl.io.siesta.kp import *
from sisl.unit.siesta import unit_convert

import numpy as np

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_kp_read_write(sisl_tmp):
    f = sisl_tmp('tmp.KP')
    g = geom.graphene()
    bz = MonkhorstPack(g, [10, 10, 10])
    kpSileSiesta(f, 'w').write_data(bz)

    kpoints, weights = kpSileSiesta(f).read_data()
    assert np.allclose(kpoints * unit_convert('Bohr', 'Ang'), bz.k)
    assert np.allclose(weights, bz.weight)
