import pytest

from sisl import geom
from sisl.physics import MonkhorstPack
from sisl.io.siesta.kp import *

import numpy as np

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_kp_read_write(sisl_tmp):
    f = sisl_tmp('tmp.KP')
    g = geom.graphene()
    bz = MonkhorstPack(g, [10, 10, 10])
    kpSileSiesta(f, 'w').write_brillouinzone(bz)

    kpoints, weights = kpSileSiesta(f).read_data(g)
    assert np.allclose(kpoints, bz.k)
    assert np.allclose(weights, bz.weight)

    kpoints, weights = kpSileSiesta(f).read_data()
    assert np.allclose(kpoints, bz.tocartesian(bz.k))
    assert np.allclose(weights, bz.weight)

    bz2 = kpSileSiesta(f).read_brillouinzone(g.sc)
    assert np.allclose(bz2.k, bz.k)
    assert np.allclose(bz2.weight, bz.weight)


def test_rkp_read_write(sisl_tmp):
    f = sisl_tmp('tmp.RKP')
    g = geom.graphene()
    bz = MonkhorstPack(g, [10, 10, 10])
    rkpSileSiesta(f, 'w').write_brillouinzone(bz)

    kpoints, weights = rkpSileSiesta(f).read_data()
    assert np.allclose(kpoints, bz.k)
    assert np.allclose(weights, bz.weight)

    bz2 = rkpSileSiesta(f).read_brillouinzone(g.sc)
    assert np.allclose(bz2.k, bz.k)
    assert np.allclose(bz2.weight, bz.weight)
