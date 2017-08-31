from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import Geometry, Atom, SuperCell, SuperCellChild
from sisl import BrillouinZone, PathBZ
from sisl import MonkhorstPackBZ


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            self.s1 = SuperCell(1, nsc=[3, 3, 1])
            self.s2 = SuperCell([2, 2, 10, 90, 90, 60], [5, 5, 1])
    return t()


@pytest.mark.brillouinzone
@pytest.mark.bz
class TestBrillouinZone(object):

    def setUp(self, setup):
        setup.s1 = SuperCell(1, nsc=[3, 3, 1])
        setup.s2 = SuperCell([2, 2, 10, 90, 90, 60], [5, 5, 1])

    def test_bz1(self, setup):
        bz = BrillouinZone(1.)
        bz.weight
        bz = BrillouinZone(setup.s1)
        assert len(bz) == 1
        assert np.allclose(bz.tocartesian([0, 0, 0]), [0] * 3)
        assert np.allclose(bz.tocartesian([0.5, 0, 0]), [m.pi, 0, 0])
        assert np.allclose(bz.toreduced([0, 0, 0]), [0] * 3)
        assert np.allclose([0.5, 0, 0], bz.tocartesian(bz.toreduced([0.5, 0, 0])))
        for k in bz:
            assert np.allclose(k, np.zeros(3))

    def test_class1(self, setup):
        class Test(SuperCellChild):
            def __init__(self, sc):
                self.set_supercell(sc)
            def eigh(self, k, *args, **kwargs):
                return np.arange(3)
            def eig(self, k, *args, **kwargs):
                return np.arange(3) - 1
        bz = BrillouinZone(Test(setup.s1))
        assert np.allclose(bz.eigh(), np.arange(3))
        assert np.allclose(bz.eig(), np.arange(3)-1)

    def test_class2(self, setup):
        class Test(SuperCellChild):
            def __init__(self, sc):
                self.set_supercell(sc)
            def eigh(self, k, *args, **kwargs):
                return np.arange(3)
            def eig(self, k, *args, **kwargs):
                return np.arange(3) - 1
        bz = BrillouinZone(Test(setup.s1))
        # Yields
        bz.yields()
        for val in bz.eigh():
            assert np.allclose(val, np.arange(3))
        for val in bz.eig():
            assert np.allclose(val, np.arange(3) - 1)
        # Average
        assert np.allclose(bz.average().eigh(), np.arange(3))

    def test_mp1(self, setup):
        bz = MonkhorstPackBZ(setup.s1, [2] * 3)
        assert len(bz) == 8
        assert bz.weight[0] == 1. / 8

    def test_mp2(self, setup):
        bz1 = MonkhorstPackBZ(setup.s1, [2] * 3)
        assert len(bz1) == 8
        bz2 = MonkhorstPackBZ(setup.s1, [2] * 3, displacement=[.5] * 3)
        assert len(bz2) == 8
        assert not np.allclose(bz1.k, bz2.k)

    def test_mp3(self, setup):
        bz1 = MonkhorstPackBZ(setup.s1, [2] * 3, size=0.5)
        assert len(bz1) == 8
        assert np.all(bz1.k < 0.25)

    def test_pbz1(self, setup):
        bz = PathBZ(setup.s1, [[0]*3, [.5]*3], 300)
        assert len(bz) == 300

        bz2 = PathBZ(setup.s1, [[0]*2, [.5]*2], 300, ['A', 'C'])
        assert len(bz) == 300

        bz3 = PathBZ(setup.s1, [[0]*2, [.5]*2], [150] * 2)
        assert len(bz) == 300
        bz.lineartick()
        bz.lineark()
        bz.lineark(True)

    def test_pbz2(self, setup):
        bz = PathBZ(setup.s1, [[0]*3, [.25]*3, [.5]*3], 300)
        assert len(bz) == 300
