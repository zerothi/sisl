from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np

from sisl import Geometry, Atom, SuperCell
from sisl import BrillouinZone, PathBZ


@attr('brillouinzone')
@attr('bz')
class TestBrillouinZone(object):

    def setUp(self):
        self.s1 = SuperCell(1, nsc=[3, 3, 1])
        self.s2 = SuperCell([2, 2, 10, 90, 90, 60], [5, 5, 1])

    def test_bz1(self):
        bz = BrillouinZone(self.s1)
        assert_true(np.all(bz([0, 0, 0]) == [0] * 3))
        assert_true(np.all(bz([0.5, 0, 0]) == [m.pi, 0, 0]))
