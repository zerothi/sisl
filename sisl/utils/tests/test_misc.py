from __future__ import print_function, division

from nose.tools import *

from sisl.utils.misc import *

import math as m


class TestMisc(object):

    def test_dir2dir_int(self):
        assert_true(dir2dir(0) == 0)
        assert_true(dir2dir(1) == 1)
        assert_true(dir2dir(2) == 2)
        assert_false(dir2dir(2) == 1)

    def test_dir2dir_str(self):
        assert_true(dir2dir('A') == 0)
        assert_true(dir2dir('B') == 1)
        assert_true(dir2dir('C') == 2)
        assert_true(dir2dir('a') == 0)
        assert_true(dir2dir('b') == 1)
        assert_true(dir2dir('c') == 2)
        assert_true(dir2dir('X') == 0)
        assert_true(dir2dir('Y') == 1)
        assert_true(dir2dir('Z') == 2)
        assert_true(dir2dir('x') == 0)
        assert_true(dir2dir('y') == 1)
        assert_true(dir2dir('z') == 2)

    def test_str2angle_r2r(self):
        assert_almost_equal(str2angle('2pi'), 2*m.pi)
        assert_almost_equal(str2angle('2pi/2'), m.pi)
        assert_almost_equal(str2angle('3pi/4'), 3*m.pi/4)

        assert_almost_equal(str2angle('a2*180'), 2*m.pi)
        assert_almost_equal(str2angle('2*180', in_radians=False), 2*m.pi)

    def test_str2angle_a2a(self):
        assert_almost_equal(str2angle('a2pia'), 360)
        assert_almost_equal(str2angle('a2pi/2a'), 180)
        assert_almost_equal(str2angle('a3pi/4a'), 3*180./4)

        assert_almost_equal(str2angle('a2pia', True, True), 360)
        assert_almost_equal(str2angle('a2pi/2a', True, False), 180)
        assert_almost_equal(str2angle('a2pi/2a', False, True), 180)
        assert_almost_equal(str2angle('a2pi/2a', False, False), 180)

        
