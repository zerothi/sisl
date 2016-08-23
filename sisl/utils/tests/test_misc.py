from __future__ import print_function, division

from nose.tools import *

from sisl.utils.misc import *

import math as m


class TestMisc(object):

    def test_direction_int(self):
        assert_true(direction(0) == 0)
        assert_true(direction(1) == 1)
        assert_true(direction(2) == 2)
        assert_false(direction(2) == 1)

    def test_direction_str(self):
        assert_true(direction('A') == 0)
        assert_true(direction('B') == 1)
        assert_true(direction('C') == 2)
        assert_true(direction('a') == 0)
        assert_true(direction('b') == 1)
        assert_true(direction('c') == 2)
        assert_true(direction('X') == 0)
        assert_true(direction('Y') == 1)
        assert_true(direction('Z') == 2)
        assert_true(direction('x') == 0)
        assert_true(direction('y') == 1)
        assert_true(direction('z') == 2)

    def test_angle_r2r(self):
        assert_almost_equal(angle('2pi'), 2*m.pi)
        assert_almost_equal(angle('2pi/2'), m.pi)
        assert_almost_equal(angle('3pi/4'), 3*m.pi/4)

        assert_almost_equal(angle('a2*180'), 2*m.pi)
        assert_almost_equal(angle('2*180', in_radians=False), 2*m.pi)

    def test_angle_a2a(self):
        assert_almost_equal(angle('a2pia'), 360)
        assert_almost_equal(angle('a2pi/2a'), 180)
        assert_almost_equal(angle('a3pi/4a'), 3*180./4)

        assert_almost_equal(angle('a2pia', True, True), 360)
        assert_almost_equal(angle('a2pi/2a', True, False), 180)
        assert_almost_equal(angle('a2pi/2a', False, True), 180)
        assert_almost_equal(angle('a2pi/2a', False, False), 180)

        
