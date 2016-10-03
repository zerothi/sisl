from __future__ import print_function, division

from nose.tools import *

from sisl.utils.ranges import *

import math as m


class TestRanges(object):

    def test_strmap(self):
        assert_equal(strmap(int, '1'), [1])
        assert_equal(strmap(int, '1,2'), [1,2])
        assert_equal(strmap(int, '1,2[0,2]'),
                     [1,(2,[0,2])])
        assert_equal(strmap(int, '1,2-3[0,2]'),
                     [1,((2,3),[0,2])])

    def test_lstranges(self):
        ranges = strmap(int, '1,2-3[0,2]')
        assert_equal(lstranges(ranges),
                     [1, [2, [0,2]], [3, [0,2]]])
        ranges = strmap(int, '1,2-4[0-4,2],6[1-3],9-10')
        assert_equal(lstranges(ranges),
                     [1,
                      [2, [0,1,2,3,4,2]],
                      [3, [0,1,2,3,4,2]],
                      [4, [0,1,2,3,4,2]],
                      [6, [1,2,3]],
                      9,10])

        
