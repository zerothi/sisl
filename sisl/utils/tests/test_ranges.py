from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

from sisl.utils.ranges import *

import math as m


class TestRanges(object):

    def test_strseq(self):
        ranges = strseq(int, '1:2:5')
        assert_equal(ranges, (1, 2, 5))

    def test_strmap1(self):
        assert_equal(strmap(int, '1'), [1])
        assert_equal(strmap(int, '1,2'), [1, 2])
        assert_equal(strmap(int, '1,2[0,2]'),
                     [1, (2, [0, 2])])
        assert_equal(strmap(int, '1,2-3[0,2]'),
                     [1, ((2, 3), [0, 2])])
        assert_equal(strmap(int, '1,[2,3][0,2]'),
                     [1, (2, [0, 2]), (3, [0, 2])])
        assert_equal(strmap(int, '[82][10]'),
                     [(82, [10])])
        assert_equal(strmap(int, '[82,83][10]'),
                     [(82, [10]),
                      (83, [10])])
        assert_equal(strmap(int, '[82,83][10-13]'),
                     [(82, [(10, 13)]),
                      (83, [(10, 13)])])

    def test_lstranges1(self):
        ranges = strmap(int, '1,2-3[0,2]')
        assert_equal(lstranges(ranges),
                     [1, [2, [0, 2]], [3, [0, 2]]])
        ranges = strmap(int, '1,2-4[0-4,2],6[1-3],9-10')
        assert_equal(lstranges(ranges),
                     [1,
                      [2, [0, 1, 2, 3, 4, 2]],
                      [3, [0, 1, 2, 3, 4, 2]],
                      [4, [0, 1, 2, 3, 4, 2]],
                      [6, [1, 2, 3]],
                      9, 10])
        ranges = strmap(int, '1,[2,4][0-4,2],6[1-3],9-10')
        assert_equal(lstranges(ranges),
                     [1,
                      [2, [0, 1, 2, 3, 4, 2]],
                      [4, [0, 1, 2, 3, 4, 2]],
                      [6, [1, 2, 3]],
                      9, 10])
        ranges = strmap(int, '1,[2,4,6-7][0,3-4,2],6[1-3],9-10')
        assert_equal(lstranges(ranges),
                     [1,
                      [2, [0, 3, 4, 2]],
                      [4, [0, 3, 4, 2]],
                      [6, [0, 3, 4, 2]],
                      [7, [0, 3, 4, 2]],
                      [6, [1, 2, 3]],
                      9, 10])
        ranges = strmap(int, '[82,83][10-13]')
        assert_equal(lstranges(ranges),
                     [[82, [10, 11, 12, 13]],
                      [83, [10, 11, 12, 13]]])
        ranges = strmap(int, ' [82,85][3]')
        assert_equal(lstranges(ranges),
                     [[82, [3]],
                      [85, [3]]])
        ranges = strmap(int, '81,[82,85][3]')
        assert_equal(lstranges(ranges),
                     [81,
                      [82, [3]],
                      [85, [3]]])

    def test_lstranges2(self):
        ranges = strmap(int, '1:2:5')
        assert_equal(lstranges(ranges), [1, 3, 5])
        ranges = strmap(int, '1-2-5')
        assert_equal(lstranges(ranges), [1, 3, 5])

    def test_fileindex1(self):
        fname = 'hello[1]'
        assert_equal(fileindex('hehlo')[1], None)
        assert_equal(fileindex(fname)[1], 1)
        assert_equal(fileindex('hehlo[1,2]')[1], [1, 2])
        assert_equal(fileindex('hehlo[1-2]')[1], [1, 2])
        assert_equal(fileindex('hehlo[1[1],2]')[1], [[1, [1]], 2])

    def test_list2range(self):
        a = list2range([2, 4, 5, 6])
        assert_equal(a, "2, 4-6")
        a = list2range([2, 4, 5, 6, 8, 9])
        assert_equal(a, "2, 4-6, 8-9")
