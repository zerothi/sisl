import pytest

import math as m
from functools import partial

from sisl.utils.ranges import *


pytestmark = pytest.mark.utils


@pytest.mark.ranges
class TestRanges:

    @pytest.mark.parametrize("sep", ['-', ':'])
    def test_strseq(self, sep):
        ranges = strseq(int, sep.join(['1', '2', '5']))
        assert ranges == (1, 2, 5)
        ranges = strseq(int, sep.join(['1', '2']))
        assert ranges == (1, 2)
        ranges = strseq(int, sep.join(['1', '']))
        assert ranges == (1, None)
        ranges = strseq(int, sep.join(['', '4']))
        assert ranges == (None, 4)
        ranges = strseq(int, sep.join(['', '4', '']))
        assert ranges == (None, 4, None)

    def test_strmap1(self):
        assert strmap(int, '1') == [1]
        assert strmap(int, '') == [None]
        assert strmap(int, '1-') == [(1, None)]
        assert strmap(int, '-') == [(None, None)]
        assert strmap(int, '-1') == [(None, 1)]
        assert strmap(int, '-1', start=1) == [(1, 1)]
        assert strmap(int, '-', start=1, end=2) == [(1, 2)]
        assert strmap(int, '1,2') == [1, 2]
        assert strmap(int, '1,2[0,2-]') == [1, (2, [0, (2, None)])]
        assert strmap(int, '1,2-[0,-2]') == [1, ((2, None), [0, (None, 2)])]
        assert strmap(int, '1,2[0,2]') == [1, (2, [0, 2])]
        assert strmap(int, '1,2-3[0,2]') == [1, ((2, 3), [0, 2])]
        assert strmap(int, '1,[2,3][0,2]') == [1, (2, [0, 2]), (3, [0, 2])]
        assert strmap(int, '[82][10]') == [(82, [10])]
        assert strmap(int, '[82,83][10]') == [(82, [10]), (83, [10])]
        assert strmap(int, '[82,83][10-13]') == [(82, [(10, 13)]), (83, [(10, 13)])]

    def test_strmap2(self):
        with pytest.raises(ValueError):
            strmap(int, '1', sep='*')

    def test_strmap3(self):
        sm = partial(strmap, sep='c')
        assert sm(int, '1') == [1]
        assert sm(int, '1,2') == [1, 2]
        assert sm(int, '1,2{0,2}') == [1, (2, [0, 2])]
        assert sm(int, '1,2-3{0,2}') == [1, ((2, 3), [0, 2])]
        assert sm(int, '1,{2,3}{0,2}') == [1, (2, [0, 2]), (3, [0, 2])]
        assert sm(int, '{82}{10}') == [(82, [10])]
        assert sm(int, '{82,83}{10}') == [(82, [10]), (83, [10])]
        assert sm(int, '{82,83}{10-13}') == [(82, [(10, 13)]), (83, [(10, 13)])]

    def test_strmap4(self):
        with pytest.raises(ValueError):
            strmap(int, '1[oestuh]]')

    def test_strmap5(self):
        r = strmap(int, '1-', end=5)
        assert r == [(1, 5)]
        r = strmap(int, '1-', start=0, end=5)
        assert r == [(1, 5)]
        r = strmap(int, '-4', start=0, end=5)
        assert r == [(0, 4)]
        r = strmap(int, '-', start=0, end=5)
        assert r == [(0, 5)]

    def test_lstranges1(self):
        ranges = strmap(int, '1,2-3[0,2]')
        assert lstranges(ranges) == [1, [2, [0, 2]], [3, [0, 2]]]
        ranges = strmap(int, '1,2-4[0-4,2],6[1-3],9-10')
        assert lstranges(ranges) == [1,
                                     [2, [0, 1, 2, 3, 4, 2]],
                                     [3, [0, 1, 2, 3, 4, 2]],
                                     [4, [0, 1, 2, 3, 4, 2]],
                                     [6, [1, 2, 3]],
                                     9, 10]
        ranges = strmap(int, '1,[2,4][0-4,2],6[1-3],9-10')
        assert lstranges(ranges) == [1,
                                     [2, [0, 1, 2, 3, 4, 2]],
                                     [4, [0, 1, 2, 3, 4, 2]],
                                     [6, [1, 2, 3]],
                                     9, 10]
        ranges = strmap(int, '1,[2,4,6-7][0,3-4,2],6[1-3],9-10')
        assert lstranges(ranges) == [1,
                                     [2, [0, 3, 4, 2]],
                                     [4, [0, 3, 4, 2]],
                                     [6, [0, 3, 4, 2]],
                                     [7, [0, 3, 4, 2]],
                                     [6, [1, 2, 3]],
                                     9, 10]
        ranges = strmap(int, '[82,83][10-13]')
        assert lstranges(ranges) == [[82, [10, 11, 12, 13]],
                                     [83, [10, 11, 12, 13]]]
        ranges = strmap(int, ' [82,85][3]')
        assert lstranges(ranges) == [[82, [3]],
                                     [85, [3]]]
        ranges = strmap(int, '81,[82,85][3]')
        assert lstranges(ranges) == [81,
                                     [82, [3]],
                                     [85, [3]]]

    def test_lstranges2(self):
        ranges = strmap(int, '1:2:5')
        assert lstranges(ranges) == [1, 3, 5]
        ranges = strmap(int, '1-2-5')
        assert lstranges(ranges) == [1, 3, 5]

    def test_fileindex1(self):
        fname = 'hello[1]'
        assert fileindex('hehlo')[1] is None
        assert fileindex(fname)[1] == 1
        assert fileindex('hehlo[1,2]')[1] == [1, 2]
        assert fileindex('hehlo[1-2]')[1] == [1, 2]
        assert fileindex('hehlo[1[1],2]')[1] == [[1, [1]], 2]

    def test_list2str(self):
        a = list2str([2, 4, 5, 6])
        assert a == "2, 4-6"
        a = list2str([2, 4, 5, 6, 8, 9])
        assert a == "2, 4-6, 8-9"
