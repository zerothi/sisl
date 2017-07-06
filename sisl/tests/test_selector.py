from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import time
import numpy as np

from sisl import TimeSelector


@attr('selector')
class TestSelector(object):

    def sleep(self, *args):
        if len(args) == 1:
            def _sleep():
                time.sleep(args[0])
            _sleep.__name__ = str(args[0])
            return _sleep
        return [self.sleep(arg) for arg in args]

    def test_selector1(self):

        sel = TimeSelector()
        sel.prepend(self.sleep(0.0001))
        sel.prepend(self.sleep(0.0002))
        sel.append(self.sleep(0.0003))

        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert_equal(i, 3)
        assert_equal(sel.best.__name__, "0.0001")

    def test_selector2(self):
        sel = TimeSelector(self.sleep(0.0001, 0.0002, 0.0003))

        while sel.best is None:
            sel()
        assert_equal(sel.best.__name__, "0.0001")

    def test_selector3(self):
        sel = TimeSelector(self.sleep(0.0003, 0.0002, 0.0001))

        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert_equal(i, 3)
        assert_equal(sel.best.__name__, "0.0001")

    def test_ordered1(self):
        sel = TimeSelector(self.sleep(0.0001, 0.0002, 0.0003), True)

        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert_equal(i, 2)
        assert_equal(sel.best.__name__, "0.0001")
        assert_equal(sel.performances[-1], None)

    def test_ordered2(self):
        sel = TimeSelector(self.sleep(0.0001, 0.0002, 0.0003), True)

        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert_equal(i, 2)
        assert_equal(sel.best.__name__, "0.0001")
        assert_equal(sel.performances[-1], None)

        sel.prepend(self.sleep(0.0002))

        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert_equal(i, 1)
        assert_equal(sel.best.__name__, "0.0001")
        assert_equal(sel.performances[-1], None)

        sel.reset()
        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert_equal(i, 3)
        assert_equal(sel.best.__name__, "0.0001")
        assert_equal(sel.performances[-1], None)

    def test_select1(self):
        sel = TimeSelector(self.sleep(0.0001, 0.0002, 0.0003))

        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert_equal(i, 3)
        assert_equal(sel.best.__name__, "0.0001")

        sel.select_best("0.0002")
        assert_equal(sel.best.__name__, "0.0002")

        idx, routine = sel.next()
        assert_equal(idx, -1)
        assert_equal(routine.__name__, "0.0002")
