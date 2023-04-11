# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest
import sys

import time
import numpy as np

from sisl import TimeSelector


@pytest.mark.selector
@pytest.mark.xfail(sys.platform.startswith("darwin"),
                   reason="Sleep on MacOS is not consistent causing erroneous fails.")
class TestSelector:

    def sleep(self, *args):
        if len(args) == 1:
            def _sleep():
                time.sleep(1. / 100 * args[0])
            _sleep.__name__ = str(args[0])
            return _sleep
        return [self.sleep(arg) for arg in args]

    def test_selector1(self):

        sel = TimeSelector()
        sel.prepend(self.sleep(1))
        sel.prepend(self.sleep(2))
        sel.append(self.sleep(3))
        str(sel)

        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert i == 3
        assert sel.best.__name__ == "1"

    def test_selector2(self):
        sel = TimeSelector(self.sleep(1, 2, 3))

        while sel.best is None:
            sel()
        assert sel.best.__name__ == "1"

    def test_selector3(self):
        sel = TimeSelector(self.sleep(3, 2, 1))

        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert i == 3
        assert sel.best.__name__ == "1"

    def test_ordered1(self):
        sel = TimeSelector(self.sleep(1, 2, 3), True)

        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert i == 2
        assert sel.best.__name__ == "1"
        assert sel.metric[-1] is None

    def test_ordered2(self):
        sel = TimeSelector(self.sleep(1, 2, 3), True)

        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert i == 2
        assert sel.best.__name__ == "1"
        assert sel.metric[-1] is None

        sel.prepend(self.sleep(2))

        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert i == 1
        assert sel.best.__name__ == "1"
        assert sel.metric[-1] is None

        sel.reset()
        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert i == 3
        assert sel.best.__name__ == "1"
        assert sel.metric[-1] is None

    def test_select1(self):
        sel = TimeSelector(self.sleep(1, 2, 3))

        i = 0
        while sel.best is None:
            sel()
            i += 1
        assert i == 3
        assert sel.best.__name__ == "1"

        sel.select_best("2")
        assert sel.best.__name__ == "2"

        idx, routine = sel.next()
        assert idx == -1
        assert routine.__name__ == "2"
