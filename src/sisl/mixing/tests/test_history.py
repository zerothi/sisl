# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

from sisl.mixing import History

pytestmark = pytest.mark.mixing


def test_simple():
    hist = History(5)

    assert len(hist) == 0
    assert hist.max_elements == 5

    hist.append(1, 2)
    assert len(hist) == 1
    assert len(hist[0]) == 2
    hist.append(1, 2, 3)
    assert len(hist) == 2
    assert len(hist[0]) == 2
    assert len(hist[1]) == 3
    hist.append(1, 2, 3)
    assert len(hist) == 3
    assert len(hist[0]) == 2
    assert len(hist[1]) == 3
    assert len(hist[2]) == 3

    # test clear
    hist.clear(0)
    assert len(hist) == 2
    hist.append(1)
    assert len(hist) == 3
    hist.clear()
    assert len(hist) == 0
