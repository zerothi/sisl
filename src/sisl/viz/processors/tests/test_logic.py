# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

from sisl.viz.processors.logic import swap

pytestmark = [pytest.mark.viz, pytest.mark.processors]


def test_swap():
    assert swap(1, (1, 2)) == 2
    assert swap(2, (1, 2)) == 1

    with pytest.raises(ValueError):
        swap(3, (1, 2))
