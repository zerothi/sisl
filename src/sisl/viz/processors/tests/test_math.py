# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.viz.processors.math import normalize

pytestmark = [pytest.mark.viz, pytest.mark.processors]


def test_normalize():
    data = [0, 1, 2]

    assert np.allclose(normalize(data), [0, 0.5, 1])

    assert np.allclose(normalize(data, vmin=-1, vmax=1), [-1, 0, 1])
