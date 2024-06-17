# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.utils._arrays import *

pytestmark = pytest.mark.utils


def test_batched_indices_2d():
    ref = np.random.rand(10, 3)
    y = ref[[0, 3, 5]]
    assert np.allclose(batched_indices(ref, y)[0], [0, 3, 5])
    assert np.allclose(batched_indices(ref, y[1])[0], [3])


def test_batched_indices_3d():
    ref = np.random.rand(10, 3, 3)
    y = np.random.rand(3, 3)
    y = ref[[0, 3, 5], [0, 1, 2]]
    idx = batched_indices(ref, y)
    assert np.allclose(idx[0], [0, 3, 5])
    assert np.allclose(idx[1], [0, 1, 2])
