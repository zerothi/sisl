# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl_toolbox.btd import BlockMatrix

pytestmark = [pytest.mark.btd]


def test_block_matrix_dtype_empty():
    assert BlockMatrix([2, 3]).dtype == np.float64


def test_block_matrix_dtype_single():
    bm = BlockMatrix([2, 3])
    bm.block_indexer[1, 1] = np.zeros([3, 3], dtype=np.complex128)
    assert bm.dtype == np.complex128


def test_block_matrix_dtype_promote():
    bm = BlockMatrix([2, 3])
    BI = bm.block_indexer
    # the float block is inserted first, the dtype must still be complex
    BI[0, 0] = np.zeros([2, 2], dtype=np.float64)
    BI[1, 1] = np.zeros([3, 3], dtype=np.complex64)
    assert bm.dtype == np.complex128
