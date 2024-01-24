# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from sisl._indices import indices
from sisl._ufuncs import register_sisl_dispatch
from sisl.typing import SparseMatrixExt

from .sparse import SparseCSR

# Nothing gets exposed here
__all__ = []
