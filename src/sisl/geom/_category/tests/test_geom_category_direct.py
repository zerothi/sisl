# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import AtomCategory

pytestmark = [pytest.mark.geom, pytest.mark.category, pytest.mark.geom_category]


def test_geom_category_direct():
    # Check that categories can be built indistinctively using the kw builder
    # or directly calling them.
    AtomCategory.kw(odd={})
    AtomCategory.kw(fx=(2, None))
