# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.io.siesta.times import *

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_h2o_dipole_times(sisl_files):
    f1 = sisl_files("siesta", "H2O_dipole", "TIMES")
    f2 = sisl_files("siesta", "H2O_dipole", "h2o_dipole.times")
    t1 = timesSileSiesta(f1).read_data()
    t2 = timesSileSiesta(f2).read_data()

    assert np.allclose(t1.to_array(), t2.to_array())
