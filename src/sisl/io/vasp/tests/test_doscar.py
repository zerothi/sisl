# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.io.vasp.doscar import *

pytestmark = [pytest.mark.io, pytest.mark.vasp]


def test_graphene_doscar(sisl_files):
    f = sisl_files("vasp", "graphene", "DOSCAR")
    E, DOS = doscarSileVASP(f).read_data()

    EHa, DOSHa = doscarSileVASP(f).read_data(units=("Ha", "Bohr"))
    assert not np.allclose(E, EHa)
    assert not np.allclose(DOS, DOSHa)
