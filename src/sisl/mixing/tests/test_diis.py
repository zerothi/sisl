# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl.mixing import AdaptiveDIISMixer, DIISMixer

pytestmark = pytest.mark.mixing


@pytest.mark.parametrize("history", [2, 10])
def test_diis_mixer(history):
    # test for different history lengths
    def scf(f):
        return np.cos(f)

    f = np.linspace(0, 7, 1000)
    mix = DIISMixer(history=history)
    s = str(mix)

    dmax = 1
    i = 0
    while dmax > 1e-6:
        i += 1
        df = scf(f) - f
        dmax = np.fabs(df).max()
        f = mix(f, df)


@pytest.mark.parametrize("history", [2, 10])
@pytest.mark.parametrize("weight", [0.5, (0.3, 0.7)])
def test_adiis_mixer(weight, history):
    # test for different history lengths
    def scf(f):
        return np.cos(f)

    f = np.linspace(0, 7, 1000)
    mix = AdaptiveDIISMixer(weight, history=history)

    dmax = 1
    i = 0
    while dmax > 1e-6:
        i += 1
        df = scf(f) - f
        dmax = np.fabs(df).max()
        f = mix(f, df)
