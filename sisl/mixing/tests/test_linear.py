# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest

import math as m
import numpy as np

from sisl.mixing import LinearMixer


pytestmark = pytest.mark.mixing


def test_linear_mixer():
    def scf(f):
        return np.cos(f)

    f = np.linspace(0, 7, 1000)
    mix = LinearMixer()
    s = str(mix)

    dmax = 1
    i = 0
    while dmax > 1e-7:
        i += 1
        df = scf(f) - f
        dmax = np.fabs(df).max()
        f = mix(f, df)
