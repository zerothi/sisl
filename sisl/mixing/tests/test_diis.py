import pytest

import math as m
import numpy as np

from sisl.mixing import DIISMixer


pytestmark = pytest.mark.mixing


@pytest.mark.parametrize("history", [2, 10])
def test_diis_mixer(history):
    # test for different history lengths
    def scf(f):
        return np.cos(f)

    f = np.zeros(1000)
    mix = DIISMixer(history=history)

    dmax = 1
    i = 0
    while dmax > 1e-6:
        i += 1
        f_out = scf(f)
        dmax = np.fabs(f_out - f).max()
        f = mix(f, f_out)
