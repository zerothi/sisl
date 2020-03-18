import pytest

import math as m
import numpy as np

from sisl.mixing import LinearMixer


pytestmark = pytest.mark.mixing


def test_linear_mixer():
    def scf(f):
        return np.cos(f)

    f = np.zeros(1000)
    mix = LinearMixer()

    dmax = 1
    i = 0
    while dmax > 1e-7:
        i += 1
        f_out = scf(f)
        dmax = np.fabs(f_out - f).max()
        f = mix(f, f_out)
