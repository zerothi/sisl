from __future__ import print_function, division

import pytest

from sisl import Geometry, Atom, Hamiltonian
from sisl.io.tbtrans import *

import os.path as osp
import math as m
import numpy as np

import common as tc

_C = type('Temporary', (object, ), {})


def setup_module(module):
    tc.setup(module._C)


def teardown_module(module):
    tc.teardown(module._C)


@pytest.mark.io
class TestTBtransnc(object):

    def test_nc1(self):
        f = osp.join(_C.d, 'gr.dH.nc')
        H = Hamiltonian(_C.gtb)
        H.construct([_C.R, _C.t])

        # annoyingly this has to be performed like this...
        with deltancSileTBtrans(f, 'w') as sile:
            H.geom.write(sile)
        with deltancSileTBtrans(f, 'a') as sile:

            # Write to level-1
            sile.write_delta(H)
            # Write to level-2
            sile.write_delta(H, k=[0, 0, .5])
            assert sile._get_lvl(2).variables['kpt'].shape == (1, 3)
            # Write to level-3
            sile.write_delta(H, E=0.1)
            assert sile._get_lvl(3).variables['E'].shape == (1, )
            sile.write_delta(H, E=0.2)
            assert sile._get_lvl(3).variables['E'].shape == (2, )
            # Write to level-4
            sile.write_delta(H, E=0.1, k=[0, 0, .5])
            assert sile._get_lvl(4).variables['kpt'].shape == (1, 3)
            assert sile._get_lvl(4).variables['E'].shape == (1, )
            sile.write_delta(H, E=0.2, k=[0, 0, .5])
            assert sile._get_lvl(4).variables['kpt'].shape == (1, 3)
            assert sile._get_lvl(4).variables['E'].shape == (2, )
            sile.write_delta(H, E=0.2, k=[0, 1., .5])
            assert sile._get_lvl(4).variables['kpt'].shape == (2, 3)
            assert sile._get_lvl(4).variables['E'].shape == (2, )

        with deltancSileTBtrans(f, 'r') as sile:

            # Read to level-1
            h = sile.read_delta()
            assert h.spsame(H)
            # Read level-2
            h = sile.read_delta(k=[0, 0, .5])
            assert h.spsame(H)
            # Read level-3
            h = sile.read_delta(E=0.1)
            assert h.spsame(H)
            # Read level-4
            h = sile.read_delta(E=0.1, k=[0, 0, .5])
            assert h.spsame(H)
            h = sile.read_delta(E=0.1, k=[0, 0., .5])
            assert h.spsame(H)
            h = sile.read_delta(E=0.2, k=[0, 1., .5])
            assert h.spsame(H)
