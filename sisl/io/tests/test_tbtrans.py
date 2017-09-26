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
        sile = deltancSileTBtrans(f, 'w')
        H.geom.write(sile)
        sile = deltancSileTBtrans(f, 'a')

        # Write to level-1
        sile.write_delta(H)
        # Write to level-2
        sile.write_delta(H, k=[0, 0, .5])
        # Write to level-3
        sile.write_delta(H, E=0.1)
        # Write to level-4
        sile.write_delta(H, E=0.1, k=[0, 0, .5])
