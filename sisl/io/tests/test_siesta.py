from __future__ import print_function, division

import pytest

from sisl import Geometry, Atom, Hamiltonian
from sisl.io.siesta import *
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
class TestSiestanc(object):

    def test_nc1(self):
        f = osp.join(_C.d, 'gr.nc')
        tb = Hamiltonian(_C.gtb)
        tb.construct([_C.R, _C.t])
        tb.write(ncSileSiesta(f, 'w'))

        ntb = ncSileSiesta(f).read_hamiltonian()

        # Assert they are the same
        assert np.allclose(tb.cell, ntb.cell)
        assert np.allclose(tb.xyz, ntb.xyz)
        assert np.allclose(tb._csr._D[:, 0], ntb._csr._D[:, 0])
        assert _C.g.atom.equal(ntb.atom, R=False)

    def test_nc2(self):
        f = osp.join(_C.d, 'gr.dH.nc')
        H = Hamiltonian(_C.gtb)
        H.construct([_C.R, _C.t])

        # annoyingly this has to be performed like this...
        sile = dHncSileTBtrans(f, 'w')
        H.geom.write(sile)
        sile = dHncSileTBtrans(f, 'a')

        # Write to level-1
        H.write(sile)
        # Write to level-2
        H.write(sile, k=[0, 0, .5])
        # Write to level-3
        H.write(sile, E=0.1)
        # Write to level-4
        H.write(sile, k=[0, 0, .5], E=0.1)

    def test_nc3(self):
        f = osp.join(_C.d, 'grS.nc')
        tb = Hamiltonian(_C.gtb, orthogonal=False)
        tb.construct([_C.R, _C.tS])
        tb.write(ncSileSiesta(f, 'w'))

        ntb = ncSileSiesta(f).read_hamiltonian()

        # Assert they are the same
        assert np.allclose(tb.cell, ntb.cell)
        assert np.allclose(tb.xyz, ntb.xyz)
        assert np.allclose(tb._csr._D, ntb._csr._D)
        assert _C.g.atom.equal(ntb.atom, R=False)
