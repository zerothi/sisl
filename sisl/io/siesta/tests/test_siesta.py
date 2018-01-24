from __future__ import print_function, division

import pytest

from sisl import Geometry, Atom, Hamiltonian
from sisl.io.siesta import *

import os.path as osp
import math as m
import numpy as np

from sisl.io.tests import common as tc

_C = type('Temporary', (object, ), {})

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def setup_module(module):
    tc.setup(module._C)


def teardown_module(module):
    tc.teardown(module._C)


def test_nc1():
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


def test_nc2():
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
