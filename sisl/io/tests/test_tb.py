from __future__ import print_function, division

import pytest

from tempfile import mkstemp, mkdtemp
import warnings as warn

from sisl import Geometry, Atom
from sisl.io.ham import *

import os.path as osp
import math as m
import numpy as np
from scipy.sparse import SparseWarning

import common as tc

_C = type('Temporary', (object, ), {})


def setup_module(module):
    tc.setup(module._C)


def teardown_module(module):
    tc.teardown(module._C)


@pytest.mark.io
class TestHAM(object):

    def test_ham1(self):
        f = osp.join(_C.d, 'gr.ham')
        _C.g.write(HamiltonianSile(f, 'w'))
        g = HamiltonianSile(f).read_geometry()

        # Assert they are the same
        assert np.allclose(g.cell, _C.g.cell)
        assert np.allclose(g.xyz, _C.g.xyz)
        assert g.atom.equal(_C.g.atom, R=False)

    def test_ham2(self):
        f = osp.join(_C.d, 'gr.ham')
        _C.ham.write(HamiltonianSile(f, 'w'))
        ham = HamiltonianSile(f).read_hamiltonian()
        assert ham.spsame(_C.ham)
