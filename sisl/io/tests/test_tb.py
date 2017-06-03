from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

from tempfile import mkstemp, mkdtemp
import warnings as warn

from sisl import Geometry, Atom
from sisl.io.ham import *

import os.path as osp
import math as m
import numpy as np
from scipy.sparse import SparseWarning

import common as tc


class TestHAM(object):
    # Base test class for MaskedArrays.

    setUp = tc.setUp
    tearDown = tc.tearDown

    def test_ham1(self):
        f = osp.join(self.d, 'gr.ham')
        self.g.write(HamiltonianSile(f, 'w'))
        g = HamiltonianSile(f).read_geometry()

        # Assert they are the same
        assert_true(np.allclose(g.cell, self.g.cell))
        assert_true(np.allclose(g.xyz, self.g.xyz))
        for ia in g:
            assert_true(g.atom[ia] == self.g.atom[ia])

    def test_ham2(self):
        with warn.catch_warnings():
            warn.simplefilter('ignore', category=SparseWarning)
            f = osp.join(self.d, 'gr.ham')
            self.ham.write(HamiltonianSile(f, 'w'))
            ham = HamiltonianSile(f).read_hamiltonian()
            assert_true(ham._data.spsame(self.ham._data))
