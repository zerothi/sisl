from __future__ import print_function, division

from nose.tools import *

from tempfile import mkstemp, mkdtemp

from sisl import Geometry, Atom
from sisl.io.xv import *

import os.path as osp
import math as m
import numpy as np

import common as tc


class TestXV(object):
    # Base test class for MaskedArrays.

    setUp = tc.setUp
    tearDown = tc.tearDown

    def test_xv1(self):
        f = osp.join(self.d, 'gr.XV')
        self.g.write(XVSile(f, 'w'))
        g = XVSile(f).read_geom()

        # Assert they are the same
        assert_true(np.allclose(g.cell, self.g.cell))
        assert_true(np.allclose(g.xyz, self.g.xyz))
        for ia in g:
            assert_true(g.atoms[ia] == self.g.atoms[ia])
