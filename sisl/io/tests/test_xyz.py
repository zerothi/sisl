from __future__ import print_function, division

from nose.tools import *

from tempfile import mkstemp, mkdtemp

from sisl import Geometry, Atom
from sisl.io.xyz import *

import os.path as osp
import math as m
import numpy as np

import common as tc


class TestXYZ(object):
    # Base test class for MaskedArrays.

    setUp = tc.setUp
    tearDown = tc.tearDown

    def test_xyz1(self):
        f = osp.join(self.d, 'gr.xyz')
        self.g.write(XYZSile(f, 'w'))
        g = XYZSile(f).read_geometry()

        # Assert they are the same
        assert_true(np.allclose(g.cell, self.g.cell))
        assert_true(np.allclose(g.xyz, self.g.xyz))
        for ia in g:
            assert_true(g.atom[ia] == self.g.atom[ia])
