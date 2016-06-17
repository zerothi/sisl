from __future__ import print_function, division

from nose.tools import *

from sisl import Geometry, Atom, TightBinding
from sisl.io.siesta import *

import os.path as osp
import math as m
import numpy as np

import common as tc


class TestSIESTAnc(object):
    # Base test class for MaskedArrays.

    setUp = tc.setUp
    tearDown = tc.tearDown

    def test_nc1(self):
        f = osp.join(self.d, 'gr.nc')
        tb = TightBinding(self.gtb)
        tb.construct(self.dR, self.t)
        tb.write(SIESTASile(f, 'w'))

        ntb = SIESTASile(f).read_tb()

        # Assert they are the same
        assert_true(np.allclose(tb.cell, ntb.cell))
        assert_true(np.allclose(tb.xyz, ntb.xyz))
        assert_true(np.allclose(tb._TB, ntb._TB))
        for ia in ntb.geom:
            assert_true(self.g.atoms[ia] == ntb.atoms[ia])
