from __future__ import print_function, division

from nose.tools import *

from sisl import Geometry, Atom, Hamiltonian
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
        tb = Hamiltonian(self.gtb)
        tb.construct(self.dR, self.t)
        tb.write(SIESTASile(f, 'w'))

        ntb = SIESTASile(f).read_es()

        # Assert they are the same
        assert_true(np.allclose(tb.cell, ntb.cell))
        assert_true(np.allclose(tb.xyz, ntb.xyz))
        assert_true(np.allclose(tb._data._D, ntb._data._D))
        for ia in ntb.geom:
            assert_true(self.g.atoms[ia] == ntb.atoms[ia])
