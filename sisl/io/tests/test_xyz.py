from __future__ import print_function, division

import pytest

from tempfile import mkstemp, mkdtemp

from sisl import Geometry, Atom
from sisl.io.xyz import *

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
class TestXYZ(object):

    def test_xyz1(self):
        f = osp.join(_C.d, 'gr.xyz')
        _C.g.write(XYZSile(f, 'w'))
        g = XYZSile(f).read_geometry()

        # Assert they are the same
        assert np.allclose(g.cell, _C.g.cell)
        assert np.allclose(g.xyz, _C.g.xyz)
        assert _C.g.atom.equal(g.atom, R=False)
