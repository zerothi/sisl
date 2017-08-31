from __future__ import print_function, division

import pytest

from tempfile import mkstemp, mkdtemp

from sisl import Geometry, Atom
from sisl.io import fdfSileSiesta

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
class TestFDF(object):

    def test_fdf1(self):
        f = osp.join(_C.d, 'gr.fdf')
        _C.g.write(fdfSileSiesta(f, 'w'))

        fdf = fdfSileSiesta(f)
        with fdf:

            fdf.readline()

            # Be sure that we can read it in a loop
            assert fdf.get('LatticeConstant') > 0.
            assert fdf.get('LatticeConstant') > 0.
            assert fdf.get('LatticeConstant') > 0.

            fdf.read_supercell()
            fdf.read_geometry()

    def test_fdf2(self):
        f = osp.join(_C.d, 'gr.fdf')
        _C.g.write(fdfSileSiesta(f, 'w'))
        g = fdfSileSiesta(f).read_geometry()

        # Assert they are the same
        assert np.allclose(g.cell, _C.g.cell)
        assert np.allclose(g.xyz, _C.g.xyz)
        for ia in g:
            assert g.atom[ia].Z == _C.g.atom[ia].Z
            assert g.atom[ia].tag == _C.g.atom[ia].tag
