from __future__ import print_function, division

import pytest

from tempfile import mkstemp, mkdtemp

from sisl import Geometry, Atom
from sisl.io import fdfSileSiesta
from sisl.unit.siesta import unit_convert

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


def test_fdf1():
    f = osp.join(_C.d, 'gr.fdf')
    _C.g.write(fdfSileSiesta(f, 'w'))

    fdf = fdfSileSiesta(f)
    print(fdf)
    with fdf:

        fdf.readline()

        # Be sure that we can read it in a loop
        assert fdf.get('LatticeConstant') > 0.
        assert fdf.get('LatticeConstant') > 0.
        assert fdf.get('LatticeConstant') > 0.

        fdf.read_supercell()
        fdf.read_geometry()


def test_fdf2():
    f = osp.join(_C.d, 'gr.fdf')
    _C.g.write(fdfSileSiesta(f, 'w'))
    g = fdfSileSiesta(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, _C.g.cell)
    assert np.allclose(g.xyz, _C.g.xyz)
    for ia in g:
        assert g.atom[ia].Z == _C.g.atom[ia].Z
        assert g.atom[ia].tag == _C.g.atom[ia].tag


def test_include():
    def d(f):
        return osp.join(_C.d, f)
    f = d('file.fdf')
    with open(f, 'w') as fh:
        fh.write('Flag1 date\n')
        fh.write('# Flag2 comment\n')
        fh.write('Flag2 date2\n')
        fh.write('# Flag3 is read through < from file hello\n')
        fh.write('Flag3 Sub < hello\n')
        fh.write('Test 1. eV\n')
        fh.write('%block Hello < hello\n')
        fh.write(' %INCLUDE file2.fdf\n')

    with open(d('hello'), 'w') as fh:
        fh.write('Flag4 hello\n')
        fh.write('# Comments should be discarded\n')
        fh.write('Flag3 test\n')
        fh.write('Sub sub-test\n')

    with open(d('file2.fdf'), 'w') as fh:
        fh.write('Flag4 non\n')
        fh.write('  %incLude file3.fdf')

    with open(d('file3.fdf'), 'w') as fh:
        fh.write('Sub level\n')
        fh.write('Third level')

    fdf = fdfSileSiesta(f, base=_C.d)
    assert fdf.includes() == [d('hello'), d('file2.fdf'), d('file3.fdf')]
    assert fdf.get('Flag1') == 'date'
    assert fdf.get('Flag2') == 'date2'
    assert fdf.get('Flag3') == 'test'
    assert fdf.get('Flag4') == 'non'
    assert fdf.get('FLAG4') == 'non'
    assert fdf.get('test') == pytest.approx(unit_convert('eV', 'Ry'))
    assert fdf.get('test', 'eV') == pytest.approx(1.)
    assert fdf.get('Sub') == 'sub-test'
    assert fdf.get('Third') == 'level'
    # Read a block
    ll = open(d('hello')).readlines()
    ll.pop(1)
    assert fdf.get('Hello') == ll
