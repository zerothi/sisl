from __future__ import print_function, division

import pytest

from tempfile import mkstemp, mkdtemp

from sisl import geom
from sisl import Geometry, Atom
from sisl.io import fdfSileSiesta, SileError
from sisl.unit.siesta import unit_convert

import os.path as osp
import math as m
import numpy as np

from sisl.io.tests import common as tc

_C = type('Temporary', (object, ), {})

pytestmark = [pytest.mark.io, pytest.mark.siesta, pytest.mark.fdf]


def setup_module(module):
    tc.setup(module._C)


def teardown_module(module):
    tc.teardown(module._C)


def d(f):
    if f == '':
        return _C.d
    return osp.join(_C.d, f)


def test_fdf1():
    f = d('gr.fdf')
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
    f = d('gr.fdf')
    _C.g.write(fdfSileSiesta(f, 'w'))
    g = fdfSileSiesta(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, _C.g.cell)
    assert np.allclose(g.xyz, _C.g.xyz)
    for ia in g:
        assert g.atom[ia].Z == _C.g.atom[ia].Z
        assert g.atom[ia].tag == _C.g.atom[ia].tag


def test_supercell():
    f = d('file.fdf')
    lines = [
        'Latticeconstant 1. Ang',
        '%block Latticevectors',
        ' 1. 1. 1.',
        ' 0. 0. 1.',
        ' 1. 0. 1.',
        '%endblock',
    ]
    with open(f, 'w') as fh:
        fh.write('\n'.join(lines))

    cell = np.array([[1.]*3, [0, 0, 1], [1, 0, 1]])
    sc = fdfSileSiesta(f).read_supercell()
    assert np.allclose(sc.cell, cell)

    lines = [
        'Latticeconstant 1. Bohr',
        '%block Latticevectors',
        ' 1. 1. 1.',
        ' 0. 0. 1.',
        ' 1. 0. 1.',
        '%endblock',
    ]
    with open(f, 'w') as fh:
        fh.write('\n'.join(lines))

    sc = fdfSileSiesta(f).read_supercell()
    assert np.allclose(sc.cell, cell * unit_convert('Bohr', 'Ang'))

    cell = np.diag([2.] * 3)
    lines = [
        'Latticeconstant 2. Ang',
        '%block Latticeparameters',
        ' 1. 1. 1. 90. 90. 90.',
        '%endblock',
    ]
    with open(f, 'w') as fh:
        fh.write('\n'.join(lines))

    sc = fdfSileSiesta(f).read_supercell()
    assert np.allclose(sc.cell, cell)


@pytest.mark.xfail(raises=SileError)
def test_supercell_fail():
    f = d('file.fdf')
    lines = [
        '%block Latticevectors',
        ' 1. 1. 1.',
        ' 0. 0. 1.',
        ' 1. 0. 1.',
        '%endblock',
    ]
    with open(f, 'w') as fh:
        fh.write('\n'.join(lines))
    fdfSileSiesta(f).read_supercell()


def test_geometry():
    f = d('file.fdf')
    sc_lines = [
        'Latticeconstant 1. Ang',
        '%block latticeparameters',
        ' 1. 1. 1. 90. 90. 90.',
        '%endblock',
    ]
    lines = [
        'NumberOfAtoms 2',
        '%block chemicalSpeciesLabel',
        ' 1 6 C',
        ' 2 12 H',
        '%endblock',
        'AtomicCoordinatesFormat Ang',
        '%block atomiccoordinatesandatomicspecies',
        ' 1. 1. 1. 1',
        ' 0. 0. 1. 1',
        ' 1. 0. 1. 2',
        '%endblock',
    ]

    with open(f, 'w') as fh:
        fh.write('\n'.join(sc_lines) + '\n')
        fh.write('\n'.join(lines))

    fdf = fdfSileSiesta(f, base=_C.d)
    g = fdf.read_geometry()
    assert g.na == 2
    assert np.allclose(g.xyz, [[1.] * 3,
                               [0, 0, 1]])
    assert g.atom[0].Z == 6
    assert g.atom[1].Z == 6

    # default read # of atoms from list
    with open(f, 'w') as fh:
        fh.write('\n'.join(sc_lines) + '\n')
        fh.write('\n'.join(lines[1:]))

    fdf = fdfSileSiesta(f, base=_C.d)
    g = fdf.read_geometry()
    assert g.na == 3
    assert np.allclose(g.xyz, [[1.] * 3,
                               [0, 0, 1],
                               [1, 0, 1]])
    assert g.atom[0].Z == 6
    assert g.atom[1].Z == 6
    assert g.atom[2].Z == 12


def test_include():
    f = d('file.fdf')
    with open(f, 'w') as fh:
        fh.write('Flag1 date\n')
        fh.write('# Flag2 comment\n')
        fh.write('Flag2 date2\n')
        fh.write('# Flag3 is read through < from file hello\n')
        fh.write('Flag3 Sub < hello\n')
        fh.write('FakeInt 1\n')
        fh.write('Test 1. eV\n')
        fh.write(' %INCLUDE file2.fdf\n')
        fh.write('TestRy 1. Ry\n')
        fh.write('%block Hello < hello\n')
        fh.write('TestLast 1. eV\n')

    with open(d('hello'), 'w') as fh:
        fh.write('Flag4 hello\n')
        fh.write('# Comments should be discarded\n')
        fh.write('Flag3 test\n')
        fh.write('Sub sub-test\n')

    with open(d('file2.fdf'), 'w') as fh:
        fh.write('Flag4 non\n')
        fh.write('FakeReal 2.\n')
        fh.write('  %incLude file3.fdf')

    with open(d('file3.fdf'), 'w') as fh:
        fh.write('Sub level\n')
        fh.write('Third level\n')
        fh.write('MyList [1 , 2 , 3]\n')

    fdf = fdfSileSiesta(f, base=_C.d)
    assert fdf.includes() == [d('hello'), d('file2.fdf'), d('file3.fdf')]
    assert fdf.get('Flag1') == 'date'
    assert fdf.get('Flag2') == 'date2'
    assert fdf.get('Flag3') == 'test'
    assert fdf.get('Flag4') == 'non'
    assert fdf.get('FLAG4') == 'non'
    assert fdf.get('Fakeint') == 1
    assert fdf.get('Fakeint', default='0') == '1'
    assert fdf.get('Fakereal') == 2.
    assert fdf.get('Fakereal', default=0.) == 2.
    assert fdf.get('test', 'eV') == pytest.approx(1.)
    assert fdf.get('test', with_unit=True)[0] == pytest.approx(1.)
    assert fdf.get('test', with_unit=True)[1] == 'eV'
    assert fdf.get('test', 'Ry') == pytest.approx(unit_convert('eV', 'Ry'))
    assert fdf.get('testRy') == pytest.approx(unit_convert('Ry', 'eV'))
    assert fdf.get('testRy', with_unit=True)[0] == pytest.approx(1.)
    assert fdf.get('testRy', with_unit=True)[1] == 'Ry'
    assert fdf.get('testRy', 'Ry') == pytest.approx(1.)
    assert fdf.get('Sub') == 'sub-test'
    assert fdf.get('Third') == 'level'
    assert fdf.get('test-last', with_unit=True)[0] == pytest.approx(1.)
    assert fdf.get('test-last', with_unit=True)[1] == 'eV'

    # Currently lists are not implemented
    #assert np.allclose(fdf.get('MyList'), np.arange(3) + 1)
    #assert np.allclose(fdf.get('MyList', default=[]), np.arange(3) + 1)

    # Read a block
    ll = open(d('hello')).readlines()
    ll.pop(1)
    assert fdf.get('Hello') == [l.replace('\n', '').strip() for l in ll]


def test_xv_preference():
    g = geom.graphene()
    g.write(d('file.fdf'))
    g.xyz[0, 0] += 1.
    g.write(d('siesta.XV'))

    g2 = fdfSileSiesta(d('file.fdf')).read_geometry(True)
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)

    g2 = fdfSileSiesta(d('file.fdf')).read_geometry(order=['fdf'])
    assert np.allclose(g.cell, g2.cell)
    g2.xyz[0, 0] += 1.
    assert np.allclose(g.xyz, g2.xyz)


def test_geom_order():
    gfdf = geom.graphene()
    gxv = gfdf.copy()
    gxv.xyz[0, 0] += 0.5
    gnc = gfdf.copy()
    gnc.xyz[0, 0] += 0.5

    gfdf.write(d('siesta.fdf'))

    # Create fdf-file
    fdf = fdfSileSiesta(d('siesta.fdf'))
    assert fdf.read_geometry(True, order=['nc']) is None
    gxv.write(d('siesta.XV'))
    gnc.write(d('siesta.nc'))

    # Should read from XV
    g = fdf.read_geometry(True)
    assert np.allclose(g.xyz, gxv.xyz)
    g = fdf.read_geometry(order=['nc', 'fdf'])
    assert np.allclose(g.xyz, gnc.xyz)
    g = fdf.read_geometry(order=['fdf', 'nc'])
    assert np.allclose(g.xyz, gfdf.xyz)
    g = fdf.read_geometry(order=['xv', 'nc'])
    assert np.allclose(g.xyz, gxv.xyz)
