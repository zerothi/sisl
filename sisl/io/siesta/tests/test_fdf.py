from pathlib import Path
import pytest
import os.path as osp
from sisl import geom
from sisl import Geometry, Atom
from sisl.io import fdfSileSiesta, SileError
from sisl.messages import SislWarning
from sisl.unit.siesta import unit_convert
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.siesta, pytest.mark.fdf,
              pytest.mark.filterwarnings("ignore:number of supercells")
]
_dir = osp.join('sisl', 'io', 'siesta')


def test_fdf1(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.fdf', _dir)
    sisl_system.g.write(fdfSileSiesta(f, 'w'))

    fdf = fdfSileSiesta(f)
    str(fdf)
    with fdf:

        fdf.readline()

        # Be sure that we can read it in a loop
        assert fdf.get('LatticeConstant') > 0.
        assert fdf.get('LatticeConstant') > 0.
        assert fdf.get('LatticeConstant') > 0.

        fdf.read_supercell()
        fdf.read_geometry()


def test_fdf2(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.fdf', _dir)
    sisl_system.g.write(fdfSileSiesta(f, 'w'))
    g = fdfSileSiesta(f).read_geometry()

    # Assert they are the same
    assert np.allclose(g.cell, sisl_system.g.cell)
    assert np.allclose(g.xyz, sisl_system.g.xyz)
    for ia in g:
        assert g.atoms[ia].Z == sisl_system.g.atoms[ia].Z
        assert g.atoms[ia].tag == sisl_system.g.atoms[ia].tag


def test_fdf_units(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.fdf', _dir)
    fdf = fdfSileSiesta(f, 'w')
    g = sisl_system.g

    for unit in ['bohr', 'ang', 'fractional', 'frac']:
        fdf.write_geometry(g, unit=unit)
        g2 = fdfSileSiesta(f).read_geometry()
        assert np.allclose(g.cell, g2.cell)
        assert np.allclose(g.xyz, g2.xyz)
        for ia in g:
            assert g.atoms[ia].Z == g2.atoms[ia].Z
            assert g.atoms[ia].tag == g2.atoms[ia].tag


def test_supercell(sisl_tmp):
    f = sisl_tmp('file.fdf', _dir)
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


def test_supercell_fail(sisl_tmp):
    f = sisl_tmp('file.fdf', _dir)
    lines = [
        '%block Latticevectors',
        ' 1. 1. 1.',
        ' 0. 0. 1.',
        ' 1. 0. 1.',
        '%endblock',
    ]
    with open(f, 'w') as fh:
        fh.write('\n'.join(lines))
    with pytest.raises(SileError):
        fdfSileSiesta(f).read_supercell()


def test_geometry(sisl_tmp):
    f = sisl_tmp('file.fdf', _dir)
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

    fdf = fdfSileSiesta(f, base=sisl_tmp.getbase())
    g = fdf.read_geometry()
    assert g.na == 2
    assert np.allclose(g.xyz, [[1.] * 3,
                               [0, 0, 1]])
    assert g.atoms[0].Z == 6
    assert g.atoms[1].Z == 6

    # default read # of atoms from list
    with open(f, 'w') as fh:
        fh.write('\n'.join(sc_lines) + '\n')
        fh.write('\n'.join(lines[1:]))

    fdf = fdfSileSiesta(f, base=sisl_tmp.getbase())
    g = fdf.read_geometry()
    assert g.na == 3
    assert np.allclose(g.xyz, [[1.] * 3,
                               [0, 0, 1],
                               [1, 0, 1]])
    assert g.atoms[0].Z == 6
    assert g.atoms[1].Z == 6
    assert g.atoms[2].Z == 12


def test_re_read(sisl_tmp):
    f = sisl_tmp('file.fdf', _dir)
    with open(f, 'w') as fh:
        fh.write('Flag1 date\n')
        fh.write('Flag1 not-date\n')
        fh.write('Flag1 not-date-2\n')
        fh.write('Flag3 true\n')

    fdf = fdfSileSiesta(f)
    for i in range(10):
        assert fdf.get('Flag1') == 'date'
    assert fdf.get('Flag3')


def test_get_set(sisl_tmp):
    f = sisl_tmp('file.fdf', _dir)
    with open(f, 'w') as fh:
        fh.write('Flag1 date\n')

    fdf = fdfSileSiesta(f)
    assert fdf.get('Flag1') == 'date'
    fdf.set('Flag1', 'not-date')
    assert fdf.get('Flag1') == 'not-date'
    fdf.set('Flag1', 'date')
    assert fdf.get('Flag1') == 'date'
    fdf.set('Flag1', 'date-date')
    assert fdf.get('Flag1') == 'date-date'
    fdf.set('Flag1', 'date-date', keep=False)


def test_get_block(sisl_tmp):
    f = sisl_tmp('file.fdf', _dir)
    with open(f, 'w') as fh:
        fh.write('%block MyBlock\n  date\n%endblock\n')

    fdf = fdfSileSiesta(f)

    assert isinstance(fdf.get('MyBlock'), list)
    assert fdf.get('MyBlock')[0] == 'date'
    assert 'block' in fdf.print("MyBlock", fdf.get("MyBlock"))


def test_include(sisl_tmp):
    f = sisl_tmp('file.fdf', _dir)
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

    hello = sisl_tmp('hello', _dir)
    with open(hello, 'w') as fh:
        fh.write('Flag4 hello\n')
        fh.write('# Comments should be discarded\n')
        fh.write('Flag3 test\n')
        fh.write('Sub sub-test\n')

    file2 = sisl_tmp('file2.fdf', _dir)
    with open(file2, 'w') as fh:
        fh.write('Flag4 non\n')
        fh.write('FakeReal 2.\n')
        fh.write('  %incLude file3.fdf')

    file3 = sisl_tmp('file3.fdf', _dir)
    with open(file3, 'w') as fh:
        fh.write('Sub level\n')
        fh.write('Third level\n')
        fh.write('MyList [1 , 2 , 3]\n')

    fdf = fdfSileSiesta(f, base=sisl_tmp.getbase())
    assert fdf.includes() == [Path(hello), Path(file2), Path(file3)]
    assert fdf.get('Flag1') == 'date'
    assert fdf.get('Flag2') == 'date2'
    assert fdf.get('Flag3') == 'test'
    assert fdf.get('Flag4') == 'non'
    assert fdf.get('FLAG4') == 'non'
    assert fdf.get('Fakeint') == 1
    assert fdf.get('Fakeint', '0') == '1'
    assert fdf.get('Fakereal') == 2.
    assert fdf.get('Fakereal', 0.) == 2.
    assert fdf.get('test', 'eV') == pytest.approx(1.)
    assert fdf.get('test', with_unit=True)[0] == pytest.approx(1.)
    assert fdf.get('test', with_unit=True)[1] == 'eV'
    assert fdf.get('test', unit='Ry') == pytest.approx(unit_convert('eV', 'Ry'))
    assert fdf.get('testRy') == pytest.approx(unit_convert('Ry', 'eV'))
    assert fdf.get('testRy', with_unit=True)[0] == pytest.approx(1.)
    assert fdf.get('testRy', with_unit=True)[1] == 'Ry'
    assert fdf.get('testRy', unit='Ry') == pytest.approx(1.)
    assert fdf.get('Sub') == 'sub-test'
    assert fdf.get('Third') == 'level'
    assert fdf.get('test-last', with_unit=True)[0] == pytest.approx(1.)
    assert fdf.get('test-last', with_unit=True)[1] == 'eV'

    # Currently lists are not implemented
    #assert np.allclose(fdf.get('MyList'), np.arange(3) + 1)
    #assert np.allclose(fdf.get('MyList', []), np.arange(3) + 1)

    # Read a block
    ll = open(sisl_tmp('hello', _dir)).readlines()
    ll.pop(1)
    assert fdf.get('Hello') == [l.replace('\n', '').strip() for l in ll]


def test_xv_preference(sisl_tmp):
    g = geom.graphene()
    g.write(sisl_tmp('file.fdf', _dir))
    g.xyz[0, 0] += 1.
    g.write(sisl_tmp('siesta.XV', _dir))

    sc = fdfSileSiesta(sisl_tmp('file.fdf', _dir)).read_supercell(True)
    g2 = fdfSileSiesta(sisl_tmp('file.fdf', _dir)).read_geometry(True)
    assert np.allclose(sc.cell, g.cell)
    assert np.allclose(g.cell, g2.cell)
    assert np.allclose(g.xyz, g2.xyz)

    g2 = fdfSileSiesta(sisl_tmp('file.fdf', _dir)).read_geometry(order=['fdf'])
    assert np.allclose(g.cell, g2.cell)
    g2.xyz[0, 0] += 1.
    assert np.allclose(g.xyz, g2.xyz)


def test_geom_order(sisl_tmp):
    gfdf = geom.graphene()
    gxv = gfdf.copy()
    gxv.xyz[0, 0] += 0.5
    gnc = gfdf.copy()
    gnc.xyz[0, 0] += 0.5

    gfdf.write(sisl_tmp('siesta.fdf', _dir))

    # Create fdf-file
    fdf = fdfSileSiesta(sisl_tmp('siesta.fdf', _dir))
    assert fdf.read_geometry(True, order=['nc']) is None
    gxv.write(sisl_tmp('siesta.XV', _dir))
    gnc.write(sisl_tmp('siesta.nc', _dir))

    # Should read from XV
    g = fdf.read_geometry(True)
    assert np.allclose(g.xyz, gxv.xyz)
    g = fdf.read_geometry(order=['nc', 'fdf'])
    assert np.allclose(g.xyz, gnc.xyz)
    g = fdf.read_geometry(order=['fdf', 'nc'])
    assert np.allclose(g.xyz, gfdf.xyz)
    g = fdf.read_geometry(order=['xv', 'nc'])
    assert np.allclose(g.xyz, gxv.xyz)


def test_geom_constraints(sisl_tmp):
    gfdf = geom.graphene().tile(2, 0).tile(2, 1)
    gfdf['CONSTRAIN'] = 0
    gfdf['CONSTRAIN-x'] = 2
    gfdf['CONSTRAIN-y'] = [1, 3, 4, 5]
    gfdf['CONSTRAIN-z'] = range(len(gfdf))

    gfdf.write(sisl_tmp('siesta.fdf', _dir))


def test_h2_dynamical_matrix(sisl_files):
    si = fdfSileSiesta(sisl_files(_dir, 'H2_dynamical_matrix.fdf'))

    trans_inv = [True, False]
    sum0 = trans_inv[:]
    hermitian = trans_inv[:]

    eV2cm = 8065.54429
    hw_true = [-88.392650, -88.392650, -0.000038, -0.000001, 0.000025, 3797.431825]

    from itertools import product
    for ti, s0, herm in product(trans_inv, sum0, hermitian):
        dyn = si.read_dynamical_matrix(trans_inv=ti, sum0=s0, hermitian=herm)
        hw = dyn.eigenvalue().hw
        if ti and s0 and herm:
            assert np.allclose(hw * eV2cm, hw_true, atol=1e-4)


def test_dry_read(sisl_tmp):
    # This test runs the read-functions. They aren't expected to actually read anything,
    # it is only a dry-run.
    file = sisl_tmp('siesta.fdf', _dir)
    geom.graphene().write(file)
    fdf = fdfSileSiesta(file)

    read_methods = set(m for m in dir(fdf) if m.startswith("read_"))
    output = dict(output=True)
    kwargs = {
        "supercell": output,
        "geometry": output,
        "grid": dict(name="rho"),
    }

    with pytest.warns(SislWarning):
        assert np.allclose(fdf.read_supercell_nsc(), (1, 1, 1))
    read_methods.remove("read_supercell_nsc")

    geom_methods = set(f"read_{x}" for x in ("basis", "supercell", "geometry"))
    read_methods -= geom_methods

    for methodname in read_methods:
        kwarg = kwargs.get(methodname[5:], dict())
        assert getattr(fdf, methodname)(**kwarg) is None

    for methodname in geom_methods:
        # Also run these, but dont assert None due to the graphene values being present
        # in the fdf. The read functions will still go dry-running through eg. nc-files.
        kwarg = kwargs.get(methodname[5:], dict())
        getattr(fdf, methodname)(**kwarg)


def test_fdf_argumentparser(sisl_tmp):
    f = sisl_tmp('file.fdf', _dir)
    with open(f, 'w') as fh:
        fh.write('Flag1 date\n')
        fh.write('Flag1 not-date\n')
        fh.write('Flag1 not-date-2\n')
        fh.write('Flag3 true\n')

    fdfSileSiesta(f).ArgumentParser()
