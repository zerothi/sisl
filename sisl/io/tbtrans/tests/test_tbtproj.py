""" pytest test configures """
from __future__ import print_function

import pytest
import os.path as osp
import numpy as np
import sisl


pytestmark = [pytest.mark.io, pytest.mark.tbtrans]
_dir = osp.join('sisl', 'io', 'tbtrans')


@pytest.mark.slow
def test_2_projection_content(sisl_files):
    tbt = sisl.get_sile(sisl_files(_dir, '2_projection.TBT.nc'))
    tbtp = sisl.get_sile(sisl_files(_dir, '2_projection.TBT.Proj.nc'))

    assert np.allclose(tbt.E, tbtp.E)
    assert np.allclose(tbt.kpt, tbtp.kpt)

    # Get geometry
    assert tbt.read_geometry() == tbtp.read_geometry()

    assert np.allclose(tbt.n_btd(), tbtp.n_btd())
    assert np.allclose(tbt.pivot(), tbtp.pivot())

    # Loop molecules
    left = tbt.elecs[0]
    right = tbt.elecs[1]
    for mol in tbtp.molecules:
        for proj in tbtp.projections(mol):
            t1 = tbtp.transmission((left, mol, proj), (right, mol, proj))
            t2 = tbtp.transmission('.'.join((left, mol, proj)), '.'.join((right, mol, proj)))
            assert np.allclose(t1, t2)

            te1 = tbtp.transmission_eig((left, mol, proj), (right, mol, proj))
            te2 = tbtp.transmission_eig('.'.join((left, mol, proj)), '.'.join((right, mol, proj)))
            assert np.allclose(te1, te2)
            assert np.allclose(t1, te1.sum(-1))
            assert np.allclose(t2, te2.sum(-1))

    # Check eigenstate
    es = tbtp.eigenstate('C60')
    assert len(es) == 3 # 1-HOMO, 2-LUMO
    assert (es.eig < 0.).nonzero()[0].size == 1
    assert (es.eig > 0.).nonzero()[0].size == 2
    assert np.allclose(es.norm2(), 1)


@pytest.mark.slow
def test_2_projection_tbtav(sisl_files, sisl_tmp):
    tbt = sisl.get_sile(sisl_files(_dir, '2_projection.TBT.Proj.nc'))
    f = sisl_tmp('2_projection.TBT.Proj.AV.nc', _dir)
    tbt.write_tbtav(f)


def test_2_projection_ArgumentParser(sisl_files, sisl_tmp):
    try:
        import matplotlib
    except ImportError:
        pytest.skip('matplotlib not available')

    # Create copy function
    from copy import deepcopy
    def copy(ns):
        if hasattr(ns, '_tbt'):
            del ns._tbt
        new = deepcopy(ns)
        new._tbt = tbt
        return new

    # Local routine to run the collected actions
    def run(ns):
        ns._actions_run = True
        # Run all so-far collected actions
        for A, Aargs, Akwargs in ns._actions:
            A(*Aargs, **Akwargs)
        ns._actions_run = False
        ns._actions = []

    tbt = sisl.get_sile(sisl_files(_dir, '2_projection.TBT.Proj.nc'))

    import argparse
    p, ns = tbt.ArgumentParser(argparse.ArgumentParser(conflict_handler='resolve'))

    p.parse_args([], namespace=copy(ns))
    out = p.parse_args(['--energy', ' -1.995:1.995'], namespace=copy(ns))
    assert not out._actions_run
    run(out)

    out = p.parse_args(['--norm', 'orbital'], namespace=copy(ns))
    run(out)
    assert out._norm == 'orbital'

    out = p.parse_args(['--norm', 'atom'], namespace=copy(ns))
    run(out)
    assert out._norm == 'atom'

    out = p.parse_args(['--atom', '10:11,14'], namespace=copy(ns))
    run(out)
    assert out._Ovalue == '10:11,14'
    # Only atom 14 is in the device region
    assert np.all(out._Orng + 1 == [14])

    out = p.parse_args(['--atom', '10:11,12,14:20'], namespace=copy(ns))
    run(out)
    assert out._Ovalue == '10:11,12,14:20'
    # Only 13-72 is in the device
    assert np.all(out._Orng + 1 == [14, 15, 16, 17, 18, 19, 20])

    out = p.parse_args(['--transmission', 'Left.C60.HOMO', 'Right.C60.HOMO'], namespace=copy(ns))
    run(out)
    assert len(out._data) == 2
    assert out._data_header[0][0] == 'E'
    assert out._data_header[1][0] == 'T'

    out = p.parse_args(['--molecules', '-P', 'C60'], namespace=copy(ns))
    run(out)

    out = p.parse_args(['--transmission', 'Left', 'Right.C60.LUMO',
                        '--transmission', 'Left.C60.LUMO', 'Right'], namespace=copy(ns))
    run(out)
    assert len(out._data) == 3
    assert out._data_header[0][0] == 'E'
    assert out._data_header[1][0] == 'T'
    assert out._data_header[2][0] == 'T'

    out = p.parse_args(['--ados', 'Left.C60.HOMO', '--ados', 'Left.C60.LUMO'], namespace=copy(ns))
    run(out)
    assert len(out._data) == 3
    assert out._data_header[0][0] == 'E'
    assert out._data_header[1][:2] == 'AD'
    assert out._data_header[2][:2] == 'AD'

    out = p.parse_args(['--transmission-eig', 'Left.C60.HOMO', 'Right.C60.LUMO'], namespace=copy(ns))
    run(out)
    assert out._data_header[0][0] == 'E'
    for i in range(1, len(out._data)):
        assert out._data_header[i][:4] == 'Teig'

    out = p.parse_args(['--info'], namespace=copy(ns))

    # Test output
    f = sisl_tmp('2_projection.dat', _dir)
    out = p.parse_args(['--transmission-eig', 'Left', 'Right.C60.HOMO', '--out', f], namespace=copy(ns))
    assert len(out._data) == 0

    f1 = sisl_tmp('2_projection_1.dat', _dir)
    f2 = sisl_tmp('2_projection_2.dat', _dir)
    out = p.parse_args(['--transmission', 'Left', 'Right.C60.HOMO', '--out', f1,
                        '--ados', 'Left.C60.HOMO',
                        '--atom', '13:2:72', '--ados', 'Left.C60.HOMO',
                        '--atom', '14:2:72', '--ados', 'Left.C60.HOMO', '--out', f2], namespace=copy(ns))

    d = sisl.io.tableSile(f1).read_data()
    assert len(d) == 2
    d = sisl.io.tableSile(f2).read_data()
    assert len(d) == 4
    assert np.allclose(d[1, :], d[2, :] + d[3, :])

    f = sisl_tmp('2_projection_T.png', _dir)
    out = p.parse_args(['--transmission', 'Left', 'Right.C60.HOMO',
                        '--transmission', 'Left.C60.HOMO', 'Right.C60.HOMO',
                        '--plot', f], namespace=copy(ns))
