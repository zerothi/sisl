""" pytest test configures """

import pytest
import os.path as osp
import sisl
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.gulp]
_dir = osp.join('sisl', 'io', 'gulp')


def test_zz_dynamical_matrix(sisl_files):
    si = sisl.get_sile(sisl_files(_dir, 'zz.gout'))
    D1 = si.read_dynamical_matrix(order=['got'], cutoff=1.e-4)
    D2 = si.read_dynamical_matrix(order=['FC'], cutoff=1.e-4)

    assert D1._csr.spsame(D2._csr)
    D1.finalize()
    D2.finalize()
    assert np.allclose(D1._csr._D, D2._csr._D, atol=1e-5)


def test_zz_sc_geom(sisl_files):
    si = sisl.get_sile(sisl_files(_dir, 'zz.gout'))
    sc = si.read_supercell()
    geom = si.read_geometry()
    assert sc == geom.sc
