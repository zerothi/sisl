""" pytest test configures """

import pytest
import os.path as osp
import numpy as np
import sisl


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def test_si_pdos_gamma(sisl_files):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_gamma.PDOS.xml'))
    geom, E, pdos = si.read_data()
    assert len(geom) == 2
    assert len(E) == 500
    assert pdos.shape == (geom.no, 500)


def test_si_pdos_gamma_xarray(sisl_files):
    pytest.importorskip("xarray", reason="xarray not available")
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_gamma.PDOS.xml'))
    X = si.read_data(as_dataarray=True)
    assert len(X.geometry) == 2
    assert len(X.E) == 500
    assert len(X.spin) == 1
    assert X.spin[0] == 'sum'
    size = np.product(X.shape[2:])
    assert size >= X.geometry.no


def test_si_pdos_kgrid(sisl_files):
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.PDOS.xml'))
    geom, E, pdos = si.read_data()
    assert len(geom) == 2
    assert len(E) == 500
    assert pdos.shape == (geom.no, 500)


def test_si_pdos_kgrid_xarray(sisl_files):
    pytest.importorskip("xarray", reason="xarray not available")
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.PDOS.xml'))
    X = si.read_data(as_dataarray=True)
    assert len(X.geometry) == 2
    assert len(X.E) == 500
    assert len(X.spin) == 1
    assert X.spin[0] == 'sum'
    size = np.product(X.shape[2:])
    assert size >= X.geometry.no
