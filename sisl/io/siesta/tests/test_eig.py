from __future__ import print_function, division

import pytest

import sisl
from sisl.io.siesta.eig import *

import numpy as np

pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = 'sisl/io/siesta'


def test_si_pdos_kgrid_eig(sisl_files):
    f = sisl_files(_dir, 'si_pdos_kgrid.EIG')
    eig = eigSileSiesta(f).read_data()

    # nspin, nk, nb
    assert np.all(eig.shape == (1, 32, 26))


def test_si_pdos_kgrid_eig_ArgumentParser(sisl_files, sisl_tmp):
    try:
        import matplotlib
    except ImportError:
        pytest.skip('matplotlib not available')
    png = sisl_tmp('si_pdos_kgrid.EIG.png', _dir)
    si = sisl.get_sile(sisl_files(_dir, 'si_pdos_kgrid.EIG'))
    p, ns = si.ArgumentParser()
    p.parse_args([], namespace=ns)
    p.parse_args(['--energy', ' -2:2'], namespace=ns)
    p.parse_args(['--energy', ' -2:2', '--plot', png], namespace=ns)
