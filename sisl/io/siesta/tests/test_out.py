from __future__ import print_function, division

import sys
import pytest
import os.path as osp
import sisl
from sisl.io.siesta.out import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.siesta, pytest.mark.only]
_dir = osp.join('sisl', 'io', 'siesta')


def test_md_nose_out(sisl_files):
    f = sisl_files(_dir, 'md_nose.out')
    out = outSileSiesta(f)

    # nspin, nk, nb
    geom0 = out.read_geometry(last=False)
    geom = out.read_geometry()
    geom1 = out.read_data(geometry=True)

    # assert it works correct
    assert isinstance(geom0, sisl.Geometry)
    assert isinstance(geom, sisl.Geometry)
    assert isinstance(geom1, sisl.Geometry)
    # assert first and last are not the same
    assert not np.allclose(geom0.xyz, geom.xyz)
    assert not np.allclose(geom0.xyz, geom1.xyz)

    # try and read all outputs
    # there are 5 outputs in this output file.
    assert len(out.read_geometry(all=True)) == 5
    assert len(out.read_force(all=True)) == 5
    assert len(out.read_stress(all=True)) == 5
    f0 = out.read_force(last=False)
    f = out.read_force()
    f1 = out.read_data(force=True)
    assert not np.allclose(f0, f)
    assert np.allclose(f1, f)

    s0 = out.read_stress(last=False)
    s = out.read_stress()
    assert not np.allclose(s0, s)

    sstatic = out.read_stress('static', all=True)
    stotal = out.read_stress('total', all=True)
    sdata = out.read_data('total', all=True, stress=True)

    for S, T, D in zip(sstatic, stotal, sdata):
        assert not np.allclose(S, T)
        assert np.allclose(D, T)


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher for ordered kwargs")
def test_md_nose_out_data(sisl_files):
    f = sisl_files(_dir, 'md_nose.out')
    out = outSileSiesta(f)

    f0, g0 = out.read_data(force=True, geometry=True)
    g1, f1 = out.read_data(geometry=True, force=True)

    assert np.allclose(f0, f1)
    assert g0 == g1
