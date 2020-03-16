import sys
import pytest
import os.path as osp
import sisl
from sisl.io.siesta.out import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.siesta]
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
    nOutputs = 5
    assert len(out.read_geometry(all=True)) == nOutputs
    assert len(out.read_force(all=True)) == nOutputs
    assert len(out.read_stress(all=True)) == nOutputs
    f0 = out.read_force(last=False)
    f = out.read_force()
    f1 = out.read_data(force=True)
    assert not np.allclose(f0, f)
    assert np.allclose(f1, f)

    #Check that we can read the different types of forces
    nAtoms = 10
    atomicF = out.read_force(all=True)
    totalF = out.read_force(total=True)
    maxF = out.read_force(max=True)
    assert atomicF.shape == (nOutputs, nAtoms, 3 )
    assert totalF.shape == (nOutputs, 3)
    assert maxF.shape == (nOutputs, )
    totalF, maxF = out.read_force(total=True, max=True)
    assert totalF.shape == (nOutputs, 3)
    assert maxF.shape == (nOutputs, )

    s0 = out.read_stress(last=False)
    s = out.read_stress()
    assert not np.allclose(s0, s)

    sstatic = out.read_stress('static', all=True)
    stotal = out.read_stress('total', all=True)
    sdata = out.read_data('total', all=True, stress=True)

    for S, T, D in zip(sstatic, stotal, sdata):
        assert not np.allclose(S, T)
        assert np.allclose(D, T)

    # Ensure SCF reads are consistent
    scf_last = out.read_scf()
    scf = out.read_scf(imd=-1)
    assert np.allclose(scf_last[-1], scf)
    for i in range(len(scf_last)):
        scf = out.read_scf(imd=i + 1)
        assert np.allclose(scf_last[i], scf)

    scf_all = out.read_scf(iscf=None, imd=-1)
    scf = out.read_scf(imd=-1)
    assert np.allclose(scf_all[-1], scf)
    for i in range(len(scf_all)):
        scf = out.read_scf(iscf=i + 1, imd=-1)
        assert np.allclose(scf_all[i], scf)


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher for ordered kwargs")
def test_md_nose_out_data(sisl_files):
    f = sisl_files(_dir, 'md_nose.out')
    out = outSileSiesta(f)

    f0, g0 = out.read_data(force=True, geometry=True)
    g1, f1 = out.read_data(geometry=True, force=True)

    assert np.allclose(f0, f1)
    assert g0 == g1
