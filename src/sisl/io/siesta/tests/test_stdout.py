# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
from pytest import approx

import sisl
from sisl.io.siesta.fdf import *
from sisl.io.siesta.stdout import *

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_mgco3_md_out(sisl_files):
    f = sisl_files("siesta", "MgCO3_md", "RUN.out")
    out = stdoutSileSiesta(f)

    geom0 = out.read_geometry()
    geom = out.read_geometry[-1]()
    geom1 = out.read_data(geometry=True, slice=-1)

    # assert it works correct
    assert isinstance(geom0, sisl.Geometry)
    assert isinstance(geom, sisl.Geometry)
    assert isinstance(geom1, sisl.Geometry)
    # assert first and last are not the same
    assert not np.allclose(geom0.xyz, geom.xyz)
    assert not np.allclose(geom0.xyz, geom1.xyz)

    # try and read all outputs (including the final section)
    nOutputs = 6
    assert len(out.read_force[:](skip_final=False)) == nOutputs
    assert len(out.read_stress[:](skip_final=False)) == nOutputs

    # there are 5 dynamics outputs in this output file.
    nOutputs = 5
    assert len(out.read_geometry[:]()) == nOutputs
    assert len(out.read_force[:]()) == nOutputs
    assert len(out.read_stress[:]()) == nOutputs

    f0 = out.read_force()
    f = out.read_force[-1]()
    f1 = out.read_data(force=True, slice=-1)
    assert not np.allclose(f0, f)
    assert np.allclose(f1, f)

    # Check that we can read the different types of forces
    nAtoms = 10
    atomicF = out.read_force[:]()
    totalF = out.read_force[:](total=True)
    maxF = out.read_force[:](max=True)
    assert atomicF.shape == (nOutputs, nAtoms, 3)
    assert totalF.shape == (nOutputs, 3)
    assert maxF.shape == (nOutputs,)
    totalF, maxF = out.read_force(total=True, max=True)
    assert totalF.shape == (3,)
    assert maxF.shape == ()

    s0 = out.read_stress()
    s = out.read_stress[-1]()
    assert not np.allclose(s0, s)

    sstatic = out.read_stress[:]("static")
    stotal = out.read_stress[:]("total")

    for S, T in zip(sstatic, stotal):
        assert not np.allclose(S, T)


def test_mgco3_md_out_scf(sisl_files):
    f = sisl_files("siesta", "MgCO3_md", "RUN.out")
    out = stdoutSileSiesta(f)

    # Ensure SCF reads are consistent
    scf_last = out.read_scf[:]()
    scf_last2, props = out.read_scf[:](ret_header=True)
    scf = out.read_scf[-1]()
    scf_props = out.read_scf[-1](ret_header=True)
    assert scf_props[1] == props
    assert np.allclose(scf, scf_props[0])
    assert np.allclose(scf_last[-1], scf)
    for i in range(len(scf_last)):
        scf = out.read_scf[i]()
        assert np.allclose(scf_last[i], scf)
        assert np.allclose(scf_last[i], scf_last2[i])

    scf_all = out.read_scf[-1](iscf=None)
    scf = out.read_scf[-1]()
    assert np.allclose(scf_all[-1], scf)
    for i in range(len(scf_all)):
        scf = out.read_scf[-1](iscf=i + 1)
        assert np.allclose(scf_all[i], scf)


def test_mgco3_md_out_data(sisl_files):
    f = sisl_files("siesta", "MgCO3_md", "RUN.out")
    out = stdoutSileSiesta(f)

    f0, g0 = out.read_data(force=True, geometry=True)
    g1, f1, e = out.read_data(geometry=True, force=True, energy=True)

    assert np.allclose(f0, f1)
    assert g0 == g1
    assert isinstance(e, sisl.utils.PropertyDict)
    assert e.fermi == approx(-3.420926)
    assert e.xc == approx(-1218.701737)
    assert e["kinetic"] == approx(3955.286834)


def test_mgco3_md_out_info(sisl_files):
    f = sisl_files("siesta", "MgCO3_md", "RUN.out")
    out = stdoutSileSiesta(f)
    assert out.info.completed
    assert out.info.spin.is_unpolarized
    geom = out.read_geometry()
    assert out.info.na == geom.na
    assert out.info.no == geom.no


def test_mgco3_md_out_dataframe(sisl_files):
    pytest.importorskip("pandas", reason="pandas not available")
    f = sisl_files("siesta", "MgCO3_md", "RUN.out")
    out = stdoutSileSiesta(f)

    data = out.read_scf[:]()
    df = out.read_scf[:](as_dataframe=True)
    # this will read all MD-steps and only latest iscf
    assert len(data) == len(df)
    assert df.index.names == ["imd"]

    df = out.read_scf[:](iscf=None, as_dataframe=True)
    assert df.index.names == ["imd", "iscf"]
    df = out.read_scf(iscf=None, as_dataframe=True)
    assert df.index.names == ["iscf"]


def test_mgco3_md_out_energy(sisl_files):
    f = sisl_files("siesta", "MgCO3_md", "RUN.out")
    energy = stdoutSileSiesta(f).read_energy()
    assert isinstance(energy, sisl.utils.PropertyDict)
    assert hasattr(energy, "basis")
    basis = energy.basis
    assert hasattr(basis, "enthalpy")


def test_mgco3_md_pao_basis(sisl_files):
    f = sisl_files("siesta", "MgCO3_md", "RUN.out")

    block = """
Mg                    3                    # Species label, number of l-shells
 n=2   0   1                         # n, l, Nzeta
   2.346
   1.000
 n=3   0   1                         # n, l, Nzeta
   7.081
   1.000
 n=2   1   1                         # n, l, Nzeta
   2.614
   1.000
C                     2                    # Species label, number of l-shells
 n=2   0   1                         # n, l, Nzeta
   4.511
   1.000
 n=2   1   1                         # n, l, Nzeta
   5.490
   1.000
O                     2                    # Species label, number of l-shells
 n=2   0   1                         # n, l, Nzeta
   3.561
   1.000
 n=2   1   1                         # n, l, Nzeta
   4.343
   1.000
    """

    atom_orbs = fdfSileSiesta._parse_pao_basis(block)
    assert len(atom_orbs) == 3
    assert len(atom_orbs["Mg"]) == 5
    assert len(atom_orbs["C"]) == 4
    assert len(atom_orbs["O"]) == 4

    atoms = stdoutSileSiesta(f).read_basis()
    for atom in atoms:
        assert atom.orbitals == atom_orbs[atom.tag]
