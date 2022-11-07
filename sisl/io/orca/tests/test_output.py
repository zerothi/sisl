# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import sys
import pytest
import os.path as osp
import sisl
from sisl.io.orca.output import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.orca]
_dir = osp.join('sisl', 'io', 'orca')


def test_tags(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    assert out.completed()
    assert out.na == 2
    assert out.no == 62

def test_read_electrons(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    N = out.read_electrons(all=True)
    assert N[0, 0] == 7.999998537730
    assert N[0, 1] == 6.999998987205
    N = out.read_electrons(all=False)
    assert N[0] == 7.999998537734
    assert N[1] == 6.999998987209

def test_charge_name(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    for name in ['mulliken', 'MULLIKEN', 'loewdin', 'Lowdin', 'LÖWDIN']:
        assert out.read_charge(name=name) is not None

def test_charge_mulliken_atom(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    C = out.read_charge(name='mulliken', projection='atom', all=True)
    S = out.read_charge(name='mulliken', projection='atom', spin=True, all=True)
    assert len(C) == 2
    assert C[0][0] == 0.029160
    assert S[0][0] == 0.687779
    assert C[0][1] == -0.029160
    assert S[0][1] == 0.312221
    assert C[1][0] == 0.029158
    assert S[1][0] == 0.687793
    assert C[1][1] == -0.029158
    assert S[1][1] == 0.312207
    C = out.read_charge(name='mulliken', projection='atom', all=False)
    S = out.read_charge(name='mulliken', projection='atom', spin=True, all=False)
    assert C[0] == 0.029158
    assert S[0] == 0.687793
    assert C[1] == -0.029158
    assert S[1] == 0.312207

def test_lowedin_atom(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    C = out.read_charge(name='loewdin', projection='atom', all=True)
    S = out.read_charge(name='loewdin', projection='atom', spin=True, all=True)
    assert len(C) == 2
    assert C[0][0] == -0.111221
    assert S[0][0] == 0.660316
    assert C[0][1] == 0.111221
    assert S[0][1] == 0.339684
    assert C[1][0] == -0.111223
    assert S[1][0] == 0.660327
    assert C[1][1] == 0.111223
    assert S[1][1] == 0.339673
    C = out.read_charge(name='loewdin', projection='atom', all=False)
    S = out.read_charge(name='loewdin', projection='atom', spin=True, all=False)
    assert C[0] == -0.111223
    assert S[0] == 0.660327
    assert C[1] == 0.111223
    assert S[1] == 0.339673

def test_charge_mulliken_reduced(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    C = out.read_charge(name='mulliken', projection='orbital', all=True)
    S = out.read_charge(name='mulliken', projection='orbital', spin=True, all=True)
    assert len(C) == 2
    # first charge block
    assert C[0][(0, 's')] == 3.915850
    assert C[0][(0, 'pz')] == 0.710261
    assert C[0][(1, 'dz2')] == 0.004147
    assert C[0][(1, 'p')] == 4.116068
    # first spin block
    assert S[0][(0, 'dx2y2')] == 0.001163
    assert S[0][(1, 'f+2')] == -0.000122
    # last charge block
    assert C[1][(0, 'pz')] == 0.710263
    assert C[1][(0, 'f0')] == 0.000681
    assert C[1][(1, 's')] == 3.860487
    # last spin block
    assert S[1][(0, 'p')] == 0.685743
    assert S[1][(1, 'dz2')] == -0.000163
    C = out.read_charge(name='mulliken', projection='orbital', all=False)
    S = out.read_charge(name='mulliken', projection='orbital', spin=True, all=False)
    # last charge block
    assert C[(0, 'pz')] == 0.710263
    assert C[(0, 'f0')] == 0.000681
    assert C[(1, 's')] == 3.860487
    # last spin block
    assert S[(0, 'p')] == 0.685743
    assert S[(1, 'dz2')] == -0.000163
    C = out.read_charge(name='mulliken', projection='orbital', orbital='pz', all=True)
    assert C[0][0] == 0.710261
    S = out.read_charge(name='mulliken', projection='orbital', orbital='f+2', spin=True, all=True)
    assert S[0][1] == -0.000122
    S = out.read_charge(name='mulliken', projection='orbital', orbital='p', spin=True, all=False)
    assert S[0] == 0.685743

def test_charge_loewdin_reduced(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    C = out.read_charge(name='loewdin', projection='orbital', all=True)
    S = out.read_charge(name='loewdin', projection='orbital', spin=True, all=True)
    assert len(S) == 2
    assert C[0][(0, 's')] == 3.553405
    assert C[0][(0, 'pz')] == 0.723111
    assert C[1][(0, 'pz')] == 0.723113
    assert S[1][(1, 'pz')] == -0.010829
    C = out.read_charge(name='loewdin', projection='orbital', all=False)
    S = out.read_charge(name='loewdin', projection='orbital', spin=True, all=False)
    assert C[(0, 'f-3')] == 0.017486
    assert S[(1, 'pz')] == -0.010829
    assert C[(0, 'pz')] == 0.723113
    assert S[(1, 'pz')] == -0.010829
    C = out.read_charge(name='loewdin', projection='orbital', orbital='s', all=True)
    assert C[0][0] == 3.553405
    C = out.read_charge(name='loewdin', projection='orbital', orbital='f-3', all=False)
    assert C[0] == 0.017486
    C = out.read_charge(name='loewdin', projection='orbital', orbital='pz', all=False)
    assert C[0] == 0.723113

def test_charge_mulliken_full(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    C = out.read_charge(name='mulliken', projection='orbital', reduced=False, all=True)
    S = out.read_charge(name='mulliken', projection='orbital', reduced=False, spin=True, all=True)
    assert len(C) == 2
    assert C[0][0] == 0.821857
    assert S[0][0] == -0.000020
    assert C[0][32] == 1.174653
    assert S[0][32] == -0.000200
    assert C[1][8] == 0.313072
    assert S[1][8] == 0.006429
    C = out.read_charge(name='mulliken', projection='orbital', reduced=False, all=False)
    S = out.read_charge(name='mulliken', projection='orbital', reduced=False, spin=True, all=False)
    assert C[8] == 0.313072
    assert S[8] == 0.006429

def test_charge_loewdin_full(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    C = out.read_charge(name='loewdin', projection='orbital', reduced=False, all=True)
    S = out.read_charge(name='loewdin', projection='orbital', reduced=False, spin=True, all=True)
    assert len(S) == 2
    assert C[0][0] == 0.894846
    assert S[0][0] == 0.000337
    assert C[0][61] == 0.006054
    assert S[0][61] == 0.004362
    assert C[1][8] == 0.312172
    assert S[1][8] == 0.005159
    C = out.read_charge(name='loewdin', projection='orbital', reduced=False, all=False)
    S = out.read_charge(name='loewdin', projection='orbital', reduced=False, spin=True, all=False)
    assert C[8] == 0.312172
    assert S[8] == 0.005159

def test_charge_atom_unpol(sisl_files):
    f = sisl_files(_dir, 'molecule2.output')
    out = outputSileORCA(f)
    C = out.read_charge(name='mulliken', projection='atom', all=True)
    S = out.read_charge(name='mulliken', projection='atom', spin=True, all=True)
    assert len(C) == 2
    assert len(S) == 0
    assert C[0][0] == -0.037652
    C = out.read_charge(name='mulliken', projection='atom', all=False)
    S = out.read_charge(name='mulliken', projection='atom', spin=True, all=False)
    assert C[0] == -0.037652
    assert S is None
    C = out.read_charge(name='loewdin', projection='atom', all=False)
    S = out.read_charge(name='loewdin', projection='atom', spin=True, all=False)
    assert C[0] == -0.259865
    assert S is None

def test_charge_orbital_reduced_unpol(sisl_files):
    f = sisl_files(_dir, 'molecule2.output')
    out = outputSileORCA(f)
    C = out.read_charge(name='mulliken', projection='orbital', all=True)
    S = out.read_charge(name='mulliken', projection='orbital', spin=True, all=True)
    assert len(C) == 2
    assert len(S) == 0
    assert C[0][(0, "py")] == 0.534313
    assert C[1][(1, "px")] == 1.346363
    C = out.read_charge(name='mulliken', projection='orbital', all=False)
    S = out.read_charge(name='mulliken', projection='orbital', spin=True, all=False)
    assert C[(0, "px")] == 0.954436
    assert S is None
    C = out.read_charge(name='mulliken', projection='orbital', orbital='px', all=False)
    S = out.read_charge(name='mulliken', projection='orbital', orbital='px', spin=True, all=False)
    assert C[0] == 0.954436
    assert S is None
    C = out.read_charge(name='loewdin', projection='orbital', all=False)
    S = out.read_charge(name='loewdin', projection='orbital', spin=True, all=False)
    assert C[(0, "d")] == 0.315910
    assert S is None
    C = out.read_charge(name='loewdin', projection='orbital', orbital='d', all=False)
    S = out.read_charge(name='loewdin', projection='orbital', orbital='d', spin=True, all=False)
    assert C[0] == 0.315910
    assert S is None

def test_charge_orbital_full_unpol(sisl_files):
    f = sisl_files(_dir, 'molecule2.output')
    out = outputSileORCA(f)
    C = out.read_charge(name='mulliken', projection='orbital', reduced=False)
    S = out.read_charge(name='mulliken', projection='orbital', reduced=False, spin=True)
    assert C is None
    assert S is None

def test_read_energy(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    E = out.read_energy(all=True, convert=False)
    assert E[0].xc == -15.222438585593
    assert E[1].xc == -15.222439217603
    E = out.read_energy()
    assert E.xc != -15.222439217603

def test_read_energy_vdw(sisl_files):
    f = sisl_files(_dir, 'molecule2.output')
    out = outputSileORCA(f)
    E = out.read_energy(all=True, convert=False)
    assert E[0].exchange == -13.310141538373
    assert E[1].exchange == -13.310144803077
    assert E[1].vdw == -0.000410877
    E = out.read_energy()
    assert E.exchange != -13.310144803077
    assert E.vdw != -0.000410877
