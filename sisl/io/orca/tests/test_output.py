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

def test_charge_name(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    for name in ['mulliken', 'MULLIKEN', 'loewdin', 'Lowdin', 'LÖWDIN']:
        assert out.read_charge(name=name) is not None

def test_charge_mulliken_atom(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    A = out.read_charge(name='mulliken', projection='atom', all=True)
    assert len(A) == 2
    assert A[0][0, 0] == 0.029160
    assert A[0][0, 1] == 0.687779
    assert A[0][1, 0] == -0.029160
    assert A[0][1, 1] == 0.312221
    assert A[1][0, 0] == 0.029158
    assert A[1][0, 1] == 0.687793
    assert A[1][1, 0] == -0.029158
    assert A[1][1, 1] == 0.312207
    A = out.read_charge(name='mulliken', projection='atom', all=False)
    assert A[0, 0] == 0.029158
    assert A[0, 1] == 0.687793
    assert A[1, 0] == -0.029158
    assert A[1, 1] == 0.312207

def test_lowedin_atom(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    A = out.read_charge(name='loewdin', projection='atom', all=True)
    assert len(A) == 2
    assert A[0][0, 0] == -0.111221
    assert A[0][0, 1] == 0.660316
    assert A[0][1, 0] == 0.111221
    assert A[0][1, 1] == 0.339684
    assert A[1][0, 0] == -0.111223
    assert A[1][0, 1] == 0.660327
    assert A[1][1, 0] == 0.111223
    assert A[1][1, 1] == 0.339673
    A = out.read_charge(name='loewdin', projection='atom', all=False)
    assert A[0, 0] == -0.111223
    assert A[0, 1] == 0.660327
    assert A[1, 0] == 0.111223
    assert A[1, 1] == 0.339673

def test_charge_mulliken_reduced(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    A = out.read_charge(name='mulliken', projection='orbital', all=True)
    assert len(A) == 2
    # first charge block
    assert A[0][0][(0, 's')] == 3.915850
    assert A[0][0][(0, 'pz')] == 0.710261
    assert A[0][0][(1, 'dz2')] == 0.004147
    assert A[0][0][(1, 'p')] == 4.116068
    # first spin block
    assert A[0][1][(0, 'dx2y2')] == 0.001163
    assert A[0][1][(1, 'f+2')] == -0.000122
    # last charge block
    assert A[1][0][(0, 'pz')] == 0.710263
    assert A[1][0][(0, 'f0')] == 0.000681
    assert A[1][0][(1, 's')] == 3.860487
    # last spin block
    assert A[1][1][(0, 'p')] == 0.685743
    assert A[1][1][(1, 'dz2')] == -0.000163
    A = out.read_charge(name='mulliken', projection='orbital', all=False)
    # last charge block
    assert A[0][(0, 'pz')] == 0.710263
    assert A[0][(0, 'f0')] == 0.000681
    assert A[0][(1, 's')] == 3.860487
    # last spin block
    assert A[1][(0, 'p')] == 0.685743
    assert A[1][(1, 'dz2')] == -0.000163
    A = out.read_charge(name='mulliken', projection='orbital', orbital='pz', all=True)
    assert A[0][0, 0] == 0.710261
    A = out.read_charge(name='mulliken', projection='orbital', orbital='f+2', all=True)
    assert A[0][1, 1] == -0.000122
    A = out.read_charge(name='mulliken', projection='orbital', orbital='p', all=False)
    assert A[0, 1] == 0.685743

def test_charge_loewdin_reduced(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    A = out.read_charge(name='loewdin', projection='orbital', all=True)
    assert len(A) == 2
    assert A[0][0][(0, 's')] == 3.553405
    assert A[0][0][(0, 'pz')] == 0.723111
    assert A[1][0][(0, 'pz')] == 0.723113
    assert A[1][1][(1, 'pz')] == -0.010829
    A = out.read_charge(name='loewdin', projection='orbital', all=False)
    assert A[0][(0, 'f-3')] == 0.017486
    assert A[1][(1, 'pz')] == -0.010829
    assert A[0][(0, 'pz')] == 0.723113
    assert A[1][(1, 'pz')] == -0.010829
    A = out.read_charge(name='loewdin', projection='orbital', orbital='s', all=True)
    assert A[0][0, 0] == 3.553405
    A = out.read_charge(name='loewdin', projection='orbital', orbital='f-3', all=False)
    assert A[0, 0] == 0.017486
    A = out.read_charge(name='loewdin', projection='orbital', orbital='pz', all=False)
    assert A[0, 0] == 0.723113

def test_charge_mulliken_full(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    A = out.read_charge(name='mulliken', projection='orbital', reduced=False, all=True)
    assert len(A) == 2
    assert A[0][0, 0] == 0.821857
    assert A[0][0, 1] == -0.000020
    assert A[0][32, 0] == 1.174653
    assert A[0][32, 1] == -0.000200
    assert A[1][8, 0] == 0.313072
    assert A[1][8, 1] == 0.006429
    A = out.read_charge(name='mulliken', projection='orbital', reduced=False, all=False)
    assert A[8, 0] == 0.313072
    assert A[8, 1] == 0.006429

def test_charge_loewdin_full(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    A = out.read_charge(name='loewdin', projection='orbital', reduced=False, all=True)
    assert len(A) == 2
    assert A[0][0, 0] == 0.894846
    assert A[0][0, 1] == 0.000337
    assert A[0][61, 0] == 0.006054
    assert A[0][61, 1] == 0.004362
    assert A[1][8, 0] == 0.312172
    assert A[1][8, 1] == 0.005159
    A = out.read_charge(name='loewdin', projection='orbital', reduced=False, all=False)
    assert A[8, 0] == 0.312172
    assert A[8, 1] == 0.005159

