# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import sys
import pytest
import os.path as osp
import sisl
from sisl.io.orca.txt import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.orca]
_dir = osp.join('sisl', 'io', 'orca')


def test_tags(sisl_files):
    f = sisl_files(_dir, 'molecule_property.txt')
    out = txtSileORCA(f)
    assert out.na == 2
    assert out.no == None

def test_read_electrons(sisl_files):
    f = sisl_files(_dir, 'molecule_property.txt')
    out = txtSileORCA(f)
    N = out.read_electrons(all=True)
    assert N[0, 0] == 7.9999985377
    assert N[0, 1] == 6.9999989872
    N = out.read_electrons(all=False)
    assert N[0] == 7.9999985377
    assert N[1] == 6.9999989872

def test_read_energy(sisl_files):
    f = sisl_files(_dir, 'molecule_property.txt')
    out = txtSileORCA(f)
    E = out.read_energy(all=True)
    assert len(E) == 2
    E = out.read_energy(all=False)
    assert pytest.approx(E.total) == -3532.4797529097723

def test_read_energy_vdw(sisl_files):
    f = sisl_files(_dir, 'molecule2_property.txt')
    out = txtSileORCA(f)
    E = out.read_energy(all=True)
    assert len(E) == 2
    assert pytest.approx(E[0].total) == -3081.2651523095283
    assert pytest.approx(E[1].total) == -3081.2651523149702
    assert pytest.approx(E[1].vdw) == -0.011180550414138613
    E = out.read_energy()
    assert pytest.approx(E.total) == -3081.2651523149702
    assert pytest.approx(E.vdw) == -0.011180550414138613

def test_read_geometry(sisl_files):
    f = sisl_files(_dir, 'molecule_property.txt')
    out = txtSileORCA(f)
    G = out.read_geometry(all=True)
    assert G[0].xyz[0, 0] == 0.421218019838
    assert G[0].xyz[1, 0] == 1.578781980162
    assert G[1].xyz[0, 0] == 0.421218210279
    assert G[1].xyz[1, 0] == 1.578781789721
    G = out.read_geometry(all=False)
    assert G.xyz[0, 0] == 0.421218210279
    assert G.xyz[1, 0] == 1.578781789721
    assert G.xyz[0, 1] == 0.0
    assert G.xyz[1, 1] == 0.0
    assert G.atoms[0].tag == 'N'
    assert G.atoms[1].tag == 'O'

def test_multiple_calls(sisl_files):
    f = sisl_files(_dir, 'molecule_property.txt')
    out = txtSileORCA(f)
    N = out.read_electrons(all=True)
    assert len(N) == 2
    E = out.read_energy(all=True)
    assert len(E) == 2
    G = out.read_geometry(all=True)
    assert len(G) == 2
    N = out.read_electrons(all=True)
    assert len(N) == 2
