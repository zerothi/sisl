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

def test_read_energy(sisl_files):
    f = sisl_files(_dir, 'molecule_property.txt')
    out = txtSileORCA(f)
    E = out.read_energy(all=True)
    assert E[0].total == -129.8161893572
    assert E[1].exchange == -14.7323176552
    assert E[1].correlation == -0.4901215624
    assert E[1].correlation_nl == 0.0
    assert E[1].xc == -15.2224392176
    assert E[1].embedding == 0.0
    assert E[1].total == -129.8161893569
    E = out.read_energy(all=False)
    assert E.total == -129.8161893569

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
    assert G.atom[0].tag == 'N'
    assert G.atom[1].tag == 'O'
