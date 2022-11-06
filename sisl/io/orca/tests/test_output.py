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


def test_completed(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    assert out.completed()
    assert out.na == 2
    assert out.no == 62
    
def test_mulliken_atom(sisl_files):
    f = sisl_files(_dir, 'molecule.output')
    out = outputSileORCA(f)
    n = 'mulliken'
    A = out.read_charge(name=n, projection='atom', all=True)
    assert A[0][0, 0] == 0.029160
    assert A[0][0, 1] == 0.687779
    assert A[0][1, 0] == -0.029160
    assert A[0][1, 1] == 0.312221
    assert A[1][0, 0] == 0.029158
    assert A[1][0, 1] == 0.687793
    assert A[1][1, 0] == -0.029158
    assert A[1][1, 1] == 0.312207
    A = out.read_charge(name=n, projection='atom', all=False)
    assert A[0, 0] == 0.029158
    assert A[0, 1] == 0.687793
    assert A[1, 0] == -0.029158
    assert A[1, 1] == 0.312207
