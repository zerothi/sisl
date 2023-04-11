# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest
import os.path as osp
from sisl.io.vasp.eigenval import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.vasp]
_dir = osp.join('sisl', 'io', 'vasp')


def test_read_eigenval(sisl_files):
    f = sisl_files(_dir, 'graphene', 'EIGENVAL')
    eigs = eigenvalSileVASP(f).read_data()
