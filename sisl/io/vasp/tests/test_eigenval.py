from __future__ import print_function, division

import pytest
import os.path as osp
from sisl.io.vasp.eigenval import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.vasp]
_dir = osp.join('sisl', 'io', 'vasp')


def test_read_eigenval(sisl_files):
    f = sisl_files(_dir, 'graphene', 'EIGENVAL')
    eigs = eigenvalSileVASP(f).read_data()
