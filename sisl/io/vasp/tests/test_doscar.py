import pytest
import os.path as osp
from sisl.io.vasp.doscar import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.vasp]
_dir = osp.join('sisl', 'io', 'vasp')


def test_graphene_doscar(sisl_files):
    f = sisl_files(_dir, 'graphene', 'DOSCAR')
    E, DOS = doscarSileVASP(f).read_data()
