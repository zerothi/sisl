import pytest
import os.path as osp
from sisl.io.siesta.orb_indx import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def test_si_pdos_kgrid_orb_indx(sisl_files):
    f = sisl_files(_dir, 'si_pdos_kgrid.ORB_INDX')
    nsc = orbindxSileSiesta(f).read_supercell_nsc()
    assert np.all(nsc > 1)
    atoms = orbindxSileSiesta(f).read_basis()

    assert len(atoms) == 2
    assert len(atoms[0]) == 13
    assert len(atoms[1]) == 13
