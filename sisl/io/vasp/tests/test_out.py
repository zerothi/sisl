import pytest
import os.path as osp
from sisl.io.vasp.out import *
import numpy as np

pytestmark = [pytest.mark.io, pytest.mark.vasp]
_dir = osp.join('sisl', 'io', 'vasp')


def test_diamond_outcar_energies(sisl_files):
    f = sisl_files(_dir, 'diamond', 'OUTCAR')
    f = outSileVASP(f)

    E = f.read_energy()
    Eall = f.read_energy(all=True)

    assert E == Eall[-1]
    assert len(Eall) > 1
    assert f.completed()


def test_diamond_outcar_cputime(sisl_files):
    f = sisl_files(_dir, 'diamond', 'OUTCAR')
    f = outSileVASP(f)

    assert f.cpu_time() > 0.
    assert f.completed()


def test_diamond_outcar_completed(sisl_files):
    f = sisl_files(_dir, 'diamond', 'OUTCAR')
    f = outSileVASP(f)

    assert f.completed()
