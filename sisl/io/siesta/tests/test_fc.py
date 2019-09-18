from __future__ import print_function, division

import pytest
import os.path as osp
from sisl.io.siesta.fc import *
from sisl.unit.siesta import unit_convert
import numpy as np

pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join('sisl', 'io', 'siesta')


def test_read_fc(sisl_tmp):
    f = sisl_tmp('test.FC', _dir)

    fc = np.random.rand(20, 6, 2, 3)
    sign = 1
    with open(f, 'w') as fh:
        fh.write('sotaeuha 2 0.5\n')
        for n in fc:
            for dx in n:
                for a in dx * sign:
                    fh.write('{} {} {}\n'.format(*a))
                sign *= -1
    fc.shape = (20, 3, 2, 2, 3)

    # Since the fc file re-creates the sign and divides by the length we have to do this
    fc2 = fcSileSiesta(f).read_force() / 0.5
    assert fc.shape == fc2.shape
    assert np.allclose(fc, fc2)


def test_read_fc_old(sisl_tmp):
    f = sisl_tmp('test2.FC', _dir)

    fc = np.random.rand(20, 6, 2, 3)
    with open(f, 'w') as fh:
        fh.write('sotaeuha\n')
        for n in fc:
            for dx in n:
                for a in dx:
                    fh.write('{} {} {}\n'.format(*a))
    fc.shape = (20, 3, 2, 2, 3)

    fc2 = fcSileSiesta(f).read_force() / (0.04 * unit_convert('Bohr', 'Ang'))
    assert fc.shape != fc2.shape
    fc2 *= np.tile([1, -1], 3).reshape(1, 3, 2, 1, 1)
    fc2.shape = (-1, 3, 2, 2, 3)
    assert fc.shape == fc2.shape
    assert np.allclose(fc, fc2)

    fc2 = fcSileSiesta(f).read_force_constant()
    assert fc.shape != fc2.shape
    fc2.shape = (-1, 3, 2, 2, 3)
    assert fc.shape == fc2.shape
    assert np.allclose(fc, fc2)

    # Specify number of atoms
    fc2 = fcSileSiesta(f).read_force_constant(2)
    assert fc.shape == fc2.shape
    assert np.allclose(fc, fc2)

    # Specify number of atoms and correction to check they are equivalent
    fc2 = fcSileSiesta(f).read_force(-1., na=2)
    assert fc.shape == fc2.shape
    fc2 *= np.tile([-1, 1], 3).reshape(1, 3, 2, 1, 1)
    assert np.allclose(fc, fc2)
