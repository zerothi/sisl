from __future__ import print_function, division

import pytest
import os.path as osp
from sisl import Hamiltonian
from sisl.io.tbtrans import *
import numpy as np


pytestmark = [pytest.mark.io, pytest.mark.tbtrans]
_dir = osp.join('sisl', 'io', 'tbtrans')


def test_tbt_delta1(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.dH.nc', _dir)
    H = Hamiltonian(sisl_system.gtb)
    H.construct([sisl_system.R, sisl_system.t])

    # annoyingly this has to be performed like this...
    with deltancSileTBtrans(f, 'w') as sile:
        H.geom.write(sile)
    with deltancSileTBtrans(f, 'a') as sile:

        # Write to level-1
        sile.write_delta(H)
        # Write to level-2
        sile.write_delta(H, k=[0, 0, .5])
        assert sile._get_lvl(2).variables['kpt'].shape == (1, 3)
        # Write to level-3
        sile.write_delta(H, E=0.1)
        assert sile._get_lvl(3).variables['E'].shape == (1, )
        sile.write_delta(H, E=0.2)
        assert sile._get_lvl(3).variables['E'].shape == (2, )
        # Write to level-4
        sile.write_delta(H, E=0.1, k=[0, 0, .5])
        assert sile._get_lvl(4).variables['kpt'].shape == (1, 3)
        assert sile._get_lvl(4).variables['E'].shape == (1, )
        sile.write_delta(H, E=0.2, k=[0, 0, .5])
        assert sile._get_lvl(4).variables['kpt'].shape == (1, 3)
        assert sile._get_lvl(4).variables['E'].shape == (2, )
        sile.write_delta(H, E=0.2, k=[0, 1., .5])
        assert sile._get_lvl(4).variables['kpt'].shape == (2, 3)
        assert sile._get_lvl(4).variables['E'].shape == (2, )

    with deltancSileTBtrans(f, 'r') as sile:

        # Read to level-1
        h = sile.read_delta()
        assert h.spsame(H)
        # Read level-2
        h = sile.read_delta(k=[0, 0, .5])
        assert h.spsame(H)
        # Read level-3
        h = sile.read_delta(E=0.1)
        assert h.spsame(H)
        # Read level-4
        h = sile.read_delta(E=0.1, k=[0, 0, .5])
        assert h.spsame(H)
        h = sile.read_delta(E=0.1, k=[0, 0., .5])
        assert h.spsame(H)
        h = sile.read_delta(E=0.2, k=[0, 1., .5])
        assert h.spsame(H)


@pytest.mark.xfail(raises=ValueError)
def test_tbt_delta_fail(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.dH.nc', _dir)
    H = Hamiltonian(sisl_system.gtb)
    H.construct([sisl_system.R, sisl_system.t])
    H.finalize()

    with deltancSileTBtrans(f, 'w') as sile:
        sile.write_delta(H, k=[0.] * 3)
        for i in range(H.no_s):
            H[0, i] = 1.
        sile.write_delta(H, k=[0.2] * 3)


def test_tbt_delta_write_read(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.dH.nc', _dir)
    H = Hamiltonian(sisl_system.gtb, dtype=np.complex64)
    H.construct([sisl_system.R, sisl_system.t])
    H.finalize()

    with deltancSileTBtrans(f, 'w') as sile:
        sile.write_delta(H)
    with deltancSileTBtrans(f, 'r') as sile:
        h = sile.read_delta()
    assert h.spsame(H)
    assert h.dkind == H.dkind


@pytest.mark.xfail(raises=ValueError)
def test_tbt_delta_fail_list_col(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.dH.nc', _dir)
    H = Hamiltonian(sisl_system.gtb)
    H.construct([sisl_system.R, sisl_system.t])

    with deltancSileTBtrans(f, 'w') as sile:
        sile.write_delta(H, E=-1.)
        edges = H.edges(0)
        i = edges.max() + 1
        del H[0, i - 1]
        H[0, i] = 1.
        sile.write_delta(H, E=1.)


@pytest.mark.xfail(raises=ValueError)
def test_tbt_delta_fail_ncol(sisl_tmp, sisl_system):
    f = sisl_tmp('gr.dH.nc', _dir)
    H = Hamiltonian(sisl_system.gtb)
    H.construct([sisl_system.R, sisl_system.t])

    with deltancSileTBtrans(f, 'w') as sile:
        sile.write_delta(H, E=-1.)
        edges = H.edges(0)
        i = edges.max() + 1
        H[0, i] = 1.
        H.finalize()
        sile.write_delta(H, E=1.)
