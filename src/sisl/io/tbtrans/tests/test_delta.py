# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import Hamiltonian
from sisl.io.tbtrans import *

pytestmark = [pytest.mark.io, pytest.mark.tbtrans]

netCDF4 = pytest.importorskip("netCDF4")


def test_tbt_delta1(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.dH.nc")
    H = Hamiltonian(sisl_system.gtb)
    H.construct([sisl_system.R, sisl_system.t])

    # annoyingly this has to be performed like this...
    with deltancSileTBtrans(f, "w") as sile:
        H.geometry.write(sile)
    with deltancSileTBtrans(f, "a") as sile:
        # Write to level-1
        sile.write_delta(H)
        # Write to level-2
        sile.write_delta(H, k=[0, 0, 0.5])
        assert sile._get_lvl(2).variables["kpt"].shape == (1, 3)
        # Write to level-3
        sile.write_delta(H, E=0.1)
        assert sile._get_lvl(3).variables["E"].shape == (1,)
        sile.write_delta(H, E=0.2)
        assert sile._get_lvl(3).variables["E"].shape == (2,)
        # Write to level-4
        sile.write_delta(H, E=0.1, k=[0, 0, 0.5])
        assert sile._get_lvl(4).variables["kpt"].shape == (1, 3)
        assert sile._get_lvl(4).variables["E"].shape == (1,)
        sile.write_delta(H, E=0.2, k=[0, 0, 0.5])
        assert sile._get_lvl(4).variables["kpt"].shape == (1, 3)
        assert sile._get_lvl(4).variables["E"].shape == (2,)
        sile.write_delta(H, E=0.2, k=[0, 1.0, 0.5])
        assert sile._get_lvl(4).variables["kpt"].shape == (2, 3)
        assert sile._get_lvl(4).variables["E"].shape == (2,)

    with deltancSileTBtrans(f, "r") as sile:
        # Read to level-1
        h = sile.read_delta()
        assert h.spsame(H)
        # Read level-2
        h = sile.read_delta(k=[0, 0, 0.5])
        assert h.spsame(H)
        # Read level-3
        h = sile.read_delta(E=0.1)
        assert h.spsame(H)
        # Read level-4
        h = sile.read_delta(E=0.1, k=[0, 0, 0.5])
        assert h.spsame(H)
        h = sile.read_delta(E=0.1, k=[0, 0.0, 0.5])
        assert h.spsame(H)
        h = sile.read_delta(E=0.2, k=[0, 1.0, 0.5])
        assert h.spsame(H)


def test_tbt_delta_fail(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.dH.nc")
    H = Hamiltonian(sisl_system.gtb)
    H.construct([sisl_system.R, sisl_system.t])
    H.finalize()

    with deltancSileTBtrans(f, "w") as sile:
        sile.write_delta(H, k=[0.0] * 3)
        for i in range(H.no_s):
            H[0, i] = 1.0
        with pytest.raises(ValueError):
            sile.write_delta(H, k=[0.2] * 3)


def test_tbt_delta_write_read(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.dH.nc")
    H = Hamiltonian(sisl_system.gtb, dtype=np.complex64)
    H.construct([sisl_system.R, sisl_system.t])
    H.finalize()

    with deltancSileTBtrans(f, "w") as sile:
        sile.write_delta(H)
    with deltancSileTBtrans(f, "r") as sile:
        h = sile.read_delta()
    assert h.spsame(H)
    assert h.dkind == H.dkind


def test_tbt_delta_fail_list_col(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.dH.nc")
    H = Hamiltonian(sisl_system.gtb)
    H.construct([sisl_system.R, sisl_system.t])

    with deltancSileTBtrans(f, "w") as sile:
        sile.write_delta(H, E=-1.0)
        edges = H.edges(0)
        i = edges.max() + 1
        del H[0, i - 1]
        H[0, i] = 1.0
        with pytest.raises(ValueError):
            sile.write_delta(H, E=1.0)


def test_tbt_delta_fail_ncol(sisl_tmp, sisl_system):
    f = sisl_tmp("gr.dH.nc")
    H = Hamiltonian(sisl_system.gtb)
    H.construct([sisl_system.R, sisl_system.t])

    with deltancSileTBtrans(f, "w") as sile:
        sile.write_delta(H, E=-1.0)
        edges = H.edges(0)
        i = edges.max() + 1
        H[0, i] = 1.0
        H.finalize()
        with pytest.raises(ValueError):
            sile.write_delta(H, E=1.0)


def test_tbt_delta_merge(sisl_tmp, sisl_system):
    f1 = sisl_tmp("gr1.dH.nc")
    f2 = sisl_tmp("gr2.dH.nc")
    fout = sisl_tmp("grmerged.dH.nc")

    H = Hamiltonian(sisl_system.gtb)
    H.construct([sisl_system.R, sisl_system.t])
    H.finalize()

    with deltancSileTBtrans(f1, "w") as sile:
        sile.write_delta(H, E=-1.0)
        sile.write_delta(H, E=-1.0, k=[0, 1, 1])
        sile.write_delta(H)
        sile.write_delta(H, k=[0, 1, 0])

    with deltancSileTBtrans(f2, "w") as sile:
        sile.write_delta(H, E=-1.0)
        sile.write_delta(H, E=-1.0, k=[0, 1, 1])
        sile.write_delta(H)
        sile.write_delta(H, k=[0, 1, 0])

    # Now merge them
    deltancSileTBtrans.merge(fout, deltancSileTBtrans(f1), f2)

    with deltancSileTBtrans(fout, "r") as sile:
        h = sile.read_delta() / 2
        assert h.spsame(H)
        h = sile.read_delta(E=-1.0) / 2
        assert h.spsame(H)
        h = sile.read_delta(E=-1.0, k=[0, 1, 1]) / 2
        assert h.spsame(H)
        h = sile.read_delta(k=[0, 1, 0]) / 2
        assert h.spsame(H)

        try:
            h = sile.read_delta(k=[0, 1, 1]) / 2
            assert False
        except:
            assert True
