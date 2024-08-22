# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_eigenmode_read(sisl_files):
    fdf = sisl.get_sile(sisl_files("siesta", "H_chain", "h_chain_vibra.fdf"))
    geometry = fdf.read_geometry()
    vectors = sisl.io.siesta.vectorsSileSiesta(
        sisl_files("siesta", "H_chain", "h_chain_vibra.vectors"),
        geometry=geometry,
    )

    blines = fdf.get("BandLines")
    nk = sum(int(l.split()[0]) for l in blines)
    nlines = len(blines)
    iklast = [sum(int(l.split()[0]) for l in blines[: i + 1]) for i in range(nlines)]
    klast = [list(map(float, l.split()[1:4])) for l in blines]

    # yield modes
    nmodes = 0
    nmodes_total = 0
    ks = []
    for state in vectors.yield_eigenmode():
        nmodes += 1
        nmodes_total += len(state)
        ks.append(state.info["k"])

    assert nmodes == nk
    assert nmodes_total == geometry.na * 3 * nk

    bz = vectors.read_brillouinzone()
    assert len(bz) == nk
    assert np.allclose(bz.k, ks)


def test_eigenmode_values(sisl_files):
    fdf = sisl.get_sile(sisl_files("siesta", "H_chain", "h_chain_vibra.fdf"))
    vectors = sisl.io.siesta.vectorsSileSiesta(
        sisl_files("siesta", "H_chain", "h_chain_vibra.vectors"),
        geometry=fdf.read_geometry(),
    )

    mode = vectors.read_eigenmode()
    assert np.allclose(mode.state[0, 0:3], [-0.7071, -0.2400e-4, -0.2400e-4])
    assert np.allclose(mode.state[0, 3:7], [0.7071, 0.2400e-4, 0.2400e-4])
