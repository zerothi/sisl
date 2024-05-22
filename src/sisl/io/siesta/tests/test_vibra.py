# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import os.path as osp

import pytest

import sisl

pytestmark = [pytest.mark.io, pytest.mark.siesta]
_dir = osp.join("sisl", "io", "siesta")


def test_eigenmode_read(sisl_files):
    fdf = sisl.get_sile(sisl_files(_dir, "si_vibra.fdf"))
    geometry = fdf.read_geometry()
    vectors = sisl.io.siesta.vectorsSileSiesta(
        sisl_files(_dir, "si_vibra.vectors"),
        geometry=geometry,
    )

    blines = fdf.get("BandLines")
    nk = sum(int(l.split()[0]) for l in blines)
    nlines = len(blines)
    iklast = [sum(int(l.split()[0]) for l in blines[: i + 1]) for i in range(nlines)]
    klast = [list(map(float, l.split()[1:4])) for l in blines]
    print(iklast, klast)

    # yield modes
    nmodes = 0
    nmodes_total = 0
    for state in vectors.yield_eigenmode():
        nmodes += 1
        nmodes_total += len(state)

    print(nmodes, nmodes_total)
    assert nmodes == nk
    assert nmodes_total == geometry.na * 3 * nk
