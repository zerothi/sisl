# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

import sisl

pytestmark = [pytest.mark.io, pytest.mark.siesta]


def test_wfsx_read(sisl_files):
    fdf = sisl.get_sile(sisl_files("siesta", "Bi2Se3_3layer", "Bi2Se3.fdf"))
    wfsx = sisl.get_sile(
        sisl_files("siesta", "Bi2Se3_3layer", "Bi2Se3.bands.WFSX"),
        parent=fdf.read_geometry(),
    )

    info = wfsx.read_info()
    sizes = wfsx.read_sizes()
    basis = wfsx.read_basis()

    # yield states
    nstates = 0
    nstates_total = 0
    for state in wfsx.yield_eigenstate():
        nstates += 1
        nstates_total += len(state)
    # in this example we a SOC calculation, so we should not muliply by spin
    assert nstates == len(info[0])
    assert nstates_total == info[2].sum()

    bz = wfsx.read_brillouinzone()
    assert len(bz) == 16
