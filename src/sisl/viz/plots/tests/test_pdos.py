# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

from sisl import Spin
from sisl.viz.data import PDOSData
from sisl.viz.plots import pdos_plot

pytestmark = [pytest.mark.viz, pytest.mark.plots]


@pytest.fixture(scope="module", params=["plotly", "matplotlib"])
def backend(request):
    pytest.importorskip(request.param)
    return request.param


@pytest.fixture(
    scope="module", params=["unpolarized", "polarized", "noncolinear", "spinorbit"]
)
def spin(request):
    return Spin(request.param)


@pytest.fixture(scope="module")
def pdos_data(spin):
    return PDOSData.toy_example(spin=spin)


def test_pdos_plot(pdos_data, backend):
    pdos_plot(pdos_data, backend=backend)
