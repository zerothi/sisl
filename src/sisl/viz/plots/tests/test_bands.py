# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

from sisl import Spin
from sisl.viz.data import BandsData
from sisl.viz.plots import bands_plot

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
def gap():
    return 2.5


@pytest.fixture(scope="module")
def bands_data(spin, gap):
    return BandsData.toy_example(spin=spin, gap=gap)


def test_bands_plot(bands_data, backend):
    bands_plot(bands_data, backend=backend)
