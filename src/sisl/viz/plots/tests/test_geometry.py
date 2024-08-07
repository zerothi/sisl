# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest

import sisl
from sisl.viz.plots import geometry_plot

pytestmark = [pytest.mark.viz, pytest.mark.plots]


@pytest.fixture(scope="module", params=["plotly", "matplotlib"])
def backend(request):
    pytest.importorskip(request.param)
    return request.param


@pytest.fixture(scope="module", params=["x", "xy", "xyz"])
def axes(request):
    return request.param


@pytest.fixture(scope="module")
def geometry():
    return sisl.geom.graphene()


def test_geometry_plot(geometry, axes, backend):
    if axes == "xyz" and backend == "matplotlib":
        with pytest.raises(NotImplementedError):
            geometry_plot(geometry, axes=axes, backend=backend)
    else:
        geometry_plot(geometry, axes=axes, backend=backend)
