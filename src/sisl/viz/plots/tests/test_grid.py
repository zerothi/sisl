# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl
from sisl import Grid
from sisl.viz.plots import grid_plot

pytestmark = [pytest.mark.viz, pytest.mark.plots]


@pytest.fixture(scope="module", params=["plotly", "matplotlib"])
def backend(request):
    pytest.importorskip(request.param)
    return request.param


@pytest.fixture(scope="module", params=["x", "xy", "xyz"])
def axes(request):
    return request.param


@pytest.fixture(scope="module")
def grid():
    geometry = sisl.geom.graphene()
    grid = Grid((10, 10, 10), geometry=geometry)

    grid.grid[:] = np.linspace(0, 1000, 1000).reshape(10, 10, 10)
    return grid


def test_grid_plot(grid, axes, backend):
    if axes == "xyz" and backend == "matplotlib":
        with pytest.raises(NotImplementedError):
            grid_plot(grid, axes=axes, backend=backend)
    else:
        grid_plot(grid, axes=axes, backend=backend)
