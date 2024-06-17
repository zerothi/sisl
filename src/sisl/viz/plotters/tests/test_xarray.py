# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import pytest
import xarray as xr

from sisl.viz.plotters.xarray import draw_xarray_xy

pytestmark = [pytest.mark.viz, pytest.mark.plotters]


def test_empty_dataset():
    ds = xr.Dataset({"x": ("dim", []), "y": ("dim", [])})

    drawings = draw_xarray_xy(ds, x="x", y="y")

    assert isinstance(drawings, list)
    assert len(drawings) == 0


def test_empty_dataarray():
    arr = xr.DataArray([], name="values", dims=["x"])

    drawings = draw_xarray_xy(arr, x="x")

    assert isinstance(drawings, list)
    assert len(drawings) == 0
