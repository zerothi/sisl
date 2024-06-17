# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Literal

import xarray as xr

from .data_source import DataSource


class EigenstateData(DataSource):
    pass


def spin_moments_from_dataset(
    axis: Literal["x", "y", "z"], data: xr.Dataset
) -> xr.DataArray:
    if "spin_moments" not in data:
        raise ValueError("The dataset does not contain spin moments")

    spin_moms = data.spin_moments.sel(axis=axis)
    spin_moms = spin_moms.rename(f"spin_moments_{axis}")
    return spin_moms


class SpinMoment(EigenstateData):
    function = staticmethod(spin_moments_from_dataset)
