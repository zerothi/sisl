# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np
from xarray import DataArray

from .data import XarrayData

__all__ = ["DOSData"]


class DOSData(XarrayData):
    """Class to store density of states data."""

    pass


def dos_from_arrays(E: np.ndarray, DOS: np.ndarray) -> DOSData:
    """Create a DOSData object from arrays.

    Parameters
    ----------
    E:
        The energy values.
    DOS:
        The density of states values.
    """
    return DOSData(DataArray(DOS, coords={"E": E}, dims=["E"]))


DOSData.new.register("array", dos_from_arrays)
