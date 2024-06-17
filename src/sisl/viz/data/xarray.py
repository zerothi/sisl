# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Union

from xarray import DataArray, Dataset

from .data import Data


class XarrayData(Data):
    _data: Union[DataArray, Dataset]

    def __init__(self, data: Union[DataArray, Dataset]):
        super().__init__(data)

    def __getattr__(self, key):
        sisl_accessor = self._data.sisl

        if hasattr(sisl_accessor, key):
            return getattr(sisl_accessor, key)

        return getattr(self._data, key)

    def __dir__(self):
        return dir(self._data.sisl) + dir(self._data)


class OrbitalData(XarrayData):
    pass
