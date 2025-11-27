# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ._singledispatch import singledispatchmethod

__all__ = ["Data", "XarrayData"]


class Data:

    def __init__(self, data):
        self._data = data

    def __init_subclass__(cls) -> None:

        @singledispatchmethod
        @classmethod
        def new(cls, data):
            return cls(data)

        cls.new = new

    def print(self):
        """Prints the data."""
        print(self._data)


class XarrayData(Data):

    def write(self, path: str):
        """Writes the data to a file.

        Parameters
        ----------
        path:
            The path to the file. For now only netcdf files are supported.
        """
        self._data.to_netcdf(path)
