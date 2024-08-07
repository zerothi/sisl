# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import TypeVar

from ..data import Data

DataInstance = TypeVar("DataInstance", bound=Data)


def accept_data(
    data: DataInstance, cls: type[Data], check: bool = True
) -> DataInstance:
    if not isinstance(data, cls):
        raise TypeError(
            f"Data must be of type {cls.__name__} and was {type(data).__name__}"
        )

    if check:
        data.sanity_check()

    return data


def extract_data(data: Data, cls: type[Data], check: bool = True):
    if not isinstance(data, cls):
        raise TypeError(
            f"Data must be of type {cls.__name__} and was {type(data).__name__}"
        )

    if check:
        data.sanity_check()

    return data._data
