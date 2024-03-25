# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Any, Union, get_args, get_origin, get_type_hints


class Data:
    """Base data class"""

    _data: Any

    def __init__(self, data):
        if isinstance(data, self.__class__):
            data = data._data

        self._data = data

    def sanity_check(self):
        def is_valid(data, expected_type) -> bool:
            if expected_type is Any:
                return True

            return isinstance(data, expected_type)

        expected_type = get_type_hints(self.__class__)["_data"]
        if get_origin(expected_type) is Union:
            valid = False
            for valid_type in get_args(expected_type):
                valid = valid | is_valid(self._data, valid_type)

        else:
            valid = is_valid(self._data, expected_type)

        assert (
            valid
        ), f"Data must be of type {expected_type} but is {type(self._data).__name__}"

    def __getattr__(self, key):
        return getattr(self._data, key)

    def __dir__(self):
        return dir(self._data)
