# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import TypeVar, Union

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def swap(val: Union[T1, T2], vals: tuple[T1, T2]) -> Union[T1, T2]:
    """Given two values, returns the one that is not the input value."""
    if val == vals[0]:
        return vals[1]
    elif val == vals[1]:
        return vals[0]
    else:
        raise ValueError(f"Value {val} not in {vals}")
