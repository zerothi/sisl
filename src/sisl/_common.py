# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

""" Global common files """
from enum import Flag, auto, unique

__all__ = ["Opt"]


@unique
class Opt(Flag):
    """Global option arguments used throughout sisl

    These flags may be combined via bit-wise operations
    """

    NONE = auto()
    ANY = auto()
    ALL = auto()
