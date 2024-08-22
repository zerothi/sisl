# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

r""" Internal sisl-only methods that should not be used outside """

# override module level, inspired by numpy

__all__ = ["set_module"]


def set_module(module):
    r"""Decorator for overriding __module__ on a function or class"""

    def deco(f_or_c):
        if module is not None:
            f_or_c.__module__ = module
        return f_or_c

    return deco
