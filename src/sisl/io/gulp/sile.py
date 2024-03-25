# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Define a common GULP Sile
"""
from sisl._internal import set_module

from ..sile import Sile, SileBin

__all__ = ["SileGULP", "SileBinGULP"]


@set_module("sisl.io.gulp")
class SileGULP(Sile):
    pass


@set_module("sisl.io.gulp")
class SileBinGULP(SileBin):
    pass
