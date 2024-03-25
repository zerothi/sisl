# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Define a common DFTB+ Sile
"""
from sisl._internal import set_module

from ..sile import Sile, SileBin

__all__ = ["SileDFTB", "SileBinDFTB"]


@set_module("sisl.io.dftb")
class SileDFTB(Sile):
    pass


@set_module("sisl.io.DFTB")
class SileBinDFTB(SileBin):
    pass
