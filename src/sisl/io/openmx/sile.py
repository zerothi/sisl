# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from sisl._internal import set_module

from ..sile import Sile, SileBin, SileCDF

__all__ = ["SileOpenMX", "SileCDFOpenMX", "SileBinOpenMX"]


@set_module("sisl.io.openmx")
class SileOpenMX(Sile):
    pass


@set_module("sisl.io.openmx")
class SileCDFOpenMX(SileCDF):
    pass


@set_module("sisl.io.openmx")
class SileBinOpenMX(SileBin):
    pass
