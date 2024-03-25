# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Define a common Wannier90 Sile
"""
from sisl._internal import set_module

from ..sile import Sile, SileBin

__all__ = ["SileWannier90", "SileBinWannier90"]


@set_module("sisl.io.wannier90")
class SileWannier90(Sile):
    pass


@set_module("sisl.io.wannier90")
class SileBinWannier90(SileBin):
    pass
