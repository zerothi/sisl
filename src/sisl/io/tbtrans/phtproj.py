# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from sisl._internal import set_module

from ..sile import add_sile
from .tbt import Ry2eV
from .tbtproj import tbtprojncSileTBtrans

__all__ = ["phtprojncSilePHtrans"]


@set_module("sisl.io.phtrans")
class phtprojncSilePHtrans(tbtprojncSileTBtrans):
    """PHtrans projection file object"""

    _trans_type = "PHT.Proj"
    _E2eV = Ry2eV**2


add_sile("PHT.Proj.nc", phtprojncSilePHtrans)
