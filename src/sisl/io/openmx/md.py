# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Sile object for reading OpenMX md files
"""
from sisl._internal import set_module

from ..sile import add_sile
from ..xyz import xyzSile
from .sile import SileOpenMX

__all__ = ["mdSileOpenMX"]


@set_module("sisl.io.openmx")
class mdSileOpenMX(xyzSile, SileOpenMX):
    pass


add_sile("md", mdSileOpenMX, gzip=True)
