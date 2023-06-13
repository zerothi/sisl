# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Sile object for reading OpenMX md files
"""
from .sile import SileOpenMX
from ..xyz import xyzSile
from ..sile import add_sile
from sisl._internal import set_module


__all__ = ["mdSileOpenMX"]


@set_module("sisl.io.openmx")
class mdSileOpenMX(xyzSile, SileOpenMX):
    r"""OpenMX MD output (equivalent to `xyzSile`) """
    pass


add_sile('md', mdSileOpenMX, gzip=True)
