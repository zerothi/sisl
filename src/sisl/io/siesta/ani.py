# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Sile object for reading ANI files
"""
from sisl._internal import set_module

from ..sile import add_sile
from ..xyz import xyzSile
from .sile import SileSiesta

__all__ = ["aniSileSiesta"]


@set_module("sisl.io.siesta")
class aniSileSiesta(xyzSile, SileSiesta):
    pass


add_sile("ANI", aniSileSiesta, gzip=True)
