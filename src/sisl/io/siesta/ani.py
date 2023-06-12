# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Sile object for reading ANI files
"""
from .sile import SileSiesta
from ..xyz import xyzSile
from ..sile import add_sile
from sisl._internal import set_module


__all__ = ["aniSileSiesta"]


@set_module("sisl.io.siesta")
class aniSileSiesta(xyzSile, SileSiesta):

    def read_geometry(self, *args, stop=None, **kwargs):
        return super().read_geometry(*args, stop=stop, **kwargs)


add_sile('ANI', aniSileSiesta, gzip=True)
