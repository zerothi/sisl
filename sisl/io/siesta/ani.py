# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Sile object for reading ANI files
"""
from sisl import GeometryCollection
from ..xyz import xyzSile
from ..sile import add_sile
from sisl._internal import set_module


__all__ = ["aniSileSiesta"]


@set_module("sisl.io.siesta")
class aniSileSiesta(xyzSile):

    def read_geometry(self, *args, **kwargs):
        if 'start' in kwargs or 'stop' in kwargs or 'step' in kwargs:
            R = super().read_geometry(*args, **kwargs)
        elif 'all' not in kwargs:
            R = super().read_geometry(*args, all=True, **kwargs)
        else:
            R = super().read_geometry(*args, **kwargs)
        return GeometryCollection(R)

add_sile('ANI', aniSileSiesta, gzip=True)
