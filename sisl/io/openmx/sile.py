# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from sisl._internal import set_module
from ..sile import Sile, SileCDF, SileBin

__all__ = ['SileOpenMX', 'SileCDFOpenMX', 'SileBinOpenMX']


@set_module("sisl.io.openmx")
class SileOpenMX(Sile):
    pass


@set_module("sisl.io.openmx")
class SileCDFOpenMX(SileCDF):
    pass


@set_module("sisl.io.openmx")
class SileBinOpenMX(SileBin):
    pass
