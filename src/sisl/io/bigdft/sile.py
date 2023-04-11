# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Define a common BigDFT Sile
"""
from sisl._internal import set_module
from ..sile import Sile, SileBin

__all__ = ["SileBigDFT", "SileBinBigDFT"]


@set_module("sisl.io.bigdft")
class SileBigDFT(Sile):
    pass


@set_module("sisl.io.bigdft")
class SileBinBigDFT(SileBin):
    pass
