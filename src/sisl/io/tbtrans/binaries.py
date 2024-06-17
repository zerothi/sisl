# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from sisl._internal import set_module
from sisl.io.siesta.binaries import _gfSileSiesta

from ..sile import add_sile

__all__ = ["tbtgfSileTBtrans"]


dic = {}
try:
    dic["__doc__"] = _gfSileSiesta.__doc__.replace(
        _gfSileSiesta.__name__, "tbtgfSileTBtrans"
    )
except Exception:
    pass

tbtgfSileTBtrans = set_module("sisl.io.tbtrans")(
    type("tbtgfSileTBtrans", (_gfSileSiesta,), dic)
)
del dic

add_sile("TBTGF", tbtgfSileTBtrans)
