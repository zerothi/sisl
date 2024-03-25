# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations

from sisl._internal import set_module
from sisl.io.siesta import MissingFDFSiestaError

from ..sile import Sile, SileBin, SileCDF, missing_input

__all__ = [
    "SileTBtrans",
    "SileCDFTBtrans",
    "SileBinTBtrans",
    "MissingFDFTBtransError",
    "missing_input_fdf",
]


@set_module("sisl.io.tbtrans")
class MissingFDFTBtransError(MissingFDFSiestaError):
    pass


@set_module("sisl.io.tbtrans")
def missing_input_fdf(
    inputs, executable: str = "siesta", when_exception: Exception = KeyError
):
    return missing_input(executable, inputs, MissingFDFTBtransError, when_exception)


@set_module("sisl.io.tbtrans")
class SileTBtrans(Sile):
    pass


@set_module("sisl.io.tbtrans")
class SileCDFTBtrans(SileCDF):
    # all netcdf output should not be masked
    def _setup(self, *args, **kwargs):
        super()._setup(*args, **kwargs)
        # all NetCDF routines actually returns masked arrays
        # this is to prevent TBtrans CDF files from doing this.
        if hasattr(self, "fh"):
            self.fh.set_auto_mask(False)


@set_module("sisl.io.tbtrans")
class SileBinTBtrans(SileBin):
    pass
