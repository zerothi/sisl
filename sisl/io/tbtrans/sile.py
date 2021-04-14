from ..sile import Sile, SileCDF, SileBin
from sisl._internal import set_module

__all__ = ['SileTBtrans', 'SileCDFTBtrans', 'SileBinTBtrans']


@set_module("sisl.io.tbtrans")
class SileTBtrans(Sile):
    pass


@set_module("sisl.io.tbtrans")
class SileCDFTBtrans(SileCDF):

    # all netcdf output should not be masked
    def _setup(self, *args, **kwargs):
        # all NetCDF routines actually returns masked arrays
        # this is to prevent TBtrans CDF files from doing this.
        if hasattr(self, "fh"):
            self.fh.set_auto_mask(False)


@set_module("sisl.io.tbtrans")
class SileBinTBtrans(SileBin):
    pass
