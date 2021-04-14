from sisl._internal import set_module
from ..sile import Sile, SileCDF, SileBin

__all__ = ['SileSiesta', 'SileCDFSiesta', 'SileBinSiesta']


@set_module("sisl.io.siesta")
class SileSiesta(Sile):
    pass


@set_module("sisl.io.siesta")
class SileCDFSiesta(SileCDF):

    # all netcdf output should not be masked
    def _setup(self, *args, **kwargs):
        # all NetCDF routines actually returns masked arrays
        # this is to prevent Siesta CDF files from doing this.
        if hasattr(self, "fh"):
            self.fh.set_auto_mask(False)


@set_module("sisl.io.siesta")
class SileBinSiesta(SileBin):
    pass
