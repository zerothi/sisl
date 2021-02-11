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
