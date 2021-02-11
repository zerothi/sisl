from sisl._internal import set_module
from ..sile import Sile, SileCDF, SileBin

__all__ = ['SileSiesta', 'SileCDFSiesta', 'SileBinSiesta']


@set_module("sisl.io.siesta")
class SileSiesta(Sile):
    pass


@set_module("sisl.io.siesta")
class SileCDFSiesta(SileCDF):
    pass


@set_module("sisl.io.siesta")
class SileBinSiesta(SileBin):
    pass
