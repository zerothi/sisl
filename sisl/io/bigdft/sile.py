"""
Define a common BigDFT Sile
"""
from sisl._internal import set_module
from ..sile import Sile, SileCDF, SileBin

__all__ = ['SileBigDFT', 'SileCDFBigDFT', 'SileBinBigDFT']


@set_module("sisl.io.bigdft")
class SileBigDFT(Sile):
    pass


@set_module("sisl.io.bigdft")
class SileCDFBigDFT(SileCDF):
    pass


@set_module("sisl.io.bigdft")
class SileBinBigDFT(SileBin):
    pass
