"""
Define a common BigDFT Sile
"""

from ..sile import Sile, SileCDF, SileBin

__all__ = ['SileBigDFT', 'SileCDFBigDFT', 'SileBinBigDFT']

class SileBigDFT(Sile):
    pass

class SileCDFBigDFT(SileCDF):
    pass

class SileBinBigDFT(SileBin):
    pass
