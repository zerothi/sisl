"""
Define a common SIESTA Sile
"""

from ..sile import Sile, SileCDF, SileBin

__all__ = ['SileSIESTA', 'SileCDFSIESTA', 'SileBinSIESTA']

class SileSIESTA(Sile):
    pass

class SileCDFSIESTA(SileCDF):
    pass

class SileBinSIESTA(SileBin):
    pass


