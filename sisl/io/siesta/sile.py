"""
Define a common SIESTA Sile
"""

from ..sile import Sile, SileCDF, SileBin

__all__ = ['SileSiesta', 'SileCDFSIESTA', 'SileBinSIESTA']

class SileSiesta(Sile):
    pass

class SileCDFSIESTA(SileCDF):
    pass

class SileBinSIESTA(SileBin):
    pass


