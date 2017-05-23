"""
Define a common ScaleUP Sile
"""

from ..sile import Sile, SileCDF, SileBin

__all__ = ['SileScaleUp', 'SileCDFScaleUp', 'SileBinScaleUp']


class SileScaleUp(Sile):
    pass


class SileCDFScaleUp(SileCDF):
    pass


class SileBinScaleUp(SileBin):
    pass
