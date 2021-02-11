"""
Define a common ScaleUP Sile
"""
from sisl._internal import set_module
from ..sile import Sile, SileCDF, SileBin

__all__ = ['SileScaleUp', 'SileCDFScaleUp', 'SileBinScaleUp']


@set_module("sisl.io.scaleup")
class SileScaleUp(Sile):
    pass


@set_module("sisl.io.scaleup")
class SileCDFScaleUp(SileCDF):
    pass


@set_module("sisl.io.scaleup")
class SileBinScaleUp(SileBin):
    pass
