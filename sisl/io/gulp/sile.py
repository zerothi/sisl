"""
Define a common GULP Sile
"""
from sisl._internal import set_module
from ..sile import Sile, SileCDF

__all__ = ['SileGULP', 'SileCDFGULP']


@set_module("sisl.io.gulp")
class SileGULP(Sile):
    pass


@set_module("sisl.io.gulp")
class SileCDFGULP(SileCDF):
    pass
