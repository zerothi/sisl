"""
Define a common Wannier90 Sile
"""
from sisl._internal import set_module
from ..sile import Sile

__all__ = ['SileWannier90']


@set_module("sisl.io.wannier90")
class SileWannier90(Sile):
    pass
