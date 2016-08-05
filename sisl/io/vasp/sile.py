"""
Define a common VASP Sile
"""

from ..sile import Sile, SileCDF

__all__ = ['SileVASP', 'SileCDFVASP']

class SileVASP(Sile):
    pass

class SileCDFVASP(SileCDF):
    pass


