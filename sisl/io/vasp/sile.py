"""
Define a common VASP Sile
"""

from ..sile import Sile, SileCDF, SileBin

__all__ = ['SileVASP', 'SileCDFVASP', 'SileBinVASP']


class SileVASP(Sile):
    pass


class SileCDFVASP(SileCDF):
    pass


class SileBinVASP(SileBin):
    pass
