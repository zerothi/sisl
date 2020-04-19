from ..sile import Sile, SileCDF, SileBin
from sisl._internal import set_module

__all__ = ['SileTBtrans', 'SileCDFTBtrans', 'SileBinTBtrans']


@set_module("sisl.io.tbtrans")
class SileTBtrans(Sile):
    pass


@set_module("sisl.io.tbtrans")
class SileCDFTBtrans(SileCDF):
    pass


@set_module("sisl.io.tbtrans")
class SileBinTBtrans(SileBin):
    pass
