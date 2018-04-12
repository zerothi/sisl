from __future__ import print_function

from ..sile import add_sile

from sisl.io.siesta.binaries import _gfSileSiesta

__all__ = ['tbtgfSileTBtrans', 'TBTGFSileTBtrans']


tbtgfSileTBtrans = type("tbtgfSileTBtrans", (_gfSileSiesta, ), {})
TBTGFSileTBtrans = tbtgfSileTBtrans

add_sile('TBTGF', tbtgfSileTBtrans)
