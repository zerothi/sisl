from __future__ import print_function

from ..sile import add_sile

from sisl.io.siesta import _GFSileSiesta

__all__ = ['TBTGFSileTBtrans']


TBTGFSileTBtrans = type("TBTGFSileTBtrans", (_GFSileSiesta, ), {})

add_sile('TBTGF', TBTGFSileTBtrans)
