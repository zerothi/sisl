from __future__ import print_function

from ..sile import add_sile

from sisl.io.siesta.binaries import _gfSileSiesta

__all__ = ['tbtgfSileTBtrans', 'TBTGFSileTBtrans']


dic = {}
try:
    dic['__doc__'] = _gfSileSiesta.__doc__.replace(_gfSileSiesta.__name__, 'tbtgfSileTBtrans')
except:
    pass
tbtgfSileTBtrans = type("tbtgfSileTBtrans", (_gfSileSiesta, ), dic)
del dic

TBTGFSileTBtrans = tbtgfSileTBtrans

add_sile('TBTGF', tbtgfSileTBtrans)
