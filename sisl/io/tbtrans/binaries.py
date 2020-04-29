from ..sile import add_sile

from sisl._internal import set_module
from sisl.io.siesta.binaries import _gfSileSiesta

__all__ = ['tbtgfSileTBtrans']


dic = {}
try:
    dic['__doc__'] = _gfSileSiesta.__doc__.replace(_gfSileSiesta.__name__, 'tbtgfSileTBtrans')
except:
    pass

tbtgfSileTBtrans = set_module("sisl.io.tbtrans")(type("tbtgfSileTBtrans", (_gfSileSiesta, ), dic))
del dic

add_sile('TBTGF', tbtgfSileTBtrans)
