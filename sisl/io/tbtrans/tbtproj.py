from __future__ import print_function, division

# Import sile objects
from ..sile import add_sile
from .tbt import tbtncSileTBtrans


__all__ = ['tbtprojncSileTBtrans', 'phtprojncSileTBtrans']


class tbtprojncSileTBtrans(tbtncSileTBtrans):
    """ TBtrans projection file object """
    _trans_type = 'TBT.Proj'


add_sile('TBT.Proj.nc', tbtprojncSileTBtrans)
# Add spin-dependent files
add_sile('TBT_DN.Proj.nc', tbtprojncSileTBtrans)
add_sile('TBT_UP.Proj.nc', tbtprojncSileTBtrans)


class phtprojncSileTBtrans(tbtprojncSileTBtrans):
    """ PHtrans projection file object """
    _trans_type = 'PHT.Proj'


add_sile('PHT.Proj.nc', phtprojncSileTBtrans)
