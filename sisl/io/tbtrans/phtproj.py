from __future__ import print_function, division

from ..sile import add_sile
from .tbt import Ry2eV
from .tbtproj import tbtprojncSileTBtrans


__all__ = ['phtprojncSilePHtrans']


class phtprojncSilePHtrans(tbtprojncSileTBtrans):
    """ PHtrans projection file object """
    _trans_type = 'PHT.Proj'
    _E2eV = Ry2eV ** 2


add_sile('PHT.Proj.nc', phtprojncSilePHtrans)
