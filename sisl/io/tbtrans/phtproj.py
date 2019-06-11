from __future__ import print_function, division

from ..sile import add_sile
from .tbtproj import tbtprojncSileTBtrans


__all__ = ['phtprojncSilePHtrans']


class phtprojncSilePHtrans(tbtprojncSileTBtrans):
    """ PHtrans projection file object """
    _trans_type = 'PHT.Proj'


# Clean up methods
for _name in ['chemical_potential', 'electron_temperature',
              'shot_noise', 'noise_power',
              'current', 'current_parameter']:
    setattr(phtprojncSilePHtrans, _name, None)


add_sile('PHT.Proj.nc', phtprojncSilePHtrans)
