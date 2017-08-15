"""
Sile object for reading TBtrans binary projection files
"""
from __future__ import print_function, division

# Import sile objects
from .tbtrans import tbtncSileSiesta
from ..sile import *
from sisl.utils import *


# Import the geometry object
from sisl.units.siesta import unit_convert

__all__ = ['tbtprojncSileSiesta', 'phtprojncSileSiesta']

Bohr2Ang = unit_convert('Bohr', 'Ang')
Ry2eV = unit_convert('Ry', 'eV')
Ry2K = unit_convert('Ry', 'K')
eV2Ry = unit_convert('eV', 'Ry')


class tbtprojncSileSiesta(tbtncSileSiesta):
    """ TBtrans projection file object """
    _trans_type = 'TBT.Proj'


add_sile('TBT.Proj.nc', tbtprojncSileSiesta)
# Add spin-dependent files
add_sile('TBT_DN.Proj.nc', tbtprojncSileSiesta)
add_sile('TBT_UP.Proj.nc', tbtprojncSileSiesta)


class phtprojncSileSiesta(tbtprojncSileSiesta):
    """ PHtrans projection file object """
    _trans_type = 'PHT.Proj'

add_sile('PHT.Proj.nc', phtprojncSileSiesta)
