"""
Library for converting units and creating numpy arrays
with automatic unit conversion.

The conversion factors are taken directly from SIESTA
which means these unit conversions should be used for SIESTA "stuff".
"""
from __future__ import print_function, division

from sisl.units import unit_group as u_group
from sisl.units import unit_convert as u_convert
from sisl.units import unit_default as u_default

__all__  = ['unit_group', 'unit_convert', 'unit_default']


_unit_table = {
    'mass': {
        'DEFAULT': 'amu',
        'kg': 1.,
        'g': 1.e-3,
        'amu': 1.66054e-27,
        },
    'length': {
        'DEFAULT': 'Bohr',
        'm': 1.,
        'cm': 0.01,
        'nm': 1.e-9,
        'Ang': 1.e-10,
        'Bohr': 0.529177e-10,
        },
    'time': {
        'DEFAULT': 'fs',
        's': 1.,
        'fs': 1.e-15,
        'ps': 1.e-12,
        'ns': 1.e-9,
        },
    'energy': {
        'DEFAULT': 'Ry',
        'J': 1.,
        'erg': 1.e-7,
        'eV': 1.60219e-19,
        'meV': 1.60219e-22,
        'Ry': 2.17991e-18,
        'mRy': 2.17991e-21,
        'Ha': 4.35982e-18,
        'Hartree': 4.35982e-18,
        'K': 1.38066e-23,
        'cm**-1': 1.986e-23,
        'kJ/mol': 1.6606e-21,
        'Hz': 6.6262e-34,
        'THz': 6.6262e-22,
        'cm-1': 1.986e-23,
        'cm^-1': 1.986e-23,
        },
    'force': {
        'DEFAULT': 'Ry/Bohr',
        'N': 1.,
        'eV/Ang': 1.60219e-9,
        'eV/Bohr': 1.60219e-9*0.529177,
        'Ry/Bohr': 4.11943e-8,
        'Ry/Ang': 4.11943e-8/0.529177,
        }
    }


def unit_group(unit, tbl=None):
    global _unit_table
    if tbl is None:
        return u_group(unit, tbl=_unit_table)
    else:
        return u_group(unit, tbl=tbl)


def unit_default(group, tbl=None):
    global _unit_table
    if tbl is None:
        return u_default(group, tbl=_unit_table)
    else:
        return u_default(group, tbl=tbl)


def unit_convert(fr, to, opts={}, tbl=None):
    global _unit_table
    if tbl is None:
        return u_convert(fr, to, opts, tbl=_unit_table)
    else:
        return u_convert(fr, to, opts, tbl=tbl)
