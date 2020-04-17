"""
Library for converting units and creating numpy arrays
with automatic unit conversion.

The conversion factors are taken directly from Siesta
which means these unit conversions should be used for Siesta "stuff".
"""

from sisl._internal import set_module
from .base import UnitParser
from .base import unit_table
from .base import unit_group as u_group
from .base import unit_convert as u_convert
from .base import unit_default as u_default

__all__  = ['unit_group', 'unit_convert', 'unit_default', 'units']


unit_table_siesta = dict({key: dict(values) for key, values in unit_table.items()})

unit_table_siesta['length'].update({
    'Bohr': 0.529177e-10,
})

unit_table_siesta['time'].update({
    'mins': 60.,
    'hours': 3600.,
    'days': 86400.,
})

unit_table_siesta['energy'].update({
    'meV': 1.60219e-22,
    'eV': 1.60219e-19,
    'mRy': 2.17991e-21,
    'Ry': 2.17991e-18,
    'mHa': 4.35982e-21,
    'Ha': 4.35982e-18,
    'Hartree': 4.35982e-18,
    'K': 1.38066e-23,
    'kJ/mol': 1.6606e-21,
    'Hz': 6.6262e-34,
    'THz': 6.6262e-22,
    'cm-1': 1.986e-23,
    'cm**-1': 1.986e-23,
    'cm^-1': 1.986e-23,
})

unit_table_siesta['force'].update({
    'eV/Ang': 1.60219e-9,
    'eV/Bohr': 1.60219e-9*0.529177,
    'Ry/Bohr': 4.11943e-8,
    'Ry/Ang': 4.11943e-8/0.529177,
})


@set_module("sisl.unit.siesta")
def unit_group(unit, tbl=None):
    global unit_table_siesta
    if tbl is None:
        return u_group(unit, unit_table_siesta)
    return u_group(unit, tbl)


@set_module("sisl.unit.siesta")
def unit_default(group, tbl=None):
    global unit_table_siesta
    if tbl is None:
        return u_default(group, unit_table_siesta)
    return u_default(group, tbl)


@set_module("sisl.unit.siesta")
def unit_convert(fr, to, opts=None, tbl=None):
    global unit_table_siesta
    if tbl is None:
        return u_convert(fr, to, opts, unit_table_siesta)
    return u_convert(fr, to, opts, tbl)

# create unit-parser
units = UnitParser(unit_table_siesta)
