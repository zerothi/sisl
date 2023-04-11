# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Physical constants
==================

Module containing a pre-set set of physical constants. The SI units are following the *new* convention
that takes effect on 20 May 2019.

The currently stored constants are (all are given in SI units):

   PhysicalConstant
   q
   c
   h
   hbar
   m_e
   m_p
   G0
   G

All constants may be used like an ordinary float (which converts it to a float):

>>> c
299792458.0 m/s
>>> c * 2
599584916

while one can just as easily convert the units (which ensures thay stay like another `PhysicalConstant`):

>>> c('Ang/ps')
2997924.58 Ang/ps
"""

from ._internal import set_module
from .unit.base import units

__all__ = ['PhysicalConstant']


@set_module("sisl")
class PhysicalConstant(float):
    """ Class to create a physical constant with unit-conversion capability, works exactly like a float.

    To change the units simply call it like a method with the desired unit:

    >>> m = PhysicalConstant(1., 'm')
    >>> m.unit
    'm'
    >>> m2nm = m('nm')
    >>> m2nm
    1000000000.0 nm
    >>> m2nm.unit
    'nm'
    >>> m2nm * 2
    1000000000.0
    """
    __slots__ = ['_unit']

    def __new__(cls, value, unit):
        constant = float.__new__(cls, value)
        return constant

    def __init__(self, value, unit):
        self._unit = unit

    @property
    def unit(self):
        """ Unit of constant """
        return self._unit

    def __str__(self):
        return '{} {}'.format(float(self), self.unit)

    def __call__(self, unit=None):
        """ Return the value for the constant in the given unit, otherwise will return the units in SI units """
        if unit is None:
            return self
        return PhysicalConstant(self * units(self.unit, unit), unit)

    def __eq__(self, other):
        if isinstance(other, PhysicalConstant):
            super().__eq__(self, other * units(other.unit, self.unit))
        return super().__eq__(self, other)


__all__ += ['q', 'c', 'h', 'hbar', 'm_e', 'm_p', 'G', 'G0', 'a0']


# These are CODATA-2018 values
#: Unit of charge [C]
q = PhysicalConstant(1.602176634e-19, "C")
#: Bohr radius [m]
a0 = PhysicalConstant(5.29177210903e-11, "m")
#: Boltzmann constant [J K^-1]
kB = PhysicalConstant(1.380649e-23, "J/K")
#: Electron mass [kg]
m_e = PhysicalConstant(9.1093837015e-31, "kg")
#: Planck constant [J Hz^-1]
h = PhysicalConstant(6.62607015e-34, "J s")
#: Reduced Planck constant [J Hz^-1]
hbar = PhysicalConstant(1.0545718176461565e-34, "J s")
#: Proton mass [kg]
m_p = PhysicalConstant(1.67262192369e-27, "kg")
#: Speed of light in vacuum [m s^-1]
c = PhysicalConstant(299792458.0, "m/s")


# Values not found in the CODATA table
#: Conductance quantum [S], or [m^2/s^2]
G0 = PhysicalConstant(2 * (q ** 2 / h), 'm^2/s^2')
#: Gravitational constant [m^3/kg/s^2]
G = PhysicalConstant(6.6740831e-11, 'm^3/kg/s^2')

