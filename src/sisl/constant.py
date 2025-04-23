# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

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
   m_n
   m_p
   G0
   G
   lambda_e
   lambda_n
   lambda_p

All constants may be used like an ordinary float (which converts it to a float):

>>> c
299792458.0 m/s
>>> c * 2
599584916

while one can just as easily convert the units (which ensures thay stay like another `PhysicalConstant`):

>>> c('Ang/ps')
2997924.58 Ang/ps
"""

from numpy import pi

from sisl.unit.codata import CODATA

from ._internal import set_module
from .unit.base import units

__all__ = ["PhysicalConstant"]


@set_module("sisl")
class PhysicalConstant(float):
    """Class to create a physical constant with unit-conversion capability, works exactly like a float.

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

    __slots__ = ["_unit"]

    def __new__(cls, value, unit):
        constant = float.__new__(cls, value)
        return constant

    def __init__(self, value, unit):
        self._unit = unit

    @property
    def unit(self):
        """Unit of constant"""
        return self._unit

    def __str__(self):
        return "{} {}".format(float(self), self.unit)

    def __call__(self, unit=None):
        """Return the value for the constant in the given unit, otherwise will return the units in SI units"""
        if unit is None:
            return self
        return PhysicalConstant(self * units(self.unit, unit), unit)

    def __eq__(self, other):
        if isinstance(other, PhysicalConstant):
            super().__eq__(self, other * units(other.unit, self.unit))
        return super().__eq__(self, other)


__all__ += ["q", "a0", "kB", "m_e", "m_n", "m_p", "h", "hbar", "c"]


# These are CODATA values depending on the user-defined ENV-var SISL_CODATA
#: Unit of charge [C]
q = PhysicalConstant(CODATA["atomic unit of charge"].value, "C")
#: Bohr radius [m]
a0 = PhysicalConstant(CODATA["Bohr radius"].value, "m")
#: Boltzmann constant [J K^-1]
kB = PhysicalConstant(CODATA["Boltzmann constant"].value, "J/K")
#: Electron mass [kg]
m_e = PhysicalConstant(CODATA["electron mass"].value, "kg")
#: Neutron mass [kg]
m_n = PhysicalConstant(CODATA["neutron mass"].value, "kg")
#: Proton mass [kg]
m_p = PhysicalConstant(CODATA["proton mass"].value, "kg")
#: Planck constant [J Hz^-1]
h = PhysicalConstant(CODATA["Planck constant"].value, "J s")
#: Reduced Planck constant [J Hz^-1]
hbar = PhysicalConstant(CODATA["Planck constant"].value / (2 * pi), "J s")
#: Speed of light in vacuum [m s^-1]
c = PhysicalConstant(CODATA["speed of light in vacuum"].value, "m/s")
#: Number of atoms/molecules in one mol [mol^-1]
NA = PhysicalConstant(CODATA["Avogadro constant"].value, "1/mol")


__all__ += ["G0", "G", "lambda_e", "lambda_p", "lambda_n"]

# Values not found in the CODATA table
#: Conductance quantum [S], or [m^2/s^2]
G0 = PhysicalConstant(2 * (q**2 / h), "m^2/s^2")
#: Gravitational constant [m^3/kg/s^2]
G = PhysicalConstant(CODATA["Newtonian constant of gravitation"].value, "m^3/kg/s^2")
#: Compton wavelength of an electron [m]
lambda_e = PhysicalConstant(h / (m_e * c), "m")
#: Compton wavelength of a proton [m]
lambda_p = PhysicalConstant(h / (m_p * c), "m")
#: Compton wavelength of a neutron [m]
lambda_n = PhysicalConstant(h / (m_n * c), "m")
