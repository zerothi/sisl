"""
Physical constants (:mod:`sisl.constant`)
=========================================

.. module:: sisl.constant
   :noindex:

Module containing a pre-set set of physical constants.

The currently stored constants are (all are given in SI units):

Constants
---------

.. autosummary::
   :toctree:
  
   c
   h
   hbar
   m_e
   m_p
   G

All constants may be used like an ordinary float:

>>> c * 2
599584916

while one can just as easily convert the units

>>> c('Ang/ps')
2997924.58
"""
from __future__ import print_function, division

from sisl.unit.base import units

__all__ = ['PhysicalConstant']


class PhysicalConstant(float):
    """ Physical constant stored in SI-units

    To change the units simply call it like a method with the desired unit:

    >>> m = PhysicalConstant(1., 'm')
    >>> m('nm')
    1e9
    """
    __slots__ = ['_unit']

    def __new__(cls, value, unit):
        c = float.__new__(cls, value)
        c._unit = unit
        return c

    @property
    def unit(self):
        return self._unit

    def __call__(self, unit=None):
        """ Return the value for the constant in the given unit, otherwise will return the units in SI units """
        if unit is None:
            return self
        return self * units(self.unit, unit)


__all__ += ['c', 'h', 'hbar', 'm_e', 'm_p', 'G']

c = PhysicalConstant(299792458, 'm/s')
h = PhysicalConstant(6.62607004081e-34, 'J s')
hbar = PhysicalConstant(1.05457180013e-34, 'J s')
m_e = PhysicalConstant(9.1093835611e-31, 'kg')
m_p = PhysicalConstant(1.67262189821e-27, 'kg')
G = PhysicalConstant(6.6740831e-11, 'm^3/kg/s^2')
