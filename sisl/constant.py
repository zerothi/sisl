"""
=========================================
Physical constants (:mod:`sisl.constant`)
=========================================

.. module:: sisl.constant

Module containing a pre-set set of physical constants. The SI units are following the *new* convention
that takes effect on 20 May 2019.

The currently stored constants are (all are given in SI units):

.. autosummary::

   PhysicalConstant
   q
   c
   h
   hbar
   m_e
   m_p
   G0
   G


.. we need to add them to the toctree to be able to create links to them


.. autosummary::
   :toctree:
   :hidden:

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
from __future__ import print_function, division

from sisl.unit.base import units

__all__ = ['PhysicalConstant']


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
        c = float.__new__(cls, value)
        c._unit = unit
        return c

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


__all__ += ['q', 'c', 'h', 'hbar', 'm_e', 'm_p', 'G', 'G0']


#: Unit of charge [C], or [A s]
q = PhysicalConstant(1.602176634e-19, 'A s')
#: Speed of light [m/s]
c = PhysicalConstant(299792458, 'm/s')
#: Plancks constant [J s]
h = PhysicalConstant(6.62607015e-34, 'J s')
#: Reduced Plancks constant [J s]
hbar = PhysicalConstant(1.0545718176461565e-34, 'J s')
#: Mass of electron [kg]
m_e = PhysicalConstant(9.1093835611e-31, 'kg')
#: Mass of proton [kg]
m_p = PhysicalConstant(1.67262189821e-27, 'kg')
#: Conductance quantum [S], or [m^2/s^2]
G0 = PhysicalConstant(2 * (q ** 2 / h), 'm^2/s^2')
#: Gravitational constant [m^3/kg/s^2]
G = PhysicalConstant(6.6740831e-11, 'm^3/kg/s^2')
