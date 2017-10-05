"""
===================================
Unit conversion (:mod:`sisl.unit`)
===================================

.. module:: sisl.unit

Generic conversion utility between different units.

All different codes unit conversion routines
should adhere to the same routine names for consistency and
readability. This package should supply a subpackage for each
code where specific unit conversions are required. I.e. if
the codes unit conversion are not the same as the sisl defaults.

Default unit conversion utilities
=================================

.. autosummary::
   :toctree:

   unit_group - which group does the unit belong to
   unit_convert - conversion factor between to units
   unit_default - the default unit in a group

All subsequent subpackages also exposes the above 3 methods. If
a subpackage method is used, the unit conversion corresponds to
the units defined in the respective code.


Siesta units (:mod:`sisl.unit.siesta`)
---------------------------------------

.. currentmodule:: sisl.unit.siesta

This subpackage implements the unit conversions used in `Siesta`_.
"""

from .base import unit_group, unit_convert, unit_default

__all__ = [s for s in dir() if not s.startswith('_')]

# Enable the siesta unit-conversion
from . import siesta

__all__ += ['siesta']
