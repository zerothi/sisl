# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Unit conversion
===============

Generic conversion utility between different units.

All different codes unit conversion routines
should adhere to the same routine names for consistency and
readability. This package should supply a subpackage for each
code where specific unit conversions are required. I.e. if
the codes unit conversion are not the same as the sisl defaults.

Default unit conversion utilities
---------------------------------

   unit_group - which group does the unit belong to
   unit_convert - conversion factor between to units
   unit_default - the default unit in a group
   units - a class used to transfer complex units (eV/Ang -> Ry/Ang etc.)

All subsequent subpackages also exposes the above 4 methods. If
a subpackage method is used, the unit conversion corresponds to
the units defined in the respective code.

The `units` object is by far the easiest version to use since it handles
complex units (Ry/kg/Bohr N) while `unit_convert` is the basic unit-conversion
table that only converts simple units. E.g. Ry to eV etc.


Siesta units (:mod:`sisl.unit.siesta`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This subpackage implements the unit conversions used in `Siesta`_.

To use the unit conversion from `Siesta`_, simply import `units` as:

>>> from sisl.unit import units
>>> from sisl.unit.siesta import units as siesta_units

in which case ``units`` will refer to default unit conversions and ``siesta_units``
will use the unit definitions in `Siesta`_.
"""
# Enable the siesta unit-conversion
from . import siesta
from .base import serialize_units_arg, unit_convert, unit_default, unit_group, units
