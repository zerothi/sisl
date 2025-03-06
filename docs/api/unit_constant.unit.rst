
Unit conversion
===============

.. currentmodule:: sisl.unit

Generic conversion utility between different units.

All different codes unit conversion routines
should adhere to the same routine names for consistency and
readability. This package should supply a subpackage for each
code where specific unit conversions are required. I.e. if
the codes unit conversion are not the same as the sisl defaults.

.. autosummary::
   :toctree: generated/

   unit_group - which group does the unit belong to
   unit_convert - conversion factor between two units
   unit_default - the default unit in a group
   units - a class used to transfer complex units (eV/Ang -> Ry/Ang etc.)

All subsequent subpackages also exposes the above 4 methods. If
a subpackage method is used, the unit conversion corresponds to
the units defined in the respective code.

The `units` object is by far the easiest version to use since it handles
complex units (Ry/kg/Bohr N) while `unit_convert` is the basic unit-conversion
table that only converts simple units. E.g. Ry to eV etc.



Siesta units
------------

.. currentmodule:: sisl.unit.siesta

This subpackage implements the unit conversions used in `Siesta`_.

To use the unit conversion from `Siesta`_, simply import `units` as:

>>> from sisl.unit import units
>>> from sisl.unit.siesta import units as siesta_units

in which case ``units`` will refer to default unit conversions and ``siesta_units``
will use the unit definitions in `Siesta`_ up to and including 4.1.X versions.
