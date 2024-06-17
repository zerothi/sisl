# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Library for converting units and creating numpy arrays
with automatic unit conversion.

The conversion factors are taken directly from Siesta
which means these unit conversions should be used for Siesta "stuff".
"""

from sisl._environ import get_environ_variable, register_environ_variable
from sisl._internal import set_module

from .base import UnitParser
from .base import unit_convert as u_convert
from .base import unit_default as u_default
from .base import unit_group as u_group
from .base import unit_table

__all__ = ["unit_group", "unit_convert", "unit_default", "units"]


register_environ_variable(
    "SISL_UNIT_SIESTA",
    "codata2018",
    "Choose default units used when parsing Siesta files. [codata2018, legacy]",
    process=str.lower,
)


unit_table_siesta_codata2018 = dict(
    {key: dict(values) for key, values in unit_table.items()}
)
unit_table_siesta_legacy = dict(
    {key: dict(values) for key, values in unit_table.items()}
)

unit_table_siesta_legacy["length"].update(
    {
        "Bohr": 0.529177e-10,
    }
)

unit_table_siesta_legacy["time"].update(
    {
        "mins": 60.0,
        "hours": 3600.0,
        "days": 86400.0,
    }
)

unit_table_siesta_legacy["energy"].update(
    {
        "meV": 1.60219e-22,
        "eV": 1.60219e-19,
        "mRy": 2.17991e-21,
        "Ry": 2.17991e-18,
        "mHa": 4.35982e-21,
        "Ha": 4.35982e-18,
        "Hartree": 4.35982e-18,
        "K": 1.38066e-23,
        "kJ/mol": 1.6606e-21,
        "Hz": 6.6262e-34,
        "THz": 6.6262e-22,
        "cm-1": 1.986e-23,
        "cm**-1": 1.986e-23,
        "cm^-1": 1.986e-23,
    }
)

unit_table_siesta_legacy["force"].update(
    {
        "eV/Ang": 1.60219e-9,
        "eV/Bohr": 1.60219e-9 * 0.529177,
        "Ry/Bohr": 4.11943e-8,
        "Ry/Ang": 4.11943e-8 / 0.529177,
    }
)


# Check for the correct handlers
_def_unit = get_environ_variable("SISL_UNIT_SIESTA")
if _def_unit in ("codata2018", "codata"):
    unit_table_siesta = unit_table_siesta_codata2018
elif _def_unit in ("legacy", "original"):
    unit_table_siesta = unit_table_siesta_legacy
else:
    raise ValueError(
        f"Could not understand SISL_UNIT_SIESTA={_def_unit}, expected one of [codata2018, legacy]"
    )


@set_module("sisl.unit.siesta")
def unit_group(unit, tbl=unit_table_siesta):
    return u_group(unit, tbl)


@set_module("sisl.unit.siesta")
def unit_default(group, tbl=unit_table_siesta):
    return u_default(group, tbl)


@set_module("sisl.unit.siesta")
def unit_convert(fr, to, opts=None, tbl=unit_table_siesta):
    return u_convert(fr, to, opts, tbl)


# create unit-parser
units = UnitParser(unit_table_siesta)
units_legacy = UnitParser(unit_table_siesta_legacy)
