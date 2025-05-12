# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional, Union

import pyparsing as pp

from sisl._internal import set_module

from .codata import CODATA

__all__ = ["unit_group", "unit_convert", "unit_default", "units", "serialize_units_arg"]


# We do not import anything as it depends on the package.
# Here we only add the conversions according to the
# standard. Other programs may use their units as they
# please with non-standard conversion factors.
# This is the CODATA following env: SISL_CODATA


UnitTableT = Mapping[str, Mapping[str, Union[float, str]]]


@set_module("sisl.unit")
@dataclass
class UnitTable:
    codata: dict = field(hash=False)
    table: dict = field(
        repr=False, default_factory=dict, compare=False, hash=False
    )  # initial table of units (of which the rest are built!)
    processed_table: dict = field(
        init=False, repr=False, default_factory=dict, compare=False, hash=True
    )

    def __post_init__(self):
        """Create the full processed table of units based of table and `codata`"""

        def add_value(dic, key, value):
            if key not in dic:
                if isinstance(value, str):
                    value = self.codata[value].value
                dic[key] = value

        mass = dict(self.table.get("mass", {}))
        mass["DEFAULT"] = "amu"
        add_value(mass, "kg", 1.0)
        assert abs(mass["kg"] - 1.0) < 1e-15, "Default unit of mass not obeyed!"
        add_value(mass, "g", 1.0e-3)
        add_value(mass, "amu", "atomic mass constant")
        add_value(mass, "m_e", "electron mass")
        add_value(mass, "m_n", "neutron mass")
        add_value(mass, "m_p", "proton mass")
        self.processed_table["mass"] = mass

        length = dict(self.table.get("length", {}))
        length["DEFAULT"] = "Ang"
        add_value(length, "m", 1.0)
        assert abs(length["m"] - 1.0) < 1e-15, "Default unit of length not obeyed!"
        add_value(length, "cm", 0.01)
        add_value(length, "km", 1e3)
        add_value(length, "nm", 1.0e-9)
        add_value(length, "Ang", 1.0e-10)
        add_value(length, "pm", 1.0e-12)
        add_value(length, "fm", 1.0e-15)
        add_value(length, "Bohr", "Bohr radius")
        self.processed_table["length"] = length

        time = dict(self.table.get("time", {}))
        time["DEFAULT"] = "fs"
        add_value(time, "s", 1.0)
        assert abs(time["s"] - 1.0) < 1e-15, "Default unit of time not obeyed!"
        add_value(time, "ns", 1e-9)
        add_value(time, "ps", 1e-12)
        add_value(time, "fs", 1e-15)
        add_value(time, "min", 60)
        add_value(time, "hour", 3600)
        add_value(time, "day", 3600 * 24)
        add_value(time, "atu", "atomic unit of time")
        self.processed_table["time"] = time

        energy = dict(self.table.get("energy", {}))
        energy["DEFAULT"] = "eV"
        add_value(energy, "J", 1.0)
        assert abs(energy["J"] - 1.0) < 1e-15, "Default unit of energy not obeyed!"
        add_value(energy, "kJ", 1e3)
        add_value(energy, "cal", 4.184)
        add_value(energy, "kcal", 4184)
        add_value(energy, "erg", 1e-7)
        add_value(energy, "K", "kelvin-joule relationship")
        add_value(energy, "Hz", "hertz-joule relationship")
        add_value(energy, "MHz", energy["Hz"] * 1e6)
        add_value(energy, "GHz", energy["Hz"] * 1e9)
        add_value(energy, "THz", energy["Hz"] * 1e12)
        value = CODATA["inverse meter-joule relationship"].value
        add_value(energy, "cm**-1", value * 100)
        add_value(energy, "cm^-1", energy["cm**-1"])
        add_value(energy, "invcm", energy["cm**-1"])
        add_value(energy, "eV", "electron volt")
        add_value(energy, "Ha", "Hartree energy")
        add_value(energy, "Ry", "Rydberg constant times hc in J")
        for e in ("eV", "Ha", "Ry"):
            add_value(energy, f"m{e}", energy[e] * 1e-3)
        self.processed_table["energy"] = energy

        force = dict(self.table.get("force", {}))
        force["DEFAULT"] = "eV/Ang"
        add_value(force, "N", 1.0)
        assert abs(force["N"] - 1.0) < 1e-15, "Default unit of force not obeyed!"
        add_value(force, "dyn", 1e5)
        for e in ("eV", "Ry", "Ha"):
            for m in ("Ang", "Bohr", "nm"):
                add_value(force, f"{e}/{m}", energy[e] / length[m])
                add_value(force, f"m{e}/{m}", energy[f"m{e}"] / length[m])
        self.processed_table["force"] = force

        velocity = dict(self.table.get("velocity", {}))
        velocity["DEFAULT"] = "Ang/fs"
        add_value(velocity, "m/s", 1.0)
        assert (
            abs(velocity["m/s"] - 1.0) < 1e-15
        ), "Default unit of velocity not obeyed!"
        add_value(velocity, "km/hour", length["km"] / time["hour"])
        for m in ("Ang", "Bohr", "nm"):
            for t in ("ns", "fs", "ps"):
                add_value(velocity, f"{m}/{t}", length[m] / time[t])
        self.processed_table["velocity"] = velocity

        pressure = dict(self.table.get("pressure", {}))
        pressure["DEFAULT"] = "eV/Ang^3"
        add_value(pressure, "Pa", 1.0)
        assert abs(pressure["Pa"] - 1.0) < 1e-15, "Default unit of pressure not obeyed!"
        add_value(pressure, "GPa", pressure["Pa"] * 1e9)
        add_value(pressure, "atm", "standard atmosphere")
        add_value(pressure, "bar", "standard-state pressure")
        add_value(pressure, "kbar", pressure["bar"] * 1e3)
        add_value(pressure, "Mbar", pressure["bar"] * 1e6)
        add_value(pressure, "eV/Ang^3", energy["eV"] / length["Ang"] ** 3)
        add_value(pressure, "Ry/Bohr^3", energy["Ry"] / length["Bohr"] ** 3)
        add_value(pressure, "Ha/Bohr^3", energy["Ha"] / length["Bohr"] ** 3)
        for unit in ("eV/Ang^3", "Ry/Bohr^3", "Ha/Bohr^3"):
            add_value(pressure, unit.replace("^", "**"), pressure[unit])
        self.processed_table["pressure"] = pressure

        efield = dict(self.table.get("efield", {}))
        efield["DEFAULT"] = "V/Ang"
        add_value(efield, "V/m", 1.0)
        assert abs(efield["V/m"] - 1.0) < 1e-15, "Default unit of efield not obeyed!"
        add_value(efield, "V/cm", efield["V/m"] * 1e2)
        add_value(efield, "V/nm", efield["V/m"] * 1e9)
        add_value(efield, "V/Ang", efield["V/m"] * 1e10)
        add_value(efield, "V/Bohr", 1 / length["Bohr"])
        self.processed_table["efield"] = efield

    # The below is useful to ensure it is `dict` compatible
    def get(self, key, default=None):
        return self.processed_table.get(key, default)

    def keys(self):
        return self.processed_table.keys()

    def items(self):
        return self.processed_table.items()

    def values(self):
        return self.processed_table.values()

    def __getitem__(self, key):
        return self.processed_table[key]

    def __contains__(self, key):
        return key in self.processed_table

    def __len__(self):
        return len(self.processed_table)

    def __iter__(self):
        yield from self.processed_table

    def groups(self) -> list[str]:
        """Form the list of groups that is present in the current unit-specification"""
        return list(self.keys())

    def group(self, unit: str) -> str:
        return unit_group(unit, tbl=self)

    def default(self, group: str) -> str:
        return unit_default(group, tbl=self)

    def units(self, group: str) -> list[str]:
        """Form the list of units within a certain unit-gruop"""
        units = list(self[group].keys())
        del units["DEFAULT"]
        return units


unit_table = UnitTable(CODATA)


@set_module("sisl.unit")
def unit_group(unit: str, tbl: UnitTableT = unit_table) -> str:
    """The group of units that `unit` belong to

    Parameters
    ----------
    unit :
      unit, e.g. kg, Ang, eV etc. returns the type of unit it is.
    tbl : dict, optional
        dictionary of units (default to the global table)

    Examples
    --------
    >>> unit_group("kg")
    "mass"
    >>> unit_group("eV")
    "energy"
    """
    for k in tbl:
        if unit in tbl[k]:
            return k
    raise ValueError(f"The unit " "{unit!s}" " could not be located in the table.")


@set_module("sisl.unit")
def unit_default(group: str, tbl: UnitTableT = unit_table) -> str:
    """The default unit of the unit group `group`.

    Parameters
    ----------
    group :
       look-up in the table for the default unit.
    tbl : dict, optional
        dictionary of units (default to the global table)

    Examples
    --------
    >>> unit_default("energy")
    "eV"
    """
    grp = tbl.get(group, {})
    if "DEFAULT" in grp:
        return grp["DEFAULT"]

    raise ValueError("The unit-group does not exist!")


@set_module("sisl.unit")
def unit_convert(
    fr: str,
    to: str,
    opts: Optional[Mapping[str, float]] = None,
    tbl: UnitTableT = unit_table,
) -> float:
    """Factor that takes `fr` to the units of `to`

    Parameters
    ----------
    fr :
        starting unit
    to :
        ending unit
    opts :
        controls whether the unit conversion is in powers or fractional units
    tbl :
        dictionary of units (default to the global table)

    Examples
    --------
    >>> unit_convert("kg","g")
    1000.0
    >>> unit_convert("eV","J")
    1.60217733e-19
    """
    if opts is None:
        opts = dict()

    # In the case that the conversion to is None, we should do nothing.
    frU = "FromNotFound"
    frV = None
    toU = "ToNotFound"
    toV = None

    # Check that the unit types live in the same
    # space
    # TODO this currently does not handle if powers are taken into
    # consideration.

    for k in tbl:
        if fr in tbl[k]:
            frU = k
            frV = tbl[k][fr]
        if to in tbl[k]:
            toU = k
            toV = tbl[k][to]
    if frU != toU:
        raise ValueError(
            f"The unit conversion is not from the same group: {frU} to {toU}"
        )

    # Calculate conversion factor
    val = frV / toV
    for opt in ("^", "power", "p"):
        if opt in opts:
            val = val ** opts[opt]
    for opt in ("*", "factor", "fac"):
        if opt in opts:
            val = val * opts[opt]
    for opt in ("/", "divide", "div"):
        if opt in opts:
            val = val / opts[opt]

    return val


# From here and on we implement the generalized parser required for
# doing complex unit-specifications (i.e. eV/Ang etc.)


@set_module("sisl.unit")
class UnitParser:
    """Object for converting between units for a set of unit-tables.

    Parameters
    ----------
    table :
       a table with the units parseable by the class
    """

    __slots__ = ("_table", "_p_left", "_left", "_p_right", "_right")

    def __init__(self, table: UnitTableT):
        self._table = table

        def value(unit: str):
            tbl = self._table
            for k in tbl:
                if unit in tbl[k]:
                    return tbl[k][unit]
            raise ValueError(f"The unit conversion did not contain unit {unit}!")

        def group(unit: str):
            tbl = self._table
            for k in tbl:
                if unit in tbl[k]:
                    return k
            raise ValueError(
                f"The unit " "{unit!s}" " could not be located in the table."
            )

        def default(group: str):
            tbl = self._table
            k = tbl.get(group, None)
            if k is None:
                raise ValueError(f"The unit-group {group} does not exist!")
            return k["DEFAULT"]

        self._left = []
        self._p_left = self.create_parser(value, default, group, self._left)
        self._right = []
        self._p_right = self.create_parser(value, default, group, self._right)

    @staticmethod
    def create_parser(value, default, group, group_table=None):
        """Routine to internally create a parser with specified unit_convert, unit_default and unit_group routines"""

        # Any length of characters will be used as a word.
        if group_table is None:

            def _value(t):
                return value(t[0])

            def _float(t):
                return float(t[0])

        else:

            def _value(t):
                group_table.append(group(t[0]))
                return value(t[0])

            def _float(t):
                f = float(t[0])
                group_table.append(f)  # append nothing
                return f

        # The unit extractor
        unit = pp.Word(pp.alphas).setParseAction(_value)

        integer = pp.Word(pp.nums)
        plusorminus = pp.oneOf("+ -")
        point = pp.Literal(".")
        e = pp.CaselessLiteral("E")
        sign_integer = pp.Combine(pp.Optional(plusorminus) + integer)
        exponent = pp.Combine(e + sign_integer)
        sign_integer = pp.Combine(pp.Optional(plusorminus) + integer)
        exponent = pp.Combine(e + sign_integer)
        number = pp.Or(
            [
                pp.Combine(point + integer + pp.Optional(exponent)),  # .[0-9][E+-[0-9]]
                pp.Combine(
                    integer
                    + pp.Optional(point + pp.Optional(integer))
                    + pp.Optional(exponent)
                ),
            ]  # [0-9].[0-9][E+-[0-9]]
        ).setParseAction(_float)

        # def _print_toks(name, op):
        #    """ May be used in pow_op.setParseAction(_print_toks("pow", "^")) to debug """
        #    def T(t):
        #        print("{}: {}".format(name, t))
        #        return op
        #    return T

        # def _fix_toks(op):
        #    """ May be used in pow_op.setParseAction(_print_toks("pow", "^")) to debug """
        #    def T(t):
        #        return op
        #    return T

        pow_op = pp.oneOf("^ **").setParseAction(lambda t: "^")
        mul_op = pp.Literal("*")
        div_op = pp.Literal("/")
        # Since any space in units are regarded as multiplication this will catch
        # those instances.
        base_op = pp.Empty()

        if group_table is None:

            def pow_action(toks):
                return toks[0][0] ** toks[0][2]

            def mul_action(toks):
                return toks[0][0] * toks[0][2]

            def div_action(toks):
                return toks[0][0] / toks[0][2]

            def base_action(toks):
                return toks[0][0] * toks[0][1]

        else:

            def pow_action(toks):
                # Fix table of units
                group = "{}^{}".format(group_table[-2], group_table.pop())
                group_table[-1] = group
                # print("^", toks[0], group_table)
                return toks[0][0] ** toks[0][2]

            def mul_action(toks):
                if isinstance(group_table[-2], float):
                    group_table.pop(-2)
                if isinstance(group_table[-1], float):
                    group_table.pop()
                # print("*", toks[0], group_table)
                return toks[0][0] * toks[0][2]

            def div_action(toks):
                if isinstance(group_table[-2], float):
                    group_table.pop(-2)
                if isinstance(group_table[-1], float):
                    group_table.pop()
                else:
                    group_table[-1] = "/{}".format(group_table[-1])
                # print("/", toks[0])
                return toks[0][0] / toks[0][2]

            def base_action(toks):
                if isinstance(group_table[-2], float):
                    group_table.pop(-2)
                if isinstance(group_table[-1], float):
                    group_table.pop()
                return toks[0][0] * toks[0][1]

        # We should parse numbers first
        parser = pp.infixNotation(
            number | unit,
            [
                (pow_op, 2, pp.opAssoc.RIGHT, pow_action),
                (mul_op, 2, pp.opAssoc.LEFT, mul_action),
                (div_op, 2, pp.opAssoc.LEFT, div_action),
                (base_op, 2, pp.opAssoc.LEFT, base_action),
            ],
        )

        return parser

    @staticmethod
    def same_group(A, B):
        """Return true if A and B have the same groups"""
        A.sort()
        B.sort()
        if len(A) != len(B):
            return False
        return all(a == b for a, b in zip(A, B))

    def _convert(self, A, B):
        """Internal routine used to convert unit `A` to unit `B`"""
        conv_A = self._p_left.parseString(A)[0]
        conv_B = self._p_right.parseString(B)[0]
        if not self.same_group(self._left, self._right):
            left = list(self._left)
            right = list(self._right)
            self._left.clear()
            self._right.clear()
            raise ValueError(
                f"The unit conversion is not from the same group: {left} to {right}!"
            )
        self._left.clear()
        self._right.clear()
        return conv_A / conv_B

    def convert(self, *units) -> Union[float, Tuple[float]]:
        """Conversion factors between units

        If 1 unit is passed a conversion to the default  will be returned.
        If 2 parameters are passed then a single float will be returned that converts from
        ``units[0]`` to ``units[1]``.
        If 3 or more parameters are passed then a tuple of floats will be returned where
        ``tuple[0]`` is the conversion from ``units[0]`` to ``units[1]``,
        ``tuple[1]`` is the conversion from ``units[1]`` to ``units[2]`` and so on.

        Parameters
        ----------
        *units : list of string
           units to be converted

        Examples
        --------
        >>> up = UnitParser(unit_table)
        >>> up.convert("kg", "g")
        1000.0
        >>> up.convert("kg", "g", "amu")
        (1000.0, 6.022140762081123e+23)

        Raises
        ------
        UnitSislError
            if the units are not commensurate
        """
        if len(units) == 2:
            # basic unit conversion
            return self._convert(units[0], units[1])

        elif len(units) == 1:
            # to default
            conv = self._p_left.parseString(units[0])[0]
            self._left.clear()
            return conv

        return tuple(self._convert(A, B) for A, B in zip(units[:-1], units[1:]))

    def __call__(self, *units):
        return self.convert(*units)

    def __getattr__(self, attr):
        """Return any attributes from the table where data is fetched from"""
        return getattr(self._table, attr)


# Create base sisl unit conversion object
units = UnitParser(unit_table)


@set_module("sisl.unit")
def serialize_units_arg(units):
    "Parse units arguments into a dictionary"

    if isinstance(units, str):
        return {unit_group(units): units}

    elif isinstance(units, (list, tuple)):
        new_units_arg = {}
        for u in units:
            g = unit_group(u)
            if g not in new_units_arg:
                # add new quantity to dictionary
                new_units_arg[g] = u
            else:
                raise ValueError(
                    f"Two units for the same quantity was specified. This is not allowed."
                )
        return new_units_arg

    else:

        return units
