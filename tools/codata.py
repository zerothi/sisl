# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np


def parse_line(line):
    """Parse a data line returning

    name, value, (error), (unit)
    """
    lines = line.strip().replace("Hz^-1", "s").split("  ")
    lines = [line.strip() for line in lines if line]
    if len(lines) in (3, 4):
        lines[1] = float(lines[1].replace(" ", "").replace("...", ""))
        if "exact" in lines[2]:
            lines[2] = 0.0
        else:
            lines[2] = float(lines[2].replace(" ", ""))
        return lines
    raise ValueError("Could not parse the line correctly.")


# Which units to store in the table
UNITS = {
    "atomic mass constant": ("mass", "amu"),
    "Bohr radius": ("length", "Bohr"),
    "atomic unit of time": ("time", "atu"),
    "atomic unit of charge": ("energy", "eV"),
    "Rydberg constant times hc in J": ("energy", "Ry"),
    "Hartree energy": ("energy", "Ha"),
    "Boltzmann constant": ("energy", "K"),
}


@dataclass
class Constant:
    doc: str
    name: str
    value: float = 0
    unit: str = ""

    def __str__(self):
        return f'#: {self.doc} [{self.unit}]\n{self.name} = PhysicalConstant({self.value}, "{self.unit}")'


CONSTANTS = {
    "speed of light in vacuum": Constant("Speed of light in vacuum", "c"),
    "atomic unit of charge": Constant("Unit of charge", "q"),
    "proton mass": Constant("Proton mass", "m_p"),
    "electron mass": Constant("Electron mass", "m_e"),
    "Planck constant": Constant("Planck constant", "h"),
    "Bohr radius": Constant("Bohr radius", "a0"),
    "Boltzmann constant": Constant("Boltzmann constant", "kB"),
}


def read_file(f):
    fh = open(f)
    start = False

    # This is the default table of units
    unit_table = {
        "mass": {
            "DEFAULT": "amu",
            "kg": 1.0,
            "g": 1.0e-3,
        },
        "length": {
            "DEFAULT": "Ang",
            "m": 1.0,
            "cm": 0.01,
            "nm": 1.0e-9,
            "Ang": 1.0e-10,
            "pm": 1.0e-12,
            "fm": 1.0e-15,
        },
        "time": {
            "DEFAULT": "fs",
            "s": 1.0,
            "ns": 1.0e-9,
            "ps": 1.0e-12,
            "fs": 1.0e-15,
            "min": 60.0,
            "hour": 3600.0,
            "day": 86400.0,
        },
        "energy": {
            "DEFAULT": "eV",
            "J": 1.0,
            "erg": 1.0e-7,
            "K": 1.380648780669e-23,
        },
        "force": {
            "DEFAULT": "eV/Ang",
            "N": 1.0,
        },
    }

    constants = []

    for line in fh:
        if "-----" in line:
            start = True
            continue
        if not start:
            continue

        name, value, *error_unit = parse_line(line)

        if name in UNITS:
            entry, key = UNITS[name]

            unit_table[entry][key] = value
            if key in ("Ry", "eV", "Ha"):
                unit_table[entry][f"m{key}"] = value / 1000

        if name in CONSTANTS:
            c = CONSTANTS[name]
            c.value = value
            if len(error_unit) == 2:
                c.unit = error_unit[1]
            else:
                c.unit = error_unit[0]

            constants.append(c)
            if c.name == "h":
                c = c.__class__(
                    f"Reduced {c.doc}", "hbar", c.value / (2 * np.pi), c.unit
                )
                constants.append(c)

    # Clarify force
    unit_table["force"][f"eV/Ang"] = (
        unit_table["energy"]["eV"] / unit_table["length"]["Ang"]
    )
    unit_table["force"][f"Ry/Bohr"] = (
        unit_table["energy"]["Ry"] / unit_table["length"]["Bohr"]
    )
    unit_table["force"][f"Ha/Bohr"] = (
        unit_table["energy"]["Ha"] / unit_table["length"]["Bohr"]
    )

    return unit_table, constants


ut, cs = read_file(sys.argv[1])

from pprint import PrettyPrinter

pp = PrettyPrinter(sort_dicts=False, compact=False, width=60)

print("Unit table:\n\n")
pp.pprint(ut)

print("\n\nConstants:\n\n")
for c in cs:
    print(str(c))
