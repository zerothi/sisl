# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from sisl._core import AtomGhost, PeriodicTable

from .data_source import DataSource


class AtomData(DataSource):
    def function(self, geometry, atoms=None):
        raise NotImplementedError("")


@AtomData.from_func
def AtomCoords(geometry, atoms=None):
    return geometry.xyz[atoms]


@AtomData.from_func
def AtomX(geometry, atoms=None):
    return geometry.xyz[atoms, 0]


@AtomData.from_func
def AtomY(geometry, atoms=None):
    return geometry.xyz[atoms, 1]


@AtomData.from_func
def AtomZ(geometry, atoms=None):
    return geometry.xyz[atoms, 2]


@AtomData.from_func
def AtomFCoords(geometry, atoms=None):
    return geometry.sub(atoms).fxyz


@AtomData.from_func
def AtomFx(geometry, atoms=None):
    return geometry.sub(atoms).fxyz[:, 0]


@AtomData.from_func
def AtomFy(geometry, atoms=None):
    return geometry.sub(atoms).fxyz[:, 1]


@AtomData.from_func
def AtomFz(geometry, atoms=None):
    return geometry.sub(atoms).fxyz[:, 2]


@AtomData.from_func
def AtomR(geometry, atoms=None):
    return geometry.sub(atoms).maxR(all=True)


@AtomData.from_func
def AtomZ(geometry, atoms=None):
    return geometry.sub(atoms).atoms.Z


@AtomData.from_func
def AtomNOrbitals(geometry, atoms=None):
    return geometry.sub(atoms).orbitals


class AtomColors(AtomData):
    _fallback_color: str = "pink"
    _atoms_colors: dict[str, str] = {}

    def function(self, geometry, atoms=None):
        return np.array(
            [
                self._atoms_colors.get(atom.symbol, self._fallback_color)
                for atom in geometry.sub(atoms).atoms
            ]
        )


class JMolAtomColors(AtomColors):

    _atoms_colors = {
        "H": "#EDEADE",
        "He": "#D9FFFF",
        "Li": "#CC80FF",
        "Be": "#C2FF00",
        "B": "#FFB5B5",
        "C": "#909090",
        "N": "#3050F8",
        "O": "#FF0D0D",
        "F": "#90E050",
        "Ne": "#B3E3F5",
        "Na": "#AB5CF2",
        "Mg": "#8AFF00",
        "Al": "#BFA6A6",
        "Si": "#F0C8A0",
        "P": "#FF8000",
        "S": "#FFFF30",
        "Cl": "#1FF01F",
        "Ar": "#80D1E3",
        "K": "#8F40D4",
        "Ca": "#3DFF00",
        "Sc": "#E6E6E6",
        "Ti": "#BFC2C7",
        "V": "#A6A6AB",
        "Cr": "#8A99C7",
        "Mn": "#9C7AC7",
        "Fe": "#E06633",
        "Co": "#F090A0",
        "Ni": "#50D050",
        "Cu": "#C88033",
        "Zn": "#7D80B0",
        "Ga": "#C28F8F",
        "Ge": "#668F8F",
        "As": "#BD80E3",
        "Se": "#FFA100",
        "Br": "#A62929",
        "Kr": "#5CB8D1",
        "Rb": "#702EB0",
        "Sr": "#00FF00",
        "Y": "#94FFFF",
        "Zr": "#94E0E0",
        "Nb": "#73C2C9",
        "Mo": "#54B5B5",
        "Tc": "#3B9E9E",
        "Ru": "#248F8F",
        "Rh": "#0A7D8C",
        "Pd": "#006985",
        "Ag": "#C0C0C0",
        "Cd": "#FFD98F",
        "In": "#A67573",
        "Sn": "#668080",
        "Sb": "#9E63B5",
        "Te": "#D47A00",
        "I": "#940094",
        "Xe": "#429EB0",
        "Cs": "#57178F",
        "Ba": "#00C900",
        "La": "#70D4FF",
        "Ce": "#FFFFC7",
        "Pr": "#D9FFC7",
        "Nd": "#C7FFC7",
        "Pm": "#A3FFC7",
        "Sm": "#8FFFC7",
        "Eu": "#61FFC7",
        "Gd": "#45FFC7",
        "Tb": "#30FFC7",
        "Dy": "#1FFFC7",
        "Ho": "#00FF9C",
        "Er": "#00E675",
        "Tm": "#00D452",
        "Yb": "#00BF38",
        "Lu": "#00AB24",
        "Hf": "#4DC2FF",
        "Ta": "#4DA6FF",
        "W": "#2194D6",
        "Re": "#267DAB",
        "Os": "#266696",
        "Ir": "#175487",
        "Pt": "#D0D0E0",
        "Au": "#FFD123",
        "Hg": "#B8B8D0",
        "Tl": "#A6544D",
        "Pb": "#575961",
        "Bi": "#9E4FB5",
        "Po": "#AB5C00",
        "At": "#754F45",
        "Rn": "#428296",
        "Fr": "#420066",
        "Ra": "#007D00",
        "Ac": "#70ABFA",
        "Th": "#00BAFF",
        "Pa": "#00A1FF",
        "U": "#008FFF",
        "Np": "#0080FF",
        "Pu": "#006BFF",
        "Am": "#545CF2",
        "Cm": "#785CE3",
        "Bk": "#8A4FE3",
        "Cf": "#A136D4",
        "Es": "#B31FD4",
        "Fm": "#B31FBA",
        "Md": "#B30DA6",
        "No": "#BD0D87",
        "Lr": "#C70066",
        "Rf": "#CC0059",
        "Db": "#D1004F",
        "Sg": "#D90045",
        "Bh": "#E00038",
        "Hs": "#E6002E",
        "Mt": "#EB0026",
    }


@AtomData.from_func
def AtomIsGhost(geometry, atoms=None, fill_true=True, fill_false=False):
    return np.array(
        [
            fill_true if isinstance(atom, AtomGhost) else fill_false
            for atom in geometry.sub(atoms).atoms
        ]
    )


@AtomData.from_func
def AtomPeriodicTable(geometry, atoms=None, what=None, pt=PeriodicTable):
    if not isinstance(pt, PeriodicTable):
        pt = pt()
    function = getattr(pt, what)
    return function(geometry.sub(atoms).atoms.Z)
