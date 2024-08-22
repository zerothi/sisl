# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NewType, Optional, TypedDict, Union

from sisl.io.sile import BaseSile
from sisl.typing import AtomsIndex

PathLike = Union[str, Path, BaseSile]

Color = NewType("Color", str)

# A colorscale can be a scale name, a sequence of colors or a sequence of
# (value, color) tuples.
Colorscale = Union[str, Sequence[Color], Sequence[tuple[float, Color]]]

Axis = Union[
    Literal["x", "y", "z", "-x", "-y", "-z", "a", "b", "c", "-a", "-b", "-c"],
    Sequence[float],
]
Axes = Sequence[Axis]


@dataclass
class StyleSpec:
    color: Optional[Color] = None
    size: Optional[float] = None
    opacity: Optional[float] = 1
    dash: Optional[
        Literal["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
    ] = None


@dataclass
class AtomsStyleSpec(StyleSpec):
    atoms: AtomsIndex = None
    vertices: Optional[float] = 15


class AtomsStyleSpecDict(TypedDict):
    atoms: AtomsIndex
    color: Optional[Color]
    size: Optional[float]
    opacity: Optional[float]
    vertices: Optional[float]


@dataclass
class Query:
    active: bool = True
    name: str = ""


Queries = Sequence[Query]

SpeciesSpec = NewType("SpeciesSpec", Optional[Sequence[str]])

OrbitalsNames = NewType("OrbitalsNames", Optional[Sequence[str]])
SpinIndex = NewType("SpinIndex", Optional[Sequence[Literal[0, 1]]])


@dataclass
class OrbitalQuery(Query):
    atoms: AtomsIndex = None
    species: SpeciesSpec = None
    orbitals: OrbitalsNames = None
    n: Optional[Sequence[int]] = None
    l: Optional[Sequence[int]] = None
    m: Optional[Sequence[int]] = None
    spin: SpinIndex = None
    scale: float = 1
    reduce: Literal["mean", "sum"] = "sum"
    spin_reduce: Literal["mean", "sum"] = "sum"


@dataclass
class OrbitalStyleQuery(StyleSpec, OrbitalQuery): ...


OrbitalQueries = Sequence[OrbitalQuery]
OrbitalStyleQueries = Sequence[OrbitalStyleQuery]


@dataclass
class ArrowSpec:
    scale: float = 1.0
    color: Any = None
    width: float = 1.0
    opacity: float = 1.0
    name: str = "arrow"
    annotate: bool = False
    arrowhead_scale: float = 0.2
    arrowhead_angle: float = 20


@dataclass
class AtomArrowSpec:
    data: Any
    atoms: AtomsIndex = None
    scale: float = 1.0
    color: Any = None
    width: float = 1.0
    opacity: float = 1.0
    name: str = "arrow"
    annotate: bool = False
    arrowhead_scale: float = 0.2
    arrowhead_angle: float = 20
