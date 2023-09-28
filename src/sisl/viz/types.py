from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NewType, Optional, Sequence, TypedDict, Union

import numpy as np
import numpy.typing as npt

import sisl
from sisl.geometry import AtomCategory, Geometry
from sisl.io.sile import BaseSile
from sisl.lattice import Lattice, LatticeChild
from sisl.typing import AtomsArgument

PathLike = Union[str, Path,  BaseSile]

Color = NewType("Color", str)

GeometryLike = Union[sisl.Geometry, Any]

Axis = Union[Literal["x", "y", "z", "-x", "-y", "-z", "a", "b", "c", "-a", "-b", "-c"],  Sequence[float]]
Axes = Sequence[Axis]

GeometryLike = Union[Geometry, PathLike]

@dataclass
class StyleSpec:
    color: Optional[Color] = None
    size: Optional[float] = None
    opacity: Optional[float] = 1
    dash: Optional[Literal["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]] = None

@dataclass
class AtomsStyleSpec(StyleSpec):
    atoms: AtomsArgument = None
    vertices: Optional[float] = 15

class AtomsStyleSpecDict(TypedDict):
    atoms: AtomsArgument
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
    atoms: AtomsArgument = None
    species : SpeciesSpec = None
    orbitals: OrbitalsNames = None
    n: Optional[Sequence[int]] = None
    l: Optional[Sequence[int]] = None
    m: Optional[Sequence[int]] = None
    spin: SpinIndex = None
    scale: float = 1
    reduce: Literal["mean", "sum"] = "sum"
    spin_reduce: Literal["mean", "sum"] = "sum"

@dataclass
class OrbitalStyleQuery(StyleSpec, OrbitalQuery):
    ...
    
OrbitalQueries = Sequence[OrbitalQuery]
OrbitalStyleQueries = Sequence[OrbitalStyleQuery]

CellLike = Union[npt.NDArray[np.float_], Lattice, LatticeChild]

@dataclass
class ArrowSpec:
    scale: float = 1.
    color: Any = None
    width: float = 1.
    opacity: float = 1.
    name: str = "arrow"
    annotate: bool = False
    arrowhead_scale: float = 0.2
    arrowhead_angle: float = 20

@dataclass
class AtomArrowSpec:
    data: Any
    atoms: AtomsArgument = None
    scale: float = 1.
    color: Any = None
    width: float = 1.
    opacity: float = 1.
    name: str = "arrow"
    annotate: bool = False
    arrowhead_scale: float = 0.2
    arrowhead_angle: float = 20

