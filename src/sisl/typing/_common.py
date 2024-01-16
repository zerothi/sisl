# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import numpy as np

import sisl._typing_ext.numpy as npt
from sisl import Atom, Atoms, BaseSile, Geometry, Grid, Lattice, Shape
from sisl._category import GenericCategory
from sisl.geom.category import AtomCategory

__all__ = [
    "AtomsLike",
    "Coord",
    "CoordOrScalar",
    "SileLike",
    "GeometryLike",
    "LatticeLike",
    "GridLike",
    "LatticeOrGeometry",
    "LatticeOrGeometryLike",
    "AtomsArgument",
    "OrbitalsArgument",
]

AtomsLike = Union[
    Atom,
    Atoms,
]

Coord = Sequence[float]
CoordOrScalar = Union[float, Coord]

SileLike = Union[
    str,
    BaseSile,
    Path,
]

GeometryLike = Union[
    SileLike,
    Geometry,
]

LatticeLike = Union[
    SileLike,
    Lattice,
]

GridLike = Union[
    SileLike,
    Grid,
]

LatticeOrGeometry = Union[
    Lattice,
    Geometry,
]

LatticeOrGeometryLike = Union[
    LatticeLike,
    GeometryLike,
]

# An atoms like argument that may be parsed by Geometry._sanitize_atoms
AtomsArgument = Union[
    npt.NDArray[Union[np.int_, np.bool_]],
    str,
    int,
    dict,
    Atom,
    AtomCategory,
    GenericCategory,
    Shape,
]

# An atoms like argument that may be parsed by Geometry._sanitize_orbs
OrbitalsArgument = Union[
    npt.NDArray[Union[np.int_, np.bool_]],
    str,
    int,
    dict,
    AtomCategory,
    Shape,
]
