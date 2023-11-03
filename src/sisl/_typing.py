# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# This contains typing things specific to sisl
from __future__ import annotations

from pathlib import Path
from typing import Union

from sisl import Atom, Atoms, Geometry, Lattice

# from typing import TYPE_CHECKING, final

AtomsLike = Union[
    Atom,
    Atoms,
]

SileLike = Union[
    "BaseSile",
    str,
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

LatticeOrGeometry = Union[
    Lattice,
    Geometry,
]

LatticeOrGeometryLike = Union[
    LatticeLike,
    Geometry,
]
