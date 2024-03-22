# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Union

import numpy as np
import scipy.sparse as sps

from . import _numpy as npt

# To prevent import cycles place any internal imports in the branch below
# and use a string literal forward reference to it in subsequent types
# https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from sisl import BaseSile, Geometry, Grid, Lattice, Shape
    from sisl._category import GenericCategory
    from sisl.geom.category import AtomCategory

__all__ = [
    "Coord",
    "CoordOrScalar",
    "FuncType",
    "GeometryLike",
    "GridLike",
    "KPoint",
    "LatticeLike",
    "LatticeOrGeometry",
    "LatticeOrGeometryLike",
    "SeqFloat",
    "SeqOrScalarFloat",
    "SileLike",
    "SparseMatrix",
    "SparseMatrixExt",
]


SeqFloat = Sequence[float]
SeqOrScalarFloat = Union[float, SeqFloat]

Coord = SeqFloat
CoordOrScalar = Union[float, Coord]

KPoint = Sequence[float]

SileLike = Union[
    str,
    "BaseSile",
    Path,
]

GeometryLike = Union[
    SileLike,
    "Geometry",
]

# Short for *any* function
FuncType = Callable[..., Any]

LatticeLike = Union[
    SileLike,
    "Lattice",
]

GridLike = Union[
    SileLike,
    "Grid",
]

LatticeOrGeometry = Union[
    "Lattice",
    "Geometry",
]

LatticeOrGeometryLike = Union[
    LatticeLike,
    GeometryLike,
]

if hasattr(sps, "sparray"):
    SparseMatrixExt = Union[
        sps.spmatrix,
        sps.sparray,
    ]
else:
    SparseMatrixExt = Union[sps.spmatrix,]

SparseMatrix = Union[
    SparseMatrixExt,
    "SparseCSR",
]
