# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Tuple, Union

import scipy.sparse as sps

# To prevent import cycles place any internal imports in the branch below
# and use a string literal forward reference to it in subsequent types
# https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from sisl import Geometry, Lattice, Quaternion, SparseAtom, SparseCSR, SparseOrbital

__all__ = [
    "Coord",
    "CoordOrScalar",
    "FuncType",
    "OrSequence",
    "KPoint",
    "SeqBool",
    "SeqOrScalarBool",
    "SeqInt",
    "SeqOrScalarInt",
    "SeqFloat",
    "SeqOrScalarFloat",
    "SparseMatrix",
    "SparseMatrixExt",
    "SparseMatrixGeometry",
    "RotationType",
]


class OrSequence:
    __slots__ = ()

    def __getitem__(self, parameter):
        return Union[Sequence[parameter], parameter]


SeqBool = Sequence[bool]
SeqOrScalarBool = Union[bool, SeqBool]
SeqInt = Sequence[int]
SeqOrScalarInt = Union[int, SeqInt]
SeqFloat = Sequence[float]
SeqOrScalarFloat = Union[float, SeqFloat]

Coord = SeqFloat
CoordOrScalar = Union[float, Coord]

KPoint = Sequence[float]

# Short for *any* function
FuncType = Callable[..., Any]

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

SparseMatrixGeometry = Union["SparseAtom", "SparseOrbital"]

# Specify a rotation around Cartesian lattice vectors
RotationType = Union[
    Tuple[float, Union[str, SeqFloat]],  # (ang, [x, y, z] | "x")
    SeqFloat,  # (ang-x, ang-y, ang-z)
    "Quaternion",  # Direct
]
RotationCartesian = Union[str, SeqFloat]
RotationDirect = Tuple[float, Union[str, Coord]]
