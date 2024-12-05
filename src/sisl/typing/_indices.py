# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Union

# To prevent import cycles place any internal imports in the branch below
# and use a string literal forward reference to it in subsequent types
# https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from sisl import Atom, Shape
    from sisl._category import GenericCategory
    from sisl.geom.category import AtomCategory

__all__ = [
    "AxisIntLiteral",
    "CellAxisStrLiteral",
    "CellAxisLiteral",
    "CartesianAxisStrLiteral",
    "CartesianAxisLiteral",
    "CellAxis",
    "CellAxes",
    "CartesianAxis",
    "CartesianAxes",
    "AnyAxis",
    "AnyAxes",
    "SimpleIndex",
    "AtomsIndex",
    "OrbitalsIndex",
]

AxisIntLiteral = Literal[0, 1, 2]
AxesIntLiteral = Sequence[AxisIntLiteral]
"""Defining axis specification in 3D space (int)"""
CellAxisStrLiteral = Literal["a", "b", "c"]
"""Defining lattice axis specification in 3D space (str)"""
CartesianAxisStrLiteral = Literal["x", "y", "z"]
"""Defining Cartesian axis specification in 3D space (str)"""

# The literal interpretations of what an axis specification can be
# Both for lattice vectors and cartesian vectors
CellAxisLiteral = Union[AxisIntLiteral, CellAxisStrLiteral]
"""Defining lattice vector allowed arguments"""
CartesianAxisLiteral = Union[AxisIntLiteral, CartesianAxisStrLiteral]
"""Defining Cartesian vector allowed arguments"""

# Axis specifications
# Some might require Cartesian direction indices
# While others will allow only Cell indices.
# And finally, some may do combinations
# Pure integer axis specification
CellAxis = CellAxisLiteral
CellAxes = Union[CellAxis, Sequence[CellAxis]]
"""A combination of multiple lattice vector arguments"""

CartesianAxis = CartesianAxisLiteral
CartesianAxes = Union[CartesianAxis, Sequence[CartesianAxis]]
"""A combination of multiple Cartesian vector arguments"""

AnyAxis = Union[CellAxis, CartesianAxis]
"""Any combination of a single Cartesian/lattice vector specification"""
AnyAxes = Union[CellAxes, CartesianAxes]
"""Any combination of one or more Cartesian/lattice vector specifications"""


SimpleIndex = Union[int, Sequence[int], Sequence[bool]]
"""Index type"""

# An atoms like argument that may be parsed by Geometry._sanitize_atoms
AtomsIndex = Union[
    SimpleIndex,
    str,  # for names
    dict,  # short-cut for categories creation
    "Atom",  # by equivalence atom
    "AtomCategory",
    "GenericCategory",
    "Shape",
    bool,  # for all or none
    None,  # typically used to default for all
]
"""Indexing atoms via various construct methods"""

# An atoms like argument that may be parsed by Geometry._sanitize_orbs
OrbitalsIndex = Union[
    SimpleIndex,
    str,
    dict,
    "Atom",
    "AtomCategory",  # gets expended to orbitals on the atoms
    "Shape",
    bool,
    None,
]
"""Indexing orbitals via various construct methods"""

try:
    from types import EllipsisType

    AtomsIndex = Union[AtomsIndex, EllipsisType]
    OrbitalsIndex = Union[OrbitalsIndex, EllipsisType]
except ImportError:
    pass
