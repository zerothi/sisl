# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import typing
from itertools import permutations
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence, Union

# To prevent import cycles place any internal imports in the branch below
# and use a string literal forward reference to it in subsequent types
# https://mypy.readthedocs.io/en/latest/common_issues.html#import-cycles
if TYPE_CHECKING:
    from sisl import Atom, Shape
    from sisl._category import GenericCategory
    from sisl.geom.category import AtomCategory

__all__ = [
    "CellAxisLiteral",
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


# The literal interpretations of what an axis specification can be
# Both for lattice vectors and cartesian vectors
CellAxisLiteral = Literal[0, 1, 2, "a", "b", "c"]
"""Defining lattice vector allowed arguments"""
CartesianAxisLiteral = Literal[0, 1, 2, "x", "y", "z"]
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
