from typing import (
    Union,
)

from ._ext.numpy import npt
from sisl import (
    Atom,
    Atoms,
    AtomCategory,
    GenericCategory,
    Shape,
    Geometry,
    SuperCell
)

# An atoms like argument that may be parsed by Geometry._sanitize_atoms
AtomsArgument = Union[
    npt.ArrayLike, # direct indices
    str, int, dict,
    Atom,
    AtomCategory,
    GenericCategory,
    Shape
]

OrbitalsArgument = Union[
    npt.ArrayLike,
    str, int, dict
]

AtomLike = Union[
    Atom,
    Atoms
]

CellOrGeometry = Union[
    SuperCell,
    Geometry
]
