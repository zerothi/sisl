from typing import (
    Union,
)

import sisl._typing_ext.numpy as npt
from sisl import (
    Atom,
    Atoms,
    Shape,
    Geometry,
    Lattice
)
from sisl._category import GenericCategory
from sisl.geom.category import AtomCategory

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
    Lattice,
    Geometry
]
