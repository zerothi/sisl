# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from typing import Union

import sisl._typing_ext.numpy as npt
from sisl import Atom, Atoms, Geometry, Lattice, Shape
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
