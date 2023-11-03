# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from pathlib import Path
from typing import Union

import numpy as np

import sisl._typing_ext.numpy as npt
from sisl import Atom, Atoms, BaseSile, Geometry, Lattice, Shape
from sisl._category import GenericCategory
from sisl._typing import *
from sisl.geom.category import AtomCategory

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

OrbitalsArgument = Union[
    npt.NDArray[Union[np.int_, np.bool_]],
    str,
    int,
    dict,
    AtomCategory,
    Shape,
]
