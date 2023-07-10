# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# This contains typing things specific to sisl
from __future__ import annotations

from typing import Union

from sisl import Atom, Atoms, Geometry, Lattice

#from typing import TYPE_CHECKING, final



# A Lattice or a Geometry
CellOrGeometry = Union[
    Lattice,
    Geometry,
]

# A basic atom or an Atom
AtomLike = Union[
    Atom,
    Atoms,
]
