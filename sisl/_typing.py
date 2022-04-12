# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# This contains typing things specific to sisl
from __future__ import annotations

from typing import Union
#from typing import TYPE_CHECKING, final

from sisl import (
    Geometry, SuperCell,
    Atom, Atoms
)


# A SuperCell or a Geometry
CellOrGeometry = Union[
    SuperCell,
    Geometry,
]

# A basic atom or an Atom
AtomLike = Union[
    Atom,
    Atoms,
]
