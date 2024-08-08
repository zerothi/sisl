# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from functools import wraps

from sisl._category import NullCategory
from sisl._core import AtomCategory, Geometry
from sisl.typing import AtomsIndex

__all__ = ["NullCategory", "AtomCategory"]


def _sanitize_loop(func):
    @wraps(func)
    def loop_func(self, geometry: Geometry, atoms: AtomsIndex = None):
        if atoms is None:
            return [func(self, geometry, ia) for ia in geometry]

        # extract based on atoms selection
        atoms = geometry._sanitize_atoms(atoms)
        if atoms.ndim == 0:
            return func(self, geometry, atoms)
        return [func(self, geometry, ia) for ia in atoms]

    return loop_func
