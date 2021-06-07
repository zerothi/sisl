# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from functools import wraps
from sisl._internal import set_module
from sisl._category import Category, NullCategory
from sisl.geometry import AtomCategory


__all__ = ["NullCategory", "AtomCategory"]


def _sanitize_loop(func):
    @wraps(func)
    def loop_func(self, geometry, atoms=None):
        if atoms is None:
            return [func(self, geometry, ia) for ia in geometry]
        # extract based on atoms selection
        atoms = geometry._sanitize_atoms(atoms)
        if atoms.ndim == 0:
            return func(self, geometry, atoms)
        return [func(self, geometry, ia) for ia in atoms]
    return loop_func

#class AtomCategory(Category)
# is defined in sisl/geometry.py since it is required in
# that instance.
