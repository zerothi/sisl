# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# To check for integers
from __future__ import annotations
from typing import List, Callable
from collections import UserList


__all__ = ["Collection"]


class Collection(UserList):
    """ Container for multiple objects in a single object

    This is primarily intended for geometry-collections and friends.
    For instance one could host a GeometryCollection + Collection(of velocities)
    associated with the geometries.
    """

    def applymap(self, func: Callable, **kwargs) -> Collection:
        """ Apply a function to all elementwise

        Applies the function `func` to each of the contained objects
        and returns a new collection with the applied function to them.

        Parameters
        ----------
        func : callable
            the function to be called on each object contained
        **kwargs : optional
            keyword arguments passed directly to `func`

        Returns
        -------
        Collection : a new collection with each geometry transformed by `func`
        """
        return self.__class__(func(g, **kwargs) for g in self)

