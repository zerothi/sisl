# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# This class manipulation goes towards using Geometries
# in a functional setting.
# It will be heavily influenced by the numpy.ufunc
# handling which posseses the __array_ufunc__
# In this instance we will be using the __geometry_ufunc__
# handler to off-load things from the class
from functools import partial

from sisl._ufuncs import register_sisl_function

from ._geometry import Geometry

__all__ = ["register_geometry_function"]


register_geometry_function = partial(
    register_sisl_function,
    name=lambda name: name[2:],
    cls=Geometry,
    module="sisl._core.geometry",
)
