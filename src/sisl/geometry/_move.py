# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from sisl.typing import AtomsArgument
else:
    AtomsArgument = Any

from ._geometry import Geometry
from ._ufuncs import register_geometry_function

__all__ = ["g_translate", "g_move"]


def _translate(
    geometry: Geometry, v, atoms: Optional[AtomsArgument] = None, cell: bool = False
) -> Geometry:
    """Translates the geometry by `v`

    One can translate a subset of the atoms by supplying `atoms`.

    Returns a copy of the structure translated by `v`.

    Parameters
    ----------
    geometry: Geometry
        the geometry to act on
    v : float or array_like
         the value or vector to displace all atomic coordinates
         It should just be broad-castable with the geometry's coordinates.
    atoms : int or array_like, optional
         only displace the given atomic indices, if not specified, all
         atoms will be displaced
    cell : bool, optional
         If True the supercell also gets enlarged by the vector
    """
    g = geometry.copy()
    if atoms is None:
        g.xyz += np.asarray(v, g.xyz.dtype)
    else:
        g.xyz[geometry._sanitize_atoms(atoms).ravel(), :] += np.asarray(v, g.xyz.dtype)
    if cell:
        g.set_lattice(g.lattice.translate(v))
    return g


g_translate = register_geometry_function(name="translate", module="sisl.geometry")(
    _translate
)
g_move = register_geometry_function(name="move", module="sisl.geometry")(_translate)
