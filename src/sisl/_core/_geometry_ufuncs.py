# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from sisl.typing import AtomsArgument
else:
    AtomsArgument = Any

import sisl._array as _a
from sisl._ufuncs import register_sisl_dispatch
from sisl.utils.mathematics import fnorm

from .geometry import Geometry

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(module="sisl")
def tile(geometry: Geometry, reps: int, axis: int) -> Geometry:
    """Tile the geometry to create a bigger one

    The atomic indices are retained for the base structure.

    Tiling and repeating a geometry will result in the same geometry.
    The *only* difference between the two is the final ordering of the atoms.

    Parameters
    ----------
    geometry:
        the geometry to act on
    reps :
       number of tiles (repetitions)
    axis :
       direction of tiling, 0, 1, 2 according to the cell-direction

    Examples
    --------
    >>> geom = Geometry([[0, 0, 0], [0.5, 0, 0]], lattice=1.)
    >>> g = geom.tile(2,axis=0)
    >>> print(g.xyz) # doctest: +NORMALIZE_WHITESPACE
    [[0.   0.   0. ]
     [0.5  0.   0. ]
     [1.   0.   0. ]
     [1.5  0.   0. ]]
    >>> g = geom.tile(2,0).tile(2,axis=1)
    >>> print(g.xyz) # doctest: +NORMALIZE_WHITESPACE
    [[0.   0.   0. ]
     [0.5  0.   0. ]
     [1.   0.   0. ]
     [1.5  0.   0. ]
     [0.   1.   0. ]
     [0.5  1.   0. ]
     [1.   1.   0. ]
     [1.5  1.   0. ]]

    See Also
    --------
    repeat : equivalent but different ordering of final structure
    untile : opposite method of this
    """
    if reps < 1:
        raise ValueError(
            f"{geometry.__class__.__name__}.tile requires a repetition above 0"
        )

    lattice = geometry.lattice.tile(reps, axis)

    # Our first repetition *must* be with
    # the former coordinate
    xyz = np.tile(geometry.xyz, (reps, 1))
    # We may use broadcasting rules instead of repeating stuff
    xyz.shape = (reps, geometry.na, 3)
    nr = _a.arangei(reps)
    nr.shape = (reps, 1, 1)
    # Correct the unit-cell offsets
    xyz += nr * geometry.cell[axis, :]
    xyz.shape = (-1, 3)

    # Create the geometry and return it (note the smaller atoms array
    # will also expand via tiling)
    return geometry.__class__(xyz, atoms=geometry.atoms.tile(reps), lattice=lattice)


@register_sisl_dispatch(module="sisl")
def translate(
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


# simple copy...
Geometry.move = Geometry.translate
