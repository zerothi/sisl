# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Optional

import numpy as np

from sisl import Atom, Geometry, Lattice
from sisl._internal import set_module
from sisl.typing import AtomsLike

from ._common import geometry_define_nsc

__all__ = ["diamond"]


@set_module("sisl.geom")
def diamond(alat: float = 3.57, atoms: Optional[AtomsLike] = None) -> Geometry:
    """Diamond lattice with 2 atoms in the unitcell

    Parameters
    ----------
    alat : float
        the lattice constant for the diamond
    atoms : Atom, optional
        atom in the lattice, may be one or two atoms. Default is Carbon
    """
    dist = alat * 3.0**0.5 / 4
    if atoms is None:
        atoms = Atom(Z=6, R=dist * 1.01)
    lattice = Lattice(
        np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], np.float64) * alat / 2
    )
    dia = Geometry(
        np.array([[0, 0, 0], [1, 1, 1]], np.float64) * alat / 4, atoms, lattice=lattice
    )

    geometry_define_nsc(dia)
    return dia
