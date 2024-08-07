# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
from scipy.sparse import issparse

import sisl


def get_orbital_sets_positions(atoms: list[sisl.Atom]) -> list[list[int]]:
    """Gets the orbital indices where an orbital set starts for each atom.

    An "orbital set" is a group of 2l + 1 orbitals with an angular momentum l
    and different m.

    Parameters
    ----------
    atoms :
        List of atoms for which the orbital sets positions are desired.
    """
    specie_orb_sets = []
    for at in atoms:
        orbitals = at.orbitals

        i_orb = 0
        positions = []
        while i_orb < len(orbitals):
            positions.append(i_orb)

            i_orb = i_orb + 1 + 2 * orbitals[i_orb].l

        specie_orb_sets.append(positions)

    return specie_orb_sets


def get_geometry_from_matrix(
    matrix: Union[sisl.SparseCSR, sisl.SparseAtom, sisl.SparseOrbital, np.ndarray],
    geometry: Optional[sisl.Geometry] = None,
):
    """Returns the geometry associated to a matrix.

    Parameters
    ----------
    matrix :
        The matrix for which the geometry is desired, which may have
        an associated geometry.
    geometry :
        The geometry to be returned. This is to be used when we already
        have a geometry and we don't want to extract it from the matrix.
    """
    if geometry is not None:
        pass
    elif hasattr(matrix, "geometry"):
        geometry = matrix.geometry

    return geometry


def matrix_as_array(
    matrix: Union[sisl.SparseCSR, sisl.SparseAtom, sisl.SparseOrbital, np.ndarray],
    dim: Optional[int] = 0,
    isc: Optional[int] = None,
    fill_value: Optional[float] = None,
) -> np.ndarray:
    """Converts any type of matrix to a numpy array.

    Parameters
    ----------
    matrix :
        The matrix to be converted.
    dim :
        If the matrix is a sisl sparse matrix and it has a third dimension, the
        index to get in that third dimension.
    isc :
        If the matrix is a sisl SparseAtom or SparseOrbital, the index of the
        cell within the auxiliary supercell.

        If None, the whole matrix is returned.
    fill_value :
        If the matrix is a sparse matrix, the value to fill the unset elements.
    """
    if isinstance(matrix, (sisl.SparseCSR, sisl.SparseAtom, sisl.SparseOrbital)):
        if dim is None:
            if isinstance(matrix, (sisl.SparseAtom, sisl.SparseOrbital)):
                matrix = matrix._csr

            matrix = matrix.todense()
        else:
            matrix = matrix.tocsr(dim=dim)

    if issparse(matrix):
        matrix = matrix.toarray()
        matrix[matrix == 0] = fill_value

    if isc is not None:
        matrix = matrix[:, matrix.shape[0] * isc : matrix.shape[0] * (isc + 1)]

    matrix = np.array(matrix)

    return matrix


def determine_color_midpoint(
    matrix: np.ndarray,
    cmid: Optional[float] = None,
    crange: Optional[tuple[float, float]] = None,
) -> Optional[float]:
    """Determines the midpoint of a colorscale given a matrix of values.

    If ``cmid`` or ``crange`` are provided, this function just returns ``cmid``.
    However, if none of them are provided, it returns 0 if the matrix has both
    positive and negative values, and None otherwise.

    Parameters
    ----------
    matrix :
        The matrix of values for which the colorscale is to be determined.
    cmid :
        Possible already determined midpoint.
    crange :
        Possible already determined range.
    """
    if cmid is not None:
        return cmid
    elif crange is not None:
        return cmid
    elif np.sum(matrix < 0) > 0 and np.sum(matrix > 0) > 0:
        return 0
    else:
        return None


def get_matrix_mode(matrix) -> Literal["atoms", "orbitals"]:
    """Returns what the elements of the matrix represent.

    If the matrix is a sisl SparseAtom, the elements are atoms.
    Otherwise, they are assumed to be orbitals.

    Parameters
    ----------
    matrix :
        The matrix for which the mode is desired.
    """
    return "atoms" if isinstance(matrix, sisl.SparseAtom) else "orbitals"


def sanitize_matrix_arrows(arrows: Union[dict, list[dict]]) -> list[dict]:
    """Sanitizes an ``arrows`` argument to a list of sanitized specifications.

    Parameters
    ----------
    arrows :
        The arrows argument to be sanitized. If it is a dictionary, it is converted to a list
        with a single element.
    """
    if isinstance(arrows, dict):
        arrows = [arrows]

    san_arrows = []
    for arrow in arrows:
        arrow = arrow.copy()
        san_arrows.append(arrow)

        if "data" in arrow:
            arrow["data"] = matrix_as_array(arrow["data"], dim=None)

            # Matrices have the y axis reverted.
            arrow["data"][..., 1] *= -1

        if "center" not in arrow:
            arrow["center"] = "middle"

    return san_arrows
