# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl
from sisl.viz.processors.matrix import (
    determine_color_midpoint,
    get_geometry_from_matrix,
    get_matrix_mode,
    get_orbital_sets_positions,
    matrix_as_array,
    sanitize_matrix_arrows,
)

pytestmark = [pytest.mark.viz, pytest.mark.processors]


def test_orbital_positions():

    C = sisl.Atom(
        6,
        orbitals=[
            sisl.AtomicOrbital("2s"),
            sisl.AtomicOrbital("2px"),
            sisl.AtomicOrbital("2py"),
            sisl.AtomicOrbital("2pz"),
            sisl.AtomicOrbital("2px"),
            sisl.AtomicOrbital("2py"),
            sisl.AtomicOrbital("2pz"),
        ],
    )

    H = sisl.Atom(1, orbitals=[sisl.AtomicOrbital("1s")])

    positions = get_orbital_sets_positions([C, H])

    assert len(positions) == 2

    assert positions[0] == [0, 1, 4]
    assert positions[1] == [0]


def test_get_geometry_from_matrix():

    geom = sisl.geom.graphene()

    matrix = sisl.Hamiltonian(geom)

    assert get_geometry_from_matrix(matrix) is geom

    geom_copy = geom.copy()

    assert get_geometry_from_matrix(matrix, geom_copy) is geom_copy

    # Check that if we pass something without an associated geometry
    # but we provide a geometry it will work
    assert get_geometry_from_matrix(np.array([1, 2]), geom) is geom


def test_matrix_as_array():

    matrix = sisl.SparseCSR((2, 2, 2))

    matrix[0, 0, 0] = 1
    matrix[0, 0, 1] = 2

    array = matrix_as_array(matrix, fill_value=0)
    assert np.allclose(array, np.array([[1, 0], [0, 0]]))

    array = matrix_as_array(matrix, dim=1, fill_value=0)
    assert np.allclose(array, np.array([[2, 0], [0, 0]]))

    array = matrix_as_array(matrix)
    assert array[0, 0] == 1
    assert np.isnan(array).sum() == 3

    # Check that it can work with auxiliary supercells
    geom = sisl.geom.graphene(
        atoms=sisl.Atom("C", orbitals=[sisl.AtomicOrbital("2pz")])
    )
    matrix = sisl.Hamiltonian(geom)

    array = matrix_as_array(matrix)
    assert array.shape == matrix.shape[:-1]

    array = matrix_as_array(matrix, isc=1)
    assert array.shape == (geom.no, geom.no)

    # Check that a numpy array is kept untouched
    matrix = np.array([[1, 2], [3, 4]])
    assert np.allclose(matrix_as_array(matrix), matrix)


def test_determine_color_midpoint():

    # With the matrix containing only positive values
    matrix = np.array([1, 2])

    assert determine_color_midpoint(matrix) is None
    assert determine_color_midpoint(matrix, cmid=1, crange=(0, 1)) == 1
    assert determine_color_midpoint(matrix, crange=(0, 1)) is None

    # With the matrix containing only negative values
    matrix = np.array([-1, -2])

    assert determine_color_midpoint(matrix) is None
    assert determine_color_midpoint(matrix, cmid=1, crange=(0, 1)) == 1
    assert determine_color_midpoint(matrix, crange=(0, 1)) is None

    # With the matrix containing both positive and negative values
    matrix = np.array([-1, 1])

    assert determine_color_midpoint(matrix) == 0
    assert determine_color_midpoint(matrix, cmid=1, crange=(-1, 1)) == 1
    assert determine_color_midpoint(matrix, crange=(-1, 1)) is None


def test_get_matrix_mode():

    geom = sisl.geom.graphene()

    matrix = sisl.SparseAtom(geom)
    assert get_matrix_mode(matrix) == "atoms"

    matrix = sisl.Hamiltonian(geom)
    assert get_matrix_mode(matrix) == "orbitals"

    matrix = sisl.SparseCSR((2, 2))
    assert get_matrix_mode(matrix) == "orbitals"

    matrix = np.array([[1, 2], [3, 4]])
    assert get_matrix_mode(matrix) == "orbitals"


def test_sanitize_matrix_arrows():

    arrows = {}
    assert sanitize_matrix_arrows(arrows) == [{"center": "middle"}]

    geom = sisl.geom.graphene()
    data = sisl.Hamiltonian(geom, dim=2)
    data[0, 0, 0] = 1
    data[0, 0, 1] = 2

    arrows = [{"data": data}]
    sanitized = sanitize_matrix_arrows(arrows)

    assert len(sanitized) == 1
    assert sanitized[0]["data"].shape == data.shape
    san_data = sanitized[0]["data"]
    assert san_data[0, 0, 0] == 1
    assert san_data[0, 0, 1] == -2
