# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from sisl import Geometry, Lattice
from sisl.geom import NeighborFinder
from sisl.geom._neighbors import (
    AtomNeighborList,
    CoordNeighborList,
    CoordsNeighborList,
    FullNeighborList,
    PartialNeighborList,
    UniqueNeighborList,
)

pytestmark = [pytest.mark.geometry, pytest.mark.geom, pytest.mark.neighbor]


tr_fixture = partial(pytest.fixture, scope="module", params=[True, False])


def request_param(request):
    return request.param


sphere_overlap = tr_fixture()(request_param)
multiR = tr_fixture()(request_param)
self_interaction = tr_fixture()(request_param)
post_setup = tr_fixture()(request_param)
pbc = tr_fixture()(request_param)


def set_pbc(geom, _pbc):
    if _pbc:
        geom.lattice.set_boundary_condition(Lattice.BC.PERIODIC)
    else:
        geom.lattice.set_boundary_condition(Lattice.BC.UNKNOWN)


@pytest.fixture(scope="module")
def neighfinder(sphere_overlap, multiR, pbc):
    geom = Geometry([[0, 0, 0], [1.2, 0, 0], [9, 0, 0]], lattice=[10, 10, 7])
    set_pbc(geom, pbc)

    R = np.array([1.1, 1.5, 1.2]) if multiR else 1.5

    neighfinder = NeighborFinder(geom, R=R, overlap=sphere_overlap)

    neighfinder.assert_consistency()

    return neighfinder


@pytest.fixture(scope="module")
def expected_neighs(sphere_overlap, multiR, self_interaction, pbc):
    first_at_neighs = []
    if not (multiR and not sphere_overlap):
        first_at_neighs.append([0, 1, 0, 0, 0])
    if self_interaction:
        first_at_neighs.append([0, 0, 0, 0, 0])
    if pbc:
        first_at_neighs.append([0, 2, -1, 0, 0])

    second_at_neighs = [[1, 0, 0, 0, 0]]
    if self_interaction:
        second_at_neighs.insert(0, [1, 1, 0, 0, 0])
    if pbc and sphere_overlap:
        second_at_neighs.append([1, 2, -1, 0, 0])

    third_at_neighs = []
    if self_interaction:
        third_at_neighs.append([2, 2, 0, 0, 0])
    if pbc:
        if sphere_overlap:
            third_at_neighs.append([2, 1, 1, 0, 0])
        third_at_neighs.append([2, 0, 1, 0, 0])

    return (
        np.array(first_at_neighs),
        np.array(second_at_neighs),
        np.array(third_at_neighs),
    )


def test_neighfinder_setup(sphere_overlap, multiR, post_setup):
    geom = Geometry([[0, 0, 0], [1, 0, 0]], lattice=[10, 10, 7])

    R = np.array([0.9, 1.5]) if multiR else 1.5

    if post_setup:
        # We are going to create a data structure with the wrong parameters,
        # and then update it.
        finder = NeighborFinder(geom, R=R - 0.7, overlap=not sphere_overlap)
        finder.setup(R=R, overlap=sphere_overlap)
    else:
        # Initialize finder with the right parameters.
        finder = NeighborFinder(geom, R=R, overlap=sphere_overlap)

    # Check that R is properly set when its a scalar and an array.
    if multiR:
        assert isinstance(finder.R, np.ndarray)
        assert np.all(finder.R == R)
    else:
        assert finder.R.ndim == 0
        assert finder.R == R

    # Assert that we have stored a copy of the geometry
    assert isinstance(finder.geometry, Geometry)
    assert finder.geometry == geom
    assert finder.geometry is not geom

    # Check that the total number of bins is correct. If sphere_overlap is
    # True, bins are much bigger.
    nbins = (1, 1, 1) if sphere_overlap else (3, 3, 2)
    assert finder.nbins == nbins

    total_bins = 1 if sphere_overlap else 18
    assert finder.total_nbins == total_bins

    # Assert that the data structure is generated.
    for k in ("_list", "_counts", "_heads"):
        assert hasattr(finder, k), k
        assert isinstance(finder._list, np.ndarray), k

    finder.assert_consistency()

    # Check that all bins are empty except one, which contains the two atoms.
    assert (finder._counts == 0).sum() == finder.total_nbins - 1
    assert finder._counts.sum() == 2


def test_neighbor_pairs(neighfinder, self_interaction, expected_neighs):
    neighs = neighfinder.find_neighbors(self_interaction=self_interaction)

    assert isinstance(neighs, FullNeighborList)

    n_neighs = [len(at_neighs) for at_neighs in expected_neighs]
    assert np.all(neighs.n_neighbors == n_neighs)

    for i, (at_neighs, expected_at_neighs) in enumerate(zip(neighs, expected_neighs)):
        assert isinstance(at_neighs, AtomNeighborList)
        assert at_neighs.atom == i
        assert len(expected_at_neighs) == at_neighs.n_neighbors
        if at_neighs.n_neighbors > 0:
            assert np.all(at_neighs.i == expected_at_neighs[:, 0])
            assert np.all(at_neighs.j == expected_at_neighs[:, 1])
            assert np.all(at_neighs.isc == expected_at_neighs[:, 2:])


def test_partial_neighbor_pairs(neighfinder, self_interaction, expected_neighs):
    neighs = neighfinder.find_neighbors(self_interaction=self_interaction, atoms=[1, 2])

    assert isinstance(neighs, PartialNeighborList)

    expected_neighs = expected_neighs[1:]

    n_neighs = [len(at_neighs) for at_neighs in expected_neighs]
    assert np.all(neighs.n_neighbors == n_neighs)

    for i, (at_neighs, expected_at_neighs) in enumerate(zip(neighs, expected_neighs)):
        assert isinstance(at_neighs, AtomNeighborList)
        assert at_neighs.atom == i + 1
        assert len(expected_at_neighs) == at_neighs.n_neighbors
        if at_neighs.n_neighbors > 0:
            assert np.all(at_neighs.i == expected_at_neighs[:, 0])
            assert np.all(at_neighs.j == expected_at_neighs[:, 1])
            assert np.all(at_neighs.isc == expected_at_neighs[:, 2:])


def test_unique_pairs(
    neighfinder, self_interaction, expected_neighs, sphere_overlap, multiR, pbc
):
    # It shouldn't work if you are not requesting sphere overlap and there are
    # multiple cutoff radius.
    if neighfinder.R.ndim == 1 and not neighfinder._overlap:
        with pytest.raises(ValueError):
            neighfinder.find_unique_pairs(self_interaction=self_interaction)
        return

    neighs = neighfinder.find_unique_pairs(self_interaction=self_interaction)

    assert isinstance(neighs, UniqueNeighborList)

    # Convert to a full neighbor list and check that everything is correct.
    full_neighs = neighs.to_full()

    expected_neighs = [
        at_neighs[np.lexsort(at_neighs[:, [1, 0]].T)] if len(at_neighs) > 0 else []
        for at_neighs in expected_neighs
    ]

    assert isinstance(full_neighs, FullNeighborList)

    for at_neighs, expected_at_neighs in zip(full_neighs, expected_neighs):
        assert isinstance(at_neighs, AtomNeighborList)
        print(at_neighs._finder_results, expected_at_neighs)
        assert len(expected_at_neighs) == at_neighs.n_neighbors
        if at_neighs.n_neighbors > 0:
            assert np.all(at_neighs.i == expected_at_neighs[:, 0])
            assert np.all(at_neighs.j == expected_at_neighs[:, 1])
            assert np.all(at_neighs.isc == expected_at_neighs[:, 2:])


def test_close(neighfinder, pbc):
    neighs = neighfinder.find_close([0.3, 0, 0])

    assert isinstance(neighs, CoordsNeighborList)

    first_point_neighs = [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]

    if pbc and neighfinder.R.ndim == 0:
        first_point_neighs.append([0, 2, -1, 0, 0])

    expected_neighs = [
        np.array(first_point_neighs),
    ]

    for point_neighs, expected_point_neighs in zip(neighs, expected_neighs):
        assert isinstance(point_neighs, CoordNeighborList)
        assert len(expected_point_neighs) == point_neighs.n_neighbors
        if point_neighs.n_neighbors > 0:
            assert np.all(point_neighs.i == expected_point_neighs[:, 0])
            assert np.all(point_neighs.j == expected_point_neighs[:, 1])
            assert np.all(point_neighs.isc == expected_point_neighs[:, 2:])


def test_close_intcoords(neighfinder):
    """Test the case when the input coordinates are integers.

    (cython routine needs floats)
    """
    neighfinder.find_close([0, 0, 0])


def test_no_neighbors(pbc):
    """Test the case where there are no neighbors, to see that it doesn't crash."""

    geom = Geometry([[0, 0, 0]])
    set_pbc(geom, pbc)

    finder = NeighborFinder(geom, R=1.5)

    neighs = finder.find_neighbors()

    assert isinstance(neighs, FullNeighborList)

    for i, at_neighs in enumerate(neighs):
        assert isinstance(at_neighs, AtomNeighborList)
        assert at_neighs.atom == i
        assert at_neighs.n_neighbors == 0

    neighs = finder.find_unique_pairs()

    assert isinstance(neighs, UniqueNeighborList)
    neighs = neighs.to_full()

    assert isinstance(neighs, FullNeighborList)

    for i, at_neighs in enumerate(neighs):
        assert isinstance(at_neighs, AtomNeighborList)
        assert at_neighs.atom == i
        assert at_neighs.n_neighbors == 0


def test_R_too_big(pbc):
    """Test the case when R is so big that it needs a bigger bin
    than the unit cell."""

    geom = Geometry([[0, 0, 0], [1, 0, 0]], lattice=[2, 10, 10])
    set_pbc(geom, pbc)

    neighfinder = NeighborFinder(geom, R=1.5)

    neighs = neighfinder.find_unique_pairs()

    assert isinstance(neighs, UniqueNeighborList)
    print(neighs._finder_results)

    neighs = neighs.to_full()
    print(neighs._finder_results)

    first_at_neighs = [[0, 1, 0, 0, 0]]
    second_at_neighs = [[1, 0, 0, 0, 0]]

    if pbc:
        first_at_neighs.insert(0, [0, 1, -1, 0, 0])
        second_at_neighs.insert(0, [1, 0, 1, 0, 0])

    expected_neighs = [np.array(first_at_neighs), np.array(second_at_neighs)]

    for at_neighs, expected_at_neighs in zip(neighs, expected_neighs):
        assert isinstance(at_neighs, AtomNeighborList)
        assert len(expected_at_neighs) == at_neighs.n_neighbors
        if at_neighs.n_neighbors > 0:
            assert np.all(at_neighs.i == expected_at_neighs[:, 0])
            assert np.all(at_neighs.j == expected_at_neighs[:, 1])
            assert np.all(at_neighs.isc == expected_at_neighs[:, 2:])

    neighfinder = NeighborFinder(geom, R=[0.6, 2.2], overlap=True)

    neighs = neighfinder.find_close([[0.5, 0, 0]])
    assert isinstance(neighs, CoordsNeighborList)

    expected_neighs = [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
    if pbc:
        expected_neighs.insert(0, [0, 1, -1, 0, 0])
    expected_neighs = [np.array(expected_neighs)]

    for point_neighs, expected_point_neighs in zip(neighs, expected_neighs):
        assert isinstance(point_neighs, CoordNeighborList)
        assert len(expected_point_neighs) == point_neighs.n_neighbors
        if point_neighs.n_neighbors > 0:
            assert np.all(point_neighs.i == expected_point_neighs[:, 0])
            assert np.all(point_neighs.j == expected_point_neighs[:, 1])
            assert np.all(point_neighs.isc == expected_point_neighs[:, 2:])


def test_bin_sizes():
    geom = Geometry([[0, 0, 0], [1, 0, 0]], lattice=[2, 10, 10])

    # We should have fewer bins along the first lattice vector
    n1 = NeighborFinder(geom, R=1.5, bin_size=2)
    n2 = NeighborFinder(geom, R=1.5, bin_size=4)

    assert n1.total_nbins > n2.total_nbins
    # When the bin is bigger than the unit cell, this situation
    # occurs
    assert n1.nbins[0] == n2.nbins[0]
    assert n1.nbins[1] > n2.nbins[1]
    assert n1.nbins[2] > n2.nbins[2]

    # We should have the same number of bins the 2nd and 3rd lattice vectors
    n3 = NeighborFinder(geom, R=1.5, bin_size=2)
    n4 = NeighborFinder(geom, R=1.5, bin_size=(2, 4, 4))

    assert n3.nbins[0] == n4.nbins[0]
    assert n3.nbins[1] > n4.nbins[1]
    assert n3.nbins[2] > n4.nbins[2]


def test_outside_box():
    geom = Geometry([[0, 0, 0], [3, 0, 0]], lattice=[2, 10, 10])

    # The neighbor finder should raise an error if an atom is outside the box
    # of the unit cell, because it is not supported for now.
    # IT SHOULD BE SUPPORTED IN THE FUTURE
    with pytest.raises(ValueError):
        n = NeighborFinder(geom, R=1.1)
