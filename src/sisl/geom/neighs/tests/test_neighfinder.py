import numpy as np
import pytest

from sisl import Geometry
from sisl.geom import NeighFinder


@pytest.fixture(scope="module", params=[True, False])
def sphere_overlap(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def multiR(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def self_interaction(request):
    return request.param


@pytest.fixture(scope="module", params=[False, True])
def post_setup(request):
    return request.param


@pytest.fixture(scope="module", params=[False, True])
def pbc(request):
    return request.param


@pytest.fixture(scope="module")
def neighfinder(sphere_overlap, multiR):
    geom = Geometry([[0, 0, 0], [1.2, 0, 0], [9, 0, 0]], lattice=np.diag([10, 10, 7]))

    R = np.array([1.1, 1.5, 1.2]) if multiR else 1.5

    neighfinder = NeighFinder(geom, R=R, sphere_overlap=sphere_overlap)

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
    geom = Geometry([[0, 0, 0], [1, 0, 0]], lattice=np.diag([10, 10, 7]))

    R = np.array([0.9, 1.5]) if multiR else 1.5

    if post_setup:
        # We are going to create a data structure with the wrong parameters,
        # and then update it.
        finder = NeighFinder(geom, R=R - 0.7, sphere_overlap=not sphere_overlap)
        finder.setup_finder(R=R, sphere_overlap=sphere_overlap)
    else:
        # Initialize finder with the right parameters.
        finder = NeighFinder(geom, R=R, sphere_overlap=sphere_overlap)

    # Check that R is properly set when its a scalar and an array.
    if multiR:
        assert isinstance(finder.R, np.ndarray)
        assert np.all(finder.R == R)
    else:
        assert isinstance(finder.R, float)
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


def test_neighbour_pairs(neighfinder, self_interaction, pbc, expected_neighs):
    neighs = neighfinder.find_neighbours(
        as_pairs=True, self_interaction=self_interaction, pbc=pbc
    )

    assert isinstance(neighs, np.ndarray)

    first_at_neighs, second_at_neighs, third_at_neighs = expected_neighs

    n_neighs = len(first_at_neighs) + len(second_at_neighs) + len(third_at_neighs)

    assert neighs.shape == (n_neighs, 5)

    assert np.all(neighs == [*first_at_neighs, *second_at_neighs, *third_at_neighs])


def test_neighbours_lists(neighfinder, self_interaction, pbc, expected_neighs):
    neighs = neighfinder.find_neighbours(
        as_pairs=False, self_interaction=self_interaction, pbc=pbc
    )

    assert isinstance(neighs, list)
    assert len(neighs) == 3

    assert all(isinstance(n, np.ndarray) for n in neighs)

    first_at_neighs, second_at_neighs, third_at_neighs = expected_neighs

    # Check shapes
    for i, i_at_neighs in enumerate(
        [first_at_neighs, second_at_neighs, third_at_neighs]
    ):
        assert neighs[i].shape == (
            len(i_at_neighs),
            4,
        ), f"Wrong shape for neighbours of atom {i}"

    # Check values
    for i, i_at_neighs in enumerate(
        [first_at_neighs, second_at_neighs, third_at_neighs]
    ):
        if len(neighs[i]) == 0:
            continue

        assert np.all(
            neighs[i] == i_at_neighs[:, 1:]
        ), f"Wrong values for neighbours of atom {i}"


def test_all_unique_pairs(neighfinder, self_interaction, pbc, expected_neighs):
    if isinstance(neighfinder.R, np.ndarray) and not neighfinder._sphere_overlap:
        with pytest.raises(ValueError):
            neighfinder.find_all_unique_pairs(
                self_interaction=self_interaction, pbc=pbc
            )
        return

    neighs = neighfinder.find_all_unique_pairs(
        self_interaction=self_interaction, pbc=pbc
    )

    first_at_neighs, second_at_neighs, third_at_neighs = expected_neighs

    all_expected_neighs = np.array(
        [*first_at_neighs, *second_at_neighs, *third_at_neighs]
    )

    unique_neighs = []
    for neigh_pair in all_expected_neighs:
        if not np.all(neigh_pair[2:] == 0):
            unique_neighs.append(neigh_pair)
        else:
            for others in unique_neighs:
                if np.all(others == [neigh_pair[1], neigh_pair[0], *neigh_pair[2:]]):
                    break
            else:
                unique_neighs.append(neigh_pair)

    assert neighs.shape == (len(unique_neighs), 5)


def test_close(neighfinder, pbc):
    neighs = neighfinder.find_close([0.3, 0, 0], as_pairs=True, pbc=pbc)

    expected_neighs = [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
    if pbc and isinstance(neighfinder.R, float):
        expected_neighs.append([0, 2, -1, 0, 0])

    assert neighs.shape == (len(expected_neighs), 5)
    assert np.all(neighs == expected_neighs)


def test_no_neighbours(pbc):
    """Test the case where there are no neighbours, to see that it doesn't crash."""

    geom = Geometry([[0, 0, 0]])

    finder = NeighFinder(geom, R=1.5)

    neighs = finder.find_neighbours(as_pairs=True, pbc=pbc)

    assert isinstance(neighs, np.ndarray)
    assert neighs.shape == (0, 5)

    neighs = finder.find_neighbours(as_pairs=False, pbc=pbc)

    assert isinstance(neighs, list)
    assert len(neighs) == 1

    assert isinstance(neighs[0], np.ndarray)
    assert neighs[0].shape == (0, 4)

    neighs = finder.find_all_unique_pairs(pbc=pbc)

    assert isinstance(neighs, np.ndarray)
    assert neighs.shape == (0, 5)


def test_R_too_big(pbc):
    """Test the case when R is so big that it needs a bigger bin
    than the unit cell."""

    geom = Geometry([[0, 0, 0], [1, 0, 0]], lattice=np.diag([2, 10, 10]))

    neighfinder = NeighFinder(geom, R=1.5)

    neighs = neighfinder.find_all_unique_pairs(pbc=pbc)

    expected_neighs = [[0, 1, 0, 0, 0]]
    if pbc:
        expected_neighs.append([0, 1, -1, 0, 0])
        expected_neighs.append([1, 0, 1, 0, 0])

    assert neighs.shape == (len(expected_neighs), 5)
    assert np.all(neighs == expected_neighs)

    neighfinder = NeighFinder(geom, R=[0.6, 2.2])

    neighs = neighfinder.find_close([[0.5, 0, 0]], as_pairs=True, pbc=pbc)

    expected_neighs = [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
    if pbc:
        expected_neighs.insert(0, [0, 1, -1, 0, 0])

    print(neighs, expected_neighs)

    assert neighs.shape == (len(expected_neighs), 5)
    assert np.all(neighs == expected_neighs)
