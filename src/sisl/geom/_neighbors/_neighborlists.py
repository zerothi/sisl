# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from functools import cached_property
from typing import Optional

import numpy as np

from sisl import Geometry

__all__ = [
    "UniqueNeighborList",
    "FullNeighborList",
    "PartialNeighborList",
    "AtomNeighborList",
    "PointsNeighborList",
    "PointNeighborList",
]


class Neighbors:

    def __init__(
        self,
        geometry: Geometry,
        finder_results: np.ndarray,
        split_indices: Optional[np.ndarray] = None,
    ):
        self.geometry = geometry
        self._finder_results = finder_results
        self._split_indices = split_indices

    @property
    def i(self) -> np.ndarray:
        """For each neighbor pair (i, j), the first index."""
        return self._finder_results[:, 0]

    @property
    def j(self) -> np.ndarray:
        """For each neighbor pair (i, j), the second index."""
        return self._finder_results[:, 1]

    @property
    def isc(self) -> np.ndarray:
        """For each neighbor pair (i, j), the supercell indices of `j`."""
        return self._finder_results[:, 2:]

    @cached_property
    def min_nsc(self) -> np.ndarray:
        """The minimum nsc for an auxiliary supercell to contain all neighbors in this object."""
        if len(self.isc) == 0:
            return np.ones(3, dtype=int)
        else:
            return np.max(np.abs(self.isc), axis=0) * 2 + 1

    @property
    def n_neighbors(self):
        raise NotImplementedError

    @cached_property
    def split_indices(self) -> np.ndarray:
        """Indices to split the interactions of each atom."""
        if self._split_indices is None:
            return np.cumsum(self.n_neighbors)
        else:
            return self._split_indices

    def __len__(self):
        return len(self.split_indices)


class AtomsNeighborList(Neighbors):
    """Base class for interactions between atoms."""

    @cached_property
    def n_neighbors(self) -> np.ndarray:
        """Number of neighbors that each atom has."""

        if self._split_indices is None:
            n_neighbors = np.zeros(self.geometry.na, dtype=int)
            index, counts = np.unique(self.i, return_counts=True)

            n_neighbors[index] = counts
            return n_neighbors
        else:
            return np.diff(self._split_indices, prepend=0)


class UniqueNeighborList(AtomsNeighborList):
    """Full neighbors list of a system, but **containing only the upper triangle of the adjacency matrix**.

    What this means, is that the the object only contains one direction of each interaction.

    This is only possible if the interaction is symmetric (there is no directionality and
    thresholds for interaction do not depend on direction).

    Examples
    --------

    You can get a unique neighbors list from the `find_unique_pairs` method of a `NeighborFinder` object.
    Then, you can retreive the neighbors from it:

    .. code-block:: python

        import sisl

        # Build a graphene supercell with a vacancy
        graphene = sisl.geom.graphene().tile(2, 0).tile(2, 1)
        graphene = graphene.remove(2).translate2uc()

        # Initialize a finder for neighbors that are within 1.5 Angstrom
        finder = sisl.geom.NeighborFinder(graphene, R=1.5)

        # Get the list of unique neighbor pairs
        neighbors = finder.find_unique_pairs()

        # You can get the neighbor pairs (i,j) from the i and j attributes
        # The supercell index of atom J is in the isc attribute.
        print("ATOM I", neighbors.i)
        print("ATOM J (NEIGHBOR)", neighbors.j)
        print("NEIGHBORS ISC:", neighbors.isc)

        # Notice that I is always smaller than J. Each connection is only
        # stored once.
        # You can convert to a full neighbors list, which will contain both
        # directions.
        full_neighbors = neighbors.to_full()

    """

    @cached_property
    def n_neighbors(self) -> np.ndarray:
        """Number of neighbors that each atom has."""

        if self._split_indices is None:
            n_neighbors = np.zeros(self.geometry.na, dtype=int)
            index, counts = np.unique([self.i, self.j], return_counts=True)

            n_neighbors[index] = counts
            return n_neighbors
        else:
            return np.diff(self._split_indices, prepend=0)

    def to_full(self) -> FullNeighborList:
        """Converts the unique neighbors list to a full neighbors list."""
        upper_tri = self._finder_results
        lower_tri = np.column_stack(
            [self.j, self.i, -self.isc[:, 0], -self.isc[:, 1], -self.isc[:, 2]]
        )

        self_interactions = (self.i == self.j) & np.all(self.isc == 0, axis=1)
        lower_tri = lower_tri[~self_interactions]

        # Concatenate the lower triangular with the upper triangular part
        all_finder_results = np.concatenate([upper_tri, lower_tri], axis=0)

        # Sort by i and then by j
        sorted_indices = np.lexsort(all_finder_results[:, [1, 0]].T)
        all_finder_results = all_finder_results[sorted_indices]

        return FullNeighborList(
            self.geometry,
            all_finder_results,
        )


class FullNeighborList(AtomsNeighborList):
    """Full neighbors list of a system.

    This class, contrary to `UniqueNeighborList`, (possibly) contains the two directions
    of an interaction between two given atoms. Notice that it is possible that there is
    a connection from atom `i` to atom `j` but not the other way around.

    Examples
    --------

    You can get a full neighbors list from the `find_neighbors` method of a `NeighborFinder` object.
    Then, you can retreive the neighbors from it:

    .. code-block:: python

        import sisl

        # Build a graphene supercell with a vacancy
        graphene = sisl.geom.graphene().tile(2, 0).tile(2, 1)
        graphene = graphene.remove(2).translate2uc()

        # Initialize a finder for neighbors that are within 1.5 Angstrom
        finder = sisl.geom.NeighborFinder(graphene, R=1.5)

        # Get the full neighbors list
        neighbors = finder.find_neighbors()

        # You can loop through atoms to get their neighbors
        for at_neighs in neighbors:
            print()
            print(f"NEIGHBORS OF ATOM {at_neighs.atom} ({at_neighs.n_neighbors} neighbors): ")
            print("J", at_neighs.j)
            print("ISC", at_neighs.isc)

        # Or get the neighbors of a particular atom:
        neighbors[0].j

    See Also
    --------
    AtomNeighborList
        The object returned by this list when iterating or indexing.

    """

    def __getitem__(self, item) -> AtomNeighborList:
        """Returns the interactions of a given atom."""
        if isinstance(item, int):

            start = 0 if item == 0 else self.split_indices[item - 1]
            end = self.split_indices[item]

            return AtomNeighborList(
                self.geometry,
                self._finder_results[start:end],
                atom=item,
            )
        else:
            raise ValueError("Only integer indexing is supported.")

    def to_unique(self) -> UniqueNeighborList:
        """Converts the full neighbors list to a unique neighbors list."""

        full_finder_results = self._finder_results
        unique_finder_results = full_finder_results[self.i <= self.j]

        # Concatenate the uc connections with the rest of the connections.
        return UniqueNeighborList(
            geometry=self.geometry, finder_results=unique_finder_results
        )


class PartialNeighborList(AtomsNeighborList):
    """Neighbors list containing only the neighbors of some atoms.

    Examples
    --------

    You can get a partial neighbors list from the `find_neighbors` method of a
    `NeighborFinder` object if you pass the `atoms` argument. Then, you can
    retreive the neighbors from it:

    .. code-block:: python

        import sisl

        # Build a graphene supercell with a vacancy
        graphene = sisl.geom.graphene().tile(2, 0).tile(2, 1)
        graphene = graphene.remove(2).translate2uc()

        # Initialize a finder for neighbors that are within 1.5 Angstrom
        finder = sisl.geom.NeighborFinder(graphene, R=1.5)

        # Get a partial neighbors list
        neighbors = finder.find_neighbors(atoms=[2, 4])

        # You can loop through atoms to get their neighbors
        # In this case, the loop will go through atoms 2 and 4
        for at_neighs in neighbors:
            print()
            print(f"NEIGHBORS OF ATOM {at_neighs.atom} ({at_neighs.n_neighbors} neighbors): ")
            print("J", at_neighs.j)
            print("ISC", at_neighs.isc)

        # Or get the neighbors of a particular atom
        neighbors[0].atom # This will be 2
        neighbors[0].j

    See Also
    --------
    AtomNeighborList
        The object returned by this list when iterating or indexing.
    """

    #: The atoms for which the neighbors are stored.
    atoms: np.ndarray

    def __init__(
        self, geometry: Geometry, finder_results, atoms: np.ndarray, split_indices=None
    ):
        self.atoms = atoms
        super().__init__(geometry, finder_results, split_indices)

    @cached_property
    def n_neighbors(self) -> np.ndarray:
        """Number of neighbors that each atom has."""

        if self._split_indices is None:
            return np.array([np.sum(self.i == at) for at in self.atoms])
        else:
            return np.diff(self._split_indices, prepend=0)

    def __getitem__(self, item) -> AtomNeighborList:
        """Returns the interactions of a given atom."""
        if isinstance(item, int):

            start = 0 if item == 0 else self.split_indices[item - 1]
            end = self.split_indices[item]

            return AtomNeighborList(
                self.geometry,
                self._finder_results[start:end],
                atom=self.atoms[item],
            )
        else:
            raise ValueError("Only integer indexing is supported.")


class AtomNeighborList(Neighbors):
    """List of atoms that are close to a given atom.

    The usual way to get an `AtomNeighborList` object is by iterating over a
    `FullNeighborList` or a `PartialNeighborList`. See their documentation for
    examples.

    See Also
    --------
    FullNeighborList, PartialNeighborList
        The lists that, when iterated, return `AtomNeighborList` objects.
    """

    #: The atom for which the neighbors are stored.
    atom: int

    def __init__(self, geometry, finder_results, atom: int):
        self.atom = atom
        super().__init__(geometry, finder_results)

    @cached_property
    def n_neighbors(self) -> int:
        """Number of neighbors of the atom."""
        return len(self._finder_results)


class PointsNeighborList(Neighbors):
    """List of atoms that are close to a set of points in space.

    Examples
    --------

    You can get a points neighbors list from the `find_close` method of a
    `NeighborFinder` object. Then, you can retreive the neighbors from it:

    .. code-block:: python

        import sisl

        # Build a graphene supercell with a vacancy
        graphene = sisl.geom.graphene().tile(2, 0).tile(2, 1)
        graphene = graphene.remove(2).translate2uc()

        # Initialize a finder for neighbors that are within 1.5 Angstrom
        finder = sisl.geom.NeighborFinder(graphene, R=1.5)

        # Get the full neighbors list
        points = [[0, 0, 0], [2, 0, 0]]
        neighbors = finder.find_close(points)

        # You can loop through points to get their neighbors
        for point_neighs in neighbors:
            print()
            print(f"NEIGHBORS OF POINT {point_neighs.point} ({point_neighs.n_neighbors} neighbors): ")
            print("J", point_neighs.j)
            print("ISC", point_neighs.isc)

        # Or get the neighbors of a particular point:
        neighbors[0].j

    See Also
    --------
    PointNeighborList
        The object returned by this list when iterating or indexing.

    """

    def __init__(
        self, geometry, points: np.ndarray, finder_results, split_indices=None
    ):
        self.points = points
        super().__init__(geometry, finder_results, split_indices)

    @cached_property
    def n_neighbors(self) -> np.ndarray:
        """Number of atoms that are close to each point"""

        if self._split_indices is None:
            n_neighbors = np.zeros(len(self.points), dtype=int)
            index, counts = np.unique(self.i, return_counts=True)
            n_neighbors[index] = counts
            return n_neighbors
        else:
            return np.diff(self._split_indices, prepend=0)

    def __getitem__(self, item):
        """Returns the interactions of a given point."""
        if isinstance(item, int):

            start = 0 if item == 0 else self.split_indices[item - 1]
            end = self.split_indices[item]

            return PointNeighborList(
                self.geometry,
                self.points[item],
                self._finder_results[start:end],
            )
        else:
            raise ValueError("Only integer indexing is supported.")


class PointNeighborList(Neighbors):
    """List of atoms that are close to a point in space.

    The usual way to get a `PointNeighborList` object is by iterating over a
    `PointsNeighborList`. See its documentation for examples.

    See Also
    --------
    PointsNeighborList
        The list that, when iterated, returns `PointNeighborList` objects.
    """

    #: The point for which the neighbors are stored.
    point: np.ndarray

    def __init__(self, geometry, point: np.ndarray, finder_results: np.ndarray):
        self.point = point
        super().__init__(geometry, finder_results)

    @cached_property
    def n_neighbors(self) -> int:
        """Number of neighbors of the point."""
        return len(self._finder_results)
