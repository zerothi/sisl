# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from functools import cached_property
from numbers import Integral
from typing import Optional

import numpy as np

from sisl import Geometry
from sisl._internal import set_module

__all__ = [
    "UniqueNeighborList",
    "FullNeighborList",
    "PartialNeighborList",
    "AtomNeighborList",
    "CoordsNeighborList",
    "CoordNeighborList",
]


@set_module("sisl.geom")
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
    def I(self) -> np.ndarray:
        """For each neighbor pair (I, J), the first index."""
        return self._finder_results[:, 0]

    @property
    def i(self) -> np.ndarray:
        """Same as `I`, provided for backwards compatibility, may be deprecated later"""
        return self.I

    @property
    def J(self) -> np.ndarray:
        """For each neighbor pair (I, J), the second index."""
        return self._finder_results[:, 1]

    @property
    def j(self) -> np.ndarray:
        """Same as `J`, provided for backwards compatibility, may be deprecated later"""
        return self.J

    @property
    def isc(self) -> np.ndarray:
        r"""For each neighbor pair (I, J), the supercell indices of :math:`J`."""
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


@set_module("sisl.geom")
class AtomsNeighborList(Neighbors):
    """Base class for interactions between atoms."""

    @cached_property
    def n_neighbors(self) -> np.ndarray:
        """Number of neighbors that each atom has."""

        if self._split_indices is None:
            n_neighbors = np.zeros(self.geometry.na, dtype=int)
            index, counts = np.unique(self.I, return_counts=True)

            n_neighbors[index] = counts
            return n_neighbors
        else:
            return np.diff(self._split_indices, prepend=0)


@set_module("sisl.geom")
class UniqueNeighborList(AtomsNeighborList):
    """Full neighbors list of a system, but **containing only the upper triangle of the adjacency matrix**.

    What this means, is that the object only contains one direction of each interaction.

    This is only possible if the interaction is symmetric (there is no directionality and
    thresholds for interaction do not depend on direction).

    Examples
    --------

    You can get a unique neighbors list from the `find_unique_pairs` method of a `NeighborFinder` object.
    Then, you can retrieve the neighbors from it:

    .. code-block:: python

        import sisl

        # Build a graphene supercell with a vacancy
        graphene = sisl.geom.graphene().tile(2, 0).tile(2, 1)
        graphene = graphene.remove(2).translate2uc()

        # Initialize a finder for neighbors that are within 1.5 Angstrom
        finder = sisl.geom.NeighborFinder(graphene, R=1.5)

        # Get the list of unique neighbor pairs
        neighbors = finder.find_unique_pairs()

        # You can get the neighbor pairs (I,J) from the I and J attributes
        # The supercell index of atom J is in the isc attribute.
        print("ATOM I", neighbors.I)
        print("ATOM J (NEIGHBOR)", neighbors.J)
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
            index, counts = np.unique([self.I, self.J], return_counts=True)

            n_neighbors[index] = counts
            return n_neighbors
        else:
            return np.diff(self._split_indices, prepend=0)

    def to_full(self) -> FullNeighborList:
        """Converts the unique neighbors list to a full neighbors list."""
        upper_tri = self._finder_results
        lower_tri = np.column_stack(
            [self.J, self.I, -self.isc[:, 0], -self.isc[:, 1], -self.isc[:, 2]]
        )

        self_interactions = (self.I == self.J) & np.all(self.isc == 0, axis=1)
        lower_tri = lower_tri[~self_interactions]

        # Concatenate the lower triangular with the upper triangular part
        all_finder_results = np.concatenate([upper_tri, lower_tri], axis=0)

        # Sort by I and then by J
        sorted_indices = np.lexsort(all_finder_results[:, [1, 0]].T)
        all_finder_results = all_finder_results[sorted_indices]

        return FullNeighborList(
            self.geometry,
            all_finder_results,
        )


@set_module("sisl.geom")
class FullNeighborList(AtomsNeighborList):
    r"""Full neighbors list of a system.

    This class, contrary to `UniqueNeighborList`, (possibly) contains the two directions
    of an interaction between two given atoms. Notice that it is possible that there is
    a connection from atom :math:`I` to atom :math:`J` but not the other way around.

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
            print("J", at_neighs.J)
            print("ISC", at_neighs.isc)

        # Or get the neighbors of a particular atom:
        neighbors[0].J

    See Also
    --------
    AtomNeighborList
        The object returned by this list when iterating or indexing.

    """

    def __getitem__(self, item) -> AtomNeighborList:
        """Returns the interactions of a given atom."""
        if isinstance(item, Integral):

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
        unique_finder_results = full_finder_results[self.I <= self.J]

        # Concatenate the uc connections with the rest of the connections.
        return UniqueNeighborList(
            geometry=self.geometry, finder_results=unique_finder_results
        )


@set_module("sisl.geom")
class PartialNeighborList(AtomsNeighborList):
    """Neighbors list containing only the neighbors of some atoms.

    Examples
    --------

    You can get a partial neighbors list from the `find_neighbors` method of a
    `NeighborFinder` object if you pass the `atoms` argument. Then, you can
    retrieve the neighbors from it:

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
            print("J", at_neighs.J)
            print("ISC", at_neighs.isc)

        # Or get the neighbors of a particular atom
        neighbors[0].atom # This will be 2
        neighbors[0].J

    See Also
    --------
    AtomNeighborList
        The object returned by this list when iterating or indexing.
    """

    atoms: np.ndarray
    """The atoms for which the neighbors are stored."""

    def __init__(
        self, geometry: Geometry, finder_results, atoms: np.ndarray, split_indices=None
    ):
        self.atoms = atoms
        super().__init__(geometry, finder_results, split_indices)

    @cached_property
    def n_neighbors(self) -> np.ndarray:
        """Number of neighbors that each atom has."""

        if self._split_indices is None:
            return np.array([np.sum(self.I == at) for at in self.atoms])
        else:
            return np.diff(self._split_indices, prepend=0)

    def __getitem__(self, item) -> AtomNeighborList:
        """Returns the interactions of a given atom."""
        if isinstance(item, Integral):

            start = 0 if item == 0 else self.split_indices[item - 1]
            end = self.split_indices[item]

            return AtomNeighborList(
                self.geometry,
                self._finder_results[start:end],
                atom=self.atoms[item],
            )
        else:
            raise ValueError("Only integer indexing is supported.")


@set_module("sisl.geom")
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

    atom: int
    """The atom for which the neighbors are stored."""

    def __init__(self, geometry: Geometry, finder_results, atom: int):
        self.atom = atom
        super().__init__(geometry, finder_results)

    @cached_property
    def n_neighbors(self) -> int:
        """Number of neighbors of the atom."""
        return len(self._finder_results)


@set_module("sisl.geom")
class CoordsNeighborList(Neighbors):
    """List of atoms that are close to a set of coordinates in space.

    Examples
    --------

    You can get a coordinates neighbors list from the `find_close` method of a
    `NeighborFinder` object. Then, you can retrieve the neighbors from it:

    .. code-block:: python

        import sisl

        # Build a graphene supercell with a vacancy
        graphene = sisl.geom.graphene().tile(2, 0).tile(2, 1)
        graphene = graphene.remove(2).translate2uc()

        # Initialize a finder for neighbors that are within 1.5 Angstrom
        finder = sisl.geom.NeighborFinder(graphene, R=1.5)

        # Get the full neighbors list
        coords = [[0, 0, 0], [2, 0, 0]]
        neighbors = finder.find_close(coords)

        # You can loop through coordinates to get their neighbors
        for coord_neighs in neighbors:
            print()
            print(f"NEIGHBORS OF COORDINATE {coord_neighs.xyz} ({coord_neighs.n_neighbors} neighbors): ")
            print("J", coord_neighs.J)
            print("ISC", coord_neighs.isc)

        # Or get the neighbors of a particular coordinate:
        neighbors[0].J

    See Also
    --------
    CoordNeighborList
        The object returned by this list when iterating or indexing.

    """

    def __init__(self, geometry, xyzs: np.ndarray, finder_results, split_indices=None):
        self.xyzs = np.atleast_2d(xyzs)
        super().__init__(geometry, finder_results, split_indices)

    @cached_property
    def n_neighbors(self) -> np.ndarray:
        """Number of atoms that are close to each coordinate"""

        if self._split_indices is None:
            n_neighbors = np.zeros(len(self.xyzs), dtype=int)
            index, counts = np.unique(self.I, return_counts=True)
            n_neighbors[index] = counts
            return n_neighbors
        else:
            return np.diff(self._split_indices, prepend=0)

    def __getitem__(self, item):
        """Returns the interactions of a given coordinate."""
        if isinstance(item, Integral):

            start = 0 if item == 0 else self.split_indices[item - 1]
            end = self.split_indices[item]

            return CoordNeighborList(
                self.geometry,
                self.xyzs[item],
                self._finder_results[start:end],
            )
        else:
            raise ValueError("Only integer indexing is supported.")


@set_module("sisl.geom")
class CoordNeighborList(Neighbors):
    """List of atoms that are close to a coordinate in space.

    The usual way to get a `CoordNeighborList` object is by iterating over a
    `CoordsNeighborList`. See its documentation for examples.

    See Also
    --------
    CoordsNeighborList
        The list that, when iterated, returns `CoordNeighborList` objects.
    """

    xyz: np.ndarray
    """The Cartesian coordinate for which the neighbors are stored."""

    def __init__(self, geometry: Geometry, xyz: np.ndarray, finder_results: np.ndarray):
        self.xyz = xyz
        super().__init__(geometry, finder_results)

    @cached_property
    def n_neighbors(self) -> int:
        """Number of neighbors of the coordinate."""
        return len(self._finder_results)
