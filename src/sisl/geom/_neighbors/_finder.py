# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from sisl import Geometry
from sisl._internal import set_module
from sisl.typing import AtomsIndex
from sisl.utils import size_to_elements

from . import _operations
from ._neighborlists import (
    CoordsNeighborList,
    FullNeighborList,
    PartialNeighborList,
    UniqueNeighborList,
)

__all__ = [
    "NeighborFinder",
]


@set_module("sisl.geom")
class NeighborFinder:
    r"""Fast and linear scaling finding of neighbors.

    Once this class is instantiated, a table is built. Then,
    the neighbor finder can be queried as many times as wished
    as long as the geometry doesn't change.

    Note that the radius (`R`) is used to build the table.
    Therefore, if one wants to look for neighbors using a different
    R, one needs to create another finder or call `setup`.

    Parameters
    ----------
    geometry: Geometry
        the geometry to find neighbors in
    R: float or array-like of shape (geometry.na), optional
        The radius to consider two atoms neighbors.
        If it is a single float, the same radius is used for all atoms.
        If it is an array, it should contain the radius for each atom.

        If not provided, an array is constructed, where the radius for
        each atom is its maxR.
    overlap: bool, optional
        If `True`, two atoms are considered neighbors if their spheres
        overlap.
        If `False`, two atoms are considered neighbors if the second atom
        is within the sphere of the first atom. Note that this implies that
        atom :math:`I` might be atom :math:`J`'s neighbor while the opposite is not true.

        If not provided, it will be `True` if `R` is an array and `False` if
        it is a single float.
    bin_size : float or tuple of float, optional
        the factor for the radius to determine how large the bins are,
        optionally along each lattice vector.
        It can minimally be 2, meaning that the maximum radius to consider
        is twice the radius considered. For larger values, more atoms will be in
        each bin (and thus fewer bins).
        Hence, this value can be used to fine-tune the memory requirement by
        decreasing number of bins, at the cost of a bit more run-time searching
        bins.

    Examples
    --------

    You need to initialize it with a geometry and the cutoff radius. Then,
    you can call the ``find_neighbors``, ``find_unique_pairs`` or ``find_close``
    methods to query the finder for neighbors. These methods will return a neighbors
    list (e.g ``find_neighbors`` returns a ``FullNeighborList``).

    Here is an example of how to find all neighbors on a graphene structure with
    a vacancy:

    .. code-block:: python

        import sisl

        # Build a graphene supercell with a vacancy
        graphene = sisl.geom.graphene().tile(2, 0).tile(2, 1)
        graphene = graphene.remove(2).translate2uc()

        # Initialize a finder for neighbors that are within 1.5 Angstrom
        finder = sisl.geom.NeighborFinder(graphene, R=1.5)

        # Find all neighbors
        neighbors = finder.find_neighbors()

        # You can get the neighbor pairs (I,J) from the I and J attributes
        # The supercell index of atom J is in the isc attribute.
        print("ATOM I SHAPE:", neighbors.I.shape)
        print("ATOM J (NEIGHBOR) SHAPE:", neighbors.J.shape)
        print("NEIGHBORS ISC:", neighbors.isc.shape)

        # You can also loop through atoms to get their neighbors
        for at_neighs in neighbors:
            print()
            print(f"NEIGHBORS OF ATOM {at_neighs.atom} ({at_neighs.n_neighbors} neighbors): ")
            print("J", at_neighs.J)
            print("ISC", at_neighs.isc)

        # Or get the neighbors of a particular atom:
        neighbors[0].J

    See Also
    --------
    FullNeighborList, UniqueNeighborList, PartialNeighborList, CoordsNeighborList:
        The neighbor lists returned by this class when neighbors are requested.

    """

    #: Memory control of the finder
    memory: tuple[str, float] = ("200MB", 1.5)
    #: Number of bins along each cell direction
    nbins: tuple[int, int, int]
    #: Total number of bins
    total_nbins: int

    #: The geometry associated with the finder
    geometry: Geometry
    # Geometry actually used for binning. Can be the provided geometry
    # or a tiled geometry if the search radius is too big (compared to the lattice size).
    _bins_geometry: Geometry

    #: The cutoff radius for each atom in the geometry.
    R: np.ndarray
    _aux_R: np.ndarray
    _overlap: bool

    # Data structure
    _list: np.ndarray  # (natoms, )
    _heads: np.ndarray  # (total_nbins, )
    _counts: np.ndarray  # (total_nbins, )

    def __init__(
        self,
        geometry: Geometry,
        R: Optional[Union[float, np.ndarray]] = None,
        overlap: bool = False,
        bin_size: Union[float, tuple[float, float, float]] = 2,
    ):
        self.setup(geometry, R=R, overlap=overlap, bin_size=bin_size)

    def setup(
        self,
        geometry: Optional[Geometry] = None,
        R: Optional[Union[float, np.ndarray]] = None,
        overlap: bool = None,
        bin_size: Union[float, tuple[float, float, float]] = 2,
    ):
        r"""Prepares everything for neighbor finding.

        **You don't need to call this method after initializing the finder**,
        this is called internally already!

        This method might be used to reset the finder with new parameters.

        Parameters
        ----------
        geometry:
            the geometry to find neighbors in.

            If not provided, the current geometry is kept.
        R: float or array-like of shape (geometry.na), optional
            The radius to consider two atoms neighbors.
            If it is a single float, the same radius is used for all atoms.
            If it is an array, it should contain the radius for each atom.

            If not provided, an array is constructed, where the radius for
            each atom is its maxR.

            Note that negative R values are allowed
        overlap:
            If `True`, two atoms are considered neighbors if their spheres
            overlap.
            If `False`, two atoms are considered neighbors if the second atom
            is within the sphere of the first atom. Note that this implies that
            atom :math:`I` might be atom :math:`J`'s neighbor while the opposite is not true.
        bin_size :
            the factor for the radius to determine how large the bins are,
            optionally along each lattice vector.
            It can minimally be 2, meaning that the maximum radius to consider
            is twice the radius considered. For larger values, more atoms will be in
            each bin (and thus fewer bins).
            Hence, this value can be used to fine-tune the memory requirement by
            decreasing number of bins, at the cost of a bit more run-time searching
            bins.
        """
        # Set the geometry. Copy it because we may need to modify the supercell size.
        if geometry is not None:

            # Warn that we do not support atoms outside the unit cell just yet.
            fxyz = geometry.fxyz
            if np.any((fxyz < -1e-8) | (fxyz > (1 + 1e-8))):
                raise ValueError(
                    f"Coordinates outside the unit cell are not supported by {self.__class__.__name__} for now. "
                    "You can do geometry.translate2uc() to move atoms to the unit cell, but note that "
                    "this will change the supercell indices of the connections and might not be compatible "
                    "with the indices of your sparse matrices, for example."
                )

            self.geometry = geometry.copy()

        # If R was not provided, build an array of Rs
        if R is None:
            R = self.geometry.atoms.maxR(all=True)
        else:
            R = np.asarray(R)

        # Set the radius
        self.R = R
        self._aux_R = R

        # If sphere overlap was not provided, set it to False if R is a single float
        # and True otherwise.
        self._overlap = overlap

        # Determine the bin_size as the maximum DIAMETER to ensure that we ALWAYS
        # only need to look one bin away for neighbors.
        max_R = np.max(self.R)
        if overlap:
            # In the case when we want to check for sphere overlap, the size should
            # be twice as big.
            max_R *= 2

        if max_R <= 0:
            raise ValueError(
                "All R values are 0 or less. Please provide some positive values"
            )

        bin_size = np.asarray(bin_size)
        if np.any(bin_size < 2):
            raise ValueError(
                "The bin_size must be larger than 2 to only search in the "
                "neighboring bins. Please increase to a value >=2"
            )

        bin_size = max_R * bin_size

        # We add a small amount to bin_size to avoid ambiguities when
        # a position is exactly at the center of a bin.
        bin_size += 0.001

        lattice_sizes = self.geometry.length

        self._R_too_big = np.any(bin_size > lattice_sizes)
        if self._R_too_big:
            # This means that nsc must be at least 5.

            # We round the amount of cells needed in each direction
            # to the closest next odd number.
            nsc = np.ceil(bin_size / lattice_sizes) // 2 * 2 + 1
            # And then set it as the number of supercells.
            self.geometry.set_nsc(nsc.astype(int))
            if self._aux_R.ndim == 1:
                self._aux_R = np.tile(self._aux_R, self.geometry.n_s)

            all_xyz = []
            for isc in self.geometry.sc_off:
                ats_xyz = self.geometry.axyz(isc=isc)
                all_xyz.append(ats_xyz)

            self._bins_geometry = Geometry(
                np.concatenate(all_xyz), atoms=self.geometry.atoms
            )

            # Recompute lattice sizes
            lattice_sizes = self._bins_geometry.length

        else:
            self._bins_geometry = self.geometry

        # Get the number of bins along each cell direction.
        nbins_float = lattice_sizes / bin_size
        self.nbins = tuple(np.floor(nbins_float).astype(int))
        self.total_nbins = np.prod(self.nbins)

        # Get the scalar bin indices of all atoms
        scalar_bin_indices = self._get_bin_indices(self._bins_geometry.fxyz)

        # Build the tables that will allow us to look for neighbors in an efficient
        # and linear scaling manner.
        self._build_table(scalar_bin_indices)

    def _build_table(self, bin_indices):
        """Builds the tables that will allow efficient linear scaling neighbor search.

        Parameters
        ----------
        bin_indices: array-like of shape (self.total_nbins, )
            Array containing the scalar bin index for each atom.
        """
        # Call the fortran routine that builds the table
        self._list, self._heads, self._counts = _operations.build_table(
            self.total_nbins, bin_indices
        )

    def assert_consistency(self):
        """Asserts that the data structure (self._list, self._heads, self._counts) is consistent.

        It also stores that the shape is consistent with the stored geometry and the store total_nbins.
        """
        # Check shapes
        assert self._list.shape == (self._bins_geometry.na,)
        assert self._counts.shape == self._heads.shape == (self.total_nbins,)

        # Check values
        for i_bin, bin_count in enumerate(self._counts):
            count = 0
            head = self._heads[i_bin]
            while head != -1:
                count += 1
                head = self._list[head]

            assert (
                count == bin_count
            ), f"Bin {i_bin} has {bin_count} atoms but we found {count} atoms"

    def _get_search_atom_counts(self, scalar_bin_indices):
        """Computes the number of atoms that will be explored for each search

        Parameters
        -----------
        scalar_bin_indices: np.ndarray of shape ([n_searches], 8)
            Array containing the bin indices for each search.
            Bin indices should be within the unit cell!

        Returns
        -----------
        np.ndarray of shape (n_searches, ):
            For each search, the number of atoms that will be involved.
        """
        return self._counts[scalar_bin_indices.ravel()].reshape(-1, 8).sum(axis=1)

    def _get_bin_indices(self, fxyz, cartesian=False, floor=True):
        """Gets the bin indices for a given fractional coordinate.

        Parameters
        -----------
        fxyz: np.ndarray of shape (N, 3)
            the fractional coordinates for which we want to get the bin indices.
        cartesian: bool, optional
            whether the indices should be returned as cartesian.
            If `False`, scalar indices are returned.
        floor: bool, optional
            whether to floor the indices or not.

            If asking for scalar indices (i.e. `cartesian=False`), the indices will
            ALWAYS be floored regardless of this argument.

        Returns
        --------
        np.ndarray:
            The bin indices. If `cartesian=True`, the shape of the array is (N, 3),
            otherwise it is (N,).
        """
        # Avoid numerical errors in coordinates
        fxyz[(fxyz <= 0) & (fxyz > -1e-8)] = 1e-8
        fxyz[(fxyz >= 1) & (fxyz < 1 + 1e-8)] = 1 - 1e-8

        bin_indices = ((fxyz) % 1) * self.nbins
        # Atoms that are exactly at the limit of the cell might have fxyz = 1
        # which would result in a bin index outside of range.
        # We just bring it back to the unit cell.
        bin_indices = bin_indices % self.nbins

        if floor or not cartesian:
            bin_indices = np.floor(bin_indices).astype(int)

        if not cartesian:
            bin_indices = self._cartesian_to_scalar_index(bin_indices)

        return bin_indices

    def _get_search_indices(self, fxyz, cartesian=False):
        r"""Gets the bin indices to explore for a given fractional coordinate.

        Given a fractional coordinate, we will need to look for neighbors
        in its own bin, and one bin away in each direction. That is, :math:`2^3=8` bins.

        Parameters
        -----------
        fxyz: np.ndarray of shape (N, 3)
            the fractional coordinates for which we want to get the search indices.
        cartesian: bool, optional
            whether the indices should be returned as cartesian.
            If `False`, scalar indices are returned.

        Returns
        --------
        np.ndarray:
            The bin indices where we need to perform the search for each
            fractional coordinate. These indices are all inside the unit cell.
            If ``cartesian=True``, cartesian indices are returned and the array
            has shape (N, 8, 3).
            If ``cartesian=False``, scalar indices are returned and the array
            has shape (N, 8).
        np.ndarray of shape (N, 8, 3):
            The supercell indices of each bin index in the search.
        """
        # Get the bin indices for the positions that are requested.
        # Note that we don't floor the indices so that we can know to which
        # border of the bin are we closer in each direction.
        bin_indices = self._get_bin_indices(fxyz, floor=False, cartesian=True)
        bin_indices = np.atleast_2d(bin_indices)

        # Determine which is the neighboring cell that we need to look for
        # along each direction.
        signs = np.ones(bin_indices.shape, dtype=int)
        signs[(bin_indices % 1) < 0.5] = -1

        # Build and arrays with all the indices that we need to look for. Since
        # we have to move one bin away in each direction, we have to look for
        # neighbors along a total of 8 bins (2**3)
        search_indices = np.tile(bin_indices.astype(int), 8).reshape(-1, 8, 3)

        search_indices[:, 1::2, 0] += signs[:, 0].reshape(-1, 1)
        search_indices[:, [2, 3, 6, 7], 1] += signs[:, 1].reshape(-1, 1)
        search_indices[:, 4:, 2] += signs[:, 2].reshape(-1, 1)

        # Convert search indices to unit cell indices, but keep the supercell indices.
        isc, search_indices = np.divmod(search_indices, self.nbins)

        if not cartesian:
            search_indices = self._cartesian_to_scalar_index(search_indices)

        return search_indices, isc

    def _cartesian_to_scalar_index(self, index):
        """Converts cartesian indices to scalar indices"""
        if not np.issubdtype(index.dtype, int):
            raise ValueError(
                "Decimal scalar indices do not make sense, please floor your cartesian indices."
            )
        return index.dot([1, self.nbins[0], self.nbins[0] * self.nbins[1]])

    def _scalar_to_cartesian_index(self, index):
        """Converts cartesian indices to scalar indices"""
        if np.min(index) < 0 or np.max(index) > self.total_nbins:
            raise ValueError(
                "Some scalar indices are not within the unit cell. We cannot uniquely convert to cartesian"
            )

        third, index = np.divmod(index, self.nbins[0] * self.nbins[1])
        second, first = np.divmod(index, self.nbins[0])
        return np.column_stack([first, second, third])

    def _correct_pairs_R_too_big(
        self,
        neighbor_pairs: np.ndarray,  # (n_pairs, 5)
        split_ind: Union[int, np.ndarray],  # (n_queried_atoms, )
    ):
        """Correction to atom and supercell indices when the binning has been done on a tiled geometry"""
        is_sc_neigh = neighbor_pairs[:, 1] >= self.geometry.na
        pbc = self.geometry.lattice.pbc

        invalid = None
        if not np.any(pbc):
            invalid = is_sc_neigh
        else:
            pbc_neighs = neighbor_pairs.copy()

            sc_neigh, uc_neigh = np.divmod(
                neighbor_pairs[:, 1][is_sc_neigh], self.geometry.na
            )
            isc_neigh = self.geometry.sc_off[sc_neigh]

            pbc_neighs[is_sc_neigh, 1] = uc_neigh
            pbc_neighs[is_sc_neigh, 2:] = isc_neigh

            if not np.all(pbc):
                invalid = pbc_neighs[:, 2:][:, ~pbc].any(axis=1)

            neighbor_pairs = pbc_neighs

        if invalid is not None:
            neighbor_pairs = neighbor_pairs[~invalid]
            if isinstance(split_ind, int):
                split_ind = split_ind - invalid.sum()
            else:
                split_ind = split_ind - np.cumsum(invalid)[split_ind - 1]

        return neighbor_pairs, split_ind

    def find_neighbors(
        self,
        atoms: AtomsIndex = None,
        self_interaction: bool = False,
    ) -> Union[FullNeighborList, PartialNeighborList]:
        """Find neighbors as specified in the finder.

        This routine only executes the action of finding neighbors,
        the parameters of the search (`geometry`, `R`, `overlap`...)
        are defined when the finder is initialized or by calling `setup`.

        Parameters
        ----------
        atoms:
            the atoms for which neighbors are desired. Anything that can
            be sanitized by `sisl.Geometry` is a valid value.

            If not provided, neighbors for all atoms are searched.
        self_interaction:
            whether to consider an atom a neighbor of itself.

        Returns
        ----------
        neighbors
            The neighbors list. It will be a partial list if `atoms` is provided.
        """
        unsanitized_atoms = atoms
        # Sanitize atoms
        atoms = self.geometry._sanitize_atoms(atoms)

        # Cast R into array of appropiate shape and type.
        thresholds = np.full(self._bins_geometry.na, self._aux_R, dtype=np.float64)

        # Get search indices
        search_indices, isc = self._get_search_indices(
            self.geometry.fxyz[atoms], cartesian=False
        )

        # Get atom counts
        at_counts = self._get_search_atom_counts(search_indices)

        max_pairs = at_counts.sum()
        if not self_interaction:
            max_pairs -= search_indices.shape[0]
        init_pairs = min(
            max_pairs,
            # 8 bytes per element (int64)
            # While this might be wrong on Win, it shouldn't matter
            size_to_elements(self.memory[0], 8),
        )

        # Find the neighbor pairs
        neighbor_pairs, split_ind = _operations.get_pairs(
            atoms,
            search_indices,
            isc,
            self._heads,
            self._list,
            self_interaction,
            self._bins_geometry.xyz,
            self._bins_geometry.cell,
            self.geometry.lattice.pbc,
            thresholds,
            self._overlap,
            init_pairs,
            self.memory[1],
        )

        # Correct neighbor indices for the case where R was too big and
        # we needed to create an auxiliary supercell.
        if self._R_too_big:
            neighbor_pairs, split_ind = self._correct_pairs_R_too_big(
                neighbor_pairs, split_ind
            )

        if unsanitized_atoms is None:
            return FullNeighborList(
                self.geometry, neighbor_pairs, split_indices=split_ind
            )
        else:
            return PartialNeighborList(
                self.geometry, neighbor_pairs, atoms=atoms, split_indices=split_ind
            )

    def find_unique_pairs(
        self,
        self_interaction: bool = False,
    ) -> UniqueNeighborList:
        r"""Find unique neighbor pairs within the geometry.

        This function only returns one direction of a given connection
        between atoms :math:`I` and :math:`J`. In particular, it returns the connection
        where :math:`I < J`.

        Note that this routine can not be called if `overlap` is
        set to `False` and the radius is not a single float. In that case,
        there is no way of defining *uniqueness* since pair :math:`IJ` can have
        a different threshold radius than pair :math:`JI`.

        Parameters
        ----------
        self_interaction :
            whether to consider an atom a neighbor of itself.
        """
        if not self._overlap and self._aux_R.ndim == 1:
            raise ValueError(
                "Unique atom pairs do not make sense if we are not looking for sphere overlaps."
                " Please setup the finder again setting `overlap` to `True` if you wish so."
            )

        # In the case where we tiled the geometry to do the binning, it is much better to
        # just find all neighbors and then drop duplicate connections. Otherwise it is a bit of a mess.
        if self._R_too_big:
            # Find all neighbors
            all_neighbors = self.find_neighbors(self_interaction=self_interaction)

            return all_neighbors.to_unique()

        # Cast R into array of appropiate shape and type.
        thresholds = np.full(self.geometry.na, self.R, dtype=np.float64)

        # Get search indices
        search_indices, isc = self._get_search_indices(
            self.geometry.fxyz, cartesian=False
        )

        # Get atom counts
        at_counts = self._get_search_atom_counts(search_indices)

        max_pairs = at_counts.sum()
        if not self_interaction:
            max_pairs -= search_indices.shape[0]
        init_pairs = min(
            max_pairs,
            # 8 bytes per element (int64)
            # While this might be wrong on Win, it shouldn't matter
            size_to_elements(self.memory[0], 8),
        )

        # Find all unique neighbor pairs
        neighbor_pairs = _operations.get_all_unique_pairs(
            search_indices,
            isc,
            self._heads,
            self._list,
            self_interaction,
            self.geometry.xyz,
            self.geometry.cell,
            self.geometry.lattice.pbc,
            thresholds,
            self._overlap,
            init_pairs,
            self.memory[1],
        )

        return UniqueNeighborList(self.geometry, neighbor_pairs)

    def find_close(
        self,
        xyz: Sequence,
    ) -> CoordsNeighborList:
        """Find all atoms that are close to some coordinates in space.

        This routine only executes the action of finding neighbors,
        the parameters of the search (`geometry`, `R`, `overlap`...)
        are defined when the finder is initialized or by calling `setup`.

        Parameters
        ----------
        xyz: array-like of shape ([npoints], 3)
            the coordinates for which neighbors are desired.
        """
        # Cast R into array of appropiate shape and type.
        thresholds = np.full(self._bins_geometry.na, self._aux_R, dtype=np.float64)

        xyz = np.atleast_2d(xyz).astype(float)
        # Get search indices
        search_indices, isc = self._get_search_indices(
            xyz.dot(self._bins_geometry.icell.T) % 1, cartesian=False
        )

        # Get atom counts
        at_counts = self._get_search_atom_counts(search_indices)

        max_pairs = at_counts.sum()
        init_pairs = min(
            max_pairs,
            # 8 bytes per element (int64)
            # While this might be wrong on Win, it shouldn't matter
            size_to_elements(self.memory[0], 8),
        )

        # Find the neighbor pairs
        neighbor_pairs, split_ind = _operations.get_close(
            xyz,
            search_indices,
            isc,
            self._heads,
            self._list,
            self._bins_geometry.xyz,
            self._bins_geometry.cell,
            self.geometry.lattice.pbc,
            thresholds,
            init_pairs,
            self.memory[1],
        )

        # Correct neighbor indices for the case where R was too big and
        # we needed to create an auxiliary supercell.
        if self._R_too_big:
            neighbor_pairs, split_ind = self._correct_pairs_R_too_big(
                neighbor_pairs, split_ind
            )

        return CoordsNeighborList(
            self.geometry, xyz, neighbor_pairs[: split_ind[-1]], split_indices=split_ind
        )
