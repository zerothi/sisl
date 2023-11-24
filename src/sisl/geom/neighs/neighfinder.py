from numbers import Real
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from sisl import Geometry
from sisl.typing import AtomsArgument
from sisl.utils.mathematics import fnorm

from . import _neigh_operations


class NeighFinder:
    """Efficient linear scaling finding of neighbours.

    Once this class is instantiated, a table is build. Then,
    the neighbour finder can be queried as many times as wished
    as long as the geometry doesn't change.

    Note that the radius (`R`) is used to build the table.
    Therefore, if one wants to look for neighbours using a different
    R, one needs to create another finder or call `setup_finder`.

    Parameters
    ----------
    geometry: sisl.Geometry
        the geometry to find neighbours in
    R: float or array-like of shape (geometry.na), optional
        The radius to consider two atoms neighbours.
        If it is a single float, the same radius is used for all atoms.
        If it is an array, it should contain the radius for each atom.

        If not provided, an array is constructed, where the radius for
        each atom is its maxR.
    sphere_overlap: bool, optional
        If `True`, two atoms are considered neighbours if their spheres
        overlap.
        If `False`, two atoms are considered neighbours if the second atom
        is within the sphere of the first atom. Note that this implies that
        atom `i` might be atom `j`'s neighbour while the opposite is not true.

        If not provided, it will be `True` if `R` is an array and `False` if
        it is a single float.
    """

    geometry: Geometry
    # Geometry actually used for binning. Can be the provided geometry
    # or a tiled geometry if the search radius is too big (compared to the lattice size).
    _bins_geometry: Geometry

    nbins: Tuple[int, int, int]
    total_nbins: int

    R: Union[float, np.ndarray]
    _aux_R: Union[
        float, np.ndarray
    ]  # If the geometry has been tiled, R is also tiled here
    _sphere_overlap: bool

    # Data structure
    _list: np.ndarray  # (natoms, )
    _heads: np.ndarray  # (total_nbins, )
    _counts: np.ndarray  # (total_nbins, )

    def __init__(
        self,
        geometry: Geometry,
        R: Union[float, np.ndarray, None] = None,
        sphere_overlap: Optional[bool] = None,
    ):
        self.setup_finder(geometry, R=R, sphere_overlap=sphere_overlap)

    def setup_finder(
        self,
        geometry: Optional[Geometry] = None,
        R: Union[float, np.ndarray, None] = None,
        sphere_overlap: Optional[bool] = None,
    ):
        """Prepares everything for neighbour finding.

        Parameters
        ----------
        geometry: sisl.Geometry, optional
            the geometry to find neighbours in.

            If not provided, the current geometry is kept.
        R: float or array-like of shape (geometry.na), optional
            The radius to consider two atoms neighbours.
            If it is a single float, the same radius is used for all atoms.
            If it is an array, it should contain the radius for each atom.

            If not provided, an array is constructed, where the radius for
            each atom is its maxR.

            Note that negative R values are allowed
        sphere_overlap: bool, optional
            If `True`, two atoms are considered neighbours if their spheres
            overlap.
            If `False`, two atoms are considered neighbours if the second atom
            is within the sphere of the first atom. Note that this implies that
            atom `i` might be atom `j`'s neighbour while the opposite is not true.

            If not provided, it will be `True` if `R` is an array and `False` if
            it is a single float.
        """
        # Set the geometry. Copy it because we may need to modify the supercell size.
        if geometry is not None:
            self.geometry = geometry.copy()

        # If R is not a single float, make sure it is a numpy array
        R_is_float = isinstance(R, Real)
        if not R_is_float:
            R = np.array(R)
        # If R was not provided, build an array of Rs
        if R is None:
            R = self.geometry.atoms.maxR(all=True)

        # Set the radius
        self.R = R
        self._aux_R = R

        # If sphere overlap was not provided, set it to False if R is a single float
        # and True otherwise.
        if sphere_overlap is None:
            sphere_overlap = not R_is_float
        self._sphere_overlap = sphere_overlap

        # Determine the bin_size as the maximum DIAMETER to ensure that we ALWAYS
        # only need to look one bin away for neighbors.
        bin_size = np.max(self.R) * 2
        # Check that the bin size is positive.
        if bin_size <= 0:
            raise ValueError(
                "All R values are 0 or less. Please provide some positive values"
            )
        if sphere_overlap:
            # In the case when we want to check for sphere overlap, the size should
            # be twice as big.
            bin_size *= 2

        # We add a small amount to bin_size to avoid ambiguities when
        # a position is exactly at the center of a bin.
        bin_size += 0.01

        lattice_sizes = fnorm(self.geometry.cell, axis=-1)

        if bin_size > np.min(lattice_sizes):
            self._R_too_big = True
            # This means that nsc must be at least 5.

            # We round the amount of cells needed in each direction
            # to the closest next odd number.
            nsc = np.ceil(bin_size / lattice_sizes) // 2 * 2 + 1
            # And then set it as the number of supercells.
            self.geometry.set_nsc(nsc.astype(int))
            if isinstance(self._aux_R, np.ndarray):
                self._aux_R = np.tile(self._aux_R, self.geometry.n_s)

            all_xyz = []
            for isc in self.geometry.sc_off:
                ats_xyz = self.geometry.axyz(isc=isc)
                all_xyz.append(ats_xyz)

            self._bins_geometry = Geometry(
                np.concatenate(all_xyz), atoms=self.geometry.atoms
            )

            # Recompute lattice sizes
            lattice_sizes = fnorm(self._bins_geometry.cell, axis=-1)

        else:
            # Nothing to modify, we can use the geometry and bin it as it is.
            self._R_too_big = False
            self._bins_geometry = self.geometry

        # Get the number of bins along each cell direction.
        nbins_float = lattice_sizes / bin_size
        self.nbins = tuple(np.floor(nbins_float).astype(int))
        self.total_nbins = np.prod(self.nbins)

        # Get the scalar bin indices of all atoms
        scalar_bin_indices = self._get_bin_indices(self._bins_geometry.fxyz)

        # Build the tables that will allow us to look for neighbours in an efficient
        # and linear scaling manner.
        self._build_table(scalar_bin_indices)

    def _build_table(self, bin_indices):
        """Builds the tables that will allow efficient linear scaling neighbour search.

        Parameters
        ----------
        bin_indices: array-like of shape (self.total_nbins, )
            Array containing the scalar bin index for each atom.
        """
        # Call the fortran routine that builds the table
        self._list, self._heads, self._counts = _neigh_operations.build_table(
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
        """Gets the bin indices to explore for a given fractional coordinate.

        Given a fractional coordinate, we will need to look for neighbours
        in its own bin, and one bin away in each direction. That is, $2^3=8$ bins.

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
            If `cartesian=True`, cartesian indices are returned and the array
            has shape (N, 8, 3).
            If `cartesian=False`, scalar indices are returned and the array
            has shape (N, 8).
        np.ndarray of shape (N, 8, 3):
            The supercell indices of each bin index in the search.
        """
        # Get the bin indices for the positions that are requested.
        # Note that we don't floor the indices so that we can know to which
        # border of the bin are we closer in each direction.
        bin_indices = self._get_bin_indices(fxyz, floor=False, cartesian=True)
        bin_indices = np.atleast_2d(bin_indices)

        # Determine which is the neighbouring cell that we need to look for
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
        return np.array([first, second, third]).T

    def _correct_pairs_R_too_big(
        self,
        neighbour_pairs: np.ndarray,  # (n_pairs, 5)
        split_ind: Union[int, np.ndarray],  # (n_queried_atoms, )
        pbc: np.ndarray,  # (3, )
    ):
        """Correction to atom and supercell indices when the binning has been done on a tiled geometry"""
        is_sc_neigh = neighbour_pairs[:, 1] >= self.geometry.na

        invalid = None
        if not np.any(pbc):
            invalid = is_sc_neigh
        else:
            pbc_neighs = neighbour_pairs.copy()

            sc_neigh, uc_neigh = np.divmod(
                neighbour_pairs[:, 1][is_sc_neigh], self.geometry.na
            )
            isc_neigh = self.geometry.sc_off[sc_neigh]

            pbc_neighs[is_sc_neigh, 1] = uc_neigh
            pbc_neighs[is_sc_neigh, 2:] = isc_neigh

            if not np.all(pbc):
                invalid = pbc_neighs[:, 2:][:, ~pbc].any(axis=1)

            neighbour_pairs = pbc_neighs

        if invalid is not None:
            neighbour_pairs = neighbour_pairs[~invalid]
            if isinstance(split_ind, int):
                split_ind = split_ind - invalid.sum()
            else:
                split_ind = split_ind - np.cumsum(invalid)[split_ind - 1]

        return neighbour_pairs, split_ind

    def find_neighbours(
        self,
        atoms: AtomsArgument = None,
        as_pairs: bool = False,
        self_interaction: bool = False,
        pbc: Union[bool, Tuple[bool, bool, bool]] = (True, True, True),
    ):
        """Find neighbours as specified in the finder.

        This routine only executes the action of finding neighbours,
        the parameters of the search (`geometry`, `R`, `sphere_overlap`...)
        are defined when the finder is initialized or by calling `setup_finder`.

        Parameters
        ----------
        atoms: optional
            the atoms for which neighbours are desired. Anything that can
            be sanitized by `sisl.Geometry` is a valid value.

            If not provided, neighbours for all atoms are searched.
        as_pairs: bool, optional
            If `True` pairs of atoms are returned. Otherwise a list containing
            the neighbours for each atom is returned.
        self_interaction: bool, optional
            whether to consider an atom a neighbour of itself.
        pbc: bool or array-like of shape (3, )
            whether periodic conditions should be considered.
            If a single bool is passed, all directions use that value.

        Returns
        ----------
        np.ndarray or list:
            If `as_pairs=True`:
                A `np.ndarray` of shape (n_pairs, 5) is returned.
                Each pair `ij` means that `j` is a neighbour of `i`.
                The three extra columns are the supercell indices of atom `j`.
            If `as_pairs=False`:
                A list containing a numpy array of shape (n_neighs, 4) for each atom.
        """
        # Sanitize atoms
        atoms = self.geometry._sanitize_atoms(atoms)

        # Cast R and pbc into arrays of appropiate shape and type.
        thresholds = np.full(self._bins_geometry.na, self._aux_R, dtype=np.float64)
        pbc = np.full(3, pbc, dtype=bool)

        # Get search indices
        search_indices, isc = self._get_search_indices(
            self.geometry.fxyz[atoms], cartesian=False
        )

        # Get atom counts
        at_counts = self._get_search_atom_counts(search_indices)

        max_pairs = at_counts.sum()
        if not self_interaction:
            max_pairs -= search_indices.shape[0]

        # Find the neighbour pairs
        neighbour_pairs, split_ind = _neigh_operations.get_pairs(
            atoms,
            search_indices,
            isc,
            self._heads,
            self._list,
            max_pairs,
            self_interaction,
            self._bins_geometry.xyz,
            self._bins_geometry.cell,
            pbc,
            thresholds,
            self._sphere_overlap,
        )

        # Correct neighbour indices for the case where R was too big and
        # we needed to create an auxiliary supercell.
        if self._R_too_big:
            neighbour_pairs, split_ind = self._correct_pairs_R_too_big(
                neighbour_pairs, split_ind, pbc
            )

        if as_pairs:
            # Just return the neighbour pairs
            return neighbour_pairs[: split_ind[-1]]
        else:
            # Split to get the neighbours of each atom
            return np.split(neighbour_pairs[:, 1:], split_ind, axis=0)[:-1]

    def find_all_unique_pairs(
        self,
        self_interaction: bool = False,
        pbc: Union[bool, Tuple[bool, bool, bool]] = (True, True, True),
    ):
        """Find all unique neighbour pairs within the geometry.

        A pair of atoms is considered unique if atoms have the same index
        and correspond to the same unit cell. As an example, the connection
        atom 0 (unit cell) to atom 5 (1, 0, 0) is not the same as the
        connection atom 5 (unit cell) to atom 0 (-1, 0, 0).

        Note that this routine can not be called if `sphere_overlap` is
        set to `False` and the radius is not a single float. In that case,
        there is no way of defining "uniqueness" since pair `ij` can have
        a different threshold radius than pair `ji`.

        Parameters
        ----------
        atoms: optional
            the atoms for which neighbours are desired. Anything that can
            be sanitized by `sisl.Geometry` is a valid value.

            If not provided, neighbours for all atoms are searched.
        as_pairs: bool, optional
            If `True` pairs of atoms are returned. Otherwise a list containing
            the neighbours for each atom is returned.
        self_interaction: bool, optional
            whether to consider an atom a neighbour of itself.
        pbc: bool or array-like of shape (3, )
            whether periodic conditions should be considered.
            If a single bool is passed, all directions use that value.

        Returns
        ----------
        np.ndarray of shape (n_pairs, 5):
            Each pair `ij` means that `j` is a neighbour of `i`.
            The three extra columns are the supercell indices of atom `j`.
        """
        if not self._sphere_overlap and not isinstance(self._aux_R, Real):
            raise ValueError(
                "Unique atom pairs do not make sense if we are not looking for sphere overlaps."
                " Please setup the finder again setting `sphere_overlap` to `True` if you wish so."
            )

        # In the case where we tiled the geometry to do the binning, it is much better to
        # just find all neighbours and then drop duplicate connections. Otherwise it is a bit of a mess.
        if self._R_too_big:
            # Find all neighbours
            all_neighbours = self.find_neighbours(
                as_pairs=True, self_interaction=self_interaction, pbc=pbc
            )

            # Find out which of the pairs are uc connections
            is_uc_neigh = ~np.any(all_neighbours[:, 2:], axis=1)

            # Create an array with unit cell connections where duplicates are removed
            unique_uc = np.unique(np.sort(all_neighbours[is_uc_neigh][:, :2]), axis=0)
            uc_neighbours = np.zeros((len(unique_uc), 5), dtype=int)
            uc_neighbours[:, :2] = unique_uc

            # Concatenate the uc connections with the rest of the connections.
            return np.concatenate((uc_neighbours, all_neighbours[~is_uc_neigh]))

        # Cast R and pbc into arrays of appropiate shape and type.
        thresholds = np.full(self.geometry.na, self.R, dtype=np.float64)
        pbc = np.full(3, pbc, dtype=bool)

        # Get search indices
        search_indices, isc = self._get_search_indices(
            self.geometry.fxyz, cartesian=False
        )

        # Get atom counts
        at_counts = self._get_search_atom_counts(search_indices)

        max_pairs = at_counts.sum()
        if not self_interaction:
            max_pairs -= search_indices.shape[0]

        # Find all unique neighbour pairs
        neighbour_pairs, n_pairs = _neigh_operations.get_all_unique_pairs(
            search_indices,
            isc,
            self._heads,
            self._list,
            max_pairs,
            self_interaction,
            self.geometry.xyz,
            self.geometry.cell,
            pbc,
            thresholds,
            self._sphere_overlap,
        )

        return neighbour_pairs[:n_pairs]

    def find_close(
        self,
        xyz: Sequence,
        as_pairs: bool = False,
        pbc: Union[bool, Tuple[bool, bool, bool]] = (True, True, True),
    ):
        """Find all atoms that are close to some coordinates in space.

        This routine only executes the action of finding neighbours,
        the parameters of the search (`geometry`, `R`, `sphere_overlap`...)
        are defined when the finder is initialized or by calling `setup_finder`.

        Parameters
        ----------
        xyz: array-like of shape ([npoints], 3)
            the coordinates for which neighbours are desired.
        as_pairs: bool, optional
            If `True` pairs are returned, where the first item is the index
            of the point and the second one is the atom index.
            Otherwise a list containing the neighbours for each point is returned.
        pbc: bool or array-like of shape (3, )
            whether periodic conditions should be considered.
            If a single bool is passed, all directions use that value.

        Returns
        ----------
        np.ndarray or list:
            If `as_pairs=True`:
                A `np.ndarray` of shape (n_pairs, 5) is returned.
                Each pair `ij` means that `j` is a neighbour of `i`.
                The three extra columns are the supercell indices of atom `j`.
            If `as_pairs=False`:
                A list containing a numpy array of shape (n_neighs, 4) for each atom.
        """
        # Cast R and pbc into arrays of appropiate shape and type.
        thresholds = np.full(self._bins_geometry.na, self._aux_R, dtype=np.float64)
        pbc = np.full(3, pbc, dtype=bool)

        xyz = np.atleast_2d(xyz)
        # Get search indices
        search_indices, isc = self._get_search_indices(
            xyz.dot(self._bins_geometry.icell.T) % 1, cartesian=False
        )

        # Get atom counts
        at_counts = self._get_search_atom_counts(search_indices)

        max_pairs = at_counts.sum()

        # Find the neighbour pairs
        neighbour_pairs, split_ind = _neigh_operations.get_close(
            xyz,
            search_indices,
            isc,
            self._heads,
            self._list,
            max_pairs,
            self._bins_geometry.xyz,
            self._bins_geometry.cell,
            pbc,
            thresholds,
        )

        # Correct neighbour indices for the case where R was too big and
        # we needed to create an auxiliary supercell.
        if self._R_too_big:
            neighbour_pairs, split_ind = self._correct_pairs_R_too_big(
                neighbour_pairs, split_ind, pbc
            )

        if as_pairs:
            # Just return the neighbour pairs
            return neighbour_pairs[: split_ind[-1]]
        else:
            # Split to get the neighbours of each position
            return np.split(neighbour_pairs[:, 1:], split_ind, axis=0)[:-1]
