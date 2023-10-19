import numpy as np
from numbers import Real

from sisl.utils.mathematics import fnorm
from . import _fneighs
from . import cneighs


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
    def __init__(self, geometry, R=None, sphere_overlap=None, fortran=True):
        if fortran:
            self._fneighs = _fneighs
        else:
            self._fneighs = cneighs

        self.setup_finder(geometry, R=R, sphere_overlap=sphere_overlap)
        
    def setup_finder(self, geometry=None, R=None, sphere_overlap=None):
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
            R = geometry.atoms.maxR(all=True)
        
        # Set the radius
        self._R = R
        
        # If sphere overlap was not provided, set it to False if R is a single float
        # and True otherwise.
        if sphere_overlap is None:
            sphere_overlap = not R_is_float
        self._sphere_overlap = sphere_overlap

        # Determine the bin_size as the maximum DIAMETER to ensure that we ALWAYS
        # only need to look one bin away for neighbors.
        bin_size = np.max(self._R) * 2
        # Check that the bin size is positive.
        if bin_size <= 0:
            raise ValueError("All R values are 0 or less. Please provide some positive values")
        if sphere_overlap:
            # In the case when we want to check for sphere overlap, the size should
            # be twice as big.
            bin_size *= 2
        
        # We add a small amount to bin_size to avoid ambiguities when
        # a position is exactly at the center of a bin.
        bin_size += 0.01
        
        # Get the number of bins along each cell direction.
        nbins_float = fnorm(geometry.cell, axis=-1) / bin_size
        self.nbins = np.floor(nbins_float).astype(int)
        self.total_nbins = np.prod(self.nbins)

        if self.total_nbins == 0:
            # This means that nsc must be at least 5.

            # We round 1/nbins (i.e. the amount of cells needed in each direction)
            # to the closest next odd number.
            nsc = np.ceil(1/nbins_float) // 2 * 2 + 1
            # And then set it as the number of supercells.
            self.geometry.set_nsc(nsc.astype(int))
            raise ValueError(
                "The diameter occupies more space than the whole unit cell,"
                "which means that we need to look for neighbours more than one unit cell away."
                f"This is not yet supported by {self.__class__.__name__}"
            )
        
        # Get the scalar bin indices of all atoms
        scalar_bin_indices = self._get_bin_indices(self.geometry.fxyz)
        
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
        self._list, self._heads, self._counts = self._fneighs.build_table(self.total_nbins, bin_indices)
    
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
        search_indices[:, [2,3,6,7], 1] += signs[:, 1].reshape(-1, 1)
        search_indices[:, 4:, 2] += signs[:, 2].reshape(-1, 1)
        
        # Convert search indices to unit cell indices, but keep the supercell indices.
        isc, search_indices = np.divmod(search_indices, self.nbins)

        if not cartesian:
            search_indices = self._cartesian_to_scalar_index(search_indices)
        
        return search_indices, isc
    
    def _cartesian_to_scalar_index(self, index):
        """Converts cartesian indices to scalar indices"""
        if not np.issubdtype(index.dtype, int):
            raise ValueError("Decimal scalar indices do not make sense, please floor your cartesian indices.")
        return index.dot([1, self.nbins[0], self.nbins[0] * self.nbins[1]])
    
    def _scalar_to_cartesian_index(self, index):
        """Converts cartesian indices to scalar indices"""
        if np.min(index) < 0 or np.max(index) > self.total_nbins:
            raise ValueError("Some scalar indices are not within the unit cell. We cannot uniquely convert to cartesian")
        
        third, index = np.divmod(index, self.nbins[0]*self.nbins[1])
        second, first = np.divmod(index, self.nbins[0])
        return np.array([first, second, third]).T

    def find_neighbours(self, atoms=None, as_pairs=False, self_interaction=False, pbc=(True, True, True)):
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
        thresholds = np.full(self.geometry.na, self._R, dtype=np.float64)
        pbc = np.full(3, pbc, dtype=np.bool)
        
        # Get search indices
        search_indices, isc = self._get_search_indices(self.geometry.fxyz[atoms], cartesian=False)
        
        # Get atom counts
        at_counts = self._get_search_atom_counts(search_indices)

        max_pairs = at_counts.sum()
        if not self_interaction:
            max_pairs -= search_indices.shape[0]
        
        # Find the neighbour pairs
        neighbour_pairs, split_ind  = self._fneighs.get_pairs(
            atoms, search_indices, isc, self._heads, self._list, max_pairs, self_interaction,
            self.geometry.xyz, self.geometry.cell, pbc, thresholds, self._sphere_overlap
        )
        
        if as_pairs:
            # Just return the neighbour pairs
            return neighbour_pairs[:split_ind[-1]]
        else:
            # Split to get the neighbours of each atom
            return np.split(neighbour_pairs[:, 1:], split_ind, axis=0)[:-1]

    def find_all_unique_pairs(self, self_interaction=False, pbc=(True, True, True)):
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
        if not self._sphere_overlap and not isinstance(self._R, Real):
            raise ValueError("Unique atom pairs do not make sense if we are not looking for sphere overlaps."
                " Please setup the finder again setting `sphere_overlap` to `True` if you wish so.")
        
        # Cast R and pbc into arrays of appropiate shape and type.
        thresholds = np.full(self.geometry.na, self._R, dtype=np.float64)
        pbc = np.full(3, pbc, dtype=np.bool)

        # Get search indices
        search_indices, isc = self._get_search_indices(self.geometry.fxyz, cartesian=False)

        # Get atom counts
        at_counts = self._get_search_atom_counts(search_indices)

        max_pairs = at_counts.sum()
        if not self_interaction:
            max_pairs -= search_indices.shape[0]

        # Find the candidate pairs
        candidate_pairs, n_pairs = self._fneighs.get_all_unique_pairs(
            search_indices, isc, self._heads, self._list, max_pairs, self_interaction,
            self.geometry.xyz, self.geometry.cell, pbc, thresholds, self._sphere_overlap
        )
        
        return candidate_pairs[:n_pairs]

    def find_close(self, xyz, as_pairs=False, pbc=(True, True, True)):
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
        thresholds = np.full(self.geometry.na, self._R, dtype=np.float64)
        pbc = np.full(3, pbc, dtype=np.bool)
        
        xyz = np.atleast_2d(xyz)
        # Get search indices
        search_indices, isc = self._get_search_indices(xyz.dot(self.geometry.icell.T) % 1, cartesian=False)
        
        # Get atom counts
        at_counts = self._get_search_atom_counts(search_indices)

        max_pairs = at_counts.sum()
        
        # Find the neighbour pairs
        neighbour_pairs, split_ind  = self._fneighs.get_close(
            xyz, search_indices, isc, self._heads, self._list, max_pairs,
            self.geometry.xyz, self.geometry.cell, pbc, thresholds
        )
        
        if as_pairs:
            # Just return the neighbour pairs
            return neighbour_pairs[:split_ind[-1]]
        else:
            # Split to get the neighbours of each position
            return np.split(neighbour_pairs[:, 1:], split_ind, axis=0)[:-1]
        return

    def find_neighbours_old(self, atoms=None, as_pairs=False, self_interaction=False):
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
        
        Returns
        ----------
        np.ndarray or list:
            If `as_pairs=True`:
                A `np.ndarray` of shape (n_pairs, 2) is returned. Each pair `ij`
                means that `j` is a neighbour of `i`.
            If `as_pairs=False`:
                A list containing a numpy array of shape (n_neighs) for each atom.
        """
        # Sanitize atoms
        atoms = self.geometry._sanitize_atoms(atoms)
        
        # Get search indices
        search_indices, isc = self._get_search_indices(self.geometry.fxyz[atoms], cartesian=False)
        
        # Get atom counts
        at_counts = self._get_search_atom_counts(search_indices)

        total_pairs = at_counts.sum()
        if not self_interaction:
            total_pairs -= search_indices.shape[0]
        
        # Find the candidate pairs
        candidate_pairs, split_ind  = self._fneighs.get_pairs_old(atoms, search_indices, self._heads, self._list, total_pairs, self_interaction)
        
        if as_pairs:
            # Just returned the filtered pairs
            return self._filter_pairs(candidate_pairs)
        else:
            # Get the mask to filter
            mask = self._filter_pairs(candidate_pairs, return_mask=True)
            
            # Split both the neighbours and mask and then filter each array
            candidate_pairs[:, 0] = mask
            return [at_pairs[:, 1][at_pairs[:,0].astype(bool)] for at_pairs in np.split(candidate_pairs, split_ind[:-1])]
    
    def find_all_unique_pairs_old(self, self_interaction=False):
        """Find all unique neighbour pairs within the geometry.
        
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
        
        Returns
        ----------
        np.ndarray or list:
            If `as_pairs=True`:
                A `np.ndarray` of shape (n_pairs, 2) is returned. Each pair `ij`
                means that `j` is a neighbour of `i`.
            If `as_pairs=False`:
                A list containing a numpy array of shape (n_neighs) for each atom.
        """
        if not self._sphere_overlap and not isinstance(self._R, Real):
            raise ValueError("Unique atom pairs do not make sense if we are not looking for sphere overlaps."
                " Please setup the finder again setting `sphere_overlap` to `True` if you wish so.")

        # Get search indices
        search_indices, isc = self._get_search_indices(self.geometry.fxyz, cartesian=False)
        
        # Get atom counts
        at_counts = self._get_search_atom_counts(search_indices)

        max_pairs = at_counts.sum()
        if not self_interaction:
            max_pairs -= search_indices.shape[0]
        
        # Find the candidate pairs
        candidate_pairs, n_pairs = self._fneighs.get_all_unique_pairs_old(search_indices, self._heads, self._list, max_pairs, self_interaction)
        candidate_pairs = candidate_pairs[:n_pairs]
        
        # Filter them and return them.
        return self._filter_pairs(candidate_pairs)

    def _filter_pairs(self, candidate_pairs, return_mask=False):
        """Filters candidate neighbour pairs.
        
        It does so according to the parameters (`geometry`, `R`, `sphere_overlap`...)
        with which the finder was setup.

        Parameters
        ----------
        candidate_pairs: np.ndarray of shape (n_pairs, 2)
            The candidate atom pairs.
        return_mask: bool, optional
            Whether to return the filtering mask.
            If `True`, the filtering is not performed. This function will just
            return the mask.

        Returns
        ----------
        np.ndarray:
            If `return_mask=True`:
                The filtering mask, a boolean array of shape (n_pairs,).
            If `return_mask=False`:
                The filtered neighbour pairs. An integer array of shape (n_filtered, 2)
        """
        
        if isinstance(self._R, Real):
            thresh = self._R
            if self._sphere_overlap:
                thresh *= 2
        else:
            thresh = self._R[candidate_pairs[:, 0]] 
            if self._sphere_overlap:
                thresh += self._R[candidate_pairs[:, 1]]

        dists = fnorm(
            self.geometry.xyz[candidate_pairs[:, 0]] - self.geometry.xyz[candidate_pairs[:, 1]],
        axis=-1)
        
        mask = dists < thresh
        
        if return_mask:
            return mask
        return candidate_pairs[mask]