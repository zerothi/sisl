cimport cython
from libc.math cimport sqrt

import numpy as np
# This enables Cython enhanced compatibilities
cimport numpy as np
@cython.boundscheck(False)
@cython.wraparound(False)
def build_table(np.int64_t nbins, np.int64_t[:] bin_indices):
    """Builds the table where the location of all atoms is encoded
    
    Additionally, it also builds an array with the number of atoms at
    each bin.
    
    Parameters
    ----------
    nbins: int
        Total number of bins
    bin_indices: array of int
        An array containing the bin index of each atom.
        
    Returns
    ----------
    list: 
        contains the list of atoms, modified to encode all bin
        locations. Each item in the list contains the index of
        the next atom that we can find in the same bin. 
        If an item is -1, it means that there are no more atoms
        in the same bin.
    heads: 
        For each bin, the index of `list` where we can find the
        first atom that is contained in it.
    counts:
        For each bin, the number of atoms that it contains.
    """
    cdef:
        np.int64_t at, bin_index
    
        np.int64_t Nat = bin_indices.shape[0]
    
        # Initialize heads and counts arrays. We don't need to initialize the list array
        # since we are going to modify all its positions.
        np.int64_t[:] list_array = np.zeros(Nat, dtype=np.int64)
        np.int64_t[:] counts = np.zeros(nbins, dtype=np.int64)
        np.int64_t[:] heads = np.ones(nbins, dtype=np.int64) * -1  
    
    # Loop through all atoms
    for at in range(Nat):
        # Get the index of the bin where this atom is located.
        bin_index = bin_indices[at]
        
        # Replace the head of this bin by the current atom index.
        # Before replacing, store the previous head in this atoms'
        # position in the list.
        list_array[at] = heads[bin_index]
        heads[bin_index] = at

        # Update the count of this bin (increment it by 1).
        counts[bin_index] = counts[bin_index] + 1
        
    # Return the memory views as numpy arrays
    return np.asarray(list_array), np.asarray(heads), np.asarray(counts)

@cython.boundscheck(False)
@cython.wraparound(False)
def get_pairs(np.int64_t[:] at_indices, np.int64_t[:, :] indices, np.int64_t[:, :, :] iscs, 
    np.int64_t[:] heads, np.int64_t[:] list_array, np.int64_t max_npairs,
    bint self_interaction, np.float64_t[:, :] xyz, np.float64_t[:, :] cell, 
    np.npy_bool[:] pbc, np.float64_t[:] thresholds, bint sphere_overlap):
    """Gets (possibly duplicated) pairs of neighbour atoms.
    
    Parameters
    ---------
    at_indices:
        The indices of the atoms that we want to get potential
        neighbours for.
    indices:
        For each atom index (first dimension), the indices of the
        8 bins that contain potential neighbours.
    iscs:
        For each bin, the supercell index.
    heads:
        For each bin, the index of `list` where we can find the
        first atom that is contained in it.
        This array is constructed by `build_table`.
    list_array:
        contains the list of atoms, modified to encode all bin
        locations. Each item in the list contains the index of
        the next atom that we can find in the same bin. 
        If an item is -1, it means that there are no more 
        atoms in the same bin.
        This array is constructed by `build_table`.
    max_npairs:
        The number of maximum pairs that can be found.
        It is used to allocate the `neighs` array. This is computed
        in python with the help of the `counts` array constructed
        by `build_table`.
    self_interaction: bool, optional
        whether to consider an atom a neighbour of itself.
    xyz:
        the cartesian coordinates of all atoms in the geometry.
    cell:
        the lattice vectors of the geometry, where cell(i, :) is the
        ith lattice vector.
    pbc:
        for each direction, whether periodic boundary conditions should
        be considered.
    thresholds:
        the threshold radius for each atom in the geometry.
    sphere_overlap:
        If true, two atoms are considered neighbours if their spheres
        overlap.
        If false, two atoms are considered neighbours if the second atom
        is within the sphere of the first atom. Note that this implies that
        atom `i` might be atom `j`'s neighbour while the opposite is not true.
    
    Returns
    --------
    neighs: 
        contains the atom indices of all pairs of neighbor atoms.
    split_indices:
        array containing the breakpoints in `neighs`. These
        breakpoints delimit the position where one can find pairs
        for each 8 bin search. It can be used later to quickly
        split `neighs` on the multiple individual searches.
    """
    cdef:
        int N_ind = at_indices.shape[0]

        int i_pair = 0
        int search_index, at, j
        bint not_unit_cell, should_not_check
        np.int64_t neigh_at, bin_index
        np.float64_t threshold, dist
        
        np.float64_t[:] ref_xyz = np.zeros(3, dtype=np.float64)
        np.int64_t[:] neigh_isc = np.zeros(3, dtype=np.int64)
    
        np.int64_t[:,:] neighs = np.zeros((max_npairs, 5), dtype=np.int64)
        np.int64_t[:] split_indices = np.zeros(N_ind, dtype=np.int64)

    for search_index in range(N_ind):
        at = at_indices[search_index]

        for j in range(8):
            # Find the bin index.
            bin_index = indices[search_index, j]

            # Get the first atom index in this bin
            neigh_at = heads[bin_index]

            # If there are no atoms in this bin, do not even bother
            # checking supercell indices.
            if neigh_at == -1:
                continue

            # Find the supercell indices for this bin
            neigh_isc[:] = iscs[search_index, j, :]
            
            # And check if this bin corresponds to the unit cell
            not_unit_cell = neigh_isc[0] != 0 or neigh_isc[1] != 0 or neigh_isc[2] != 0

            ref_xyz[:] = xyz[at, :]
            if not_unit_cell:
                # If we are looking at a neighbouring cell in a direction
                # where there are no periodic boundary conditions, go to
                # next bin.
                should_not_check = False
                for i in range(3):
                    if not pbc[i] and neigh_isc[i] != 0:
                        should_not_check = True
                if should_not_check:
                    continue
                # Otherwise, move the atom to the neighbor cell. We do this 
                # instead of moving potential neighbors to the unit cell 
                # because in this way we reduce the number of operations.
                for i in range(3):
                    ref_xyz[i] = ref_xyz[i] - cell[0, i] * neigh_isc[0] - cell[1, i] * neigh_isc[1] - cell[2, i] * neigh_isc[2]

            # Loop through all atoms that are in this bin. 
            # If neigh_at == -1, this means no more atoms are in this bin.
            while neigh_at >= 0:
                # If this is a self interaction and the user didn't want them,
                # go to next atom.
                if not self_interaction and at == neigh_at:
                    neigh_at = list_array[neigh_at]
                    continue
                
                # Calculate the distance between the atom and the potential
                # neighbor.
                dist = 0.0
                for i in range(3):
                    dist += (xyz[neigh_at, i] - ref_xyz[i])**2
                dist = sqrt(dist)

                # Get the threshold for this pair of atoms
                threshold = thresholds[at]
                if sphere_overlap:
                    # If the user wants to check for sphere overlaps, we have
                    # to sum the radius of the neighbor to the threshold
                    threshold = threshold + thresholds[neigh_at]
                
                if dist < threshold:
                    # Store the pair of neighbours.
                    neighs[i_pair, 0] = at
                    neighs[i_pair, 1] = neigh_at
                    neighs[i_pair, 2] = neigh_isc[0]
                    neighs[i_pair, 3] = neigh_isc[1]
                    neighs[i_pair, 4] = neigh_isc[2]

                    # Increment the pair index
                    i_pair = i_pair + 1

                # Get the next atom in this bin. Sum 1 to get fortran index.
                neigh_at = list_array[neigh_at]

        # We have finished this search, store the breakpoint.
        split_indices[search_index] = i_pair

    return np.asarray(neighs), np.asarray(split_indices)

@cython.boundscheck(False)
@cython.wraparound(False)
def get_all_unique_pairs(np.int64_t[:, :] indices, np.int64_t[:, :, :] iscs, 
    np.int64_t[:] heads, np.int64_t[:] list_array, np.int64_t max_npairs,
    bint self_interaction, np.float64_t[:, :] xyz, np.float64_t[:, :] cell, 
    np.npy_bool[:] pbc, np.float64_t[:] thresholds, bint sphere_overlap):
    """Gets all unique pairs of atoms that are neighbours.
    
    Parameters
    ---------
    indices:
        For each atom index (first dimension), the indices of the
        8 bins that contain potential neighbours.
    iscs:
        For each bin, the supercell index.
    heads:
        For each bin, the index of `list` where we can find the
        first atom that is contained in it.
        This array is constructed by `build_table`.
    list_array:
        contains the list of atoms, modified to encode all bin
        locations. Each item in the list contains the index of
        the next atom that we can find in the same bin. 
        If an item is -1, it means that there are no more 
        atoms in the same bin.
        This array is constructed by `build_table`.
    max_npairs:
        The number of maximum pairs that can be found.
        It is used to allocate the `neighs` array. This is computed
        in python with the help of the `counts` array constructed
        by `build_table`.
    self_interaction: bool, optional
        whether to consider an atom a neighbour of itself.
    xyz:
        the cartesian coordinates of all atoms in the geometry.
    cell:
        the lattice vectors of the geometry, where cell(i, :) is the
        ith lattice vector.
    pbc:
        for each direction, whether periodic boundary conditions should
        be considered.
    thresholds:
        the threshold radius for each atom in the geometry.
    sphere_overlap:
        If true, two atoms are considered neighbours if their spheres
        overlap.
        If false, two atoms are considered neighbours if the second atom
        is within the sphere of the first atom. Note that this implies that
        atom `i` might be atom `j`'s neighbour while the opposite is not true.
    
    Returns
    --------
    neighs: 
        contains the atom indices of all unique neighbor pairs
        First column of atoms is sorted in increasing order. 
        Columns 3 to 5 contain the supercell indices of the neighbour
        atom (the one in column 2, column 1 atoms are always in the
        unit cell).
      npairs:
        The number of unique pairs that have been found, which
        will be less than `max_npairs`. This should presumably
        be used to slice the `neighs` array once in python.
    """
    cdef:
        int N_ats = list_array.shape[0]

        int i_pair = 0
        int at, j
        bint not_unit_cell, should_not_check
        np.int64_t neigh_at, bin_index
        np.float64_t threshold, dist
        
        np.float64_t[:] ref_xyz = np.zeros(3, dtype=np.float64)
        np.int64_t[:] neigh_isc = np.zeros(3, dtype=np.int64)
    
        np.int64_t[:,:] neighs = np.zeros((max_npairs, 5), dtype=np.int64)

    for at in range(N_ats):
        for j in range(8):
            # Find the bin index.
            bin_index = indices[at, j]

            # Get the first atom index in this bin
            neigh_at = heads[bin_index]

            # If there are no atoms in this bin, do not even bother
            # checking supercell indices.
            if neigh_at == -1:
                continue

            # Find the supercell indices for this bin
            neigh_isc[:] = iscs[at, j, :]
            
            # And check if this bin corresponds to the unit cell
            not_unit_cell = neigh_isc[0] != 0 or neigh_isc[1] != 0 or neigh_isc[2] != 0

            ref_xyz[:] = xyz[at, :]
            if not_unit_cell:
                # If we are looking at a neighbouring cell in a direction
                # where there are no periodic boundary conditions, go to
                # next bin.
                should_not_check = False
                for i in range(3):
                    if not pbc[i] and neigh_isc[i] != 0:
                        should_not_check = True
                if should_not_check:
                    continue
                # Otherwise, move the atom to the neighbor cell. We do this 
                # instead of moving potential neighbors to the unit cell 
                # because in this way we reduce the number of operations.
                for i in range(3):
                    ref_xyz[i] = ref_xyz[i] - cell[0, i] * neigh_isc[0] - cell[1, i] * neigh_isc[1] - cell[2, i] * neigh_isc[2]
                
            # Loop through all atoms that are in this bin. 
            # If neigh_at == -1, this means no more atoms are in this bin.
            while neigh_at >= 0:
                # If neigh_at is smaller than at, we already stored
                # this pair when performing the search for neigh_at.
                # The following atoms will have even lower indices
                # So we can just move to the next bin. However, if
                # we are checking a neighboring cell, this connection
                # will always be unique.
                if (not not_unit_cell and neigh_at <= at):
                    break

                # Calculate the distance between the atom and the potential
                # neighbor.
                dist = 0.0
                for i in range(3):
                    dist += (xyz[neigh_at, i] - ref_xyz[i])**2
                dist = sqrt(dist)

                # Get the threshold for this pair of atoms
                threshold = thresholds[at]
                if sphere_overlap:
                    # If the user wants to check for sphere overlaps, we have
                    # to sum the radius of the neighbor to the threshold
                    threshold = threshold + thresholds[neigh_at]
                
                if dist < threshold:
                    # Store the pair of neighbours.
                    neighs[i_pair, 0] = at
                    neighs[i_pair, 1] = neigh_at
                    neighs[i_pair, 2] = neigh_isc[0]
                    neighs[i_pair, 3] = neigh_isc[1]
                    neighs[i_pair, 4] = neigh_isc[2]

                    # Increment the pair index
                    i_pair = i_pair + 1

                # Get the next atom in this bin. Sum 1 to get fortran index.
                neigh_at = list_array[neigh_at]
            
            if j == 1 and self_interaction:
                # Add the self interaction
                neighs[i_pair, 0] = at
                neighs[i_pair, 1] = neigh_at
                neighs[i_pair, 2:] = 0
                
                # Increment the pair index
                i_pair = i_pair + 1

    # Return the array of neighbours, but only the filled part
    return np.asarray(neighs[:i_pair + 1]), i_pair

@cython.boundscheck(False)
@cython.wraparound(False)
def get_close(np.float64_t[:, :] search_xyz, np.int64_t[:, :] indices, np.int64_t[:, :, :] iscs, 
    np.int64_t[:] heads, np.int64_t[:] list_array, np.int64_t max_npairs,
    np.float64_t[:, :] xyz, np.float64_t[:, :] cell, 
    np.npy_bool[:] pbc, np.float64_t[:] thresholds):
    """Gets the atoms that are close to given positions
    
    Parameters
    ---------
    search_xyz:
        The coordinates for which we want to look for atoms that
        are close.
    indices:
        For each point (first dimension), the indices of the
        8 bins that contain potential neighbours.
    iscs:
        For each bin, the supercell index.
    heads:
        For each bin, the index of `list` where we can find the
        first atom that is contained in it.
        This array is constructed by `build_table`.
    list_array:
        contains the list of atoms, modified to encode all bin
        locations. Each item in the list contains the index of
        the next atom that we can find in the same bin. 
        If an item is -1, it means that there are no more 
        atoms in the same bin.
        This array is constructed by `build_table`.
    max_npairs:
        The number of maximum pairs that can be found.
        It is used to allocate the `neighs` array. This is computed
        in python with the help of the `counts` array constructed
        by `build_table`.
    self_interaction: bool, optional
        whether to consider an atom a neighbour of itself.
    xyz:
        the cartesian coordinates of all atoms in the geometry.
    cell:
        the lattice vectors of the geometry, where cell(i, :) is the
        ith lattice vector.
    pbc:
        for each direction, whether periodic boundary conditions should
        be considered.
    thresholds:
        the threshold radius for each atom in the geometry.
    sphere_overlap:
        If true, two atoms are considered neighbours if their spheres
        overlap.
        If false, two atoms are considered neighbours if the second atom
        is within the sphere of the first atom. Note that this implies that
        atom `i` might be atom `j`'s neighbour while the opposite is not true.
    
    Returns
    --------
    neighs: 
        contains the atom indices of all pairs of neighbor atoms.
    split_indices:
        array containing the breakpoints in `neighs`. These
        breakpoints delimit the position where one can find pairs
        for each 8 bin search. It can be used later to quickly
        split `neighs` on the multiple individual searches.
    """
    cdef:
        int N_ind = search_xyz.shape[0]

        int i_pair = 0
        int search_index, j
        bint not_unit_cell, should_not_check
        np.int64_t neigh_at, bin_index
        np.float64_t threshold, dist
        
        np.float64_t[:] ref_xyz = np.zeros(3, dtype=np.float64)
        np.int64_t[:] neigh_isc = np.zeros(3, dtype=np.int64)
    
        np.int64_t[:,:] neighs = np.zeros((max_npairs, 5), dtype=np.int64)
        np.int64_t[:] split_indices = np.zeros(N_ind, dtype=np.int64)

    for search_index in range(N_ind):
        for j in range(8):
            # Find the bin index.
            bin_index = indices[search_index, j]

            # Get the first atom index in this bin
            neigh_at = heads[bin_index]

            # If there are no atoms in this bin, do not even bother
            # checking supercell indices.
            if neigh_at == -1:
                continue

            # Find the supercell indices for this bin
            neigh_isc[:] = iscs[search_index, j, :]
            
            # And check if this bin corresponds to the unit cell
            not_unit_cell = neigh_isc[0] != 0 or neigh_isc[1] != 0 or neigh_isc[2] != 0

            ref_xyz[:] = search_xyz[search_index, :]
            if not_unit_cell:
                # If we are looking at a neighbouring cell in a direction
                # where there are no periodic boundary conditions, go to
                # next bin.
                should_not_check = False
                for i in range(3):
                    if not pbc[i] and neigh_isc[i] != 0:
                        should_not_check = True
                if should_not_check:
                    continue
                # Otherwise, move the atom to the neighbor cell. We do this 
                # instead of moving potential neighbors to the unit cell 
                # because in this way we reduce the number of operations.
                for i in range(3):
                    ref_xyz[i] = ref_xyz[i] - cell[0, i] * neigh_isc[0] - cell[1, i] * neigh_isc[1] - cell[2, i] * neigh_isc[2]

            # Loop through all atoms that are in this bin. 
            # If neigh_at == -1, this means no more atoms are in this bin.
            while neigh_at >= 0:
                # Calculate the distance between the atom and the potential
                # neighbor.
                dist = 0.0
                for i in range(3):
                    dist += (xyz[neigh_at, i] - ref_xyz[i])**2
                dist = sqrt(dist)

                # Get the threshold for the potential neighbour
                threshold = thresholds[neigh_at]
                
                if dist < threshold:
                    # Store the pair of neighbours.
                    neighs[i_pair, 0] = search_index
                    neighs[i_pair, 1] = neigh_at
                    neighs[i_pair, 2] = neigh_isc[0]
                    neighs[i_pair, 3] = neigh_isc[1]
                    neighs[i_pair, 4] = neigh_isc[2]

                    # Increment the pair index
                    i_pair = i_pair + 1

                # Get the next atom in this bin. Sum 1 to get fortran index.
                neigh_at = list_array[neigh_at]

        # We have finished this search, store the breakpoint.
        split_indices[search_index] = i_pair

    return np.asarray(neighs), np.asarray(split_indices)
