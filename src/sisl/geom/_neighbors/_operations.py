# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""File meant to be compiled with Cython so that operations are much faster."""

from __future__ import annotations

import cython
import cython.cimports.numpy as cnp
import numpy as np
from cython.cimports.libc.math import sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
def build_table(nbins: cnp.int64_t, bin_indices: cnp.int64_t[:]):
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
    n_atoms = bin_indices.shape[0]

    list_array_obj = np.zeros(n_atoms, dtype=np.int64)
    list_array: cnp.int64_t[:] = list_array_obj

    counts_obj = np.zeros(nbins, dtype=np.int64)
    counts: cnp.int64_t[:] = counts_obj

    heads_obj = np.full(nbins, -1, dtype=np.int64)
    heads: cnp.int64_t[:] = heads_obj

    # Loop through all atoms
    for at in range(n_atoms):
        # Get the index of the bin where this atom is located.
        bin_index = bin_indices[at]

        # Replace the head of this bin by the current atom index.
        # Before replacing, store the previous head in this atoms'
        # position in the list.
        list_array[at] = heads[bin_index]
        heads[bin_index] = at

        # Update the count of this bin (increment it by 1).
        counts[bin_index] += 1

    # Return the memory views as numpy arrays
    return list_array_obj, heads_obj, counts_obj


@cython.boundscheck(False)
@cython.wraparound(False)
def get_pairs(
    at_indices: cnp.int64_t[:],
    indices: cnp.int64_t[:, :],
    iscs: cnp.int64_t[:, :, :],
    heads: cnp.int64_t[:],
    list_array: cnp.int64_t[:],
    self_interaction: cython.bint,
    xyz: cnp.float64_t[:, :],
    cell: cnp.float64_t[:, :],
    pbc: cnp.npy_bool[:],
    thresholds: cnp.float64_t[:],
    overlap: cython.bint,
    init_npairs: cython.size_t,
    grow_factor: cnp.float64_t,
):
    r"""Gets (possibly duplicated) pairs of neighbor atoms.

    Parameters
    ---------
    at_indices:
        The indices of the atoms that we want to get potential
        neighbors for.
    indices:
        For each atom index (first dimension), the indices of the
        8 bins that contain potential neighbors.
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
    self_interaction: bool, optional
        whether to consider an atom a neighbor of itself.
    xyz:
        the cartesian coordinates of all atoms in the geometry.
    cell:
        the lattice vectors of the geometry, where ``cell[i, :]`` is the
        :math:`i`th lattice vector.
    pbc:
        for each direction, whether periodic boundary conditions should
        be considered.
    thresholds:
        the threshold radius for each atom in the geometry.
    overlap:
        If true, two atoms are considered neighbors if their spheres
        overlap.
        If false, two atoms are considered neighbors if the second atom
        is within the sphere of the first atom. Note that this implies that
        atom :math:`I` might be atom :math:`J`'s neighbor while the opposite is not true.
    init_npairs:
        The initial number of pairs that can be found.
        It is used to allocate the `neighs` array. This is computed
        in python with the help of the `counts` array constructed
        by `build_table`.
        This is not limiting the final size, but is used to pre-allocate
        a growing array.
    grow_factor:
        the grow factor of the size when the neighbor list needs
        to grow.

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
    N_ind = at_indices.shape[0]

    ref_xyz: cnp.float64_t[:] = np.empty(3, dtype=np.float64)
    neigh_isc: cnp.int64_t[:] = np.empty(3, dtype=np.int64)

    def grow():
        nonlocal neighs_obj, neighs, grow_factor

        n: cython.size_t = neighs.shape[0]
        new_neighs_obj = np.empty([int(n * grow_factor), 5], dtype=np.int64)
        new_neighs_obj[:n, :] = neighs_obj[:, :]
        neighs_obj = new_neighs_obj
        neighs = neighs_obj

    neighs_obj: cnp.ndarray = np.empty([init_npairs, 5], dtype=np.int64)
    neighs: cnp.int64_t[:, :] = neighs_obj

    split_indices_obj: cnp.ndarray = np.zeros(N_ind, dtype=np.int64)
    split_indices: cnp.int64_t[:] = split_indices_obj

    # Counter for filling neighs
    i_pair: cython.size_t = 0

    for search_index in range(N_ind):
        at = at_indices[search_index]

        for j in range(8):
            # Find the bin index.
            bin_index: cnp.int64_t = indices[search_index, j]

            # Get the first atom index in this bin
            neigh_at: cnp.int64_t = heads[bin_index]

            # If there are no atoms in this bin, do not even bother
            # checking supercell indices.
            if neigh_at == -1:
                continue

            # Find the supercell indices for this bin
            neigh_isc[:] = iscs[search_index, j, :]

            # And check if this bin corresponds to the unit cell
            not_unit_cell: cython.bint = (
                neigh_isc[0] != 0 or neigh_isc[1] != 0 or neigh_isc[2] != 0
            )

            ref_xyz[:] = xyz[at, :]
            if not_unit_cell:
                # If we are looking at a neighboring cell in a direction
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
                    ref_xyz[i] -= (
                        cell[0, i] * neigh_isc[0]
                        + cell[1, i] * neigh_isc[1]
                        + cell[2, i] * neigh_isc[2]
                    )

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
                dist: float = 0.0
                for i in range(3):
                    dist += (xyz[neigh_at, i] - ref_xyz[i]) ** 2
                dist = sqrt(dist)

                # Get the threshold for this pair of atoms
                threshold: float = thresholds[at]
                if overlap:
                    # If the user wants to check for sphere overlaps, we have
                    # to sum the radius of the neighbor to the threshold
                    threshold = threshold + thresholds[neigh_at]

                if dist < threshold:

                    if i_pair >= neighs.shape[0]:
                        grow()

                    # Store the pair of neighbors.
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

    # We copy to allow GC to remove the full array
    return neighs_obj[:i_pair, :].copy(), split_indices_obj


@cython.boundscheck(False)
@cython.wraparound(False)
def get_all_unique_pairs(
    indices: cnp.int64_t[:, :],
    iscs: cnp.int64_t[:, :, :],
    heads: cnp.int64_t[:],
    list_array: cnp.int64_t[:],
    self_interaction: cython.bint,
    xyz: cnp.float64_t[:, :],
    cell: cnp.float64_t[:, :],
    pbc: cnp.npy_bool[:],
    thresholds: cnp.float64_t[:],
    overlap: cython.bint,
    init_npairs: cython.size_t,
    grow_factor: cnp.float64_t,
):
    r"""Gets all unique pairs of atoms that are neighbors.

    Parameters
    ---------
    indices:
        For each atom index (first dimension), the indices of the
        8 bins that contain potential neighbors.
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
    self_interaction: bool, optional
        whether to consider an atom a neighbor of itself.
    xyz:
        the cartesian coordinates of all atoms in the geometry.
    cell:
        the lattice vectors of the geometry, where ``cell[i, :]`` is the
        ith lattice vector.
    pbc:
        for each direction, whether periodic boundary conditions should
        be considered.
    thresholds:
        the threshold radius for each atom in the geometry.
    overlap:
        If true, two atoms are considered neighbors if their spheres
        overlap.
        If false, two atoms are considered neighbors if the second atom
        is within the sphere of the first atom. Note that this implies that
        atom :math:`I` might be atom :math:`J`'s neighbor while the opposite is not true.
    init_npairs:
        The initial number of pairs that can be found.
        It is used to allocate the `neighs` array. This is computed
        in python with the help of the `counts` array constructed
        by `build_table`.
        This is not limiting the final size, but is used to pre-allocate
        a growing array.
    grow_factor:
        the grow factor of the size when the neighbor list needs
        to grow.

    Returns
    --------
    neighs:
        contains the atom indices of all unique neighbor pairs
        First column of atoms is sorted in increasing order.
        Columns 3 to 5 contain the supercell indices of the neighbor
        atom (the one in column 2, column 1 atoms are always in the
        unit cell).
    """

    N_ats = list_array.shape[0]

    ref_xyz: cnp.float64_t[:] = np.empty(3, dtype=np.float64)
    neigh_isc: cnp.int64_t[:] = np.empty(3, dtype=np.int64)

    def grow():
        nonlocal neighs_obj, neighs, grow_factor

        n: cython.size_t = neighs.shape[0]
        new_neighs_obj = np.empty([int(n * grow_factor), 5], dtype=np.int64)
        new_neighs_obj[:n, :] = neighs_obj[:, :]
        neighs_obj = new_neighs_obj
        neighs = neighs_obj

    neighs_obj: cnp.ndarray = np.empty([init_npairs, 5], dtype=np.int64)
    neighs: cnp.int64_t[:, :] = neighs_obj

    # Counter for filling neighs
    i_pair: cython.size_t = 0

    for at in range(N_ats):
        if self_interaction:
            if i_pair >= neighs.shape[0]:
                grow()

            # Add the self interaction
            neighs[i_pair, 0] = at
            neighs[i_pair, 1] = at
            neighs[i_pair, 2:] = 0

            # Increment the pair index
            i_pair += 1

        for j in range(8):
            # Find the bin index.
            bin_index: cnp.int64_t = indices[at, j]

            # Get the first atom index in this bin
            neigh_at: cnp.int64_t = heads[bin_index]

            # If there are no atoms in this bin, do not even bother
            # checking supercell indices.
            if neigh_at == -1:
                continue

            # Find the supercell indices for this bin
            neigh_isc[:] = iscs[at, j, :]

            # And check if this bin corresponds to the unit cell
            not_unit_cell: cython.bint = (
                neigh_isc[0] != 0 or neigh_isc[1] != 0 or neigh_isc[2] != 0
            )

            ref_xyz[:] = xyz[at, :]
            if not_unit_cell:
                # If we are looking at a neighboring cell in a direction
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
                    ref_xyz[i] -= (
                        cell[0, i] * neigh_isc[0]
                        + cell[1, i] * neigh_isc[1]
                        + cell[2, i] * neigh_isc[2]
                    )

            # Loop through all atoms that are in this bin.
            # If neigh_at == -1, this means no more atoms are in this bin.
            while neigh_at >= 0:
                # If neigh_at is smaller than at, we already stored
                # this pair when performing the search for neigh_at.
                # The following atoms will have even lower indices
                # So we can just move to the next bin.
                if neigh_at <= at:
                    break

                # Calculate the distance between the atom and the potential
                # neighbor.
                dist: float = 0.0
                for i in range(3):
                    dist += (xyz[neigh_at, i] - ref_xyz[i]) ** 2
                dist = sqrt(dist)

                # Get the threshold for this pair of atoms
                threshold: float = thresholds[at]
                if overlap:
                    # If the user wants to check for sphere overlaps, we have
                    # to sum the radius of the neighbor to the threshold
                    threshold = threshold + thresholds[neigh_at]

                if dist < threshold:
                    if i_pair >= neighs.shape[0]:
                        grow()

                    # Store the pair of neighbors.
                    neighs[i_pair, 0] = at
                    neighs[i_pair, 1] = neigh_at
                    neighs[i_pair, 2] = neigh_isc[0]
                    neighs[i_pair, 3] = neigh_isc[1]
                    neighs[i_pair, 4] = neigh_isc[2]

                    # Increment the pair index
                    i_pair += 1

                # Get the next atom in this bin. Sum 1 to get fortran index.
                neigh_at = list_array[neigh_at]

    # Return the array of neighbors, but only the filled part
    # We copy to remove the unneeded data-sizes
    return neighs_obj[:i_pair].copy()


@cython.boundscheck(False)
@cython.wraparound(False)
def get_close(
    search_xyz: cnp.float64_t[:, :],
    indices: cnp.int64_t[:, :],
    iscs: cnp.int64_t[:, :, :],
    heads: cnp.int64_t[:],
    list_array: cnp.int64_t[:],
    xyz: cnp.float64_t[:, :],
    cell: cnp.float64_t[:, :],
    pbc: cnp.npy_bool[:],
    thresholds: cnp.float64_t[:],
    init_npairs: cython.size_t,
    grow_factor: cnp.float64_t,
):
    r"""Gets the atoms that are close to given positions

    Parameters
    ---------
    search_xyz:
        The coordinates for which we want to look for atoms that
        are close.
    indices:
        For each point (first dimension), the indices of the
        8 bins that contain potential neighbors.
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
    xyz:
        the cartesian coordinates of all atoms in the geometry.
    cell:
        the lattice vectors of the geometry, where ``cell[i, :]`` is the
        ith lattice vector.
    pbc:
        for each direction, whether periodic boundary conditions should
        be considered.
    thresholds:
        the threshold radius for each atom in the geometry.
    init_npairs:
        The initial number of pairs that can be found.
        It is used to allocate the `neighs` array. This is computed
        in python with the help of the `counts` array constructed
        by `build_table`.
        This is not limiting the final size, but is used to pre-allocate
        a growing array.
    grow_factor:
        the grow factor of the size when the neighbor list needs
        to grow.

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

    N_ind = search_xyz.shape[0]

    ref_xyz: cnp.float64_t[:] = np.empty(3, dtype=np.float64)
    neigh_isc: cnp.int64_t[:] = np.empty(3, dtype=np.int64)

    def grow():
        nonlocal neighs_obj, neighs, grow_factor

        n: cython.size_t = neighs.shape[0]
        new_neighs_obj = np.empty([int(n * grow_factor), 5], dtype=np.int64)
        new_neighs_obj[:n, :] = neighs_obj[:, :]
        neighs_obj = new_neighs_obj
        neighs = neighs_obj

    neighs_obj: cnp.ndarray = np.empty([init_npairs, 5], dtype=np.int64)
    neighs: cnp.int64_t[:, :] = neighs_obj

    split_indices_obj: cnp.ndarray = np.zeros(N_ind, dtype=np.int64)
    split_indices: cnp.int64_t[:] = split_indices_obj

    i_pair: cython.size_t = 0

    for search_index in range(N_ind):
        for j in range(8):
            # Find the bin index.
            bin_index: cnp.int64_t = indices[search_index, j]

            # Get the first atom index in this bin
            neigh_at: cnp.int64_t = heads[bin_index]

            # If there are no atoms in this bin, do not even bother
            # checking supercell indices.
            if neigh_at == -1:
                continue

            # Find the supercell indices for this bin
            neigh_isc[:] = iscs[search_index, j, :]

            # And check if this bin corresponds to the unit cell
            not_unit_cell: cython.bint = (
                neigh_isc[0] != 0 or neigh_isc[1] != 0 or neigh_isc[2] != 0
            )

            ref_xyz[:] = search_xyz[search_index, :]
            if not_unit_cell:
                # If we are looking at a neighboring cell in a direction
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
                    ref_xyz[i] -= (
                        cell[0, i] * neigh_isc[0]
                        + cell[1, i] * neigh_isc[1]
                        + cell[2, i] * neigh_isc[2]
                    )

            # Loop through all atoms that are in this bin.
            # If neigh_at == -1, this means no more atoms are in this bin.
            while neigh_at >= 0:
                # Calculate the distance between the atom and the potential
                # neighbor.
                dist: float = 0.0
                for i in range(3):
                    dist += (xyz[neigh_at, i] - ref_xyz[i]) ** 2
                dist = sqrt(dist)

                # Get the threshold for the potential neighbor
                threshold: float = thresholds[neigh_at]

                if dist < threshold:

                    if i_pair >= neighs.shape[0]:
                        grow()

                    # Store the pair of neighbors.
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

    return neighs_obj[:i_pair, :].copy(), split_indices_obj
