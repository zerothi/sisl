"""This file implements the cython functions that help building the DM efficiently."""


import cython

complex_or_float = cython.fused_type(cython.complex, cython.floating)

@cython.boundscheck(False)
@cython.wraparound(False)
def add_cnc_diag_spin(state: complex_or_float[:, :], DM_ptr: cython.int[:], DM_col_uc: cython.int[:],
                    occs: cython.floating[:], DM_kpoint: complex_or_float[:], occtol: float = 1e-9):
    """Adds the cnc contributions of all orbital pairs to the DM given a array of states.
    
    To be used for the case of diagonal spins (unpolarized or polarized spin.)
    
    Parameters
    ----------
    state:
        The coefficients of all eigenstates for this contribution.
        Array of shape (n_eigenstates, n_basisorbitals)
    DM_ptr:
        The pointer to row array of the sparse DM. 
        Shape (no + 1, ), where no is the number of orbitals in the unit cell.
    DM_col_uc:
        The orbital col indices of the sparsity pattern, but converted to the unit cell.
        Shape (nnz, ), where nnz is the number of nonzero elements in the sparsity pattern.
    occs:
        Array with the occupations for each eigenstate. Shape (n_eigenstates, )
    DM_kpoint:
        Array where contributions should be stored.
        Shape (nnz, ), where nnz is the number of nonzero elements in the sparsity pattern.
    occtol:
        Threshold below which the contribution of a state is not even added to the
        DM.
    """
    # The wavefunction (i) and orbital (u, v) indices
    i: cython.int
    u: cython.int
    v: cython.int

    # Number of orbitals in the unit cell
    no: cython.int = DM_ptr.shape[0] - 1
    ival: cython.int

    # Loop lengths
    n_wfs: cython.int = state.shape[0]
        
    # Variable to store the occupation of each state 
    occ: float
    
    # Loop through all eigenstates
    for i in range(n_wfs):
        # Find the occupation for this eigenstate
        occ = occs[i]
        # If the occupation is lower than the tolerance, skip the state
        if occ < occtol:
            continue
        
        # Loop over all non zero elements in the sparsity pattern
        for u in range(no):
            for ival in range(DM_ptr[u], DM_ptr[u+1]):
                v = DM_col_uc[ival]
                # Add the contribution of this eigenstate to the DM_{u,v} element
                DM_kpoint[ival] = DM_kpoint[ival] + state[i, u] * occ * state[i, v].conjugate()

            
@cython.boundscheck(False)
@cython.wraparound(False)
def add_cnc_nc(state: cython.complex[:, :, :], DM_ptr: cython.int[:], DM_col_uc: cython.int[:], 
                    occs: cython.floating[:], DM_kpoint: cython.complex[:, :, :], occtol: float = 1e-9):
    """Adds the cnc contributions of all orbital pairs to the DM given a array of states.
    
    To be used for the case of non-diagonal spins (non-colinear or spin orbit).
    
    Parameters
    ----------
    state:
        The coefficients of all eigenstates for this contribution.
        Array of shape (n_eigenstates, n_basisorbitals, 2), where the last dimension is the spin
        "up"/"down" dimension.
    DM_ptr:
        The pointer to row array of the sparse DM. 
        Shape (no + 1, ), where no is the number of orbitals in the unit cell.
    DM_col_uc:
        The orbital col indices of the sparsity pattern, but converted to the unit cell.
        Shape (nnz, ), where nnz is the number of nonzero elements in the sparsity pattern.
    occs:
        Array with the occupations for each eigenstate. Shape (n_eigenstates, )
    DM_kpoint:
        Array where contributions should be stored.
        Shape (nnz, 2, 2), where nnz is the number of nonzero elements in the sparsity pattern
        and the 2nd and 3rd dimensions account for the 2x2 spin box. 
    occtol:
        Threshold below which the contribution of a state is not even added to the
        DM.
    """
    # The wavefunction (i) and orbital (u, v) indices
    i: cython.int
    u: cython.int
    v: cython.int
    ival: cython.int

    # Number of orbitals in the unit cell
    no: cython.int = DM_ptr.shape[0] - 1

    # The spin box indices.
    Di: cython.int
    Dj: cython.int

    # Loop lengths
    n_wfs: cython.int = state.shape[0]
        
    # Variable to store the occupation of each state 
    occ: float
    
    # Loop through all eigenstates
    for i in range(n_wfs):
        # Find the occupation for this eigenstate
        occ = occs[i]
        # If the occupation is lower than the tolerance, skip the state
        if occ < occtol:
            continue

        # Loop over all non zero elements in the sparsity pattern
        for u in range(no):
            for ival in range(DM_ptr[u], DM_ptr[u+1]):
                v = DM_col_uc[ival]
            
                # Add to spin box
                for Di in range(2):
                    for Dj in range(2):
                        DM_kpoint[ival, Di, Dj] = DM_kpoint[ival, Di, Dj] + state[i, u, Di] * occ * state[i, v, Dj].conjugate()