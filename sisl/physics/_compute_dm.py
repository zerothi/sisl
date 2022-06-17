"""This file implements the cython functions that help building the DM efficiently."""


import cython

complex_or_float = cython.fused_type(cython.complex, cython.floating)

@cython.boundscheck(False)
@cython.wraparound(False)
def add_cnc_diag_spin(state: complex_or_float[:, :], row_orbs: cython.int[:], col_orbs_uc: cython.int[:], 
                    occs: cython.floating[:], DM_kpoint: complex_or_float[:], occtol: float = 1e-9):
    """Adds the cnc contributions of all orbital pairs to the DM given a array of states.
    
    To be used for the case of diagonal spins (unpolarized or polarized spin.)
    
    Parameters
    ----------
    state:
        The coefficients of all eigenstates for this contribution.
        Array of shape (n_eigenstates, n_basisorbitals)
    row_orbs:
        The orbital row indices of the sparsity pattern. 
        Shape (nnz, ), where nnz is the number of nonzero elements in the sparsity pattern.
    col_orbs_uc:
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
    ipair: cython.int

    # Loop lengths
    n_wfs: cython.int = state.shape[0]
    n_opairs: cython.int = row_orbs.shape[0]
        
    # Variable to store the occupation of each state 
    occ: float
    
    # Loop through all eigenstates
    for i in range(n_wfs):
        # Find the occupation for this eigenstate
        occ = occs[i]
        # If the occupation is lower than the tolerance, skip the state
        if occ < occtol:
            continue
        
        # The occupation is above the tolerance threshold, loop through all overlaping orbital pairs
        for ipair in range(n_opairs):
            # Get the orbital indices of this pair
            u = row_orbs[ipair]
            v = col_orbs_uc[ipair]
            # Add the contribution of this eigenstate to the DM_{u,v} element
            DM_kpoint[ipair] = DM_kpoint[ipair] + state[i, u] * occ * state[i, v].conjugate()
            
@cython.boundscheck(False)
@cython.wraparound(False)
def add_cnc_nc(state: cython.complex[:, :, :], row_orbs: cython.int[:], col_orbs_uc: cython.int[:], 
                    occs: cython.floating[:], DM_kpoint: cython.complex[:, :, :], occtol: float = 1e-9):
    """Adds the cnc contributions of all orbital pairs to the DM given a array of states.
    
    To be used for the case of non-diagonal spins (non-colinear or spin orbit).
    
    Parameters
    ----------
    state:
        The coefficients of all eigenstates for this contribution.
        Array of shape (n_eigenstates, n_basisorbitals, 2), where the last dimension is the spin
        "up"/"down" dimension.
    row_orbs:
        The orbital row indices of the sparsity pattern. 
        Shape (nnz, ), where nnz is the number of nonzero elements in the sparsity pattern.
    col_orbs_uc:
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
    ipair: cython.int
    # The spin box indices.
    Di: cython.int
    Dj: cython.int

    # Loop lengths
    n_wfs: cython.int = state.shape[0]
    n_opairs: cython.int = row_orbs.shape[0]
        
    # Variable to store the occupation of each state 
    occ: float
    
    # Loop through all eigenstates
    for i in range(n_wfs):
        # Find the occupation for this eigenstate
        occ = occs[i]
        # If the occupation is lower than the tolerance, skip the state
        if occ < occtol:
            continue
        
        # The occupation is above the tolerance threshold, loop through all overlaping orbital pairs
        for ipair in range(n_opairs):
            # Get the orbital indices of this pair
            u = row_orbs[ipair]
            v = col_orbs_uc[ipair]
            
            # Add to spin box
            for Di in range(2):
                for Dj in range(2):
                    DM_kpoint[ipair, Di, Dj] = DM_kpoint[ipair, Di, Dj] + state[i, u, Di] * occ * state[i, v, Dj].conjugate()