import numpy as np
import tqdm
from typing import Callable, Optional

from sisl import BrillouinZone, DensityMatrix, get_distribution, unit_convert
from ._compute_dm import add_cnc_diag_spin, add_cnc_nc

def compute_dm(bz: BrillouinZone, occ_distribution: Optional[Callable] = None, 
            occtol: float = 1e-9, fermi_dirac_T: float = 300., eta: bool = True):
    """Computes the DM from the eigenstates of a Hamiltonian along a BZ.
    
    Parameters
    ----------
    bz: BrillouinZone
        The brillouin zone object containing the Hamiltonian of the system
        and the k-points to be sampled.
    occ_distribution: function, optional
        The distribution that will determine the occupations of states. It will
        receive an array of energies (in eV, referenced to fermi level) and it should
        return an array of floats.
        If not provided, a fermi_dirac distribution will be considered, being the 
        fermi_dirac_T parameter the electronic temperature.
    occtol: float, optional
        Threshold below which the contribution of a state is not even added to the
        DM.
    fermi_dirac_T: float, optional
        If an occupation distribution is not provided, a fermi-dirac distribution centered
        at the chemical potential is assumed. This argument controls the electronic temperature (in K).
    eta: bool, optional
        Whether a progress bar should be displayed or not.
    """
    # Get the hamiltonian
    H = bz.parent

    # Geometry
    geom = H.geometry

    # Sparsity pattern information
    row_orbs, col_orbs = H.nonzero()
    col_orbs_uc = H.osc2uc(col_orbs)
    col_isc = col_orbs // H.no
    sc_offsets = H.sc_off.dot(H.cell)

    # Initialize the density matrix using the sparsity pattern of the Hamiltonian.
    last_dim = H.dim 
    S = None
    if not H.orthogonal:
        last_dim -= 1
        S = H.tocsr(dim=last_dim)
    DM = DensityMatrix.fromsp(geom, [H.tocsr(dim=idim) for idim in range(last_dim)], S=S)
    # Keep a reference to its data array so that we can have
    # direct access to it (instead of through orbital indexing).
    vals = DM._csr.data
    # And set all values to 0
    if DM.orthogonal:
        vals[:, :] = 0
    else:
        # Don't touch the overlap values
        vals[:, :-1] = 0

    # For spin polarized calculations, we need to iterate over the two spin components.
    # If spin is unpolarized, we will multiply the contributions by 2.
    if DM.spin.is_polarized:
        spin_iterator = (0, 1)
        spin_factor = 1
    else:
        spin_iterator = (0,)
        spin_factor = 2
    
    # Set the distribution that will compute occupations (or more generally, weights)
    # for eigenstates. If not provided, use a fermi-dirac
    if occ_distribution is None:
        kT = unit_convert("K", "eV") * fermi_dirac_T
        occ_distribution = get_distribution("fermi_dirac", smearing=kT, x0=0)
        
    # Loop over spins
    for ispin in spin_iterator:
        # Create the eigenstates generator
        eigenstates = bz.apply.eigenstate(spin=ispin)
        
        # Zip it with the weights so that we can scale the contribution of each k point.
        k_it = zip(bz.weight, eigenstates)
        # Provide progress bar if requested
        if eta:
            k_it = tqdm.tqdm(k_it, total=len(bz.weight))

        # Now, loop through all k points
        for k_weight, k_eigs in k_it:
            # Get the k point for which this state has been calculated (in fractional coordinates)
            k = k_eigs.info['k']
            # Convert the k points to 1/Ang
            k = k.dot(geom.rcell)

            # Ensure R gauge so that we can use supercell phases. Much faster and less memory requirements
            # than using the r gauge, because we just have to compute the phase one time for each sc index.
            k_eigs.change_gauge("R")

            # Calculate all phases, this will be a (nnz, ) shaped array.
            sc_phases = np.exp(-1j * sc_offsets.dot(k))
            phases = sc_phases[col_isc]

            # Now find out the occupations for each wavefunction
            occs = k_eigs.occupation(occ_distribution)

            state = k_eigs.state

            if DM.spin.is_diagonal:
                # Calculate the matrix elements contributions for this k point.
                DM_kpoint = np.zeros(row_orbs.shape[0], dtype=k_eigs.state.dtype)
                add_cnc_diag_spin(state, row_orbs, col_orbs_uc, occs, DM_kpoint, occtol=occtol)

                # Apply phases
                DM_kpoint = DM_kpoint * phases

                # Take only the real part, weighting the contribution
                vals[:, ispin] += k_weight * DM_kpoint.real * spin_factor

            else:
                # Non colinear eigenstates contain an array of coefficients
                # of shape (n_wfs, no * 2), where n_wfs is also no * 2.
                # However, we only have "no" basis orbitals. The extra factor of 2 accounts for a hidden dimension
                # corresponding to spin "up"/"down". We reshape the array to uncover this extra dimension.
                state = state.reshape(-1, state.shape[1] // 2, 2)

                # Calculate the matrix elements contributions for this k point. For each matrix element
                # we allocate a 2x2 spin box.
                DM_kpoint = np.zeros((row_orbs.shape[0], 2, 2), dtype=np.complex128)
                add_cnc_nc(state, row_orbs, col_orbs_uc, occs, DM_kpoint, occtol=occtol)

                # Apply phases
                DM_kpoint *= phases.reshape(-1, 1, 1)

                # Now, each matrix element is a 2x2 spin box of complex numbers. That is, 4 complex numbers
                # i.e. 8 real numbers. What we do is to store these 8 real numbers separately in the DM.
                # However, in the non-colinear case (no spin orbit), since H is spin box hermitian we can force
                # the DM to also be spin-box hermitian. This means that DM[:, 0, 1] and DM[:, 1, 0] are complex 
                # conjugates and we can store only 4 numbers while keeping the same information. 
                # Here is how the spin-box can be reconstructed from the stored values:
                # D[j, i] = 
                # NON-COLINEAR
                # [[ D[j, i, 0],                D[j, i, 2] -i D[j, i, 3] ],
                #  [ D[j, i, 2] + i D[j, i, 3], D[j, i, 1]               ]]
                # SPIN-ORBIT
                # [[ D[j, i, 0],                D[j, i, 6] + i D[j, i, 7]],
                #  [ D[j, i, 2] -i D[j, i, 3],  D[j, i, 1]               ]]

                # Force DM spin-box to be hermitian in the non-colinear case.
                if DM.spin.is_noncolinear:
                    DM_kpoint[:, 1, 0] = 0.5 * (DM_kpoint[:, 1, 0] + DM_kpoint[:, 0, 1].conj())

                # Add each contribution to its location
                vals[:, 0] += DM_kpoint[:, 0, 0].real * k_weight
                vals[:, 1] += DM_kpoint[:, 1, 1].real * k_weight
                vals[:, 2] += DM_kpoint[:, 1, 0].real * k_weight
                vals[:, 3] -= DM_kpoint[:, 1, 0].imag * k_weight

                if DM.spin.is_spinorbit:
                    vals[:, 4] -= DM_kpoint[:, 0, 0].imag * k_weight
                    vals[:, 5] -= DM_kpoint[:, 1, 1].imag * k_weight
                    vals[:, 6] += DM_kpoint[:, 0, 1].real * k_weight
                    vals[:, 7] -= DM_kpoint[:, 0, 1].imag * k_weight
                
    return DM