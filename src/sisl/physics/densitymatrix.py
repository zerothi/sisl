# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from numbers import Integral
from typing import Literal, Optional

import numpy as np
from numpy import add, dot, logical_and, repeat, subtract, unique
from scipy.sparse import csr_matrix
from scipy.sparse import hstack as ss_hstack
from scipy.sparse import tril, triu

import sisl._array as _a
from sisl import BoundaryCondition as BC
from sisl import Geometry, Grid, Lattice
from sisl._core.sparse import SparseCSR, _ncol_to_indptr
from sisl._core.sparse_geometry import SparseOrbital
from sisl._indices import indices_fabs_le, indices_le
from sisl._internal import set_module
from sisl._math_small import xyz_to_spherical_cos_phi
from sisl.messages import deprecate_argument, progressbar, warn
from sisl.typing import GaugeType, KPoint

from .sparse import SparseOrbitalBZSpin, _get_spin
from .spin import Spin

__all__ = ["DensityMatrix"]


class _densitymatrix(SparseOrbitalBZSpin):

    def mulliken(self, projection: Literal["orbital", "atom"] = "orbital"):
        r""" Calculate Mulliken charges from the density matrix

        See :ref:`here <math_convention>` for details on the mathematical notation.
        Matrices :math:`\boldsymbol\rho` and :math:`\mathbf S` are density
        and overlap matrices, respectively.

        For polarized calculations the Mulliken charges are calculated as
        (for each spin-channel)

        .. math::

             \mathbf m_i &= \sum_{i} [\boldsymbol\rho_{ij} \mathbf S_{ij}]
             \\
             \mathbf m_I &= \sum_{i\in I} \mathbf m_i

        For non-colinear calculations (including spin-orbit) they are calculated
        as above but using the spin-box per orbital (:math:`\sigma` is spin)

        .. math::
             \mathbf m_i &= \sum_\sigma\sum_j [\boldsymbol\rho_{ij,\sigma\sigma} \mathbf S_{ij,\sigma\sigma}]
             \\
             \mathbf m^{S^x}_i &= \sum_j \Re [\boldsymbol\rho_{ij,\uparrow\downarrow} \mathbf S_{ij,\uparrow\downarrow}] +
                          \Re [\boldsymbol\rho_{ij,\downarrow\uparrow} \mathbf S_{ij,\downarrow\uparrow}]
             \\
             \mathbf m^{S^y}_i &= \sum_j \Im [\boldsymbol\rho_{ij,\uparrow\downarrow} \mathbf S_{ij,\uparrow\downarrow}] -
                          \Im [\boldsymbol\rho_{ij,\downarrow\uparrow} \mathbf S_{ij,\downarrow\uparrow}]
             \\
             \mathbf m^{S^z}_i &= \sum_{j} \Re [\boldsymbol\rho_{ij,\uparrow\uparrow} \mathbf S_{ij,\uparrow\uparrow}] -
                          \Re [\boldsymbol\rho_{ij,\downarrow\downarrow} \mathbf S_{ij,\downarrow\downarrow}]

        Parameters
        ----------
        projection :
            how the Mulliken charges are returned.
            Can be atom-resolved, orbital-resolved or the
            charge matrix (off-diagonal elements)

        Returns
        -------
        numpy.ndarray
            Contains Mulliken charges for each orbital/atom (depending on
            `projection`).
            The first dimension holds the spin-components, total charge for un-polarized,
            (T, Sz) for polarized. (T, Sx, Sy, Sz) for all non-colinear cases.
        """
        # Mulliken population, diagonal for orthogonal, otherwise multiply
        # with overlap matrix
        if self.orthogonal:
            D = np.stack(
                [self._csr.tocsr(i).diagonal() for i in range(self.shape[2])], axis=1
            )
        else:
            D = self._csr.copy(range(self.shape[2] - 1))
            D._D *= self._csr._D[:, -1].reshape(-1, 1)
            D = np.sum(D, axis=1)

        spin = self.spin
        # The trace will return a complex (if complex)
        Q = _get_spin(D, spin, what="trace").reshape(-1, 1).real
        S = None

        def convert(D):
            return D

        if spin.is_unpolarized:
            D = Q

            def convert(D):
                return D[0]

        elif spin.is_polarized:
            S = _get_spin(D, spin, what="vector")
            D = np.concatenate([Q, S[..., [2]]], axis=1)
        else:
            S = _get_spin(D, spin, what="vector")
            D = np.concatenate([Q, S], axis=1)
        del Q, S

        projection = projection.lower()
        if projection.startswith("orbital"):  # allows orbitals
            return convert(D.T)

        elif projection.startswith("atom"):  # allows atoms
            # Now perform summation per atom
            geom = self.geometry
            M = np.zeros([geom.na, D.shape[1]], dtype=D.dtype)
            np.add.at(M, geom.o2a(np.arange(geom.no)), D)
            del D

            return convert(M.T)

        raise ValueError(
            f"{self.__class__.__name__}.mulliken only allows projection [orbital, atom]"
        )

    @deprecate_argument(
        "tol",
        "atol",
        "argument tol has been deprecated in favor of atol, please update your code.",
        "0.15",
        "0.17",
    )
    def density(
        self,
        grid: Grid,
        spinor=None,
        atol: float = 1e-7,
        eta: Optional[bool] = False,
        method: Literal["pre-compute", "direct"] = "pre-compute",
        **kwargs,
    ):
        r"""Expand the density matrix to the charge density on a grid

        This routine calculates the real-space density components on a specified grid.

        This is an *in-place* operation that *adds* to the current values in the grid.

        Note: To calculate :math:`\boldsymbol\rho(\mathbf r)` in a unit-cell different from the
        originating geometry, simply pass a grid with a unit-cell different than the originating
        supercell.

        The real-space density is calculated as:

        .. math::
            \boldsymbol\rho(\mathbf r) = \sum_{ij}\phi_i(\mathbf r)\phi_j(\mathbf r) \mathbf D_{ij}

        While for non-collinear/spin-orbit calculations the density is determined from the
        spinor component (`spinor`) by

        .. math::
           \boldsymbol\rho_{\boldsymbol\sigma}(\mathbf r) = \sum_{ij}\phi_i(\mathbf r)\phi_j(\mathbf r) \sum_\alpha [\boldsymbol\sigma \boldsymbol \rho_{ij}]_{\alpha\alpha}

        Here :math:`\boldsymbol\sigma` corresponds to a spinor operator to extract relevant quantities. By passing the identity matrix the total charge is added. By using the Pauli matrix :math:`\boldsymbol\sigma_x`
        only the :math:`x` component of the density is added to the grid (see `Spin.X`).

        Parameters
        ----------
        grid :
           the grid on which to add the density (the density is in ``e/Ang^3``)
        spinor : (2,) or (2, 2), optional
           the spinor matrix to obtain the diagonal components of the density. For un-polarized density matrices
           this keyword has no influence. For spin-polarized it *has* to be either 1 integer or a vector of
           length 2 (defaults to total density).
           For non-collinear/spin-orbit density matrices it has to be a 2x2 matrix (defaults to total density).
        atol :
           DM tolerance for accepted values. For all density matrix elements with absolute values below
           the tolerance, they will be treated as strictly zeros.
        eta :
           show a progressbar on stdout
        method:
           It determines if the orbital values are computed on the fly (direct) or they are all pre-computed
           on the grid at the beginning (pre-compute).
           Pre computing orbitals results in a faster computation, but it requires more memory.

        Notes
        -----

        The `method` argument may change at will since this is an experimental feature.
        """
        # Translate the density matrix to have all the unit cell atoms actually inside
        # the unit cell, since this will facilitate things greatly and it gives the
        # same result.
        uc_dm = self.translate2uc()

        if method == "pre-compute":
            # Compute orbital values on the grid
            psi_values = uc_dm.geometry._orbital_values(grid.shape)

            # Here we just set the nsc to whatever the psi values have.
            # If the nsc is bigger in the DM, then some elements of the DM will be discarded.
            # If the nsc is smaller in the DM, then the DM is just "padded" with 0s.
            if not np.all(uc_dm.nsc == psi_values.geometry.nsc):
                uc_dm.set_nsc(psi_values.geometry.nsc)

        # Get the DM components with which we want to compute the density
        csr = uc_dm._csr
        if self.spin.kind > Spin.POLARIZED:
            if spinor is None:
                # Default to the total density
                spinor = np.identity(2, dtype=np.complex128)
            else:
                spinor = _a.arrayz(spinor)
            if spinor.size != 4 or spinor.ndim != 2:
                raise ValueError(
                    f"{self.__class__.__name__}.density with NC/SO spin, requires a 2x2 matrix."
                )

            DM = _a.emptyz([self.nnz, 2, 2])
            idx = _a.array_arange(csr.ptr[:-1], n=csr.ncol)
            if self.spin.is_noncolinear:
                # non-collinear
                DM[:, 0, 0] = csr._D[idx, 0]
                DM[:, 0, 1] = csr._D[idx, 2] + 1j * csr._D[idx, 3]
                DM[:, 1, 0] = np.conj(DM[:, 0, 1])
                DM[:, 1, 1] = csr._D[idx, 1]
            else:
                # spin-orbit
                DM[:, 0, 0] = csr._D[idx, 0] + 1j * csr._D[idx, 4]
                DM[:, 0, 1] = csr._D[idx, 2] + 1j * csr._D[idx, 3]
                DM[:, 1, 0] = csr._D[idx, 6] + 1j * csr._D[idx, 7]
                DM[:, 1, 1] = csr._D[idx, 1] + 1j * csr._D[idx, 5]

            # Perform dot-product with spinor, and take out the diagonal real part
            DM = dot(DM, spinor.T)[:, [0, 1], [0, 1]].sum(1).real

            # Create the DM csr matrix.
            csrDM = csr_matrix(
                (DM, csr.col[idx], _ncol_to_indptr(csr.ncol)),
                shape=(uc_dm.shape[:2]),
                dtype=DM.dtype,
            )

            del idx, DM

        elif self.spin.kind == Spin.POLARIZED:
            if spinor is None:
                spinor = _a.onesd(2)

            elif isinstance(spinor, Integral):
                # extract the provided spin-polarization
                s = _a.zerosd(2)
                s[spinor] = 1.0
                spinor = s
            else:
                spinor = _a.arrayd(spinor)

            if spinor.size != 2 or spinor.ndim != 1:
                raise ValueError(
                    f"{self.__class__.__name__}.density with polarized spin, requires spinor "
                    "argument as an integer, or a vector of length 2"
                )

            csrDM = csr.tocsr(dim=0) * spinor[0] + csr.tocsr(dim=1) * spinor[1]

        elif self.spin.is_nambu:
            raise NotImplementedError("Nambu spin configuration not implemneted")
        else:
            csrDM = csr.tocsr(dim=0)

        if method == "pre-compute":
            try:
                psi_values.reduce_orbital_products(
                    csrDM, uc_dm.lattice, out=grid.grid, **kwargs
                )
            except MemoryError:
                raise MemoryError(
                    "Ran out of memory while computing the density with the 'pre-compute'"
                    " method. Try using method='direct', which is slower but requires much"
                    " less memory."
                )
        elif method == "direct":
            self._density_direct(grid, csrDM, atol=atol, eta=eta)

    def _density_direct(
        self, grid: Grid, csrDM, atol: float = 1e-7, eta: Optional[bool] = None
    ):
        r"""Compute the density by calculating the orbital values on the fly.

        Parameters
        ----------
        grid : Grid
           the grid on which to add the density (the density is in ``e/Ang^3``)
        spinor : (2,) or (2, 2), optional
           the spinor matrix to obtain the diagonal components of the density. For un-polarized density matrices
           this keyword has no influence. For spin-polarized it *has* to be either 1 integer or a vector of
           length 2 (defaults to total density).
           For non-collinear/spin-orbit density matrices it has to be a 2x2 matrix (defaults to total density).
        atol : float, optional
           DM tolerance for accepted values. For all density matrix elements with absolute values below
           the tolerance, they will be treated as strictly zeros.
        eta : bool, optional
           show a progressbar on stdout
        """
        geometry = self.geometry

        # Extract sub variables used throughout the loop
        shape = _a.asarrayi(grid.shape)
        dcell = grid.dcell

        # In the following we don't care about division
        # So 1) save error state, 2) turn off divide by 0, 3) calculate, 4) turn on old error state
        old_err = np.seterr(divide="ignore", invalid="ignore")

        # To heavily speed up the construction of the density we can recreate
        # the sparse csrDM matrix by summing the lower and upper triangular part.
        # This means we only traverse the sparse UPPER part of the DM matrix
        # I.e.:
        #    psi_i * DM_{ij} * psi_j + psi_j * DM_{ji} * psi_i
        # is equal to:
        #    psi_i * (DM_{ij} + DM_{ji}) * psi_j
        # Secondly, to ease the loops we extract the main diagonal (on-site terms)
        # and store this for separate usage
        csr_sum = [None] * geometry.n_s
        no = geometry.no
        primary_i_s = geometry.sc_index([0, 0, 0])
        for i_s in range(geometry.n_s):
            # Extract the csr matrix
            o_start, o_end = i_s * no, (i_s + 1) * no
            csr = csrDM[:, o_start:o_end]
            if i_s == primary_i_s:
                csr_sum[i_s] = triu(csr) + tril(csr, -1).transpose()
            else:
                csr_sum[i_s] = csr

        # Recreate the column-stacked csr matrix
        csrDM = ss_hstack(csr_sum, format="csr")
        del csr, csr_sum

        # Remove all zero elements (note we use the tolerance here!)
        csrDM.data = np.where(np.fabs(csrDM.data) > atol, csrDM.data, 0.0)

        # Eliminate zeros and sort indices etc.
        csrDM.eliminate_zeros()
        csrDM.sort_indices()
        csrDM.prune()

        # 1. Ensure the grid has a geometry associated with it
        lattice = grid.lattice.copy()
        # Find the periodic directions
        pbc = [
            bc == BC.PERIODIC or geometry.nsc[i] > 1
            for i, bc in enumerate(grid.lattice.boundary_condition[:, 0])
        ]
        if grid.geometry is None:
            # Create the actual geometry that encompass the grid
            ia, xyz, _ = geometry.within_inf(lattice, periodic=pbc)
            if len(ia) > 0:
                grid.set_geometry(Geometry(xyz, geometry.atoms[ia], lattice=lattice))

        # Instead of looping all atoms in the supercell we find the exact atoms
        # and their supercell indices.
        add_R = _a.fulld(3, geometry.maxR())
        # Calculate the required additional vectors required to increase the fictitious
        # supercell by add_R in each direction.
        # For extremely skewed lattices this will be way too much, hence we make
        # them square.
        o = lattice.to.Cuboid(orthogonal=True)
        lattice = Lattice(o._v + np.diag(2 * add_R), origin=o.origin - add_R)

        # Retrieve all atoms within the grid supercell
        # (and the neighbors that connect into the cell)
        IA, XYZ, ISC = geometry.within_inf(lattice, periodic=pbc)
        XYZ -= grid.lattice.origin.reshape(1, 3)

        # Retrieve progressbar
        eta = progressbar(len(IA), f"{self.__class__.__name__}.density", "atom", eta)

        cell = geometry.cell
        atoms = geometry.atoms
        axyz = geometry.axyz
        a2o = geometry.a2o

        def xyz2spherical(xyz, offset):
            """Calculate the spherical coordinates from indices"""
            rx = xyz[:, 0] - offset[0]
            ry = xyz[:, 1] - offset[1]
            rz = xyz[:, 2] - offset[2]

            # Calculate radius ** 2
            xyz_to_spherical_cos_phi(rx, ry, rz)
            return rx, ry, rz

        def xyz2sphericalR(xyz, offset, R):
            """Calculate the spherical coordinates from indices"""
            rx = xyz[:, 0] - offset[0]
            idx = indices_fabs_le(rx, R)
            ry = xyz[idx, 1] - offset[1]
            ix = indices_fabs_le(ry, R)
            ry = ry[ix]
            idx = idx[ix]
            rz = xyz[idx, 2] - offset[2]
            ix = indices_fabs_le(rz, R)
            ry = ry[ix]
            rz = rz[ix]
            idx = idx[ix]
            if len(idx) == 0:
                return [], [], [], []
            rx = rx[idx]

            # Calculate radius ** 2
            ix = indices_le(rx**2 + ry**2 + rz**2, R**2)
            idx = idx[ix]
            if len(idx) == 0:
                return [], [], [], []
            rx = rx[ix]
            ry = ry[ix]
            rz = rz[ix]
            xyz_to_spherical_cos_phi(rx, ry, rz)
            return idx, rx, ry, rz

        # Looping atoms in the sparse pattern is better since we can pre-calculate
        # the radial parts and then add them.
        # First create a SparseOrbital matrix, then convert to SparseAtom
        spO = SparseOrbital(geometry, dtype=np.int16)
        spO._csr = SparseCSR(csrDM)
        spA = spO.toSparseAtom(dtype=np.int16)
        del spO
        na = geometry.na
        # Remove the diagonal part of the sparse atom matrix
        off = na * primary_i_s
        for ia in range(na):
            del spA[ia, off + ia]

        # Get pointers and delete the atomic sparse pattern
        # The below complexity is because we are not finalizing spA
        csr = spA._csr
        a_ptr = _ncol_to_indptr(csr.ncol)
        a_col = csr.col[_a.array_arange(csr.ptr, n=csr.ncol)]
        del spA, csr

        # Get offset in supercell in orbitals
        off = geometry.no * primary_i_s
        origin = grid.origin
        # TODO sum the non-origin atoms to the csrDM matrix
        #      this would further decrease the loops required.

        # Loop over all atoms in the grid-cell
        for ia, ia_xyz, isc in zip(IA, XYZ, ISC):
            # Get current atom
            ia_atom = atoms[ia]
            IO = a2o(ia)
            IO_range = range(ia_atom.no)
            cell_offset = (cell * isc.reshape(3, 1)).sum(0) - origin

            # Extract maximum R
            R = ia_atom.maxR()
            if R <= 0.0:
                warn(f"Atom '{ia_atom}' does not have a wave-function, skipping atom.")
                eta.update()
                continue

            # Retrieve indices of the grid for the atomic shape
            idx = grid.index(ia_atom.to.Sphere(center=ia_xyz))

            # Now we have the indices for the largest orbital on the atom

            # Subsequently we have to loop the orbitals and the
            # connecting orbitals
            # Then we find the indices that overlap with these indices
            # First reduce indices to inside the grid-cell
            idx[idx[:, 0] < 0, 0] = 0
            idx[shape[0] <= idx[:, 0], 0] = shape[0] - 1
            idx[idx[:, 1] < 0, 1] = 0
            idx[shape[1] <= idx[:, 1], 1] = shape[1] - 1
            idx[idx[:, 2] < 0, 2] = 0
            idx[shape[2] <= idx[:, 2], 2] = shape[2] - 1

            idx = unique(idx, axis=0)
            if len(idx) == 0:
                eta.update()
                continue

            # Get real-space coordinates for the current atom
            # as well as the radial parts
            grid_xyz = dot(idx, dcell)

            # Perform loop on connection atoms
            # Allocate the DM_pj arrays
            # This will have a size equal to number of elements times number of
            # orbitals on this atom
            # In this way we do not have to calculate the psi_j multiple times
            DM_io = csrDM[IO : IO + ia_atom.no, :].tolil()
            DM_pj = _a.zerosd([ia_atom.no, grid_xyz.shape[0]])

            # Now we perform the loop on the connections for this atom
            # Remark that we have removed the diagonal atom (it-self)
            # As that will be calculated in the end
            for ja in a_col[a_ptr[ia] : a_ptr[ia + 1]]:
                # Retrieve atom (which contains the orbitals)
                ja_atom = atoms[ja % na]
                JO = a2o(ja)
                jR = ja_atom.maxR()
                # Get actual coordinate of the atom
                ja_xyz = axyz(ja) + cell_offset

                # Reduce the ia'th grid points to those that connects to the ja'th atom
                ja_idx, ja_r, ja_theta, ja_cos_phi = xyz2sphericalR(
                    grid_xyz, ja_xyz, jR
                )

                if len(ja_idx) == 0:
                    # Quick step
                    continue

                # Loop on orbitals on this atom
                for jo in range(ja_atom.no):
                    o = ja_atom.orbitals[jo]
                    oR = o.R

                    # Downsize to the correct indices
                    if jR - oR < 1e-6:
                        ja_idx1 = ja_idx
                        ja_r1 = ja_r
                        ja_theta1 = ja_theta
                        ja_cos_phi1 = ja_cos_phi
                    else:
                        ja_idx1 = indices_le(ja_r, oR)
                        if len(ja_idx1) == 0:
                            # Quick step
                            continue

                        # Reduce arrays
                        ja_r1 = ja_r[ja_idx1]
                        ja_theta1 = ja_theta[ja_idx1]
                        ja_cos_phi1 = ja_cos_phi[ja_idx1]
                        ja_idx1 = ja_idx[ja_idx1]

                    # Calculate the psi_j component
                    psi = o.psi_spher(ja_r1, ja_theta1, ja_cos_phi1, cos_phi=True)

                    # Now add this orbital to all components
                    for io in IO_range:
                        DM_pj[io, ja_idx1] += DM_io[io, JO + jo] * psi

                # Temporary clean up
                del ja_idx, ja_r, ja_theta, ja_cos_phi
                del ja_idx1, ja_r1, ja_theta1, ja_cos_phi1, psi

            # Now we have all components for all orbitals connection to all orbitals on atom
            # ia. We simply need to add the diagonal components

            # Loop on the orbitals on this atom
            ia_r, ia_theta, ia_cos_phi = xyz2spherical(grid_xyz, ia_xyz)
            del grid_xyz
            for io in IO_range:
                # Only loop halve the range.
                # This is because: triu + tril(-1).transpose()
                # removes the lower half of the on-site matrix.
                for jo in range(io + 1, ia_atom.no):
                    DM = DM_io[io, off + IO + jo]

                    oj = ia_atom.orbitals[jo]
                    ojR = oj.R

                    # Downsize to the correct indices
                    if R - ojR < 1e-6:
                        ja_idx1 = slice(None)
                        ja_r1 = ia_r
                        ja_theta1 = ia_theta
                        ja_cos_phi1 = ia_cos_phi
                    else:
                        ja_idx1 = indices_le(ia_r, ojR)
                        if len(ja_idx1) == 0:
                            # Quick step
                            continue

                        # Reduce arrays
                        ja_r1 = ia_r[ja_idx1]
                        ja_theta1 = ia_theta[ja_idx1]
                        ja_cos_phi1 = ia_cos_phi[ja_idx1]

                    # Calculate the psi_j component
                    DM_pj[io, ja_idx1] += DM * oj.psi_spher(
                        ja_r1, ja_theta1, ja_cos_phi1, cos_phi=True
                    )

                # Calculate the psi_i component
                # Note that this one *also* zeroes points outside the shell
                # I.e. this step is important because it "nullifies" all but points where
                # orbital io is defined.
                psi = ia_atom.orbitals[io].psi_spher(
                    ia_r, ia_theta, ia_cos_phi, cos_phi=True
                )
                DM_pj[io, :] += DM_io[io, off + IO + io] * psi
                DM_pj[io, :] *= psi

            # Temporary clean up
            ja_idx1 = ja_r1 = ja_theta1 = ja_cos_phi1 = None
            del ia_r, ia_theta, ia_cos_phi, psi, DM_io

            # Now add the density
            grid.grid[idx[:, 0], idx[:, 1], idx[:, 2]] += DM_pj.sum(0)

            # Clean-up
            del DM_pj, idx

            eta.update()
        eta.close()

        # Reset the error code for division
        np.seterr(**old_err)


@set_module("sisl.physics")
class DensityMatrix(_densitymatrix):
    """Sparse density matrix object

    Assigning or changing elements is as easy as with standard `numpy` assignments:

    >>> DM = DensityMatrix(...)
    >>> DM.D[1,2] = 0.1

    which assigns 0.1 as the density element between orbital 2 and 3.
    (remember that Python is 0-based elements).

    For spin matrices the elements are defined with an extra dimension.

    For a polarized matrix:

    >>> M = DensityMatrix(..., spin="polarized")
    >>> M[0, 0, 0] = # onsite spin up
    >>> M[0, 0, 1] = # onsite spin down

    For non-colinear the indices are a bit more tricky:

    >>> M = DensityMatrix(..., spin="non-colinear")
    >>> M[0, 0, M.M11] = # Re(up-up)
    >>> M[0, 0, M.M22] = # Re(down-down)
    >>> M[0, 0, M.M12r] = # Re(up-down)
    >>> M[0, 0, M.M12i] = # Im(up-down)

    For spin-orbit it looks like this:

    >>> M = DensityMatrix(..., spin="spin-orbit")
    >>> M[0, 0, M.M11r] = # Re(up-up)
    >>> M[0, 0, M.M11i] = # Im(up-up)
    >>> M[0, 0, M.M22r] = # Re(down-down)
    >>> M[0, 0, M.M22i] = # Im(down-down)
    >>> M[0, 0, M.M12r] = # Re(up-down)
    >>> M[0, 0, M.M12i] = # Im(up-down)
    >>> M[0, 0, M.M21r] = # Re(down-up)
    >>> M[0, 0, M.M21i] = # Im(down-up)

    Thus the number of *orbitals* is unchanged but a sub-block exists for
    the spin-block.

    When transferring the matrix to a k-point the spin-box is local to each
    orbital, meaning that the spin-box for orbital i will be:

    >>> Dk = M.Dk()
    >>> Dk[i*2:(i+1)*2, i*2:(i+1)*2]

    Parameters
    ----------
    geometry : Geometry
      parent geometry to create a density matrix from. The density matrix will
      have size equivalent to the number of orbitals in the geometry
    dim : int or Spin, optional
      number of components per element, may be a `Spin` object
    dtype : np.dtype, optional
      data type contained in the density matrix. See details of `Spin` for default values.
    nnzpr : int, optional
      number of initially allocated memory per orbital in the density matrix.
      For increased performance this should be larger than the actual number of entries
      per orbital.
    spin : Spin, optional
      equivalent to `dim` argument. This keyword-only argument has precedence over `dim`.
    orthogonal : bool, optional
      whether the density matrix corresponds to a non-orthogonal basis. In this case
      the dimensionality of the density matrix is one more than `dim`.
      This is a keyword-only argument.
    """

    def __init__(self, geometry, dim=1, dtype=None, nnzpr=None, **kwargs):
        """Initialize density matrix"""
        super().__init__(geometry, dim, dtype, nnzpr, **kwargs)
        self._reset()

    def _reset(self):
        super()._reset()
        self.Dk = self.Pk
        self.dDk = self.dPk
        self.ddDk = self.ddPk

    @property
    def D(self):
        r"""Access the density matrix elements"""
        self._def_dim = self.UP
        return self

    def orbital_momentum(
        self,
        projection: Literal["orbital", "atom"] = "orbital",
        method: Literal["onsite"] = "onsite",
    ):
        r"""Calculate orbital angular momentum on either atoms or orbitals

        Currently this implementation equals the Siesta implementation in that
        the on-site approximation is enforced thus limiting the calculated quantities
        to obey the following conditions:

        1. Same atom
        2. :math:`l>0`
        3. :math:`l_i \equiv l_j`
        4. :math:`m_i \neq m_j`
        5. :math:`\zeta_i \equiv \zeta_j`

        This allows one to sum the orbital angular moments on a per atom site.

        Parameters
        ----------
        projection :
            whether the angular momentum is resolved per atom, or per orbital
        method :
            method used to calculate the angular momentum

        Returns
        -------
        numpy.ndarray
            orbital angular momentum with the last dimension equalling the :math:`L_x`, :math:`L_y` and :math:`L_z` components
        """
        # Check that the spin configuration is correct
        if not (self.spin.is_spinorbit or self.spin.is_nambu):
            raise ValueError(
                f"{self.__class__.__name__}.orbital_momentum requires minimum a spin-orbit matrix"
            )

        # First we calculate
        orb_lmZ = _a.emptyi([self.no, 3])
        for atom, idx in self.geometry.atoms.iter(True):
            # convert to FIRST orbital index per atom
            oidx = self.geometry.a2o(idx)
            # loop orbitals
            for io, orb in enumerate(atom):
                orb_lmZ[oidx + io, :] = orb.l, orb.m, orb.zeta

        # Now we need to calculate the stuff
        DM = self.copy()
        # The Siesta convention *only* calculates contributions
        # in the primary unit-cell.
        DM.set_nsc([1] * 3)
        geom = DM.geometry
        csr = DM._csr

        # The siesta moments are only *on-site* per atom.
        # 1. create a logical index for the matrix elements
        #    that is true for ia-ia interaction and false
        #    otherwise
        idx = repeat(_a.arangei(geom.no), csr.ncol)
        aidx = geom.o2a(idx)

        # Sparse matrix indices for data
        sidx = _a.array_arange(csr.ptr[:-1], n=csr.ncol, dtype=np.int32)
        jdx = csr.col[sidx]
        ajdx = geom.o2a(jdx)

        # Now only take the elements that are *on-site* and which are *not*
        # having the same m quantum numbers (if the orbital index is the same
        # it means they have the same m quantum number)
        #
        # 1. on the same atom
        # 2. l > 0
        # 3. same quantum number l
        # 4. different quantum number m
        # 5. same zeta
        onsite_idx = (
            (aidx == ajdx)
            & (orb_lmZ[idx, 0] > 0)
            & (orb_lmZ[idx, 0] == orb_lmZ[jdx, 0])
            & (orb_lmZ[idx, 1] != orb_lmZ[jdx, 1])
            & (orb_lmZ[idx, 2] == orb_lmZ[jdx, 2])
        ).nonzero()[0]
        # clean variables we don't need
        del aidx, ajdx

        # Now reduce arrays to the orbital connections that obey the
        # above criteria
        idx = idx[onsite_idx]
        idx_l = orb_lmZ[idx, 0]
        idx_m = orb_lmZ[idx, 1]
        jdx = jdx[onsite_idx]
        jdx_m = orb_lmZ[jdx, 1]
        sidx = sidx[onsite_idx]

        # Sum the spin-box diagonal imaginary parts
        DM = csr._D[sidx][:, [4, 5]].sum(1)

        # Define functions to calculate L projections
        def La(idx_l, DM, sub):
            if len(sub) == 0:
                return []
            return (idx_l[sub] * (idx_l[sub] + 1) * 0.5) ** 0.5 * DM[sub]

        def Lb(idx_l, DM, sub):
            if len(sub) == 0:
                return []
            return (idx_l[sub] * (idx_l[sub] + 1) - 2) ** 0.5 * 0.5 * DM[sub]

        def Lc(idx, idx_l, DM, sub):
            if len(sub) == 0:
                return [], []
            sub = sub[idx_l[sub] >= 3]
            if len(sub) == 0:
                return [], []
            return idx[sub], (idx_l[sub] * (idx_l[sub] + 1) - 6) ** 0.5 * 0.5 * DM[sub]

        # construct for different m
        # in Siesta the spin orbital angular momentum
        # is calculated by swapping i and j indices.
        # This is somewhat confusing to me, so I reversed everything.
        # This will probably add to the confusion when comparing the two
        # Additionally Siesta calculates L for <i|L|j> and then does:
        #    L(:) = [L(3), -L(2), -L(1)]
        # Here we *directly* store the quantities used.
        # Pre-allocate the L_xyz quantity per orbital.
        L = np.zeros([3, geom.no])
        L0 = L[0]
        L1 = L[1]
        L2 = L[2]

        # Pre-calculate all those which have m_i + m_j == 0
        b = (idx_m + jdx_m == 0).nonzero()[0]
        subtract.at(L2, idx[b], idx_m[b] * DM[b])
        del b

        #   mi == 0
        i_m = idx_m == 0
        #     mj == -1
        sub = logical_and(i_m, jdx_m == -1).nonzero()[0]
        subtract.at(L0, idx[sub], La(idx_l, DM, sub))
        #     mj == 1
        sub = logical_and(i_m, jdx_m == 1).nonzero()[0]
        add.at(L1, idx[sub], La(idx_l, DM, sub))

        #   mi == 1
        i_m = idx_m == 1
        #     mj == -2
        sub = logical_and(i_m, jdx_m == -2).nonzero()[0]
        subtract.at(L0, idx[sub], Lb(idx_l, DM, sub))
        #     mj == 0
        sub = logical_and(i_m, jdx_m == 0).nonzero()[0]
        subtract.at(L1, idx[sub], La(idx_l, DM, sub))
        #     mj == 2
        sub = logical_and(i_m, jdx_m == 2).nonzero()[0]
        add.at(L1, idx[sub], Lb(idx_l, DM, sub))

        #   mi == -1
        i_m = idx_m == -1
        #     mj == -2
        sub = logical_and(i_m, jdx_m == -2).nonzero()[0]
        add.at(L1, idx[sub], Lb(idx_l, DM, sub))
        #     mj == 0
        sub = logical_and(i_m, jdx_m == 0).nonzero()[0]
        add.at(L0, idx[sub], La(idx_l, DM, sub))
        #     mj == 2
        sub = logical_and(i_m, jdx_m == 2).nonzero()[0]
        add.at(L0, idx[sub], Lb(idx_l, DM, sub))

        #   mi == 2
        i_m = idx_m == 2
        #     mj == -3
        sub = logical_and(i_m, jdx_m == -3).nonzero()[0]
        subtract.at(L0, *Lc(idx, idx_l, DM, sub))
        #     mj == -1
        sub = logical_and(i_m, jdx_m == -1).nonzero()[0]
        subtract.at(L0, idx[sub], Lb(idx_l, DM, sub))
        #     mj == 1
        sub = logical_and(i_m, jdx_m == 1).nonzero()[0]
        subtract.at(L1, idx[sub], Lb(idx_l, DM, sub))
        #     mj == 3
        sub = logical_and(i_m, jdx_m == 3).nonzero()[0]
        add.at(L1, *Lc(idx, idx_l, DM, sub))

        #   mi == -2
        i_m = idx_m == -2
        #     mj == -3
        sub = logical_and(i_m, jdx_m == -3).nonzero()[0]
        add.at(L1, *Lc(idx, idx_l, DM, sub))
        #     mj == -1
        sub = logical_and(i_m, jdx_m == -1).nonzero()[0]
        subtract.at(L1, idx[sub], Lb(idx_l, DM, sub))
        #     mj == 1
        sub = logical_and(i_m, jdx_m == 1).nonzero()[0]
        add.at(L0, idx[sub], Lb(idx_l, DM, sub))
        #     mj == 3
        sub = logical_and(i_m, jdx_m == 3).nonzero()[0]
        add.at(L0, *Lc(idx, idx_l, DM, sub))

        #   mi == -3
        i_m = idx_m == -3
        #     mj == -2
        sub = logical_and(i_m, jdx_m == -2).nonzero()[0]
        subtract.at(L1, *Lc(idx, idx_l, DM, sub))
        #     mj == 2
        sub = logical_and(i_m, jdx_m == 2).nonzero()[0]
        add.at(L0, *Lc(idx, idx_l, DM, sub))

        #   mi == 3
        i_m = idx_m == 3
        #     mj == -2
        sub = logical_and(i_m, jdx_m == -2).nonzero()[0]
        subtract.at(L0, *Lc(idx, idx_l, DM, sub))
        #     mj == 2
        sub = logical_and(i_m, jdx_m == 2).nonzero()[0]
        subtract.at(L1, *Lc(idx, idx_l, DM, sub))

        projection = projection.lower()
        if projection.startswith("orbital"):  # allows orbitals
            return L

        if projection.startswith("atom"):  # allows atoms
            # Now perform summation per atom
            l = np.zeros([3, geom.na], dtype=L.dtype)
            add.at(l.T, geom.o2a(np.arange(geom.no)), L.T)
            return l

        raise ValueError(
            f"{self.__class__.__name__}.orbital_momentum only allows projection [orbital, atom]"
        )

    def Dk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format="csr",
        *args,
        **kwargs,
    ):
        r"""Setup the density matrix for a given k-point

        Creation and return of the density matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \mathbf D(\mathbf k) = \mathbf D_{ij} e^{i \mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \mathbf D(\mathbf k) = \mathbf D_{ij} e^{i \mathbf k\cdot\mathb r}

        where :math:`\mathbf r` is the distance between the orbitals.

        Parameters
        ----------
        k :
           the k-point to setup the density matrix at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge, ``lattice`` for cell vector gauge, and ``atomic`` for atomic distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).
           Prefixing with 'sc:', or simply 'sc' returns the matrix in supercell format
           with phases.
        spin : int, optional
           if the density matrix is a spin polarized one can extract the specific spin direction
           matrix by passing an integer (0 or 1). If the density matrix is not `Spin.POLARIZED`
           this keyword is ignored.

        See Also
        --------
        dDk : Density matrix derivative with respect to `k`
        ddDk : Density matrix double derivative with respect to `k`

        Returns
        -------
        matrix : numpy.ndarray or scipy.sparse.*_matrix
            the density matrix at :math:`\mathbf k`. The returned object depends on `format`.
        """
        pass

    def dDk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format="csr",
        *args,
        **kwargs,
    ):
        r"""Setup the density matrix derivative for a given k-point

        Creation and return of the density matrix derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_k \mathbf D_\alpha(\mathbf k) = i \mathbf R_\alpha \mathbf D_{ij} e^{i \mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.
        And :math:`\alpha` is one of the Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \nabla_k \mathbf D_\alpha(\mathbf k) = i \mathbf r_\alpha \mathbf D_{ij} e^{i \mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is the distance between the orbitals.

        Parameters
        ----------
        k :
           the k-point to setup the density matrix at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge, ``lattice`` for cell vector gauge, and ``atomic`` for atomic distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).
        spin : int, optional
           if the density matrix is a spin polarized one can extract the specific spin direction
           matrix by passing an integer (0 or 1). If the density matrix is not `Spin.POLARIZED`
           this keyword is ignored.

        See Also
        --------
        Dk : Density matrix with respect to `k`
        ddDk : Density matrix double derivative with respect to `k`

        Returns
        -------
        tuple
             for each of the Cartesian directions a :math:`\partial \mathbf D(\mathbf k)/\partial\mathbf k` is returned.
        """
        pass

    def ddDk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format="csr",
        *args,
        **kwargs,
    ):
        r"""Setup the density matrix double derivative for a given k-point

        Creation and return of the density matrix double derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_k^2 \mathbf D^{(alpha\beta)}(\mathbf k) = - \mathbf R_\alpha \mathbf R_\beta \mathbf D_{ij} e^{i \mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.
        And :math:`\alpha` and :math:`\beta` are one of the Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \nabla_k^2 \mathbf D^{(\alpha\beta)}(\mathbf k) = - \mathbf r^{(i)} \mathbf r^{\beta} \mathbf D_{ij} e^{i\mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is the distance between the orbitals.

        Parameters
        ----------
        k :
           the k-point to setup the density matrix at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge, ``lattice`` for cell vector gauge, and ``atomic`` for atomic distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).
        spin : int, optional
           if the density matrix is a spin polarized one can extract the specific spin direction
           matrix by passing an integer (0 or 1). If the density matrix is not `Spin.POLARIZED`
           this keyword is ignored.

        See Also
        --------
        Dk : Density matrix with respect to `k`
        dDk : Density matrix derivative with respect to `k`

        Returns
        -------
        list of matrices
            for each of the Cartesian directions (in Voigt representation); xx, yy, zz, zy, xz, xy
        """
        pass

    @staticmethod
    def read(sile, *args, **kwargs):
        """Reads density matrix from `Sile` using `read_density_matrix`.

        Parameters
        ----------
        sile : Sile, str or pathlib.Path
            a `Sile` object which will be used to read the density matrix
            and the overlap matrix (if any)
            if it is a string it will create a new sile using `get_sile`.
        * : args passed directly to ``read_density_matrix(,**)``
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import BaseSile, get_sile

        if isinstance(sile, BaseSile):
            return sile.read_density_matrix(*args, **kwargs)
        else:
            with get_sile(sile, mode="r") as fh:
                return fh.read_density_matrix(*args, **kwargs)
