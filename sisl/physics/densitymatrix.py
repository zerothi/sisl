# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from numbers import Integral
import math as m
from scipy.sparse import csr_matrix, triu, tril
from scipy.sparse import hstack as ss_hstack
import numpy as np
from numpy import repeat, logical_and
from numpy import dot, unique, add, subtract

from sisl._internal import set_module
from sisl.geometry import Geometry
from sisl.supercell import SuperCell
import sisl._array as _a
from sisl._indices import indices_le, indices_fabs_le
from sisl._math_small import xyz_to_spherical_cos_phi
from sisl.messages import warn, progressbar
from sisl.utils.ranges import array_arange
from .spin import Spin
from sisl.sparse import SparseCSR, _ncol_to_indptr
from sisl.sparse_geometry import SparseOrbital
from .sparse import SparseOrbitalBZSpin

__all__ = ['DensityMatrix']


class _densitymatrix(SparseOrbitalBZSpin):

    def spin_rotate(self, angles, rad=False):
        r""" Rotates spin-boxes by fixed angles around the :math:`x`, :math:`y` and :math:`z` axis, respectively.

        The angles are with respect to each spin-boxes initial angle.
        One should use `spin_align` to fix all angles along a specific direction.

        Notes
        -----
        For a polarized matrix:
        The returned matrix will be in non-collinear spin-configuration in case
        the angles does not reflect a pure flip of spin in the :math:`z`-axis.

        Parameters
        ----------
        angles : (3,)
           angle to rotate spin boxes, :math:`x`, :math:`y` and :math`z`, respectively
        rad : bool, optional
           Determines the unit of `angles`, for true it is in radians

        See Also
        --------
        spin_align : align all spin-boxes along a specific direction

        Returns
        -------
        object
             a new object with rotated spins
        """
        angles = _a.asarrayd(angles)
        if not rad:
            angles = angles / 180 * np.pi

        def cos_sin(a):
            return m.cos(a), m.sin(a)
        calpha, salpha = cos_sin(angles[0])
        cbeta, sbeta = cos_sin(angles[1])
        cgamma, sgamma = cos_sin(angles[2])
        del cos_sin

        # define rotation matrix
        R = (np.array([[cgamma, -sgamma, 0],
                      [sgamma, cgamma, 0],
                      [0, 0, 1]])
             .dot([[cbeta, 0, sbeta],
                   [0, 1, 0],
                   [-sbeta, 0, cbeta]])
             .dot([[1, 0, 0],
                   [0, calpha, -salpha],
                   [0, salpha, calpha]])
        )

        if self.spin.is_noncolinear:
            A = np.empty([len(self._csr._D), 3], dtype=self.dtype)

            D = self._csr._D
            Q = (D[:, 0] + D[:, 1]) * 0.5
            A[:, 0] = 2 * D[:, 2]
            A[:, 1] = - 2 * D[:, 3]
            A[:, 2] = D[:, 0] - D[:, 1]

            A = R.dot(A.T).T * 0.5

            out = self.copy()
            D = out._csr._D
            D[:, 0] = Q + A[:, 2]
            D[:, 1] = Q - A[:, 2]
            D[:, 2] = A[:, 0]
            D[:, 3] = -A[:, 1]

        elif self.spin.is_spinorbit:
            # Since this spin-matrix has all 8 components we will take
            # each half and align individually.
            # I believe this should retain most of the physics in its
            # intrinsic form and thus be a bit more accurate than
            # later re-creating the matrix by some scaling factor.
            A = np.empty([len(self._csr._D), 2, 3], dtype=self.dtype)

            D = self._csr._D
            # we align each part individually
            # this *should* give us the same magnitude...
            Q = (D[:, 0] + D[:, 1]) * 0.5
            A[:, :, 2] = (D[:, 0] - D[:, 1]).reshape(-1, 1)
            A[:, 0, 0] = 2 * D[:, 2]
            A[:, 1, 0] = 2 * D[:, 6]
            A[:, 0, 1] = - 2 * D[:, 3]
            A[:, 1, 1] = 2 * D[:, 7]

            A = R.dot(A.reshape(-1, 3).T).T.reshape(-1, 2, 3) * 0.5

            out = self.copy()
            D = out._csr._D
            D[:, 0] = Q + A[:, :, 2].sum(1) * 0.5
            D[:, 1] = Q - A[:, :, 2].sum(1) * 0.5
            D[:, 2] = A[:, 0, 0]
            D[:, 3] = - A[:, 0, 1]
            # 4 and 5 are diagonal imaginary part (un-changed)
            # Since we copy, we don't need to do anything
            #D[:, 4] =
            #D[:, 5] =
            D[:, 6] = A[:, 1, 0]
            D[:, 7] = A[:, 1, 1]

        elif self.spin.is_polarized:
            def close(a, v):
                return abs(abs(a) - v) < np.pi / 1080

            # figure out if this is only rotating 180 for x or y
            if close(angles[0], np.pi) and close(angles[1], 0) or \
               close(angles[0], 0) and close(angles[1], np.pi):
                # flip spin
                out = self.copy()
                out._csr._D[:, [0, 1]] = out._csr._D[:, [1, 0]]

            else:
                spin = Spin("nc", dtype=self.dtype)
                out = self.__class__(self.geometry, dtype=self.dtype, spin=spin,
                                     orthogonal=self.orthogonal)
                out._csr.ptr[:] = self._csr.ptr[:]
                out._csr.ncol[:] = self._csr.ncol[:]
                out._csr.col = self._csr.col.copy()
                out._csr._nnz = self._csr._nnz

                if self.orthogonal:
                    out._csr._D = np.zeros([len(self._csr._D), 4], dtype=self.dtype)
                    out._csr._D[:, [0, 1]] = self._csr._D[:, :]
                else:
                    out._csr._D = np.zeros([len(self._csr._D), 5], dtype=self.dtype)
                    out._csr._D[:, [0, 1, 4]] = self._csr._D[:, :]
                out = out.spin_rotate(angles, rad=True)

        else:
            raise ValueError(f"{self.__class__.__name__}.spin_rotate requires a matrix with some spin configuration, not an unpolarized matrix.")

        return out

    def spin_align(self, vec):
        r""" Aligns *all* spin along the vector `vec`

        In case the matrix is polarized and `vec` is not aligned at the z-axis, the returned
        matrix will be a non-collinear spin configuration.

        Parameters
        ----------
        vec : (3,)
           vector to align the spin boxes against

        See Also
        --------
        spin_rotate : rotate spin-boxes by a fixed amount (does not align spins)

        Returns
        -------
        object
            a new object with aligned spins
        """
        vec = _a.asarrayd(vec)
        # normalize vector
        vec = vec / (vec ** 2).sum() ** 0.5

        if self.spin.is_noncolinear:
            A = np.empty([len(self._csr._D), 3], dtype=self.dtype)

            D = self._csr._D
            Q = (D[:, 0] + D[:, 1]) * 0.5
            A[:, 0] = 2 * D[:, 2]
            A[:, 1] = - 2 * D[:, 3]
            A[:, 2] = D[:, 0] - D[:, 1]

            # align with vector
            # add factor 1/2 here (instead when unwrapping)
            A[:, :] = 0.5 * vec.reshape(1, 3) * (np.sum(A ** 2, axis=1)
                                                 .reshape(-1, 1)) ** 0.5

            out = self.copy()
            D = out._csr._D
            D[:, 0] = Q + A[:, 2]
            D[:, 1] = Q - A[:, 2]
            D[:, 2] = A[:, 0]
            D[:, 3] = -A[:, 1]

        elif self.spin.is_spinorbit:
            # Since this spin-matrix has all 8 components we will take
            # each half and align individually.
            # I believe this should retain most of the physics in its
            # intrinsic form and thus be a bit more accurate than
            # later re-creating the matrix by some scaling factor.
            A = np.empty([len(self._csr._D), 2, 3], dtype=self.dtype)

            D = self._csr._D
            # we align each part individually
            # this *should* give us the same magnitude...
            Q = (D[:, 0] + D[:, 1]) * 0.5
            A[:, :, 2] = (D[:, 0] - D[:, 1]).reshape(-1, 1)
            A[:, 0, 0] = 2 * D[:, 2]
            A[:, 0, 1] = - 2 * D[:, 3]
            A[:, 1, 0] = 2 * D[:, 6]
            A[:, 1, 1] = 2 * D[:, 7]

            # align with vector
            # add factor 1/2 here (instead when unwrapping)
            A[:, :, :] = 0.5 * vec.reshape(1, 1, 3) * (np.sum(A ** 2, axis=2)
                                                       .reshape(-1, 2, 1)) ** 0.5

            out = self.copy()
            D = out._csr._D
            D[:, 0] = Q + A[:, :, 2].sum(1) * 0.5
            D[:, 1] = Q - A[:, :, 2].sum(1) * 0.5
            D[:, 2] = A[:, 0, 0]
            D[:, 3] = - A[:, 0, 1]
            # 4 and 5 are diagonal imaginary part (un-changed)
            # Since we copy, we don't need to do anything
            #D[:, 4] =
            #D[:, 5] =
            D[:, 6] = A[:, 1, 0]
            D[:, 7] = A[:, 1, 1]

        elif self.spin.is_polarized:
            if abs(vec.sum() - vec[2]) > 1e-6:
                spin = Spin("nc", dtype=self.dtype)
                out = self.__class__(self.geometry, dtype=self.dtype, spin=spin,
                                     orthogonal=self.orthogonal)
                out._csr.ptr[:] = self._csr.ptr[:]
                out._csr.ncol[:] = self._csr.ncol[:]
                out._csr.col = self._csr.col.copy()
                out._csr._nnz = self._csr._nnz

                if self.orthogonal:
                    out._csr._D = np.zeros([len(self._csr._D), 4], dtype=self.dtype)
                    out._csr._D[:, [0, 1]] = self._csr._D[:, :]
                else:
                    out._csr._D = np.zeros([len(self._csr._D), 5], dtype=self.dtype)
                    out._csr._D[:, [0, 1, 4]] = self._csr._D[:, :]
                out = out.spin_align(vec)

            elif vec[2] < 0:
                # flip spin
                out = self.copy()
                out._csr._D[:, [0, 1]] = out._csr._D[:, [1, 0]]
            else:
                out = self.copy()

        else:
            raise ValueError(f"{self.__class__.__name__}.spin_align requires a matrix with some spin configuration, not an unpolarized matrix.")

        return out

    def mulliken(self, projection='orbital'):
        r""" Calculate Mulliken charges from the density matrix

        In the following :math:`\nu` and :math:`\mu` are orbital indices.
        Atomic indices noted by :math:`\alpha`, :math:`\beta`.
        Matrices :math:`\boldsymbol\rho` and :math:`\mathbf S` are density
        and overlap matrices, respectively.

        For polarized calculations the Mulliken charges are calculated as
        (for each spin-channel)

        .. math::

             M_{\nu} &= \sum_mu [\boldsymbol\rho \mathbf S]_{\nu\mu}
             \\
             M_{\alpha} &= \sum_{\nu\in\alpha} M_{\nu}

        For non-colinear calculations (including spin-orbit) they are calculated
        as above but using the spin-box per orbital (:math:`\sigma` is spin)

        .. math::
             M_{\nu} &= \sum_\sigma\sum_mu [\boldsymbol\rho \mathbf S]_{\nu\mu,\sigma\sigma}
             \\
             S_{\nu}^x &= \sum_mu \Re [\boldsymbol\rho \mathbf S]_{\nu\mu,\uparrow\downarrow} +
                          \Re [\boldsymbol\rho \mathbf S]_{\nu\mu,\downarrow\uparrow}
             \\
             S_{\nu}^y &= \sum_mu \Im [\boldsymbol\rho \mathbf S]_{\nu\mu,\uparrow\downarrow} -
                          \Im [\boldsymbol\rho \mathbf S]_{\nu\mu,\downarrow\uparrow}
             \\
             S_{\nu}^z &= \sum_mu \Re [\boldsymbol\rho \mathbf S]_{\nu\mu,\uparrow\uparrow} -
                          \Re [\boldsymbol\rho \mathbf S]_{\nu\mu,\downarrow\downarrow}

        Parameters
        ----------
        projection : {'orbital', 'atom'}
            how the Mulliken charges are returned.
            Can be atom-resolved, orbital-resolved or the
            charge matrix (off-diagonal elements)

        Returns
        -------
        numpy.ndarray
            if `projection` does not contain matrix, the first dimension contains the orbitals, and
            the 2nd the spin information
        """
        def _convert(M):
            """ Converts a non-colinear DM from [11, 22, Re(12), Im(12)] -> [T, Sx, Sy, Sz] """
            if M.shape[-1] == 8:
                # We need to calculate the corresponding values
                M[:, 2] = 0.5 * (M[:, 2] + M[:, 6])
                M[:, 3] = 0.5 * (M[:, 3] - M[:, 7]) # sign change again below
                M = M[:, :4]
            if M.shape[-1] == 4:
                m = np.empty_like(M)
                m[:, 0] = M[:, 0] + M[:, 1]
                m[:, 3] = M[:, 0] - M[:, 1]
                m[:, 1] = 2 * M[:, 2]
                m[:, 2] = - 2 * M[:, 3]
            else:
                return M
            return m

        if "orbital" == projection:
            # Orbital Mulliken population
            if self.orthogonal:
                D = np.array([self._csr.tocsr(i).diagonal() for i in range(self.shape[2])]).T
            else:
                D = self._csr.copy(range(self.shape[2] - 1))
                D._D *= self._csr._D[:, -1].reshape(-1, 1)
                D = np.sum(D, axis=1)

            return _convert(D)

        elif "atom" == projection:
            # Atomic Mulliken population
            if self.orthogonal:
                D = np.array([self._csr.tocsr(i).diagonal() for i in range(self.shape[2])]).T
            else:
                D = self._csr.copy(range(self.shape[2] - 1))
                D._D *= self._csr._D[:, -1].reshape(-1, 1)
                D = np.sum(D, axis=1)

            # Now perform summation per atom
            geom = self.geometry
            M = np.zeros([geom.na, D.shape[1]], dtype=D.dtype)
            np.add.at(M, geom.o2a(np.arange(geom.no)), D)
            del D

            return _convert(M)

        raise NotImplementedError(f"{self.__class__.__name__}.mulliken only allows projection [orbital, atom]")

    def density(self, grid, spinor=None, tol=1e-7, eta=None):
        r""" Expand the density matrix to the charge density on a grid

        This routine calculates the real-space density components on a specified grid.

        This is an *in-place* operation that *adds* to the current values in the grid.

        Note: To calculate :math:`\rho(\mathbf r)` in a unit-cell different from the
        originating geometry, simply pass a grid with a unit-cell different than the originating
        supercell.

        The real-space density is calculated as:

        .. math::
            \rho(\mathbf r) = \sum_{\nu\mu}\phi_\nu(\mathbf r)\phi_\mu(\mathbf r) D_{\nu\mu}

        While for non-collinear/spin-orbit calculations the density is determined from the
        spinor component (`spinor`) by

        .. math::
           \rho_{\boldsymbol\sigma}(\mathbf r) = \sum_{\nu\mu}\phi_\nu(\mathbf r)\phi_\mu(\mathbf r) \sum_\alpha [\boldsymbol\sigma \mathbf \rho_{\nu\mu}]_{\alpha\alpha}

        Here :math:`\boldsymbol\sigma` corresponds to a spinor operator to extract relevant quantities. By passing the identity matrix the total charge is added. By using the Pauli matrix :math:`\boldsymbol\sigma_x`
        only the :math:`x` component of the density is added to the grid (see `Spin.X`).

        Parameters
        ----------
        grid : Grid
           the grid on which to add the density (the density is in ``e/Ang^3``)
        spinor : (2,) or (2, 2), optional
           the spinor matrix to obtain the diagonal components of the density. For un-polarized density matrices
           this keyword has no influence. For spin-polarized it *has* to be either 1 integer or a vector of
           length 2 (defaults to total density).
           For non-collinear/spin-orbit density matrices it has to be a 2x2 matrix (defaults to total density).
        tol : float, optional
           DM tolerance for accepted values. For all density matrix elements with absolute values below
           the tolerance, they will be treated as strictly zeros.
        eta : bool, optional
           show a progressbar on stdout
        """
        try:
            # Once unique has the axis keyword, we know we can safely
            # use it in this routine
            # Otherwise we raise an ImportError
            unique([[0, 1], [2, 3]], axis=0)
        except:
            raise NotImplementedError(f"{self.__class__.__name__}.density requires numpy >= 1.13, either update "
                                      "numpy or do not use this function!")

        geometry = self.geometry
        # Check that the atomic coordinates, really are all within the intrinsic supercell.
        # If not, it may mean that the DM does not conform to the primary unit-cell paradigm
        # of matrix elements. It complicates things.
        fxyz = geometry.fxyz
        f_min = fxyz.min()
        f_max = fxyz.max()
        del fxyz, f_min, f_max

        # Extract sub variables used throughout the loop
        shape = _a.asarrayi(grid.shape)
        dcell = grid.dcell

        # Sparse matrix data
        csr = self._csr

        # In the following we don't care about division
        # So 1) save error state, 2) turn off divide by 0, 3) calculate, 4) turn on old error state
        old_err = np.seterr(divide='ignore', invalid='ignore')

        # Placeholder for the resulting coefficients
        DM = None
        if self.spin.kind > Spin.POLARIZED:
            if spinor is None:
                # Default to the total density
                spinor = np.identity(2, dtype=np.complex128)
            else:
                spinor = _a.arrayz(spinor)
            if spinor.size != 4 or spinor.ndim != 2:
                raise ValueError(f"{self.__class__.__name__}.density with NC/SO spin, requires a 2x2 matrix.")

            DM = _a.emptyz([self.nnz, 2, 2])
            idx = array_arange(csr.ptr[:-1], n=csr.ncol)
            if self.spin.kind == Spin.NONCOLINEAR:
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

        elif self.spin.kind == Spin.POLARIZED:
            if spinor is None:
                spinor = _a.onesd(2)

            elif isinstance(spinor, Integral):
                # extract the provided spin-polarization
                s = _a.zerosd(2)
                s[spinor] = 1.
                spinor = s
            else:
                spinor = _a.arrayd(spinor)

            if spinor.size != 2 or spinor.ndim != 1:
                raise ValueError(f"{self.__class__.__name__}.density with polarized spin, requires spinor "
                                 "argument as an integer, or a vector of length 2")

            idx = array_arange(csr.ptr[:-1], n=csr.ncol)
            DM = csr._D[idx, 0] * spinor[0] + csr._D[idx, 1] * spinor[1]

        else:
            idx = array_arange(csr.ptr[:-1], n=csr.ncol)
            DM = csr._D[idx, 0]

        # Create the DM csr matrix.
        csrDM = csr_matrix((DM, csr.col[idx], _ncol_to_indptr(csr.ncol)),
                           shape=(self.shape[:2]), dtype=DM.dtype)

        # Clean-up
        del idx, DM

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
        csrDM = ss_hstack(csr_sum, format='csr')
        del csr, csr_sum

        # Remove all zero elements (note we use the tolerance here!)
        csrDM.data = np.where(np.fabs(csrDM.data) > tol, csrDM.data, 0.)

        # Eliminate zeros and sort indices etc.
        csrDM.eliminate_zeros()
        csrDM.sort_indices()
        csrDM.prune()

        # 1. Ensure the grid has a geometry associated with it
        sc = grid.sc.copy()
        # Find the periodic directions
        pbc = [bc == grid.PERIODIC or geometry.nsc[i] > 1 for i, bc in enumerate(grid.bc[:, 0])]
        if grid.geometry is None:
            # Create the actual geometry that encompass the grid
            ia, xyz, _ = geometry.within_inf(sc, periodic=pbc)
            if len(ia) > 0:
                grid.set_geometry(Geometry(xyz, geometry.atoms[ia], sc=sc))

        # Instead of looping all atoms in the supercell we find the exact atoms
        # and their supercell indices.
        add_R = _a.fulld(3, geometry.maxR())
        # Calculate the required additional vectors required to increase the fictitious
        # supercell by add_R in each direction.
        # For extremely skewed lattices this will be way too much, hence we make
        # them square.
        o = sc.toCuboid(True)
        sc = SuperCell(o._v + np.diag(2 * add_R), origin=o.origin - add_R)

        # Retrieve all atoms within the grid supercell
        # (and the neighbours that connect into the cell)
        IA, XYZ, ISC = geometry.within_inf(sc, periodic=pbc)
        XYZ -= grid.sc.origin.reshape(1, 3)

        # Retrieve progressbar
        eta = progressbar(len(IA), f"{self.__class__.__name__}.density", "atom", eta)

        cell = geometry.cell
        atoms = geometry.atoms
        axyz = geometry.axyz
        a2o = geometry.a2o

        def xyz2spherical(xyz, offset):
            """ Calculate the spherical coordinates from indices """
            rx = xyz[:, 0] - offset[0]
            ry = xyz[:, 1] - offset[1]
            rz = xyz[:, 2] - offset[2]

            # Calculate radius ** 2
            xyz_to_spherical_cos_phi(rx, ry, rz)
            return rx, ry, rz

        def xyz2sphericalR(xyz, offset, R):
            """ Calculate the spherical coordinates from indices """
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
            ix = indices_le(rx ** 2 + ry ** 2 + rz ** 2, R ** 2)
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
        a_col = csr.col[array_arange(csr.ptr, n=csr.ncol)]
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
            if R <= 0.:
                warn(f"Atom '{ia_atom}' does not have a wave-function, skipping atom.")
                eta.update()
                continue

            # Retrieve indices of the grid for the atomic shape
            idx = grid.index(ia_atom.toSphere(ia_xyz))

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

            # Remove duplicates, requires numpy >= 1.13
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
            DM_io = csrDM[IO:IO+ia_atom.no, :].tolil()
            DM_pj = _a.zerosd([ia_atom.no, grid_xyz.shape[0]])

            # Now we perform the loop on the connections for this atom
            # Remark that we have removed the diagonal atom (it-self)
            # As that will be calculated in the end
            for ja in a_col[a_ptr[ia]:a_ptr[ia+1]]:
                # Retrieve atom (which contains the orbitals)
                ja_atom = atoms[ja % na]
                JO = a2o(ja)
                jR = ja_atom.maxR()
                # Get actual coordinate of the atom
                ja_xyz = axyz(ja) + cell_offset

                # Reduce the ia'th grid points to those that connects to the ja'th atom
                ja_idx, ja_r, ja_theta, ja_cos_phi = xyz2sphericalR(grid_xyz, ja_xyz, jR)

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
                        DM_pj[io, ja_idx1] += DM_io[io, JO+jo] * psi

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
                for jo in range(io+1, ia_atom.no):
                    DM = DM_io[io, off+IO+jo]

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
                    DM_pj[io, ja_idx1] += DM * oj.psi_spher(ja_r1, ja_theta1, ja_cos_phi1, cos_phi=True)

                # Calculate the psi_i component
                # Note that this one *also* zeroes points outside the shell
                # I.e. this step is important because it "nullifies" all but points where
                # orbital io is defined.
                psi = ia_atom.orbitals[io].psi_spher(ia_r, ia_theta, ia_cos_phi, cos_phi=True)
                DM_pj[io, :] += DM_io[io, off+IO+io] * psi
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
    """ Sparse density matrix object

    Assigning or changing elements is as easy as with standard `numpy` assignments:

    >>> DM = DensityMatrix(...)
    >>> DM.D[1,2] = 0.1

    which assigns 0.1 as the density element between orbital 2 and 3.
    (remember that Python is 0-based elements).

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
        """ Initialize density matrix """
        super().__init__(geometry, dim, dtype, nnzpr, **kwargs)
        self._reset()

    def _reset(self):
        super()._reset()
        self.Dk = self.Pk
        self.dDk = self.dPk
        self.ddDk = self.ddPk

    @property
    def D(self):
        r""" Access the density matrix elements """
        self._def_dim = self.UP
        return self

    def orbital_momentum(self, projection='orbital', method='onsite'):
        r""" Calculate orbital angular momentum on either atoms or orbitals

        Currently this implementation equals the Siesta implementation in that
        the on-site approximation is enforced thus limiting the calculated quantities
        to obey the following conditions:

        1. Same atom
        2. :math:`l>0`
        3. :math:`l_\nu \equiv l_\mu`
        4. :math:`m_\nu \neq m_\mu`
        5. :math:`\zeta_\nu \equiv \zeta_\mu`

        This allows one to sum the orbital angular moments on a per atom site.

        Parameters
        ----------
        projection : {'orbital', 'atom'}
            whether the angular momentum is resolved per atom, or per orbital
        method : {'onsite'}
            method used to calculate the angular momentum

        Returns
        -------
        numpy.ndarray
            orbital angular momentum with the last dimension equalling the :math:`L_x`, :math:`L_y` and :math:`L_z` components
        """
        # Check that the spin configuration is correct
        if not self.spin.is_spinorbit:
            raise ValueError(f"{self.__class__.__name__}.orbital_momentum requires a spin-orbit matrix")

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
        sidx = array_arange(csr.ptr[:-1], n=csr.ncol, dtype=np.int32)
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
        onsite_idx = ((aidx == ajdx) & \
                      (orb_lmZ[idx, 0] > 0) & \
                      (orb_lmZ[idx, 0] == orb_lmZ[jdx, 0]) & \
                      (orb_lmZ[idx, 1] != orb_lmZ[jdx, 1]) & \
                      (orb_lmZ[idx, 2] == orb_lmZ[jdx, 2])).nonzero()[0]
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
                return
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
        L = np.zeros([geom.no, 3])
        L0 = L[:, 0]
        L1 = L[:, 1]
        L2 = L[:, 2]

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

        if "orbital" == projection:
            return L
        elif "atom" == projection:
            # Now perform summation per atom
            l = np.zeros([geom.na, 3], dtype=L.dtype)
            add.at(l, geom.o2a(np.arange(geom.no)), L)
            return l
        raise ValueError(f"{self.__class__.__name__}.orbital_momentum must define projection to be 'orbital' or 'atom'.")

    def Dk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the density matrix for a given k-point

        Creation and return of the density matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \mathbf D(k) = \mathbf D_{\nu\mu} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`\nu`, :math:`\mu` are orbital indices.

        Another possible gauge is the orbital distance which can be written as

        .. math::
           \mathbf D(k) = \mathbf D_{\nu\mu} e^{i k r}

        where :math:`r` is the distance between the orbitals.

        Parameters
        ----------
        k : array_like
           the k-point to setup the density matrix at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge : {'R', 'r'}
           the chosen gauge, `R` for cell vector gauge, and `r` for orbital distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the ``scipy.sparse.csr_matrix``,
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
            the density matrix at :math:`k`. The returned object depends on `format`.
        """
        pass

    def dDk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the density matrix derivative for a given k-point

        Creation and return of the density matrix derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_k \mathbf D_\alpha(k) = i R_\alpha \mathbf D_{\nu\mu} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`\nu`, :math:`\mu` are orbital indices.
        And :math:`\alpha` is one of the Cartesian directions.

        Another possible gauge is the orbital distance which can be written as

        .. math::
           \nabla_k \mathbf D_\alpha(k) = i r_\alpha \mathbf D_{\nu\mu} e^{i k r}

        where :math:`r` is the distance between the orbitals.

        Parameters
        ----------
        k : array_like
           the k-point to setup the density matrix at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge : {'R', 'r'}
           the chosen gauge, `R` for cell vector gauge, and `r` for orbital distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the ``scipy.sparse.csr_matrix``,
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
             for each of the Cartesian directions a :math:`\partial \mathbf D(k)/\partial k` is returned.
        """
        pass

    def ddDk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the density matrix double derivative for a given k-point

        Creation and return of the density matrix double derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_k^2 \mathbf D_{\alpha\beta}(k) = - R_\alpha R_\beta \mathbf D_{\nu\mu} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`\nu`, :math:`\mu` are orbital indices.
        And :math:`\alpha` and :math:`\beta` are one of the Cartesian directions.

        Another possible gauge is the orbital distance which can be written as

        .. math::
           \nabla_k^2 \mathbf D_{\alpha\beta}(k) = - r_\alpha r_\beta \mathbf D_{\nu\mu} e^{i k r}

        where :math:`r` is the distance between the orbitals.

        Parameters
        ----------
        k : array_like
           the k-point to setup the density matrix at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge : {'R', 'r'}
           the chosen gauge, `R` for cell vector gauge, and `r` for orbital distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the ``scipy.sparse.csr_matrix``,
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
        """ Reads density matrix from `Sile` using `read_density_matrix`.

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
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_density_matrix(*args, **kwargs)
        else:
            with get_sile(sile) as fh:
                return fh.read_density_matrix(*args, **kwargs)

    def write(self, sile, *args, **kwargs):
        """ Writes a density matrix to the `Sile` as implemented in the :code:`Sile.write_density_matrix` method """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_density_matrix(self, *args, **kwargs)
        else:
            with get_sile(sile, 'w') as fh:
                fh.write_density_matrix(self, *args, **kwargs)
