from __future__ import print_function, division

from warnings import warn
from numbers import Integral
from functools import partial
from scipy.sparse import csr_matrix
import numpy as np
from numpy import int32
from numpy import dot, argsort, where, floor, take
from numpy import logical_and as log_and


import sisl._array as _a
from sisl.utils.ranges import array_arange
from sisl._help import _zip as zip, _range as range
from sisl.utils.mathematics import cart2spher
from sisl.shape import Sphere
from .spin import Spin
from .sparse import SparseOrbitalBZSpin

__all__ = ['DensityMatrix']


class DensityMatrix(SparseOrbitalBZSpin):
    """ DensityMatrix object containing the density matrix elements

    The object contains information regarding the
     - geometry
     - density matrix elements between orbitals

    Assigning or changing elements is as easy as with
    standard `numpy` assignments:

    >>> DM = DensityMatrix(...) # doctest: +SKIP
    >>> DM.D[1,2] = 0.1 # doctest: +SKIP

    which assigns 0.1 as the density element between orbital 2 and 3.
    (remember that Python is 0-based elements).
    """

    def __init__(self, geom, dim=1, dtype=None, nnzpr=None, **kwargs):
        """Create DensityMatrix model from geometry

        Initializes a DensityMatrix using the ``geom`` object.
        """
        super(DensityMatrix, self).__init__(geom, dim, dtype, nnzpr, **kwargs)

        if self.spin.is_unpolarized:
            self.Dk = self._Pk_unpolarized
        elif self.spin.is_polarized:
            self.Dk = self._Pk_polarized
        elif self.spin.is_noncolinear:
            self.Dk = self._Pk_non_colinear
        elif self.spin.is_spinorbit:
            self.Dk = self._Pk_spin_orbit

    def Dk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the density matrix for a given k-point

        Creation and return of the density matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
          D(k) = D_{ij} e^{i k R}

        where :math:`R` is an integer times the cell vector and :math:`i`, :math:`j` are orbital indices.

        Another possible gauge is the orbital distance which can be written as

        .. math::
          D(k) = D_{ij} e^{i k r}

        where :math:`r` is the distance between the orbitals :math:`i` and :math:`j`.
        Currently the second gauge is not implemented (yet).

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
           return in `numpy.ndarray` (`'array'`) or `numpy.matrix` (`'dense'`).
        spin : int, optional
           if the density matrix is a spin polarized one can extract the specific spin direction
           matrix by passing an integer (0 or 1). If the density matrix is not `Spin.POLARIZED`
           this keyword is ignored.
        """
        pass

    def _get_D(self):
        self._def_dim = self.UP
        return self

    def _set_D(self, key, value):
        if len(key) == 2:
            self._def_dim = self.UP
        self[key] = value

    D = property(_get_D, _set_D)

    def rho(self, grid, spinor=None):
        r""" Expand the density matrix to a density on the grid

        This routine calculates the real-space density components in the
        specified grid. 

        This is an *in-place* operation that *adds* to the current values in the grid.

        Note: To calculate :math:`\rho(\mathbf r)` in a unit-cell different from the
        originating geometry, simply pass a grid with a unit-cell smaller than the originating
        supercell.

        The real-space density is calculated as:

        .. math::
            \psi(\mathbf r) = \sum_{\nu\mu}\phi_\nu(\mathbf r)\phi_\mu(\mathbf r) \mathbf \rho_{\nu\mu}

        While for non-colinear/spin-orbit calculations the wavefunctions are determined from the
        spinor component (`spinor`) by

        .. math::
           \psi_{\sigma'}(\mathbf r) = \sum_{\nu\mu}\phi_\nu(\mathbf r)\phi_\mu(\mathbf r) \sum_\alpha [\sigma' \mathbf \rho_{\nu\mu}]_{\alpha\alpha}

        so to get only the :math:`x` component of the density one should pass the Pauli :math:`\sigma_x` matrix (`Spin.X`).

        Parameters
        ----------
        grid : Grid
           the grid on which to add the density (the density is in ``e/Ang^3``)
        spinor : (2, ) or (2, 2), optional
           the spinor matrix to obtain the diagonal components of the density. For un-polarized density matrices
           this keyword has no influence. For spin-polarized it *has* to be either 1 integer or a vector of
           length 2 (defaults to total density).
           For non-colinear/spin-orbit density matrices it has to be a 2x2 matrix (defaults to total density).
        """
        geom = self.geom
        cell = self.geom.cell

        if geom is None:
            geom = grid.geometry

        # Extract sub variables used throughout the loop
        csr = self._csr
        o2a = geom.o2a
        dcell = grid.dcell
        frcell = grid.rcell.T / (2 * np.pi)

        # In the following we don't care about division
        # So 1) save error state, 2) turn off divide by 0, 3) calculate, 4) turn on old error state
        old_err = np.seterr(divide='ignore', invalid='ignore')

        spin_pol = Spin('p')

        DM = None
        if self.spin > spin_pol:
            if spinor is None:
                spinor = _a.arrayz([[1., 0], [0., 1.]])
            spinor = _a.arrayz(spinor)
            if spinor.size != 4 or spinor.ndim != 2:
                raise ValueError(self.__class__.__name__ + '.rho with NC/SO spin, requires a 2x2 matrix.')

            DM = _a.emptyz([self.nnz, 2, 2])
            idx = array_arange(csr.ptr[:-1], n=csr.ncol)
            if self.spin == Spin('NC'):
                # non-colinear
                DM[:, 0, 0] = csr._D[idx, 0]
                DM[:, 1, 1] = csr._D[idx, 1]
                DM[:, 1, 0] = csr._D[idx, 2] - 1j * csr._D[idx, 3]
                DM[:, 0, 1] = np.conj(DM[:, 1, 0])
            else:
                # spin-orbit
                DM[:, 0, 0] = csr._D[idx, 0] + 1j * csr._D[idx, 4]
                DM[:, 1, 1] = csr._D[idx, 1] + 1j * csr._D[idx, 5]
                DM[:, 1, 0] = csr._D[idx, 2] - 1j * csr._D[idx, 3]
                DM[:, 0, 1] = csr._D[idx, 6] + 1j * csr._D[idx, 7]

            # Perform dot-product with spinor, and take out the diagonal real part
            DM = dot(DM, spinor.T)[:, [0, 1], [0, 1]].sum(1).real

        elif self.spin == spin_pol:
            if spinor is None:
                spinor = _a.arrayd([1., 1.])
            if isinstance(spinor, Integral):
                # extract the provided spin-polarization
                s = [0.] * 2
                s[spinor] = 1.
                spinor = s
            spinor = _a.arrayd(spinor)

            if spinor.size != 2 or spinor.ndim != 1:
                raise ValueError(self.__class__.__name__ + '.rho with polarized spin, requires an integer, or a vector of length 2')

            DM = _a.emptyd([self.nnz, 2])
            idx = array_arange(csr.ptr[:-1], n=csr.ncol)
            DM[:, 0] = csr._D[idx, 0]
            DM[:, 1] = csr._D[idx, 1]
            DM = dot(DM, spinor)

        else:
            idx = array_arange(csr.ptr[:-1], n=csr.ncol)
            DM = csr._D[idx, 0]

        # Create the DM csr matrix.
        csrDM = csr_matrix((DM, csr.col[idx], np.insert(np.cumsum(csr.ncol), 0, 0)),
                           shape=(self.shape[:2]), dtype=DM.dtype)
        csrDM.eliminate_zeros()
        csrDM.sort_indices()
        csrDM.prune()

        # Clean-up
        del idx, DM

        # The shape of the grid, in a reshaped version
        shape = _a.arrayi(grid.shape).reshape(1, 3)
        # Retrieve all supercell atoms' coordinates
        origo = grid.origo.reshape(1, 3)
        all_xyz = (geom.axyz(np.arange(geom.na_s)) - origo).reshape(-1, 1, 3)

        def add_DM(ia, atomi, xyzi, cscDM, ja, atomj, xyzj, s):

            # Find all indices for the grid (they may be outside the cell).
            idx = grid.index(s)
            if len(idx) == 0:
                return

            # TODO currently we don't allow unit-cells larger than the calculating one
            # becouse we would then need to figure out the relative atomic indices

            # The grid unit-cell and the geometry unit-cell are the same
            # Hence we can easily determine whether they are inside or out
            idx2 = log_and(log_and.reduce(0 <= idx, axis=1),
                           log_and.reduce(idx < shape, axis=1)).nonzero()[0]
            if len(idx2) == 0:
                return

            idx = take(idx, idx2, axis=0)

            # Calculate the positions of the indices according to the grid
            # voxel cells
            rxyz = dot(idx, dcell)

            del idx2

            # Figure out orbitals, instead of using all=True, we
            # do it like this because it is fancy indexing which is easier
            # on memory.
            o1, o2 = geom.a2o([ja, ja+1])

            # Retrieve the matrix that connects the two atoms (i in unit-cell, j in supercell)
            ijDM = cscDM[:, o1:o2]
            ijptr = ijDM.indptr.view()
            ijind = ijDM.indices.view()
            ijdm = ijDM.data.view()

            # Get the two atoms spherical coordinates
            rri, thetai, cos_phii = cart2spher(rxyz - xyzi, cos_phi=True)
            if ia == ja:
                rrj = rri.view()
                thetaj = thetai.view()
                cos_phij = cos_phii.view()
            else:
                rrj, thetaj, cos_phij = cart2spher(rxyz - xyzj, cos_phi=True)

            # Clean-up to reduce memory...
            del rxyz

            # Now loop on all connections between the two atoms
            psi = _a.emptyd(rri.shape)
            for c in range(ijDM.shape[1]):
                psi.fill(0.)
                sl = slice(ijptr[c], ijptr[c+1])
                for r, dm in zip(ijind[sl], ijdm[sl]):
                    psi += dm * atomi.orbital[r].psi_spher(rri, thetai, cos_phii, cos_phi=True)
                grid.grid[idx[:, 0], idx[:, 1], idx[:, 2]] += psi * atomj.orbital[c].psi_spher(rrj, thetaj, cos_phij, cos_phi=True)

        def skip_atom(a):
            if a.maxR() <= 0.:
                warnings.warn("Atom '{}' does not have a wave-function, skipping atom.".format(a))
                # Skip this atom
                return True
            return False

        def unique_atom_edge(csrDM, ia):
            slo = slice(csrDM.indptr[geom.firsto[ia]],
                        csrDM.indptr[geom.lasto[ia] + 1])
            ja = geom.o2a(csrDM.indices[slo], uniq=True)
            return zip(ja, ja % geom.na)

        # Loop over all atoms in unitcell
        for ia in geom:

            # Get current atom
            atomi = geom.atom[ia]
            if skip_atom(atomi):
                # Note we don't check atomj, because they should be the
                # same.
                continue

            # Get information about this atom
            xyzi = all_xyz[ia, :, :]
            si = atomi.toSphere()
            si.set_center(xyzi)

            # Extract all connections to this atom and convert to CSC matrix
            # This is just because we can then loop on the atom orbitals in add_DM
            cscDM = csrDM[geom.firsto[ia]:geom.lasto[ia] + 1, :].tocsc()

            # Figure out all connecting atoms
            for ja, JA in unique_atom_edge(csrDM, ia):
                # Get connecting atom (in supercell format)
                atomj = geom.atom[JA]

                # Get information about this atom
                xyzj = all_xyz[ja, :, :]
                sj = atomj.toSphere()
                sj.set_center(xyzj)

                # Add the density matrix for atom ia -> ja
                add_DM(ia, atomi, xyzi, cscDM, ja, atomj, xyzj, si & sj)

        # Reset the error code for division
        np.seterr(**old_err)

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads density matrix from `Sile` using `read_density_matrix`.

        Parameters
        ----------
        sile : `Sile`, str
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
