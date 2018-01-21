from __future__ import print_function, division

from numbers import Integral
from functools import partial
import numpy as np
from numpy import dot, argsort, where, mod

import sisl._array as _a
from sisl._help import _zip as zip
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
        if geom is None:
            geom = grid.geometry

        # Extract sub variables used throughout the loop
        csr = self._csr
        o2a = geom.o2a
        dcell = grid.dcell

        # In the following we don't care about division
        # So 1) save error state, 2) turn off divide by 0, 3) calculate, 4) turn on old error state
        old_err = np.seterr(divide='ignore', invalid='ignore')

        spin_pol = Spin('p')

        if self.spin > spin_pol:
            if spinor is None:
                spinor = _a.arrayz([[1., 0], [0., 1.]])
            if spinor.size != 4 or spinor.ndim != 2:
                raise ValueError(self.__class__.__name__ + '.rho with NC/SO spin, requires a 2x2 matrix.')

            # TODO I am not sure whether the below dot product
            # requires dot(DM, spinor) or dot(DM, spinor.T)
            # I think it has to be spinor.T
            if self.spin == Spin('NC'):
                def extract_row_DM(io):
                    """ First construct the spin-box DM, then do DM . spinor, and lastly, sum diagonals """
                    sl = slice(csr.ptr[io], csr.ptr[io] + csr.ncol[io])
                    col = csr.col[sl]
                    DM = _a.emptyz([csr.ncol[io], 2, 2])
                    DM[:, 0, 0] = csr._D[sl, 0]
                    DM[:, 1, 1] = csr._D[sl, 1]
                    DM[:, 1, 0] = csr._D[sl, 2] - 1j * csr._D[sl, 3]
                    DM[:, 0, 1] = np.conj(DM[:, 1, 0])
                    DM = dot(DM, spinor.T)[:, [0, 1], [0, 1]].sum(1).real
                    idx = DM.nonzero()[0]
                    return col[idx], DM[idx]
            else:
                def extract_row_DM(io):
                    """ First construct the spin-box DM, then do DM . spinor, and lastly, sum diagonals """
                    sl = slice(csr.ptr[io], csr.ptr[io] + csr.ncol[io])
                    col = csr.col[sl]
                    DM = _a.emptyz([csr.ncol[io], 2, 2])
                    DM[:, 0, 0] = csr._D[sl, 0] + 1j * csr._D[sl, 4]
                    DM[:, 1, 1] = csr._D[sl, 1] + 1j * csr._D[sl, 5]
                    DM[:, 1, 0] = csr._D[sl, 2] - 1j * csr._D[sl, 3]
                    DM[:, 0, 1] = csr._D[sl, 6] + 1j * csr._D[sl, 7]
                    DM = dot(DM, spinor.T)[:, [0, 1], [0, 1]].sum(1).real
                    idx = DM.nonzero()[0]
                    return col[idx], DM[idx]

        elif self.spin == spin_pol:
            if spinor is None:
                spinor = _a.arrayd([1., 1.])
            if isinstance(spinor, Integral):
                # extract the provided spin-polarization
                s = [0.] * 2
                s[spinor] = 1.
                spinor = s
            spinor = _a.arrayd(spinor)

            if spinor.size != 2:
                raise ValueError(self.__class__.__name__ + '.rho with polarized spin, requires an integer, or a vector of length 2')

            if self.orthogonal:
                def extract_row_DM(io):
                    sl = slice(csr.ptr[io], csr.ptr[io] + csr.ncol[io])
                    col = csr.col[sl]
                    DM = dot(csr._D[sl, :], spinor)
                    idx = DM.nonzero()[0]
                    return col[idx], DM[idx]
            else:
                def extract_row_DM(io):
                    sl = slice(csr.ptr[io], csr.ptr[io] + csr.ncol[io])
                    col = csr.col[sl]
                    DM = dot(csr._D[sl, :-1], spinor)
                    idx = DM.nonzero()[0]
                    return col[idx], DM[idx]

        else:
            # spin-unpolarized
            def extract_row_DM(io):
                sl = slice(csr.ptr[io], csr.ptr[io] + csr.ncol[io])
                col = csr.col[sl]
                DM = csr._D[sl, 0]
                idx = DM.nonzero()[0]
                return col[idx], DM[idx]

        log_and = np.logical_and
        log_andr = log_and.reduce
        shape = _a.arrayi(grid.shape).reshape(1, 3)

        all_xyz = (geom.axyz(np.arange(geom.na_s)) - grid.origo.reshape(1, 3)).reshape(-1, 1, 3)
        c2s = partial(cart2spher, cos_phi=True)
        def add_DM(xyz, io, orb):
            s0 = orb.toSphere()
            s0.set_center(xyz)

            # Now loop on all connections (and skip the diagonal, since it is in the above loop)
            # Note that extract_row_DM also removes the diagonal element
            col, DM_col = extract_row_DM(io)
            for ja, orb2, DM in zip(o2a(col), geom.atom.orbital(col), DM_col):

                # Create the unified sphere
                s2 = orb2.toSphere()
                xyz2 = all_xyz[ja, :, :]
                s2.set_center(xyz2)
                s = s0 & s2

                # Find indices of overlapping spheres
                idx = grid.index(s)
                if len(idx) == 0:
                    continue

                rxyz = dot(idx, dcell)
                mod(idx, shape, out=idx)
                grid.grid[idx[:, 0], idx[:, 1], idx[:, 2]] += DM * (
                    orb.psi_spher(*c2s(rxyz - xyz), cos_phi=True) *
                    orb2.psi_spher(*c2s(rxyz - xyz2), cos_phi=True))

        # Loop over all atoms in supercell structure
        io = -1
        for ia in geom:

            # Get atomic coordinate
            xyz = all_xyz[ia, :, :]
            # Get current atom
            atom = geom.atom[ia]

            # Extract maximum R
            R = atom.maxR()
            if R <= 0.:
                warnings.warn("Atom '{}' does not have a wave-function, skipping atom.".format(atom))
                # Skip this atom
                io += atom.no
                continue

            # Loop on orbitals on this atom, grouped by radius
            for os in atom.iter(True):

                # Get the radius of orbitals (os)
                oR = os[0].R

                if oR <= 0.:
                    warnings.warn("Orbital(s) '{}' does not have a wave-function, skipping orbital.".format(os))
                    # Skip these orbitals
                    io += len(os)
                    continue

                # Loop orbitals with the same radius
                for o in os:
                    io += 1
                    #print('{} / {}'.format(io, geom.no))

                    # Now loop each connection orbital
                    add_DM(xyz, io, o)

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
