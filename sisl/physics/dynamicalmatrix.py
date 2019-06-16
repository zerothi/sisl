from __future__ import print_function, division

import numpy as np
from scipy.sparse import lil_matrix

from .sparse import SparseOrbitalBZ
from .phonon import EigenvaluePhonon, EigenmodePhonon

__all__ = ['DynamicalMatrix']


def _correct_hw(hw):
    idx = (hw < 0).nonzero()[0]
    HW = hw.copy()
    HW[idx] *= -1
    HW[:] = np.sqrt(HW)
    HW[idx] *= -1
    return HW


class DynamicalMatrix(SparseOrbitalBZ):
    """ Dynamical matrix of a geometry """

    def __init__(self, geometry, dim=1, dtype=None, nnzpr=None, **kwargs):
        super(DynamicalMatrix, self).__init__(geometry, dim, dtype, nnzpr, **kwargs)
        self._reset()

    def _reset(self):
        super(DynamicalMatrix, self)._reset()
        self.Dk = self.Pk
        self.dDk = self.dPk
        self.ddDk = self.ddPk

        self.Dk = self._Pk
        self.dDk = self.dPk
        self.ddDk = self.ddPk

    def Dk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the dynamical matrix for a given k-point

        Creation and return of the dynamical matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \mathbf D(k) = \mathbf D_{i_\alpha j_\beta} e^{i q R}

        where :math:`R` is an integer times the cell vector and :math:`\alpha`, :math:`\beta` are Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \mathbf D(k) = \mathbf D_{\nu\mu} e^{i k r}

        where :math:`r` is the distance between the atoms.

        Parameters
        ----------
        k : array_like
           the k-point to setup the dynamical matrix at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge : {'R', 'r'}
           the chosen gauge, `R` for cell vector gauge, and `r` for atomic distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the ``scipy.sparse.csr_matrix``,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).

        See Also
        --------
        dDk : dynamical matrix derivative with respect to `k`
        ddDk : dynamical matrix double derivative with respect to `k`

        Returns
        -------
        object : the dynamical matrix at :math:`k`. The returned object depends on `format`.
        """
        pass

    def dDk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the dynamical matrix derivative for a given k-point

        Creation and return of the dynamical matrix derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_k \mathbf D_\gamma(k) = i R_\gamma \mathbf D_{i_\alpha j_\beta} e^{i q R}

        where :math:`R` is an integer times the cell vector and :math:`\alpha`, :math:`\beta` are atomic indices.
        And :math:`\gamma` is one of the Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
          \nabla_k \mathbf D_\gamma(k) = i r_\gamma \mathbf D_{i_\alpha j_\beta} e^{i k r}

        where :math:`r` is the distance between the atoms.

        Parameters
        ----------
        k : array_like
           the k-point to setup the dynamical matrix at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge : {'R', 'r'}
           the chosen gauge, `R` for cell vector gauge, and `r` for atomic distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the ``scipy.sparse.csr_matrix``,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).

        See Also
        --------
        Dk : dynamical matrix with respect to `k`
        ddDk : dynamical matrix double derivative with respect to `k`

        Returns
        -------
        tuple : for each of the Cartesian directions a :math:`\partial \mathbf D(k)/\partial k_\gamma` is returned.
        """
        pass

    def ddDk(self, k=(0, 0, 0), dtype=None, gauge='R', format='csr', *args, **kwargs):
        r""" Setup the dynamical matrix double derivative for a given k-point

        Creation and return of the dynamical matrix double derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_k^2 \mathbf D_{\gamma\sigma}(k) = - R_\gamma R_\sigma \mathbf D_{i_\alpha j_\beta} e^{i q R}

        where :math:`R` is an integer times the cell vector and :math:`\alpha`, :math:`\beta` are Cartesian directions.
        And :math:`\gamma`, :math:`\sigma` are one of the Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \nabla_k^2 \mathbf D_{\gamma\sigma}(k) = - r_\gamma r_\sigma \mathbf D_{i_\alpha j_\beta} e^{i k r}

        where :math:`r` is atomic distance.

        Parameters
        ----------
        k : array_like
           the k-point to setup the dynamical matrix at
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

        See Also
        --------
        Dk : dynamical matrix with respect to `k`
        dDk : dynamical matrix derivative with respect to `k`

        Returns
        -------
        tuple of tuples : for each of the Cartesian directions
        """
        pass

    def _get_D(self):
        self._def_dim = 0
        return self

    def _set_D(self, key, value):
        if len(key) == 2:
            self._def_dim = 0
        self[key] = value

    D = property(_get_D, _set_D, doc="Access elements to the sparse dynamical matrix")

    def apply_newton(self):
        """ Sometimes the dynamical matrix does not obey Newtons 3rd law.

        We correct the dynamical matrix by imposing zero force.

        Correcting for Newton forces the matrix to be finalized.
        """
        # Create UC dynamical matrix
        dyn_sc = self.tocsr(0)
        no = self.no
        d_uc = lil_matrix((no, no), dtype=dyn_sc.dtype)

        for i, _ in self.sc:
            d_uc[:, :] += dyn_sc[:, i*no: (i+1)*no]

        # A CSC matrix is faster to slice for columns
        d_uc = d_uc.tocsc()

        # we need to correct the dynamical matrix such that Newtons 3rd law
        # is obeyed (action == reaction)
        om = np.sqrt(self.mass)
        MM = np.empty([len(om)], np.float64)

        for ja in self.geometry:

            # Create conversion to force-constant in units of the on-site mass scaled
            # dynamical matrix.
            MM[:] = om[:] / om[ja]
            jo = ja * 3

            # Unroll...
            D = self.D[jo, jo]
            self.D[jo, jo] = D - d_uc[jo, ::3].multiply(MM).sum()
            D = self.D[jo, jo + 1]
            self.D[jo, jo + 1] = D - d_uc[jo, 1::3].multiply(MM).sum()
            D = self.D[jo, jo + 2]
            self.D[jo, jo + 2] = D - d_uc[jo, 2::3].multiply(MM).sum()

            D = self.D[jo + 1, jo]
            self.D[jo + 1, jo] = D - d_uc[jo + 1, ::3].multiply(MM).sum()
            D = self.D[jo + 1, jo + 1]
            self.D[jo + 1, jo + 1] = D - d_uc[jo + 1, 1::3].multiply(MM).sum()
            D = self.D[jo + 1, jo + 2]
            self.D[jo + 1, jo + 2] = D - d_uc[jo + 1, 2::3].multiply(MM).sum()

            D = self.D[jo + 2, jo]
            self.D[jo + 2, jo] = D - d_uc[jo + 2, ::3].multiply(MM).sum()
            D = self.D[jo + 2, jo + 1]
            self.D[jo + 2, jo + 1] = D - d_uc[jo + 2, 1::3].multiply(MM).sum()
            D = self.D[jo + 2, jo + 2]
            self.D[jo + 2, jo + 2] = D - d_uc[jo + 2, 2::3].multiply(MM).sum()

        del d_uc

    def eigenvalue(self, k=(0, 0, 0), gauge='R', **kwargs):
        """ Calculate the eigenvalues at `k` and return an `EigenvaluePhonon` object containing all eigenvalues for a given `k`

        Parameters
        ----------
        k : array_like*3, optional
            the k-point at which to evaluate the eigenvalues at
        gauge : str, optional
            the gauge used for calculating the eigenvalues
        sparse : bool, optional
            if ``True``, `eigsh` will be called, else `eigh` will be
            called (default).
        **kwargs : dict, optional
            passed arguments to the `eigh` routine

        See Also
        --------
        eigh : eigenvalue routine
        eigsh : eigenvalue routine

        Returns
        -------
        EigenvaluePhonon
        """
        if kwargs.pop('sparse', False):
            hw = self.eigsh(k, gauge=gauge, eigvals_only=True, **kwargs)
        else:
            hw = self.eigh(k, gauge, eigvals_only=True, **kwargs)
        info = {'k': k, 'gauge': gauge}
        return EigenvaluePhonon(_correct_hw(hw), self, **info)

    def eigenmode(self, k=(0, 0, 0), gauge='R', **kwargs):
        """ Calculate the eigenmodes at `k` and return an `EigenmodePhonon` object containing all eigenmodes

        Parameters
        ----------
        k : array_like*3, optional
            the k-point at which to evaluate the eigenmodes at
        gauge : str, optional
            the gauge used for calculating the eigenmodes
        sparse : bool, optional
            if ``True``, `eigsh` will be called, else `eigh` will be
            called (default).
        **kwargs : dict, optional
            passed arguments to the `eigh` routine

        See Also
        --------
        eigh : eigenvalue routine
        eigsh : eigenvalue routine

        Returns
        -------
        EigenmodePhonon
        """
        if kwargs.pop('sparse', False):
            hw, v = self.eigsh(k, gauge=gauge, eigvals_only=False, **kwargs)
        else:
            hw, v = self.eigh(k, gauge, eigvals_only=False, **kwargs)
        info = {'k': k, 'gauge': gauge}
        # Since eigh returns the eigenvectors [:, i] we have to transpose
        return EigenmodePhonon(v.T, _correct_hw(hw), self, **info)

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads dynamical matrix from `Sile` using `read_dynamical_matrix`.

        Parameters
        ----------
        sile : Sile, str or pathlib.Path
            a `Sile` object which will be used to read the dynamical matrix.
            If it is a string it will create a new sile using `get_sile`.
        * : args passed directly to ``read_dynamical_matrix(,**)``
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_dynamical_matrix(*args, **kwargs)
        else:
            with get_sile(sile) as fh:
                return fh.read_dynamical_matrix(*args, **kwargs)

    def write(self, sile, *args, **kwargs):
        """ Writes a dynamical matrix to the `Sile` as implemented in the :code:`Sile.write_dynamical_matrix` method """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_dynamical_matrix(self, *args, **kwargs)
        else:
            with get_sile(sile, 'w') as fh:
                fh.write_dynamical_matrix(self, *args, **kwargs)

    def velocity(self, k=(0, 0, 0), **kwargs):
        r""" Calculate the velocity for the eigenmodes for a given `k` point

        Parameters
        ----------
        k : array_like, optional
            k-point at which the velocities are calculated
        **kwargs : optional
            additional parameters passed to the `eigenmode` routine

        See Also
        --------
        eigenmode : method used to calculate the eigenmodes
        displacement : Calculate mode displacements
        EigenmodePhonon.velocity : Underlying method used to calculate the velocity
        """
        return self.eigenmode(k, **kwargs).velocity()

    def displacement(self, k=(0, 0, 0), **kwargs):
        r""" Calculate the displacement for the eigenmodes for a given `k` point

        Parameters
        ----------
        k : array_like, optional
            k-point at which the displacement are calculated
        **kwargs : optional
            additional parameters passed to the `eigenmode` routine

        See Also
        --------
        eigenmode : method used to calculate the eigenmodes
        velocity : Calculate mode velocity
        EigenmodePhonon.displacement : Underlying method used to calculate the velocity
        """
        return self.eigenmode(k, **kwargs).displacement()

    def DOS(self, E, k=(0, 0, 0), distribution='gaussian', **kwargs):
        r""" Calculate the DOS at the given energies for a specific `k` point

        Parameters
        ----------
        E : array_like
            energies to calculate the DOS at
        k : array_like, optional
            k-point at which the DOS is calculated
        distribution : func or str, optional
            a function that accepts :math:`E-\hbar\omega` as argument and calculates the
            distribution function.
        **kwargs : optional
            additional parameters passed to the `eigenvalue` routine

        See Also
        --------
        sisl.physics.distribution : setup a distribution function, see details regarding the `distribution` argument
        eigenvalue : method used to calculate the eigenvalues
        PDOS : Calculate projected DOS
        EigenvaluePhonon.DOS : Underlying method used to calculate the DOS
        """
        return self.eigenvalue(k, **kwargs).DOS(E, distribution)

    def PDOS(self, E, k=(0, 0, 0), distribution='gaussian', **kwargs):
        r""" Calculate the projected DOS at the given energies for a specific `k` point

        Parameters
        ----------
        E : array_like
            energies to calculate the projected DOS at
        k : array_like, optional
            k-point at which the projected DOS is calculated
        distribution : func or str, optional
            a function that accepts :math:`E-\epsilon` as argument and calculates the
            distribution function.
        **kwargs : optional
            additional parameters passed to the `eigenmode` routine

        See Also
        --------
        sisl.physics.distribution : setup a distribution function, see details regarding the `distribution` argument
        eigenmode : method used to calculate the eigenmodes
        DOS : Calculate total DOS
        EigenmodePhonon.PDOS : Underlying method used to calculate the projected DOS
        """
        return self.eigenmode(k, **kwargs).PDOS(E, distribution)
