# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix

from sisl._internal import set_module
from sisl.typing import GaugeType, KPoint

from .phonon import EigenmodePhonon, EigenvaluePhonon
from .sparse import SparseOrbitalBZ

__all__ = ["DynamicalMatrix"]


def _correct_hw(hw):
    idx = (hw < 0).nonzero()[0]
    HW = hw.copy()
    HW[idx] *= -1
    HW[:] = np.sqrt(HW)
    HW[idx] *= -1
    return HW


@set_module("sisl.physics")
class DynamicalMatrix(SparseOrbitalBZ):
    r"""Dynamical matrix of a geometry

    The dynamical matrix is defined as the mass-reduced quantity.
    Hence the quantities stored in this matrix are expected to contain
    a factor:

    .. math::
        \frac1{\sqrt{M_IM_J}}

    for the elements that contains couplings between atoms :math:`I`
    and :math:`J`.
    """

    def __init__(self, geometry, dim=1, dtype=None, nnzpr=None, **kwargs):
        super().__init__(geometry, dim, dtype, nnzpr, **kwargs)
        self._reset()

    def _reset(self):
        super()._reset()
        self.Dk = self._Pk
        self.dDk = self.dPk
        self.ddDk = self.ddPk

    @property
    def D(self):
        r"""Access the dynamical matrix elements"""
        self._def_dim = 0
        return self

    def Dk(
        self,
        k: KPoint = (0, 0, 0),
        dtype=None,
        gauge: GaugeType = "lattice",
        format="csr",
        *args,
        **kwargs,
    ):
        r"""Setup the dynamical matrix for a given k-point

        Creation and return of the dynamical matrix for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \mathbf D(\mathbf k) = \mathbf D_{I_\alpha J_\beta} e^{i\mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`\alpha`, :math:`\beta` are Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \mathbf D(\mathbf k) = \mathbf D_{I_\alpha J_\beta} e^{i\mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is the distance between the atoms.

        Parameters
        ----------
        k :
           the k-point to setup the dynamical matrix at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge, `lattice` for lattice vector gauge, and `atomic` for atomic distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).
           Prefixing with 'sc:', or simply 'sc' returns the matrix in supercell format
           with phases.

        See Also
        --------
        dDk : dynamical matrix derivative with respect to `k`
        ddDk : dynamical matrix double derivative with respect to `k`

        Returns
        -------
        matrix : numpy.ndarray or scipy.sparse.*_matrix
            the dynamical matrix at :math:`\mathbf k`. The returned object depends on `format`.
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
        r"""Setup the dynamical matrix derivative for a given k-point

        Creation and return of the dynamical matrix derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_{\mathbf k} \mathbf D_\gamma(\mathbf k) = i \mathbf R_\gamma \mathbf D_{I_\alpha J_\beta} e^{i\mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`\alpha`, :math:`\beta` are atomic indices.
        And :math:`\gamma` is one of the Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
          \nabla_{\mathbf k} \mathbf D_\gamma(\mathbf k) = i \mathbf r_\gamma \mathbf D_{I_\alpha J_\beta} e^{i\mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is the distance between the atoms.

        Parameters
        ----------
        k :
           the k-point to setup the dynamical matrix at
        dtype : numpy.dtype , optional
           the data type of the returned matrix. Do NOT request non-complex
           data-type for non-Gamma k.
           The default data-type is `numpy.complex128`
        gauge :
           the chosen gauge, `lattice` for lattice vector gauge, and `atomic` for atomic distance
           gauge.
        format : {'csr', 'array', 'dense', 'coo', ...}
           the returned format of the matrix, defaulting to the `scipy.sparse.csr_matrix`,
           however if one always requires operations on dense matrices, one can always
           return in `numpy.ndarray` (`'array'`/`'dense'`/`'matrix'`).

        See Also
        --------
        Dk : dynamical matrix with respect to `k`
        ddDk : dynamical matrix double derivative with respect to `k`

        Returns
        -------
        tuple
            for each of the Cartesian directions a :math:`\partial \mathbf D(\mathbf k)/\partial \mathbf k_\gamma` is returned.
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
        r"""Setup the dynamical matrix double derivative for a given k-point

        Creation and return of the dynamical matrix double derivative for a given k-point (default to Gamma).

        Notes
        -----

        Currently the implemented gauge for the k-point is the cell vector gauge:

        .. math::
           \nabla_{\mathbf k^2} \mathbf D_{\gamma\sigma}(\mathbf k) = - \mathbf R_\gamma \mathbf R_\sigma \mathbf D_{I_\alpha J_\beta} e^{i\mathbf k\cdot\mathbf R}

        where :math:`\mathbf R` is an integer times the cell vector and :math:`\alpha`, :math:`\beta` are Cartesian directions.
        And :math:`\gamma`, :math:`\sigma` are one of the Cartesian directions.

        Another possible gauge is the atomic distance which can be written as

        .. math::
           \nabla_{\mathbf k^2} \mathbf D_{\gamma\sigma}(\mathbf k) = - \mathbf r_\gamma \mathbf r_\sigma \mathbf D_{I_\alpha J_\beta} e^{i\mathbf k\cdot\mathbf r}

        where :math:`\mathbf r` is atomic distance.

        Parameters
        ----------
        k :
           the k-point to setup the dynamical matrix at
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

        See Also
        --------
        Dk : dynamical matrix with respect to `k`
        dDk : dynamical matrix derivative with respect to `k`

        Returns
        -------
        list of matrices
            for each of the Cartesian directions (in Voigt representation); xx, yy, zz, zy, xz, xy
        """
        pass

    def apply_newton(self) -> None:
        """Sometimes the dynamical matrix does not obey Newtons 3rd law.

        We correct the dynamical matrix by imposing zero force.

        Correcting for Newton forces the matrix to be finalized.

        Notes
        -----
        This is an in-place operation.
        """
        # Create UC dynamical matrix
        dyn_sc = self.tocsr(0)
        no = self.no
        d_uc = lil_matrix((no, no), dtype=dyn_sc.dtype)

        for i, _ in self.lattice:
            d_uc[:, :] += dyn_sc[:, i * no : (i + 1) * no]

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

    def eigenvalue(
        self, k: KPoint = (0, 0, 0), gauge: GaugeType = "lattice", **kwargs
    ) -> EigenvaluePhonon:
        """Calculate the eigenvalues at `k` and return an `EigenvaluePhonon` object containing all eigenvalues for a given `k`

        Parameters
        ----------
        k :
            the k-point at which to evaluate the eigenvalues at
        gauge :
            the gauge used for calculating the eigenvalues
        sparse : bool, optional
            if ``True``, `eigsh` will be called, else `eigh` will be
            called (default).
        **kwargs : dict, optional
            passed arguments to the eigenvalue calculator routine

        See Also
        --------
        eigh : dense eigenvalue routine
        eigsh : sparse eigenvalue routine

        Returns
        -------
        EigenvaluePhonon
        """
        if kwargs.pop("sparse", False):
            hw = self.eigsh(k, gauge=gauge, eigvals_only=True, **kwargs)
        else:
            hw = self.eigh(k, gauge, eigvals_only=True, **kwargs)
        info = {"k": k, "gauge": gauge}
        return EigenvaluePhonon(_correct_hw(hw), self, **info)

    def eigenmode(
        self, k: KPoint = (0, 0, 0), gauge: GaugeType = "lattice", **kwargs
    ) -> EigenmodePhonon:
        r"""Calculate the eigenmodes at `k` and return an `EigenmodePhonon` object containing all eigenmodes

        Notes
        -----
        Note that the phonon modes are *not* mass-scaled.

        Parameters
        ----------
        k :
            the k-point at which to evaluate the eigenmodes at
        gauge :
            the gauge used for calculating the eigenmodes
        sparse : bool, optional
            if ``True``, `eigsh` will be called, else `eigh` will be
            called (default).
        **kwargs : dict, optional
            passed arguments to the eigenvalue calculator routine

        See Also
        --------
        eigh : dense eigenvalue routine (returns hw ** 2)
        eigsh : sparse eigenvalue routine (returns hw ** 2)

        Returns
        -------
        EigenmodePhonon
        """
        if kwargs.pop("sparse", False):
            hw, v = self.eigsh(k, gauge=gauge, eigvals_only=False, **kwargs)
        else:
            hw, v = self.eigh(k, gauge, eigvals_only=False, **kwargs)
        info = {"k": k, "gauge": gauge}
        # Since eigh returns the eigenvectors [:, i] we have to transpose
        return EigenmodePhonon(v.T, _correct_hw(hw), self, **info)

    @staticmethod
    def read(sile, *args, **kwargs) -> DynamicalMatrix:
        """Reads dynamical matrix from `Sile` using `read_dynamical_matrix`.

        Parameters
        ----------
        sile : Sile, str or pathlib.Path
            a `Sile` object which will be used to read the dynamical matrix.
            If it is a string it will create a new sile using `get_sile`.
        * : args passed directly to ``read_dynamical_matrix(,**)``
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import BaseSile, get_sile

        if isinstance(sile, BaseSile):
            return sile.read_dynamical_matrix(*args, **kwargs)
        else:
            with get_sile(sile, mode="r") as fh:
                return fh.read_dynamical_matrix(*args, **kwargs)
