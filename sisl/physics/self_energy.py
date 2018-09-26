from __future__ import print_function, division

import numpy as np
from numpy import dot, amax, conjugate, zeros_like
from numpy import abs as _abs


from sisl.messages import warn
from sisl.utils.ranges import array_arange
import sisl._array as _a
import sisl.linalg as lin
from sisl.linalg import solve

__all__ = ['SelfEnergy', 'SemiInfinite']
__all__ += ['RecursiveSI']


class SelfEnergy(object):
    """ Self-energy object able to calculate the dense self-energy for a given sparse matrix

    The self-energy object contains a `SparseGeometry` object which, in it-self
    contains the geometry.

    This is the base class for self-energies.
    """

    def __init__(self, *args, **kwargs):
        """ Self-energy class for constructing a self-energy. """
        pass

    def _setup(self, *args, **kwargs):
        """ Class specific setup routine """
        pass

    def self_energy(self, E, k=(0, 0, 0)):
        raise NotImplementedError

    def __getattr__(self, attr):
        """ Overload attributes from the hosting object """
        pass


class SemiInfinite(SelfEnergy):
    """ Self-energy object able to calculate the dense self-energy for a given `SparseGeometry` in a semi-infinite chain.

    Parameters
    ----------
    spgeom : SparseGeometry
       any sparse geometry matrix which may return matrices
    infinite : str
       axis specification for the semi-infinite direction (`+A`/`-A`/`+B`/`-B`/`+C`/`-C`)
    eta : float, optional
       the default imaginary part of the self-energy calculation
    """

    def __init__(self, spgeom, infinite, eta=1e-6):
        """ Create a `SelfEnergy` object from any `SparseGeometry` """
        self.eta = eta

        # Determine whether we are in plus/minus direction
        if infinite.startswith('+'):
            self.semi_inf_dir = 1
        elif infinite.startswith('-'):
            self.semi_inf_dir = -1
        else:
            raise ValueError(self.__class__.__name__ + ": infinite keyword does not start with `+` or `-`.")

        # Determine the direction
        INF = infinite.upper()
        if INF.endswith('A'):
            self.semi_inf = 0
        elif INF.endswith('B'):
            self.semi_inf = 1
        elif INF.endswith('C'):
            self.semi_inf = 2

        # Check that the Hamiltonian does have a non-zero V along the semi-infinite direction
        if spgeom.geometry.sc.nsc[self.semi_inf] == 1:
            warn('Creating a semi-infinite self-energy with no couplings along the semi-infinite direction')

        # Finalize the setup by calling the class specific routine
        self._setup(spgeom)

    def __str__(self):
        """ String representation of the SemiInfinite """
        direction = {-1: '-', 1: '+'}
        axis = {0: 'A', 1: 'B', 2: 'C'}
        return self.__class__.__name__ + '{{direction: {0}{1}\n}}'.format(
            direction[self.semi_inf_dir], axis[self.semi_inf])

    def _correct_k(self, k=None):
        """ Return a corrected k-point

        Notes
        -----
        This is strictly not required because any `k` along the semi-infinite direction
        is *integrated* out and thus the self-energy is the same for all k along the
        semi-infinite direction.
        """
        if k is None:
            k = _a.zerosd([3])
        else:
            k = self._fill(k, np.float64)
            k[self.semi_inf] = 0.
        return k


class RecursiveSI(SemiInfinite):
    """ Self-energy object using the Lopez-Sancho Lopez-Sancho algorithm """

    def __getattr__(self, attr):
        """ Overload attributes from the hosting object """
        return getattr(self.spgeom0, attr)

    def __str__(self):
        """ Representation of the RecursiveSI model """
        direction = {-1: '-', 1: '+'}
        axis = {0: 'A', 1: 'B', 2: 'C'}
        return self.__class__.__name__ + '{{direction: {0}{1},\n {2}\n}}'.format(
            direction[self.semi_inf_dir], axis[self.semi_inf],
            str(self.spgeom0).replace('\n', '\n '),
        )

    def _setup(self, spgeom):
        """ Setup the Lopez-Sancho internals for easy axes """

        # Create spgeom0 and spgeom1
        self.spgeom0 = spgeom.copy()
        nsc = np.copy(spgeom.geometry.sc.nsc)
        nsc[self.semi_inf] = 1
        self.spgeom0.set_nsc(nsc)

        # For spgeom1 we have to do it slightly differently
        old_nnz = spgeom.nnz
        self.spgeom1 = spgeom.copy()
        nsc[self.semi_inf] = 3

        # Already now limit the sparse matrices
        self.spgeom1.set_nsc(nsc)
        if self.spgeom1.nnz < old_nnz:
            warn("RecursiveSI: SparseGeometry has connections across the first neighbouring cell. "
                 "These values will be forced to 0 as the principal cell-interaction is a requirement")

        # I.e. we will delete all interactions that are un-important
        n_s = self.spgeom1.geometry.sc.n_s
        n = self.spgeom1.shape[0]
        # Figure out the matrix columns we should set to zero
        nsc = [None] * 3
        nsc[self.semi_inf] = self.semi_inf_dir
        # Get all supercell indices that we should delete
        idx = np.delete(_a.arangei(n_s),
                        _a.arrayi(spgeom.geometry.sc.sc_index(nsc)))

        cols = array_arange(idx * n, (idx + 1) * n)
        # Delete all values in columns, but keep them to retain the supercell information
        self.spgeom1._csr.delete_columns(cols, keep_shape=True)

    def self_energy(self, E, k=None, dtype=None, eps=1e-14, bulk=False):
        r""" Return a dense matrix with the self-energy at energy `E` and k-point `k` (default Gamma).

        Parameters
        ----------
        E : float/complex
          energy at which the calculation will take place
        k : array_like, optional
          k-point at which the self-energy should be evaluated.
          the k-point should be in units of the reciprocal lattice vectors, and
          the semi-infinite component will be automatically set to zero.
        dtype : numpy.dtype
          the resulting data type
        eps : float, optional
          convergence criteria for the recursion
        bulk : bool, optional
          if true, :math:`E\cdot \mathbf S - \mathbf H -\boldsymbol\Sigma` is returned, else
          :math:`\boldsymbol\Sigma` is returned (default).

        Returns
        -------
        self-energy : the self-energy corresponding to the semi-infinite direction
        """
        if E.imag == 0.:
            E = E.real + 1j * self.eta

        # Get k-point
        k = self._correct_k(k)

        if dtype is None:
            dtype = np.complex128

        sp0 = self.spgeom0
        sp1 = self.spgeom1

        # As the SparseGeometry inherently works for
        # orthogonal and non-orthogonal basis, there is no
        # need to have two algorithms.
        GB = (sp0.Sk(k, dtype=dtype) * E - sp0.Pk(k, dtype=dtype)).toarray()

        if sp1.orthogonal:
            alpha = sp1.Pk(k, dtype=dtype, format='array')
            beta  = conjugate(alpha.T)
        else:
            M = sp1.Pk(k, dtype=dtype)
            S = sp1.Sk(k, dtype=dtype)
            alpha = (M - S * E).toarray()
            beta  = (M.getH() - S.getH() * E).toarray()
            del M, S

        # Surface Green function (self-energy)
        if bulk:
            GS = GB.copy()
        else:
            GS = zeros_like(GB)

        while True:
            tA = solve(GB, alpha)
            tB = solve(GB, beta)

            tmp = dot(alpha, tB)
            # Update bulk Green function
            GB -= tmp + dot(beta, tA)
            # Update surface self-energy
            GS -= tmp

            # Update forward/backward
            alpha = dot(alpha, tA)
            beta = dot(beta, tB)

            # Convergence criteria, it could be stricter
            if _abs(tmp).max() < eps:
                # Return the pristine Green function
                del tA, tB, alpha, beta, GB
                if bulk:
                    return GS
                return - GS

        raise ValueError(self.__class__.__name__+': could not converge self-energy calculation')

    def self_energy_lr(self, E, k=None, dtype=None, eps=1e-14, bulk=False):
        r""" Return two dense matrices with the left/right self-energy at energy `E` and k-point `k` (default Gamma).

        Note calculating the LR self-energies simultaneously requires that their chemical potentials are the same.
        I.e. only when the reference energy is equivalent in the left/right schemes does this make sense.

        Parameters
        ----------
        E : float/complex
          energy at which the calculation will take place, if complex, the hosting ``eta`` won't be used.
        k : array_like, optional
          k-point at which the self-energy should be evaluated.
          the k-point should be in units of the reciprocal lattice vectors, and
          the semi-infinite component will be automatically set to zero.
        dtype : numpy.dtype
          the resulting data type
        eps : float, optional
          convergence criteria for the recursion
        bulk : bool, optional
          if true, :math:`E\cdot \mathbf S - \mathbf H -\boldsymbol\Sigma` is returned, else
          :math:`\boldsymbol\Sigma` is returned (default).

        Returns
        -------
        left : the left self-energy
        right : the right self-energy
        """
        if E.imag == 0.:
            E = E.real + 1j * self.eta

        # Get k-point
        k = self._correct_k(k)

        if dtype is None:
            dtype = np.complex128

        sp0 = self.spgeom0
        sp1 = self.spgeom1

        # As the SparseGeometry inherently works for
        # orthogonal and non-orthogonal basis, there is no
        # need to have two algorithms.
        SmH0 = sp0.Sk(k, dtype=dtype) * E - sp0.Pk(k, dtype=dtype)
        GB = SmH0.toarray()

        if sp1.orthogonal:
            alpha = sp1.Pk(k, dtype=dtype, format='array')
            beta  = conjugate(alpha.T)
        else:
            M = sp1.Pk(k, dtype=dtype)
            S = sp1.Sk(k, dtype=dtype)
            alpha = (M - S * E).toarray()
            beta  = (M.getH() - S.getH() * E).toarray()
            del M, S

        # Surface Green function (self-energy)
        if bulk:
            GS = GB.copy()
        else:
            GS = zeros_like(GB)

        while True:
            tA = solve(GB, alpha)
            tB = solve(GB, beta)

            tmp = dot(alpha, tB)
            # Update bulk Green function
            GB -= tmp + dot(beta, tA)
            # Update surface self-energy
            GS -= tmp

            # Update forward/backward
            alpha = dot(alpha, tA)
            beta = dot(beta, tB)

            # Convergence criteria, it could be stricter
            if _abs(tmp).max() < eps:
                # Return the pristine Green function
                del tA, tB, alpha, beta
                if self.semi_inf_dir == 1:
                    # GS is the "right" self-energy
                    if bulk:
                        return GB - GS + SmH0.toarray(), GS
                    return GS - GB + SmH0.toarray(), - GS
                # GS is the "left" self-energy
                if bulk:
                    return GS, GB - GS + SmH0.toarray()
                return - GS, GS - GB + SmH0.toarray()

        raise ValueError(self.__class__.__name__+': could not converge self-energy (LR) calculation')
