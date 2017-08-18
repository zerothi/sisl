"""
Self-energy class for calculating self-energies.
"""
from __future__ import print_function, division

import warnings

import numpy as np
from numpy import dot
from scipy.linalg import solve

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

    def __call__(self, *args, **kwargs):
        """ Return the calculated self-energy """
        raise NotImplementedError


class SemiInfinite(SelfEnergy):
    """ Self-energy object able to calculate the dense self-energy for a given `SparseGeometry` in a semi-infinite chain. """

    def __init__(self, spgeom, infinite, eta=1e-6, bloch=None):
        """ Create a `SelfEnergy` object from any `SparseGeometry`

        This enables the calculation of the self-energy for a semi-infinite chain.

        Parameters
        ----------
        spgeom : SparseGeometry
           any sparse geometry matrix which may return matrices
        infinite : str
           axis specification for the semi-infinite direction (`+A`/`-A`/`+B`/`-B`/`+C`/`-C`)
        eta : float, optional
           the default imaginary part of the self-energy calculation
        bloch : array_like, optional
           Bloch-expansion for each of the lattice vectors (`1` for no expansion)
           The resulting self-energy will have dimension
           equal to `len(obj) * np.product(bloch)`.
        """
        self.eta = eta
        if bloch is None:
            self.bloch = np.ones([3], np.int16)
        else:
            self.bloch = np.array(bloch, np.int16)

        # Determine whether we are in plus/minus direction
        if infinite.startswith('+'):
            self.semi_inf_dir = 1
        elif infinite.startswith('-'):
            self.semi_inf_dir = -1
        else:
            raise ValueError("SemiInfinite: infinite keyword does not start with `+` or `-`.")

        # Determine the direction
        INF = infinite.upper()
        if INF.endswith('A'):
            self.semi_inf = 0
        elif INF.endswith('B'):
            self.semi_inf = 1
        elif INF.endswith('C'):
            self.semi_inf = 2

        # Check that the Hamiltonian does have a non-zero V along the semi-infinite direction
        if spgeom.geom.sc.nsc[self.semi_inf] == 1:
            raise ValueError(("SemiInfinite: SparseGeometry does not have couplings along the "
                              "semi-infinite direction."))

        # Finalize the setup by calling the class specific routine
        self._setup(spgeom)

    def _correct_k(self, k=None):
        """ Return a corrected k-point

        Note
        ----
        This is strictly not required because any k along the semi-infinite direction
        is *integrated* out and thus the self-energy is the same for all k along the
        semi-infinite direction.
        """
        if k is None:
            k = np.zeros([3], np.float64)
        else:
            k = self._fill(k, np.float64)
            k[self.semi_inf] = 0.
        return k


class RecursiveSI(SemiInfinite):
    """ Self-energy object using the Lopez-Sancho Lopez-Sancho algorithm """

    def _setup(self, spgeom):
        """ Setup the Lopez-Sancho internals for easy axes """

        # Create spgeom0 and spgeom1
        self.spgeom0 = orig.copy()
        nsc = np.copy(self.geom.sc.nsc)
        nsc[self.semi_inf] = 1
        self.spgeom0.set_nsc(nsc)

        # For spgeom1 we have to do it slightly differently
        old_nnz = spgeom.nnz
        self.spgeom1 = spgeom.copy()
        nsc[self.semi_inf] = 3

        # Already now limit the sparse matrices
        self.spgeom1.set_nsc(nsc)
        if self.spgeom1.nnz < old_nnz:
            warnings.warn(("RecursiveSI: SparseGeometry has connections across the first neighbouring cell. "
                           "These values will be forced to 0 as the principal cell-interaction is a requirement"))

        # I.e. we will delete all interactions that are un-important
        n_s = self.spgeom1.geom.sc.n_s
        n = self.spgeom1.shape[0]
        # Figure out the matrix columns we should set to zero
        nsc = [None] * 3
        nsc[self.semi_inf] = self.semi_inf_dir
        # Get all supercell indices that we should delete
        idx = np.delete(n_.arangei(n_s),
                        n_.arrayi(spgeom.geom.sc.sc_index(nsc)))

        cols = array_arange(idx * n, (idx + 1) * n)
        # Delete all values in columns, but keep them to retain
        self.spgeom1._csr.delete_columns(cols, keep=True)

    def __call__(self, E, k=None, eta=None, dtype=None, eps=1e-14):
        """ Return a dense matrix with the self-energy at energy `E` and k-point `k` (default Gamma).

        Parameters
        ----------
        E : float
          energy at which the calculation will take place (should *not* be complex)
        k : array_like, optional
          k-point at which the self-energy should be evaluated.
          the k-point should be in units of the reciprocal lattice vectors, and
          the semi-infinite component will be automatically set to zero.
        eta : float, optional
          the imaginary value to evaluate the self-energy with. Defaults to the
          value with which the object was created
        dtype : numpy.dtype
          the resulting data type
        eps : float, optional
          convergence criteria for the recursion
        """
        if eta is None:
            eta = self.eta
        E = E + 1j * eta

        # Get k-point
        k = self._correct_k(k)

        if dtype is None:
            dtype = np.complex128

        # It could be that we are dealing with a
        # non-orthogonal system
        sp0 = self.spgeom0
        sp1 = self.spgeom1
        kw = {'dtype': dtype,
              'format': 'array'}
        def herm(m):
            return np.transpose(np.conjugate(m))
        try:
            if sp0.orthogonal:
                raise AttributeError('pass')
            # non-orthogonal case
            GB = sp0.Sk(k, **kw) * E - sp0.Pk(k, **kw)

            M = sp1.Pk(k, dtype=dtype)
            S = sp1.Sk(k, dtype=dtype)
            alpha = (M - S * E).asformat('array')
            beta  = (M.getH() - S.getH() * E).asformat('array')
            del M, S
        except AttributeError:
            # Bulk Green function
            GB = - sp0.Pk(k, **kw)
            GB += np.eye(len(GB), dtype=dtype) * E

            # The two forward/backward arrays (orthogonal basis-set)
            alpha = sp1.Pk(k, **kw)
            beta = alpha.transpose().conj().copy()

        # Surface Green function (self-energy)
        GS = np.zeros_like(GB)

        i = 0
        while True:
            i += 1

            # Do not allow overwrite
            tA = solve(GB, alpha, overwrite_a=False, overwrite_b=False)
            tB = solve(GB, beta, overwrite_a=False, overwrite_b=False)

            tmp = dot(alpha, tB)
            # Update surface self-energy
            GS += tmp
            # Update bulk Green function
            GB -= tmp + dot(beta, tA)

            # Update forward/backward
            alpha = dot(alpha, tA)
            beta = dot(beta, tB)

            # Convergence criteria, it could be stricter
            if np.amax(np.abs(alpha) + np.abs(beta)) < eps:
                # Return the pristine Green function
                return GS

        raise ValueError('SemiInfinite: could not converge self-energy calculation')
