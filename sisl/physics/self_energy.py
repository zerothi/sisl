"""
Self-energy class for calculating self-energies.
"""
from __future__ import print_function, division

import warnings
from numbers import Integral

import numpy as np
import scipy.linalg as sli
import scipy.sparse.linalg as ssli

from sisl._help import get_dtype, is_python3
from .hamiltonian import Hamiltonian

__all__ = ['SelfEnergy', 'SemiInfinite']


if not is_python3:
    from itertools import izip as zip


class SelfEnergy(object):
    """ Self-energy object able to calculate the dense self-energy for a given Hamiltonian.

    The self-energy object contains a `Hamiltonian` object which, in it-self
    contains the geometry.

    This is the base class for self-energies.
    """

    def __init__(self, *args, **kwargs):
        """ Self-energy class for constructing a self-energy. """
        pass

    def __call__(self, *args, **kwargs):
        """ Return the calculated self-energy """
        raise NotImplementedError


class SemiInfinite(SelfEnergy):
    """ Self-energy object able to calculate the dense self-energy for a given Hamiltonian in a semi-infinite chain.

    The self-energy object contains a `Hamiltonian` object which, in it-self
    contains the geometry.
    """

    def __init__(self, hamiltonian, infinite, eta=1e-6, bloch=None):
        """ Create a SelfEnergy object from a Hamiltonian.

        This enables the calculation of the self-energy for a semi-infinite chain.

        Parameters
        ----------
        hamiltonian : `Hamiltonian`
           the Hamiltonian of the semi-infinite chain
        infinite : `str`
           axis specification for the semi-infinite direction (`+A`/`-A`/`+B`/`-B`/`+C`/`-C`)
        eta : `float=1e-6`
           the default imaginary part of the self-energy calculation
        bloch : `array_like=[1,1,1]`
           Bloch-expansion for each of the lattice vectors (`1` for no expansion)
           The resulting self-energy will have dimension
           equal to `hamiltonian.no * np.product(bloch)`.
        """

        self.hamiltonian = hamiltonian
        self.geom = self.hamiltonian.geom
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
        sc_off = [0] * 3
        sc_off[self.semi_inf] = self.semi_inf_dir
        try:
            self.geom.sc.sc_index(sc_off)
        except:
            raise ValueError(("SemiInfinite: Hamiltonian does not have couplings along the "
                              "semi-infinite direction."))

        # Try and see if we have connections extending more than 1
        sc_off[self.semi_inf] = self.semi_inf_dir * 2
        try:
            self.geom.sc.sc_index(sc_off)
            warnings.warn(("SemiInfinite: Hamiltonian has connections across the first neighbouring cell. "
                           "These values will be forced to 0 as the principal cell-interaction is a requirement"))
        except:
            pass # GOOD, no connections across the first coupling

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

    def __call__(self, E, k=None, eta=None, dtype=None, eps=None):
        """ Return a dense matrix with the self-energy at energy `E` and k-point `k` (default Gamma).

        Parameters
        ----------
        E : `float`
          energy at which the calculation will take place
        k : `array_like=[0,0,0]`
          k-point at which the self-energy should be evaluated.
          the k-point should be in units of the reciprocal lattice vectors, and
          the semi-infinite component will be automatically set to zero.
        eta : `float=<>`
          the imaginary value to evaluate the self-energy with. Defaults to the initial
          value
        eps : float
          convergence criteria.
        """
        if eta is None:
            eta = self.eta
        E = E + 1j * eta

        # Get k-point
        k = self._correct_k(k)

        # Get H0 and H1
        sc0 = [None, None, None]
        sc1 = [None, None, None]
        sc0[self.semi_inf] = 0
        sc1[self.semi_inf] = self.semi_inf_dir

        # Now we may calculate the actual self-energy
        return self._Sancho(E, dtype=dtype, eps=eps)

    def _Sancho(self, H0, H1, E, dtype=None, eps=1e-14):
        """ Calculate the self-energy according to the Sancho-Sancho algorithm """

        # Faster calls
        dot = np.dot
        solve = la.solve

        if dtype is None:
            dtype = np.complex128

        if H0.orthogonal:
            # Bulk Green function
            GB = np.eye(len(H0)) * E - H0.Hk(k, dtype=dtype)

            # The two forward/backward arrays (orthogonal basis-set)
            alpha = H1.Hk(k, dtype=dtype).todense()
            beta = alpha.T.conj().copy()

        else:
            # non-orthogonal case

            GB = H0.Sk(k, dtype=dtype) * E - H0.Hk(k, dtype=dtype).todense()

            H = H1.Hk(k, dtype=dtype)
            S = H1.Sk(k, dtype=dtype)
            alpha = H.todense() - S.todense() * E
            beta  = H.T.conj().todense() - S.T.conj().todense() * E
            del H, S

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
