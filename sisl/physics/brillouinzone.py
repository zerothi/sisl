# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Brillouin zone classes
=========================

The Brillouin zone objects are all special classes enabling easy manipulation
of an underlying physical quantity.

Quite often a physical quantity will be required to be averaged, or calculated individually
over a number of k-points. In this regard the Brillouin zone objects can help.

The BrillouinZone object allows direct looping of contained k-points while invoking
particular methods from the contained object.
This is best shown with an example:

>>> H = Hamiltonian(...)
>>> bz = BrillouinZone(H)
>>> bz.apply.array.eigh()

This will calculate eigenvalues for all k-points associated with the `BrillouinZone` and
return everything as an array. The `~sisl.physics.BrillouinZone.dispatch` property of
the `BrillouinZone` object has several use cases (here ``array`` is shown).

This may be extremely convenient when calculating band-structures:

>>> H = Hamiltonian(...)
>>> bs = BandStructure(H, [[0, 0, 0], [0.5, 0, 0]], 100)
>>> bs_eig = bs.apply.array.eigh()
>>> plt.plot(bs.lineark(), bs_eig)

and then you have all eigenvalues for all the k-points along the path.

Sometimes one may want to post-process the data for each k-point.
As an example lets post-process the DOS on a per k-point basis while
calculating the average:
 
>>> H = Hamiltonian(...)
>>> mp = MonkhorstPack(H, [10, 10, 10])
>>> E = np.linspace(-2, 2, 100)
>>> def wrap_DOS(eigenstate):
...    # Calculate the DOS for the eigenstates
...    DOS = eigenstate.DOS(E)
...    # Calculate the velocity for the eigenstates
...    v = eigenstate.velocity()
...    V = (v ** 2).sum(1)
...    return DOS.reshape(-1, 1) * v ** 2 / V.reshape(-1, 1)
>>> DOS = mp.apply.average.eigenstate(wrap=wrap_DOS, eta=True)

This will, calculate the Monkhorst pack k-averaged DOS split into 3 Cartesian
directions based on the eigenstates velocity direction. This method of manipulating
the result can be extremely powerful to calculate many quantities while running an
efficient `BrillouinZone` average. The `eta` flag will print, to stdout, a progress-bar.
The usage of the ``wrap`` method are also passed optional arguments, ``parent`` which is
``H`` in the above example. ``k`` and ``weight`` are the current k-point and weight of the
corresponding k-point. An example could be to manipulate the DOS depending on the k-point and
weight:

>>> H = Hamiltonian(...)
>>> mp = MonkhorstPack(H, [10, 10, 10])
>>> E = np.linspace(-2, 2, 100)
>>> def wrap_DOS(eigenstate, k, weight):
...    # Calculate the DOS for the eigenstates and weight by k_x and weight
...    return eigenstate.DOS(E) * k[0] * weight
>>> DOS = mp.apply.sum.eigenstate(wrap=wrap_DOS, eta=True)

When using wrap to calculate more than one quantity per eigenstate it may be advantageous
to use `~sisl.oplist` to handle cases of `BrillouinZone.apply.average` and `BrillouinZone.apply.sum`.

>>> H = Hamiltonian(...)
>>> mp = MonkhorstPack(H, [10, 10, 10])
>>> E = np.linspace(-2, 2, 100)
>>> def wrap_multiple(eigenstate):
...    # Calculate DOS/PDOS for eigenstates
...    DOS = eigenstate.DOS(E)
...    PDOS = eigenstate.PDOS(E)
...    # Calculate velocity for the eigenstates
...    v = eigenstate.velocity()
...    return oplist([DOS, PDOS, v])
>>> DOS, PDOS, v = mp.apply.average.eigenstate(wrap=wrap_multiple, eta=True)

Which does mathematical operations (averaging/summing) using `~sisl.oplist`.


Parallel calculations
---------------------

The ``apply`` method looping k-points may be explicitly parallelized.
To run parallel do:

>>> H = Hamiltonian(...)
>>> mp = MonkhorstPack(H, [10, 10, 10])
>>> with mp.apply.renew(pool=True) as par:
...     par.eigh()

This requires you also have the package ``pathos`` available.
The above will run in parallel using a default number of processors
in priority:

1. Environment variable ``SISL_NUM_PROCS``
2. Return value of ``os.cpu_count()``.

Note that this may interfere with BLAS implementation which defaults
to use all CPU's for threading. The total processors/threads that will
be created is ``SISL_NUM_PROCS * OMP_NUM_THREADS``. Try and ensure this is below
or equal to the actual core-count of your machine (or the number of requested
cores in a HPC environment).


Alternatively one can control the number of processors locally by doing:

>>> H = Hamiltonian(...)
>>> mp = MonkhorstPack(H, [10, 10, 10])
>>> with mp.apply.renew(pool=2) as par:
...     par.eigh()

which will request 2 processors (regardless of core-count).
As a last resort you can pass your own ``Pool`` of workers that
will be used for the parallel processing.

>>> from multiprocessing import Pool
>>> pool = Pool(4)
>>> H = Hamiltonian(...)
>>> mp = MonkhorstPack(H, [10, 10, 10])
>>> with mp.apply.renew(pool=pool) as par:
...     par.eigh()

The ``Pool`` should implement some standard methods that are
existing in the ``pathos`` enviroment such as ``Pool.restart`` and ``Pool.terminate``
and ``imap`` and ``uimap`` methods. See the ``pathos`` documentation for detalis.


   BrillouinZone
   MonkhorstPack
   BandStructure

"""

from numbers import Integral, Real
import itertools
from functools import reduce

from numpy import pi
from numpy import sum, dot, argsort
import numpy as np

from sisl._internal import set_module
from sisl.oplist import oplist
from sisl.unit import units
from sisl.quaternion import Quaternion
from sisl.utils.mathematics import cart2spher, fnorm
from sisl.utils import batched_indices
import sisl._array as _a
from sisl.messages import info, warn, SislError, progressbar
from sisl.supercell import SuperCell
from sisl.grid import Grid
from sisl._dispatcher import ClassDispatcher


__all__ = ["BrillouinZone", "MonkhorstPack", "BandStructure", "linspace_bz"]


class BrillouinZoneDispatcher(ClassDispatcher):
    r""" Loop over all k-points by applying `parent` methods for all k.

    This allows potential for running and collecting various computationally
    heavy methods from a single point on all k-points.

    The `apply` method will *dispatch* the parent methods through all k-points
    and passing `k` as arguments to the parent methods in a straight-forward manner.

    For instance to iterate over all eigenvalues of a Hamiltonian

    >>> H = Hamiltonian(...)
    >>> bz = BrillouinZone(H)
    >>> for ik, eigh in enumerate(bz.apply.eigh()):
    ...    # do something with eigh which corresponds to bz.k[ik]

    By default the `apply` method exposes a set of dispatch methods:

    - `apply.iter`, the default iterator module
    - `apply.average` reduced result by averaging (using `BrillouinZone.weight` as the weight per k-point.
    - `apply.sum` reduced result without weighing
    - `apply.array` return a single array with all values; has `len` equal to number of k-points
    - `apply.none`, specialized method that is mainly useful when wrapping methods
    - `apply.list` same as `apply.array` but using Python list as return value
    - `apply.oplist` using `sisl.oplist` allows greater flexibility for mathematical operations element wise
    - `apply.datarray` if `xarray` is available one can retrieve an `xarray.DataArray` instance

    Please see :ref:`physics.brillouinzone` for further examples.
    """
    pass


@set_module("sisl.physics")
def linspace_bz(bz, stop=None, jumps=None, jump_dk=0.05):
    r""" Convert points from a BZ object into a linear spacing of maximum value `stop`

    Parameters
    ----------
    bz : BrillouinZone, or ndarray
       the object containing the k-points
    stop : int or None, optional
       maximum value in the linear space, or if None, will return the cumulative
       distance of the k-points in the Brillouin zone
    jumps: array_like, optional
       whether there are any jumps for the k-points that should not be taken into account
    jump_dk: float or array_like, optional
       how much total distance the jump points will take

    Returns
    -------

    """
    if isinstance(bz, BrillouinZone):
        cart = bz.tocartesian(bz.k)
    else:
        cart = bz
    # calculate vectors between each neighbouring points
    dcart = np.diff(cart, axis=0, prepend=cart[0].reshape(1, -1))
    # calculate distances
    dist = (dcart ** 2).sum(1) ** 0.5

    if jumps is not None:
        # calculate the total distance
        total_dist = dist.sum()

        # Zero out the jumps
        dist[jumps] = 0.
        total_dist = dist.sum()
        # correct jumps
        dist[jumps] = total_dist * np.asarray(jump_dk)

    # convert to linear scale
    if stop is None:
        return np.cumsum(dist)

    total_dist = dist.sum() / stop
    # Scale to total length of `stop`
    return np.cumsum(dist) / total_dist


@set_module("sisl.physics")
class BrillouinZone:
    """ A class to construct Brillouin zone related quantities

    It takes any object (which has access to cell-vectors) as an argument
    and can then return the k-points in non-reduced units from reduced units.

    The object associated with the BrillouinZone object *has* to implement
    at least two different properties:

    1. `cell` which is the lattice vector
    2. `rcell` which is the reciprocal lattice vectors.

    The object may also be an array of floats in which case an internal
    `SuperCell` object will be created from the cell vectors (see `SuperCell` for
    details).

    Parameters
    ----------
    parent : object or array_like
       An object with associated ``parent.cell`` and ``parent.rcell`` or
       an array of floats which may be turned into a `SuperCell`
    k : array_like, optional
       k-points that this Brillouin zone represents
    weight : array_like, optional
       weights for the k-points. Must have the same length as `k`.
    """

    def __init__(self, parent, k=None, weight=None):
        self.set_parent(parent)
        # define a bz_attr as though it has not been set
        self._bz_attr = ("", None)

        # Gamma point
        if k is None:
            self._k = _a.zerosd([1, 3])
            self._w = _a.onesd(1)
        else:
            self._k = _a.arrayd(k).reshape(-1, 3)
            if weight is None:
                n = self._k.shape[0]
                self._w = _a.fulld(n, 1. / n)
            else:
                self._w = _a.arrayd(weight).ravel()
        if len(self.k) != len(self.weight):
            raise ValueError(f'{self.__class__.__name__}.__init__ requires input k-points and weights to be of equal length.')

    apply = BrillouinZoneDispatcher("apply",
                                    # Do not allow class dispatching
                                    type_dispatcher=None,
                                    obj_getattr=lambda obj, key: getattr(obj.parent, key))

    def set_parent(self, parent):
        """ Update the parent associated to this object

        Parameters
        ----------
        parent : object or array_like
           an object containing cell vectors
        """
        try:
            # It probably has the supercell attached
            parent.cell
            parent.rcell
            self.parent = parent
        except Exception:
            self.parent = SuperCell(parent)

    def __getstate__(self):
        """ Return dictionary with the current state """
        return {
            'parent_class': self.parent.__class__,
            'parent': self.parent.__getstate__(),
            'k': self._k.copy(),
            'weight': self._w.copy()
        }

    def __setstate__(self, state):
        """ Reset state of the object """
        self._k = state['k']
        self._w = state['weight']
        parent = state['parent_class'].__new__(state['parent_class'])
        parent.__setstate__(state['parent'])
        self.set_parent(parent)

    def __str__(self):
        """ String representation of the BrillouinZone """
        parent = self.parent
        if isinstance(parent, SuperCell):
            parent = str(parent).replace("\n", "\n ")
        else:
            parent = str(parent.sc).replace("\n", "\n ")
        return f"{self.__class__.__name__}{{nk: {len(self)},\n {parent}\n}}"

    def volume(self, ret_dim=False, periodic=None):
        """ Calculate the volume of the full Brillouin zone of the parent

        This will return the volume depending on the dimensions of the system.
        Here the dimensions of the system is determined by how many dimensions
        have auxilliary supercells that can contribute to Brillouin zone integrals.
        Therefore the returned value will have differing units depending on
        dimensionality.

        Parameters
        ----------
        ret_dim: bool, optional
           also return the dimensionality of the system
        periodic : array_like of int, optional
           estimate the volume using only the directions indexed by this array.
           The default value is `(self.parent.nsc > 1).nonzero()[0]`.

        Returns
        -------
        vol :
           the volume of the Brillouin zone. Units are Ang^D with D being the dimensionality.
           For 0D it will return 0.
        dimensionality : int
           the dimensionality of the volume
        """
        # default periodic array
        if periodic is None:
            periodic = (self.parent.nsc > 1).nonzero()[0]

        dim = len(periodic)
        vol = 0.
        if dim == 3:
            vol = self.parent.volume
        elif dim == 2:
            vol = self.parent.area(*periodic)
        elif dim == 1:
            vol = self.parent.length[periodic[0]]

        if ret_dim:
            return vol, dim
        return vol

    @staticmethod
    def parametrize(parent, func, N, *args, **kwargs):
        """ Generate a new `BrillouinZone` object with k-points parameterized via the function `func` in `N` separations

        Generator of a parameterized Brillouin zone object that contains a parameterized k-point
        list.

        Parameters
        ----------
        parent : SuperCell, or SuperCellChild
           the object that the returned object will contain as parent
        func : callable
           method that parameterizes the k-points, *must* at least accept three arguments,
           1. ``parent``: object
           2. ``N``: total number of k-points
           3. ``i``: current index of the k-point (starting from 0)

           the function must return a k-point in 3 dimensions.
        N : int or list of int
           number of k-points generated using the parameterization,
           or a list of integers that will be looped over.
           In this case arguments ``N`` and ``i`` in `func` will be
           lists accordingly.
        *args :
           additional arguments passed directly to `func`
        **kwargs :
           additional keyword arguments passed directly to `func`


        Examples
        --------
        Simple linear k-points

        >>> def func(sc, N, i):
        ...    return [i/N, 0, 0]
        >>> bz = BrillouinZone.parametrize(1, func, 10)
        >>> assert len(bz) == 10
        >>> assert np.allclose(bz.k[-1, :], [9./10, 0, 0])

        For double looping, say to create your own grid

        >>> def func(sc, N, i):
        ...    return [i[0]/N[0], i[1]/N[1], 0]
        >>> bz = BrillouinZone.parametrize(1, func, [10, 5])
        >>> assert len(bz) == 50

        """
        if isinstance(N, Integral):
            k = np.empty([N, 3], np.float64)
            for i in range(N):
                k[i] = func(parent, N, i, *args, **kwargs)
        else:
            # N must be some-kind of list like thingy
            Nk = np.prod(N)
            k = np.empty([Nk, 3], np.float64)
            for i, indices in enumerate(itertools.product(*map(range, N))):
                k[i] = func(parent, N, indices, *args, **kwargs)
        return BrillouinZone(parent, k)

    @staticmethod
    def param_circle(parent, N_or_dk, kR, normal, origin, loop=False):
        r""" Create a parameterized k-point list where the k-points are generated on a circle around an origin

        The generated circle is a perfect circle in the reciprocal space (Cartesian coordinates).
        To generate a perfect circle in units of the reciprocal lattice vectors one can
        generate the circle for a diagonal supercell with side-length :math:`2\pi`, see
        example below.

        Parameters
        ----------
        parent : SuperCell, or SuperCellChild
           the parent object
        N_or_dk : int
           number of k-points generated using the parameterization (if an integer),
           otherwise it specifies the discretization length on the circle (in 1/Ang),
           If the latter case will use less than 4 points a warning will be raised and
           the number of points increased to 4.
        kR : float
           radius of the k-point. In 1/Ang
        normal : array_like of float
           normal vector to determine the circle plane
        origin : array_like of float
           origin of the circle used to generate the circular parameterization
        loop : bool, optional
           whether the first and last point are equal

        Examples
        --------

        >>> sc = SuperCell([1, 1, 10, 90, 90, 60])
        >>> bz = BrillouinZone.param_circle(sc, 10, 0.05, [0, 0, 1], [1./3, 2./3, 0])

        To generate a circular set of k-points in reduced coordinates (reciprocal

        >>> sc = SuperCell([1, 1, 10, 90, 90, 60])
        >>> bz = BrillouinZone.param_circle(sc, 10, 0.05, [0, 0, 1], [1./3, 2./3, 0])
        >>> bz_rec = BrillouinZone.param_circle(2*np.pi, 10, 0.05, [0, 0, 1], [1./3, 2./3, 0])
        >>> bz.k[:, :] = bz_rec.k[:, :]

        Returns
        -------
        BrillouinZone
            with the parameterized k-points.
        """
        if isinstance(N_or_dk, Integral):
            N = N_or_dk
        else:
            # Calculate the required number of points
            N = int(kR ** 2 * pi / N_or_dk + 0.5)
            if N < 4:
                N = 4
                info('BrillouinZone.param_circle increased the number of circle points to 4.')

        # Conversion object
        bz = BrillouinZone(parent)

        normal = _a.asarrayd(normal)
        origin = _a.asarrayd(origin)
        k_n = bz.tocartesian(normal)
        k_o = bz.tocartesian(origin)

        # Generate a preset list of k-points on the unit-circle
        if loop:
            radians = _a.aranged(N) / (N-1) * 2 * np.pi
        else:
            radians = _a.aranged(N) / N * 2 * np.pi
        k = _a.emptyd([N, 3])
        k[:, 0] = np.cos(radians)
        k[:, 1] = np.sin(radians)
        k[:, 2] = 0.

        # Now generate the rotation
        _, theta, phi = cart2spher(k_n)
        if theta != 0:
            pv = _a.arrayd([k_n[0], k_n[1], 0])
            pv /= fnorm(pv)
            q = Quaternion(phi, pv, rad=True) * Quaternion(theta, [0, 0, 1], rad=True)
        else:
            q = Quaternion(0., [0, 0, k_n[2] / abs(k_n[2])], rad=True)

        # Calculate k-points
        k = q.rotate(k)
        k *= kR / fnorm(k).reshape(-1, 1)
        k = bz.toreduced(k + k_o)

        # The sum of weights is equal to the BZ area
        W = np.pi * kR ** 2
        w = np.repeat([W / N], N)

        return BrillouinZone(parent, k, w)

    def copy(self, parent=None):
        """ Create a copy of this object, optionally changing the parent

        Parameters
        ----------
        parent : optional
           change the parent
        """
        if parent is None:
            parent = self.parent
        bz = self.__class__(parent, self._k, self.weight)
        bz._k = self._k.copy()
        bz._w = self._w.copy()
        return bz

    @property
    def k(self):
        """ A list of all k-points (if available) """
        return self._k

    @property
    def weight(self):
        """ Weight of the k-points in the `BrillouinZone` object """
        return self._w

    @property
    def cell(self):
        return self.parent.cell

    @property
    def rcell(self):
        return self.parent.rcell

    def tocartesian(self, k):
        """ Transfer a k-point in reduced coordinates to the Cartesian coordinates

        Parameters
        ----------
        k : list of float
           k-point in reduced coordinates

        Returns
        -------
        numpy.ndarray
            in units of 1/Ang
        """
        return dot(k, self.rcell)

    def toreduced(self, k):
        """ Transfer a k-point in Cartesian coordinates to the reduced coordinates

        Parameters
        ----------
        k : list of float
           k-point in Cartesian coordinates

        Returns
        -------
        numpy.ndarray
            in units of reciprocal lattice vectors ]-0.5 ; 0.5] (if k is in the primitive cell)
        """
        return dot(k, self.cell.T / (2 * pi))

    @staticmethod
    def in_primitive(k):
        """ Move the k-point into the primitive point(s) ]-0.5 ; 0.5]

        Parameters
        ----------
        k : array_like
           k-point(s) to move into the primitive cell

        Returns
        -------
        numpy.ndarray
            all k-points moved into the primitive cell
        """
        k = _a.arrayd(k) % 1.

        # Ensure that we are in the interval ]-0.5; 0.5]
        k[k > 0.5] -= 1

        return k

    def iter(self, ret_weight=False):
        """ An iterator for the k-points and (possibly) the weights

        Parameters
        ----------
        ret_weight : bool, optional
          if true, also yield the weight for the respective k-point

        Yields
        ------
        kpt : k-point
        weight : weight of k-point, only if `ret_weight` is true.
       """
        if ret_weight:
            for i in range(len(self)):
                yield self.k[i], self.weight[i]
        else:
            yield from self.k

    __iter__ = iter

    def __len__(self):
        return len(self._k)

    def write(self, sile, *args, **kwargs):
        """ Writes k-points to a `~sisl.io.tableSile`.

        This allows one to pass a `tableSile` or a file-name.
        """
        from sisl.io import tableSile
        kw = np.concatenate((self.k, self.weight.reshape(-1, 1)), axis=1)
        if isinstance(sile, tableSile):
            sile.write_data(kw.T, *args, **kwargs)
        else:
            with tableSile(sile, 'w') as fh:
                fh.write_data(kw.T, *args, **kwargs)


@set_module("sisl.physics")
class MonkhorstPack(BrillouinZone):
    r""" Create a Monkhorst-Pack grid for the Brillouin zone

    Parameters
    ----------
    parent : object or array_like
       An object with associated `parent.cell` and `parent.rcell` or
       an array of floats which may be turned into a `SuperCell`
    nkpt : array_like of ints
       a list of number of k-points along each cell direction
    displacement : float or array_like of float, optional
       the displacement of the evenly spaced grid, a single floating
       number is the displacement for the 3 directions, else they
       are the individual displacements
    size : float or array_like of float, optional
       the size of the Brillouin zone sampled. This reduces the boundaries
       of the Brillouin zone around the displacement to the fraction specified.
       I.e. `size` must be of values :math:`]0 ; 1]`. Defaults to the entire BZ.
       Note that this will also reduce the weights such that the weights
       are normalized to the entire BZ.
    centered : bool, optional
       whether the k-points are :math:`\Gamma`-centered (for zero displacement)
    trs : bool, optional
       whether time-reversal symmetry exists in the Brillouin zone.

    Examples
    --------
    >>> sc = SuperCell(3.)
    >>> MonkhorstPack(sc, 10) # 10 x 10 x 10 (with TRS)
    >>> MonkhorstPack(sc, [10, 5, 5]) # 10 x 5 x 5 (with TRS)
    >>> MonkhorstPack(sc, [10, 5, 5], trs=False) # 10 x 5 x 5 (without TRS)
    """

    def __init__(self, parent, nkpt, displacement=None, size=None, centered=True, trs=True):
        super().__init__(parent)

        if isinstance(nkpt, Integral):
            nkpt = np.diag([nkpt] * 3)
        elif isinstance(nkpt[0], Integral):
            nkpt = np.diag(nkpt)

        # Now we have a matrix of k-points
        if np.any(nkpt - np.diag(np.diag(nkpt)) != 0):
            raise NotImplementedError(f"{self.__class__.__name__} with off-diagonal components is not implemented yet")

        if displacement is None:
            displacement = np.zeros(3, np.float64)
        elif isinstance(displacement, Real):
            displacement = _a.fulld(3, displacement)
        else:
            displacement = _a.asarrayd(displacement)

        if size is None:
            size = _a.onesd(3)
        elif isinstance(size, Real):
            size = _a.fulld(3, size)
        else:
            size = _a.asarrayd(size)

        # Retrieve the diagonal number of values
        Dn = np.diag(nkpt).astype(np.int32)
        if np.any(Dn) == 0:
            raise ValueError(f'{self.__class__.__name__} *must* be initialized with '
                             'diagonal elements different from 0.')

        i_trs = -1
        if trs:
            # Figure out which direction to TRS
            nmax = 0
            for i in [0, 1, 2]:
                if displacement[i] in [0., 0.5] and Dn[i] > nmax:
                    nmax = Dn[i]
                    i_trs = i
            if nmax == 1:
                i_trs = -1
            if i_trs == -1:
                # If we still haven't decided (say for weird displacements)
                # simply take the one with the maximum number of k-points.
                i_trs = np.argmax(Dn)

        # Calculate k-points and weights along all directions
        kw = [self.grid(Dn[i], displacement[i], size[i], centered, i == i_trs) for i in (0, 1, 2)]

        # Now figure out if we have a 0 point along the TRS direction
        if trs:
            # Figure out if the first value is zero
            if abs(kw[i_trs][0][0]) < 1e-10:
                # Find indices we want to delete
                ik1, ik2 = (i_trs + 1) % 3, (i_trs + 2) % 3
                k1, k2 = kw[ik1][0], kw[ik2][0]
                k_dup = _a.emptyd([k1.size, k2.size, 2])
                k_dup[:, :, 0] = k1.reshape(-1, 1)
                k_dup[:, :, 1] = k2.reshape(1, -1)
                # Figure out the duplicate values
                # To do this we calculate the norm matrix
                # Note for a 100 x 100 k-point sampling this will produce
                # a 100 ^ 4 matrix ~ 93 MB
                # For larger k-point samplings this is probably not so good (300x300 -> 7.5 GB)
                k_dup = k_dup.reshape(k1.size, k2.size, 1, 1, 2) + k_dup.reshape(1, 1, k1.size, k2.size, 2)
                k_dup = ((k_dup[..., 0] ** 2 + k_dup[..., 1] ** 2) ** 0.5 < 1e-10).nonzero()
                # At this point we have found all duplicate points, to only take one
                # half of the points we only take the lower half
                # Also, the Gamma point is *always* zero, so we shouldn't do <=!
                # Now check the case where one of the directions is (only) the Gamma-point
                if kw[ik1][0].size == 1 and kw[ik1][0][0] == 0.:
                    # We keep all indices for the ik1 direction (since it is the Gamma-point!
                    rel = (k_dup[1] > k_dup[3]).nonzero()[0]
                elif kw[ik2][0].size == 1 and kw[ik2][0][0] == 0.:
                    # We keep all indices for the ik2 direction (since it is the Gamma-point!
                    rel = (k_dup[0] > k_dup[2]).nonzero()[0]
                else:
                    rel = np.logical_and(k_dup[0] > k_dup[2], k_dup[1] > k_dup[3])
                k_dup = (k_dup[0][rel], k_dup[1][rel], k_dup[2][rel], k_dup[3][rel])
                del rel, k1, k2
            else:
                # To signal we can't do this
                k_dup = None

        self._k = _a.emptyd((kw[0][0].size, kw[1][0].size, kw[2][0].size, 3))
        self._w = _a.onesd(self._k.shape[:-1])
        for i in (0, 1, 2):
            k = kw[i][0].reshape(-1, 1, 1)
            w = kw[i][1].reshape(-1, 1, 1)
            self._k[..., i] = np.rollaxis(k, 0, i + 1)
            self._w[...] *= np.rollaxis(w, 0, i + 1)

        del kw
        # Now clean up a few of the points
        if trs and k_dup is not None:
            # Create the correct indices in the ravelled indices
            k = [0] * 3
            k[ik1] = k_dup[2]
            k[ik2] = k_dup[3]
            k_del = np.ravel_multi_index(tuple(k), self._k.shape[:-1])
            k[ik1] = k_dup[0]
            k[ik2] = k_dup[1]
            k_dup = np.ravel_multi_index(tuple(k), self._k.shape[:-1])
            del k

        self._k.shape = (-1, 3)
        self._w.shape = (-1,)

        if trs and k_dup is not None:
            self._k = np.delete(self._k, k_del, 0)
            self._w[k_dup] += self._w[k_del]
            self._w = np.delete(self._w, k_del)
            del k_dup, k_del

        # Store information regarding size and diagonal elements
        # This information is basically only necessary when
        # we want to replace special k-points
        self._diag = Dn # vector
        self._displ = displacement # vector
        self._size = size # vector
        self._centered = centered
        self._trs = i_trs

    def __str__(self):
        """ String representation of MonkhorstPack """
        if isinstance(self.parent, SuperCell):
            p = self.parent
        else:
            p = self.parent.sc
        return ('{cls}{{nk: {nk:d}, size: [{size[0]:.5f} {size[1]:.5f} {size[0]:.5f}], trs: {trs},'
                '\n diagonal: [{diag[0]:d} {diag[1]:d} {diag[2]:d}], displacement: [{disp[0]:.5f} {disp[1]:.5f} {disp[2]:.5f}],'
                '\n {sc}\n}}').format(cls=self.__class__.__name__, nk=len(self),
                                      size=self._size, trs={0: 'A', 1: 'B', 2: 'C'}.get(self._trs, 'no'),
                                      diag=self._diag, disp=self._displ, sc=str(p).replace('\n', '\n '))

    def __getstate__(self):
        """ Return dictionary with the current state """
        state = super().__getstate__()
        state['diag'] = self._diag
        state['displ'] = self._displ
        state['size'] = self._size
        state['centered'] = self._centered
        state['trs'] = self._trs
        return state

    def __setstate__(self, state):
        """ Reset state of the object """
        super().__setstate__(state)
        self._diag = state['diag']
        self._displ = state['displ']
        self._size = state['size']
        self._centered = state['centered']
        self._trs = state['trs']

    def copy(self, parent=None):
        """ Create a copy of this object, optionally changing the parent

        Parameters
        ----------
        parent : optional
           change the parent
        """
        if parent is None:
            parent = self.parent
        bz = self.__class__(parent, self._diag, self._displ, self._size, self._centered, self._trs >= 0)
        # this is required due to replace calls
        bz._k = self._k.copy()
        bz._w = self._w.copy()
        return bz

    @classmethod
    def grid(cls, n, displ=0., size=1., centered=True, trs=False):
        r""" Create a grid of `n` points with an offset of `displ` and sampling `size` around `displ`

        The :math:`k`-points are :math:`\Gamma` centered.

        Parameters
        ----------
        n : int
           number of points in the grid. If `trs` is ``True`` this may be smaller than `n`
        displ : float, optional
           the displacement of the grid
        size : float, optional
           the total size of the Brillouin zone to sample
        centered : bool, optional
           if the points are centered
        trs : bool, optional
           whether time-reversal-symmetry is applied

        Returns
        -------
        k : numpy.ndarray
           the list of k-points in the Brillouin zone to be sampled
        w : numpy.ndarray
           weights for the k-points
        """
        # First ensure that displ is in the Brillouin
        displ = displ % 1.
        if displ > 0.5:
            displ -= 1.
        if displ < -0.5:
            displ += 1.

        # Centered _only_ has effect IFF
        #  displ == 0. and size == 1
        # Otherwise we resort to other schemes
        if displ != 0. or size != 1.:
            centered = False

        # size *per k-point*
        dsize = size / n

        # We create the full grid, then afterwards we figure out TRS
        n_half = n // 2
        if n % 2 == 1:
            k = _a.aranged(-n_half, n_half + 1) * dsize + displ
        else:
            k = _a.aranged(-n_half, n_half) * dsize + displ
            if not centered:
                # Shift everything by halve the size each occupies
                k += dsize / 2

        # Move k to the primitive cell and generate weights
        k = cls.in_primitive(k)
        w = _a.fulld(n, dsize)

        # Check for TRS points
        if trs and np.any(k < 0.):
            # Make all positive to remove the double conting terms
            k_pos = np.fabs(k)

            # Sort k-points and weights
            idx = argsort(k_pos)

            # Re-arange according to k value
            k_pos = k_pos[idx]
            w = w[idx]

            # Find indices of all equivalent k-points (tolerance of 1e-10 in reciprocal units)
            # Use the dsize to estimate the difference in positions
            idx_same = (np.diff(k_pos) < dsize * 1e-3).nonzero()[0]

            # The above algorithm should never create more than two duplicates.
            # Hence we can simply remove all idx_same and double the weight for all
            # idx_same + 1.
            w[idx_same + 1] *= 2
            # Delete the duplicated k-points (they are already sorted)
            k = np.delete(k_pos, idx_same, axis=0)
            w = np.delete(w, idx_same)
        else:
            # Sort them, because it makes more visual sense
            idx = argsort(k)
            k = k[idx]
            w = w[idx]

        # Return values
        return k, w

    def replace(self, k, mp, displacement=False, as_index=False, check_vol=True):
        r""" Replace a k-point with a new set of k-points from a Monkhorst-Pack grid

        This method tries to replace an area corresponding to `mp.size` around the k-point `k`
        such that the k-points are replaced.
        This enables one to zoom in on specific points in the Brillouin zone for detailed analysis.

        Parameters
        ----------
        k : array_like
           k-point in this object to replace, if `as_index` is true, it will be regarded as integer
           positions of the k-points to replace, otherwise the indices of the k-points will be located
           individually (in chunks of 200 MB).
        mp : MonkhorstPack
           object containing the replacement k-points.
        displacement : array_like or bool, optional
           the displacment of the `mp` k-points. Needed for doing *lots* of replacements due to efficiency.
           Defaults to not displace anything. The inserted k-points will be `mp.k + displacement`.
           If True, it will use `k` as the displacement vector. For multiple k-point replacements
           each k-point will be replaced my `mp` with k as the displacement.
        as_index : bool, optional
           whether `k` is input as reciprocal k-points, or as indices of k-points in this object.
        check_vol : bool, optional
           whether to check the volume of the replaced k-point(s); by default the volume of each k-point
           is determined by the original ``size`` and ``nkpt`` values. However, when doing
           replacements of k-points these values are not kept for the individual k-points
           that were replaced, so subsequent replacements of these points will cause errors that
           effectively are not valid.

        Examples
        --------

        This example creates a zoomed-in view of the :math:`\Gamma`-point by replacing it with
        a 3x3x3 Monkhorst-Pack grid.

        >>> sc = SuperCell(1.)
        >>> mp = MonkhorstPack(sc, [3, 3, 3])
        >>> mp.replace([0, 0, 0], MonkhorstPack(sc, [3, 3, 3], size=1./3))

        This example creates a zoomed-in view of the :math:`\Gamma`-point by replacing it with
        a 4x4x4 Monkhorst-Pack grid.

        >>> sc = SuperCell(1.)
        >>> mp = MonkhorstPack(sc, [3, 3, 3])
        >>> mp.replace([0, 0, 0], MonkhorstPack(sc, [4, 4, 4], size=1./3))

        This example creates a zoomed-in view of the :math:`\Gamma`-point by replacing it with
        a 4x4x1 Monkhorst-Pack grid.

        >>> sc = SuperCell(1.)
        >>> mp = MonkhorstPack(sc, [3, 3, 3])
        >>> mp.replace([0, 0, 0], MonkhorstPack(sc, [4, 4, 1], size=1./3))

        Raises
        ------
        SislError
            if the size of the replacement `MonkhorstPack` grid is not compatible with the k-point spacing in this object.
        """
        # First we find all k-points within k +- mp.size
        # Those are the points we wish to remove.
        # Secondly we need to ensure that the k-points we remove are occupying *exactly*
        # the Brillouin zone we wish to replace.
        if not isinstance(mp, MonkhorstPack):
            raise ValueError("Object 'mp' is not a MonkhorstPack object")

        if check_vol:
            # We can easily figure out the BZ that each k-point is averaging
            k_vol = self._size / self._diag
            print(np.prod(k_vol), self.weight.min())

            # Compare against the size of this one
            # Since we can remove more than one k-point, we require that the
            # size of the replacement MP is an integer multiple of the
            # k-point volumes.
            k_int = mp._size / k_vol
            if not np.allclose(np.rint(k_int), k_int):
                raise SislError(f"{self.__class__.__name__}.reduce could not replace k-point, BZ "
                                "volume replaced is not equivalent to the inherent k-point volume.")

        # the size of the k-points that will be added
        s_size2 = self._size / 2
        mp_size2 = mp._size / 2
        dk = np.where(mp_size2 < s_size2, mp_size2, s_size2)
        dk.shape = (1, 3)

        # determine indices of k-point inputs
        k = np.asarray(k)

        if as_index:
            idx = k.ravel()
            k = self.k[idx]
        else:
            # find k-points in batches of 200 MB
            k = k.reshape(-1, 3)
            idx = batched_indices(self.k, k, atol=dk, batch_size=200,
                                  diff_func=self.in_primitive)[0]

        # Idea of fast replacements is attributed @ahkole in #454, but the resulting code needed some
        # changes since that code was not stable againts *wrong input*, i.e. k=[0, 0, 0]
        # replacements.

        # determine the displacement vector
        if isinstance(displacement, bool):
            if displacement:
                displacement = k
            else:
                displacement = None

        elif displacement is not None:
            # convert to array
            displacement = _a.asarray(displacement).reshape(-1, 3)

        if displacement is None:
            displ_nk = 1
        else:
            displ_nk = len(displacement)

        # Now we have the k-points we need to remove
        # Figure out if the total weight is consistent
        total_weight = self.weight[idx].sum()
        replace_weight = mp.weight.sum() * displ_nk
        atol = min(total_weight, replace_weight) * 1e-4
        if abs(total_weight - replace_weight) < atol:
            weight_factor = 1.
        elif abs(total_weight - replace_weight * 2) < atol:
            weight_factor = 2.
            if self._trs < 0:
                info(f"{self.__class__.__name__}.reduce assumes that the replaced k-point has double weights.")
        else:
            #print("k-point to replace: ", k.ravel())
            #print("delta-k: ", dk.ravel())
            #print("Found k-indices that will be replaced:")
            #print(idx)
            #print("k-points replaced:")
            #print(self.k[idx, :])
            #print("weights replaced:")
            #print(self.weight[idx])
            #print(self.weight.min(), self.weight.max())
            #print(mp.weight.min(), mp.weight.max())
            #print("Summed weights vs. replaced summed weights: ")
            #print(total_weight, replace_weight)
            #print(mp)
            raise SislError(f"{self.__class__.__name__}.reduce found inconsistent replacement weights "
                            f"self={total_weight} vs. mp={replace_weight}.")

        # delete and append new k-points and weights
        if displacement is None:
            self._k = np.concatenate((np.delete(self._k, idx, axis=0), mp._k), axis=0)
        else:
            self._k = np.concatenate((np.delete(self._k, idx, axis=0),
                                      self.in_primitive(mp.k + displacement.reshape(-1, 1, 3)).reshape(-1, 3)), axis=0)
        self._w = np.concatenate((np.delete(self._w, idx), np.tile(mp._w * weight_factor, displ_nk)))


@set_module("sisl.physics")
class BandStructure(BrillouinZone):
    """ Create a path in the Brillouin zone for plotting band-structures etc.

    Parameters
    ----------
    parent : object or array_like
       An object with associated `parent.cell` and `parent.rcell` or
       an array of floats which may be turned into a `SuperCell`
    points : array_like of float
       a list of points that are the *corners* of the path
    divisions : int or array_like of int
       number of divisions in each segment.
       If a single integer is passed it is the total number
       of points on the path (equally separated).
       If it is an array_like input it must have length one
       less than `point`, in this case the total number of points
       will be ``sum(divisions) + 1`` due to the end-point constraint.
    names : array_like of str
       the associated names of the points on the Brillouin Zone path
    jump_dk: float or array_like, optional
       Percentage of ``self.lineark()[-1]`` that is used as separation between discontinued
       jumps in the band-structure.
       For band-structures with disconnected jumps the `lineark` and `lineartick` methods
       returns a separation between the disconnected points according to this percentage.
       Default value is 5% of the total distance. Alternatively an array equal to the
       number of discontinuity jumps may be passed for individual percentages.
       Keyword only, argument.

    Examples
    --------
    >>> sc = SuperCell(10)
    >>> bs = BandStructure(sc, [[0] * 3, [0.5] * 3], 200)
    >>> bs = BandStructure(sc, [[0] * 3, [0.5] * 3, [1.] * 3], 200)
    >>> bs = BandStructure(sc, [[0] * 3, [0.5] * 3, [1.] * 3], 200, ['Gamma', 'M', 'Gamma'])

    A disconnected band structure may be created by either having a point of 0 length, or None.
    Note that the number of names does not contain the empty points (they are simply removed).
    Such a band-structure may be useful when one is not interested in a fully connected band structure.

    >>> bs = BandStructure(sc, [[0, 0, 0], [0, 0.5, 0], None, [0.5, 0, 0], [0.5, 0.5, 0]], 200)
    """

    def __init__(self, parent, *args, **kwargs):
        #points, divisions, names=None):
        super().__init__(parent)

        points = kwargs.get("points")
        if points is None:
            if len(args) > 0:
                points, *args = args
            else:
                raise ValueError(f"{self.__class__.__name__} 'points' argument missing")

        divisions = kwargs.get("divisions")
        if divisions is None:
            if len(args) > 0:
                divisions, *args = args
            else:
                raise ValueError(f"{self.__class__.__name__} 'divisions' argument missing")

        names = kwargs.get("names")
        if names is None:
            if len(args) > 0:
                names, *args = args

        if len(args) > 0:
            raise ValueError(f"{self.__class__.__name__} unknown arguments after parsing 'points', 'divisions' and 'names': {args}")

        # Store empty split size
        self._jump_dk = np.asarray(kwargs.get("jump_dk", 0.05))

        # Copy over points
        # Check if any of the points is None or has length 0
        # In that case it is a disconnected path
        def is_empty(ix):
            try:
                return len(ix[1]) == 0
            except Exception:
                return ix[1] is None

        # filter out jump directions
        jump_idx = _a.arrayi([i for i, _ in filter(is_empty, enumerate(points))])

        # store only *valid* points
        self.points = _a.arrayd([p for i, p in enumerate(points) if i not in jump_idx])

        # remove erroneous jumps
        if len(points) - 1 in jump_idx:
            jump_idx = jump_idx[:-1]
        if 0 in jump_idx:
            jump_idx = jump_idx[1:]

        if self._jump_dk.size > 1 and jump_idx.size != self._jump_dk.size:
            raise ValueError(f"{self.__class__.__name__} got inconsistent argument lengths (jump_dk does not match jumps in points)")

        # The jump-idx is equal to using np.split(self.points, jump_idx)
        # which then returns continuous sections
        # correct for removed indices
        jump_idx -= np.arange(len(jump_idx))
        self._jump_idx = jump_idx

        # If the array has fewer points we try and determine
        if self.points.shape[1] < 3:
            if self.points.shape[1] != np.sum(self.parent.nsc > 1):
                raise ValueError('Could not determine the non-periodic direction')

            # fix the points where there are no periodicity
            for i in (0, 1, 2):
                if self.parent.nsc[i] == 1:
                    self.points = np.insert(self.points, i, 0., axis=1)

        # Ensure the shape is correct
        self.points.shape = (-1, 3)

        # Now figure out what to do with the divisions
        if isinstance(divisions, Integral):

            if divisions < len(self.points):
                raise ValueError(f"Can not evenly split {len(self.points)} points into {divisions} divisions, ensure division>=len(points)")

            # Get length between different k-points with a total length
            # of division
            dists = np.diff(linspace_bz(self.tocartesian(self.points), jumps=jump_idx, jump_dk=0.))

            # Get floating point divisions
            divs_r = dists * divisions / dists.sum()
            # Convert to integers
            divs = np.rint(divs_r).astype(np.int32)
            # ensure at least 1 point along each division
            # 1 division means only the starting point
            divs[divs == 0] = 1
            divs[jump_idx-1] = 1
            divs_sum = divs.sum()
            while divs_sum != divisions - 1:
                # only check indices where divs > 1
                idx = (divs > 1).nonzero()[0]
                dk = dists[idx] / divs[idx]
                if divs_sum >= divisions:
                    divs[idx[np.argmin(dk)]] -= 1
                else:
                    divs[idx[np.argmax(dk)]] += 1
                divs_sum = divs.sum()

            divisions = divs[:]

        elif len(divisions) + 1 != len(self.points):
            raise ValueError(f"inconsistent number of elements in 'points' and 'divisions' argument. One less 'divisions' elements.")

        self.divisions = _a.arrayi(divisions).ravel()

        if names is None:
            self.names = 'ABCDEFGHIJKLMNOPQRSTUVXYZ'[:len(self.points)]
        else:
            self.names = names
        if len(self.names) != len(self.points):
            raise ValueError(f"inconsistent number of elements in 'points' and 'names' argument")

        # Calculate points
        dpoint = np.diff(self.points, axis=0)
        k = _a.emptyd([self.divisions.sum() + 1, 3])
        i = 0
        for ik, (divs, dk) in enumerate(zip(self.divisions, dpoint)):
            k[i:i+divs, :] = self.points[ik] + dk * _a.aranged(divs).reshape(-1, 1) / divs
            i += divs
        k[-1] = self.points[-1]
        # sanity check that should always be obeyed
        assert i + 1 == len(k)

        self._k = k
        self._w = _a.fulld(len(self.k), 1 / len(self.k))

    def copy(self, parent=None):
        """ Create a copy of this object, optionally changing the parent

        Parameters
        ----------
        parent : optional
           change the parent
        """
        if parent is None:
            parent = self.parent
        bz = self.__class__(parent, self.points, self.divisions, self.names, jump_dk=self._jump_dk)
        return bz

    def __getstate__(self):
        """ Return dictionary with the current state """
        state = super().__getstate__()
        state['points'] = self.points.copy()
        state['divisions'] = self.divisions.copy()
        state['jump_idx'] = self._jump_idx.copy()
        state['names'] = list(self.names)
        state['jump_dk'] = self._jump_dk
        return state

    def __setstate__(self, state):
        """ Reset state of the object """
        super().__setstate__(state)
        self.points = state['points']
        self.divisions = state['divisions']
        self.names = state['names']
        self._jump_dk = state['jump_dk']
        self._jump_idx = state['jump_idx']

    def insert_jump(self, *arrays, value=np.nan):
        """ Return a copy of `arrays` filled with `value` at indices of discontinuity jumps

        Arrays with `value` in jumps is easier to plot since those lines will be naturally discontinued.
        For band structures without discontinuity jumps in the Brillouin zone the `arrays` will
        be return as is.

        It will insert `value` along the first dimension matching the length of `self`.
        For each discontinuity jump an element will be inserted.

        This may be useful for plotting since `np.nan` gets interpreted as a discontinuity
        in the graph thus removing connections between the segments.

        Parameters
        ----------
        *arrays : array_like
           arrays will get `value` inserted where there are jumps in the band structure
        value : optional
           the value to be inserted at the jump points in the data array

        Examples
        --------
        Create a bandrstructure with a discontinuity.

        >>> gr = geom.graphene()
        >>> bs = BandStructure(gr, [[0, 0, 0], [0.5, 0, 0], None, [0, 0, 0], [0, 0.5, 0]], 4)
        >>> data = np.zeros([len(bs), 10])
        >>> data_with_jump = bs.insert_jump(data)
        >>> assert data_with_jump.shape == (len(bs)+1, 10)
        >>> np.all(data_with_jump[2] == np.nan)
        True
        """
        # quick return if nothing needs changed
        if len(self._jump_idx) == 0:
            if len(arrays) == 1:
                return arrays[0]
            return arrays

        nk = len(self)
        full_jumps = np.cumsum(self.divisions)[self._jump_idx-1]
        def _insert(array):
            array = np.asarray(array)
            # ensure dtype is equivalent as input array
            nans = np.empty(len(full_jumps), dtype=array.dtype)
            nans.fill(value)
            axis = array.shape.index(nk)
            shape = list(1 for _ in array.shape)
            shape[axis] = -1
            return np.insert(array, full_jumps, nans.reshape(shape), axis=axis)

        # convert all
        arrays = tuple(_insert(array) for array in arrays)
        if len(arrays) == 1:
            return arrays[0]
        return arrays

    def lineartick(self):
        """ The tick-marks corresponding to the linear-k values

        Returns
        -------
        numpy.ndarray
            the positions in reciprocal space determined by the distance between points

        See Also
        --------
        lineark : Routine used to calculate the tick-marks.
        """
        return self.lineark(True)[1:3]

    def tolinear(self, k, ret_index=False, tol=1e-4):
        """ Convert a k-point into the equivalent linear k-point via the distance

        Finds the index of the k-point in `self.k` that is closests to `k`.
        The returned value is then the equivalent index in `lineark`.

        This is very useful for extracting certain points along the band structure.

        Parameters
        ----------
        k : array_like
           the k-point(s) to locate in the linear values
        ret_index : bool, optional
           whether the indices are also returned
        tol : float, optional
           when the found k-point has a distance (in Cartesian coordinates)
           is differing by more than `tol` a warning will be issued.
           The tolerance is in units 1/Ang.
        """
        # Faster than to do sqrt all the time
        tol = tol ** 2
        # first convert to the cartesian coordinates (for proper distances)
        ks = self.tocartesian(np.atleast_2d(k))
        kk = self.tocartesian(self.k)

        # find closest values
        def find(k):
            dist = ((kk - k) ** 2).sum(-1)
            idx = np.argmin(dist)
            if dist[idx] > tol:
                warn(f"{self.__class__.__name__}.tolinear could not find a k-point within given tolerance ({self.toreduced(k)})")
            return idx

        idxs = [find(k) for k in ks]
        if ret_index:
            return self.lineark()[idxs], idxs
        return self.lineark()[idxs]

    def lineark(self, ticks=False):
        """ A 1D array which corresponds to the delta-k values of the path

        This is mainly meant for plotting but may be useful for finding out
        distances in the reciprocal lattice.

        Examples
        --------

        >>> p = BandStructure(...)
        >>> eigs = Hamiltonian.eigh(p)
        >>> for i in range(len(Hamiltonian)):
        ...     plt.plot(p.lineark(), eigs[:, i])

        >>> p = BandStructure(...)
        >>> eigs = Hamiltonian.eigh(p)
        >>> lk, kt, kl = p.lineark(True)
        >>> plt.xticks(kt, kl)
        >>> for i in range(len(Hamiltonian)):
        ...     plt.plot(lk, eigs[:, i])

        Parameters
        ----------
        ticks : bool, optional
           if `True` the ticks for the points are also returned

        See Also
        --------
        linspace_bz : converts k-points into a linear distance parameterization

        Returns
        -------
        linear_k : numpy.ndarray
            the positions in reciprocal space determined by the distance between points
        ticks : numpy.ndarray
            linear k-positions of the points, only returned if `ticks` is ``True``
        ticklabels : list of str
            labels at `ticks`, only returned if `ticks` is ``True``
        """
        cum_divs = np.cumsum(self.divisions)
        # Calculate points
        # First we also need to calculate the jumps
        dK = linspace_bz(self, jumps=cum_divs[self._jump_idx-1], jump_dk=self._jump_dk)

        # Get label tick, in case self.names is a single string 'ABCD'
        if ticks:
            # Get number of points
            xtick = np.zeros(len(self.points), dtype=int)
            xtick[1:] = cum_divs
            # Ensure the returned label_tick is a copy
            return dK, dK[xtick], [a for a in self.names]
        return dK
