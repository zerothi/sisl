from __future__ import print_function, division

import types
from numbers import Integral, Real

from numpy import pi
import numpy as np
from numpy import sum, dot

import sisl._array as _a
from sisl.messages import info, SislError, tqdm_eta
from sisl.supercell import SuperCell


__all__ = ['BrillouinZone', 'MonkhorstPack', 'BandStructure']


def _do_nothing(x):
    return x


class BrillouinZone(object):
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
    """

    def __init__(self, parent):
        try:
            # It probably has the supercell attached
            parent.cell
            parent.rcell
            self.parent = parent
        except:
            self.parent = SuperCell(parent)

        # Gamma point
        self._k = _a.zerosd([1, 3])
        self._w = _a.onesd(1)

        # Instantiate the array call
        self.asarray()

    def __repr__(self):
        """ String representation of the BrillouinZone """
        if isinstance(self.parent, SuperCell):
            return self.__class__.__name__ + '{{nk: {},\n {}\n}}'.format(len(self), repr(self.parent).replace('\n', '\n '))
        return self.__class__.__name__ + '{{nk: {},\n {}\n}}'.format(len(self), repr(self.parent.sc).replace('\n', '\n '))

    def set_parent(self, parent):
        """ Update the parent associated to this object

        Parameter
        ---------
        parent : object
           an object containing cell vectors
        """
        self.parent = parent

    def copy(self):
        """ Create a copy of this object """
        bz = self.__class__(self.parent)
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
        """
        return dot(k, self.rcell)

    def toreduced(self, k):
        """ Transfer a k-point in Cartesian coordinates to the reduced coordinates

        Parameters
        ----------
        k : list of float
           k-point in Cartesian coordinates
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
        k : all k-points moved into the primitive cell
        """
        k = _a.arrayd(k) % 1.

        # Ensure that we are in the interval ]-0.5; 0.5]
        idx = (k.ravel() > 0.5).nonzero()[0]
        while len(idx) > 0:
            k[np.unravel_index(idx, k.shape)] -= 1.
            idx = (k.ravel() > 0.5).nonzero()[0]

        idx = (k.ravel() < -0.5).nonzero()[0]
        while len(idx) > 0:
            k[np.unravel_index(idx, k.shape)] += 1.
            idx = (k.ravel() < -0.5).nonzero()[0]

        return k

    __attr = None

    def __getattr__(self, attr):
        try:
            getattr(self.parent, attr)
            self.__attr = attr
            return self
        except AttributeError:
            raise AttributeError("'{}' does not exist in class '{}'".format(
                attr, self.parent.__class__.__name__))

    # Implement wrapper calls
    def asarray(self, dtype=np.float64):
        """ Return `self` with `numpy.ndarray` returned quantities

        This forces the `__call__` routine to return a single array.

        Notes
        -----
        All invocations of sub-methods are added these keyword-only arguments:

        eta : bool, optional
           if true a progress-bar is created, default false.
        wrap : callable, optional
           a function that accepts the output of the given routine and post-process
           it. Defaults to ``lambda x: x``.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            the data-type to cast the values to

        Examples
        --------
        >>> obj = BrillouinZone(...) # doctest: +SKIP
        >>> obj.asarray().eigh(eta=True) # doctest: +SKIP

        See Also
        --------
        asyield : all output returned through an iterator
        asaverage : take the average (with k-weights) of the Brillouin zone
        aslist : all output returned as a Python list
        """

        def _call(self, *args, **kwargs):
            func = getattr(self.parent, self.__attr)
            wrap = kwargs.pop('wrap', _do_nothing)
            eta = tqdm_eta(len(self), self.__class__.__name__ + '.asarray',
                           'k', kwargs.pop('eta', False))
            for i, k in enumerate(self):
                if i == 0:
                    v = wrap(func(*args, k=k, **kwargs))
                    if len(self) == 1:
                        return v
                    shp = [len(self)]
                    shp.extend(v.shape)
                    a = np.empty(shp, dtype=dtype)
                    a[i, :] = v[:]
                    del v
                else:
                    a[i, :] = wrap(func(*args, k=k, **kwargs))
                eta.update()
            eta.close()
            return a
        # Set instance __call__
        setattr(self, '__call__', types.MethodType(_call, self))
        return self

    def asnone(self):
        """ Return `self` with None, this may be done for instance when wrapping the function calls.

        This forces the `__call__` routine to return ``None``. This usage is mainly intended when
        creating custom `wrap` function calls.

        Notes
        -----
        All invocations of sub-methods are added these keyword-only arguments:

        eta : bool, optional
           if true a progress-bar is created, default false.
        wrap : callable, optional
           a function that accepts the output of the given routine and post-process
           it. Defaults to ``lambda x: x``.

        Examples
        --------
        >>> obj = BrillouinZone(...) # doctest: +SKIP
        >>> obj.asnone().eigh(eta=True) # doctest: +SKIP

        See Also
        --------
        asyield : all output returned through an iterator
        asaverage : take the average (with k-weights) of the Brillouin zone
        aslist : all output returned as a Python list
        """

        def _call(self, *args, **kwargs):
            func = getattr(self.parent, self.__attr)
            wrap = kwargs.pop('wrap', _do_nothing)
            eta = tqdm_eta(len(self), self.__class__.__name__ + '.asnone',
                           'k', kwargs.pop('eta', False))
            for i, k in enumerate(self):
                wrap(func(*args, k=k, **kwargs))
                eta.update()
            eta.close()
        # Set instance __call__
        setattr(self, '__call__', types.MethodType(_call, self))
        return self

    def aslist(self):
        """ Return `self` with `list` returned quantities

        This forces the `__call__` routine to return a list with returned values.

        Notes
        -----
        All invocations of sub-methods are added these keyword-only arguments:

        eta : bool, optional
           if true a progress-bar is created, default false.
        wrap : callable, optional
           a function that accepts the output of the given routine and post-process
           it. Defaults to ``lambda x: x``.

        Examples
        --------
        >>> obj = BrillouinZone(...) # doctest: +SKIP
        >>> def first_ten(es):
        ...    return es.sub(range(10))
        >>> obj.aslist().eigenstate(eta=True, wrap=first_ten) # doctest: +SKIP

        See Also
        --------
        asarray : all output as a single array
        asyield : all output returned through an iterator
        asaverage : take the average (with k-weights) of the Brillouin zone
        """

        def _call(self, *args, **kwargs):
            func = getattr(self.parent, self.__attr)
            wrap = kwargs.pop('wrap', _do_nothing)
            eta = tqdm_eta(len(self), self.__class__.__name__ + '.aslist',
                           'k', kwargs.pop('eta', False))
            a = [None] * len(self)
            for i, k in enumerate(self):
                a[i] = wrap(func(*args, k=k, **kwargs))
                eta.update()
            eta.close()
            return a
        # Set instance __call__
        setattr(self, '__call__', types.MethodType(_call, self))
        return self

    def asyield(self, dtype=np.float64):
        """ Return `self` with yielded quantities

        This forces the `__call__` routine to return a an iterator which may
        yield the quantities calculated.

        Notes
        -----
        All invocations of sub-methods are added these keyword-only arguments:

        eta : bool, optional
           if true a progress-bar is created, default false.
        wrap : callable, optional
           a function that accepts the output of the given routine and post-process
           it. Defaults to ``lambda x: x``.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            the data-type to cast the values to

        Examples
        --------
        >>> obj = BrillouinZone(Hamiltonian) # doctest: +SKIP
        >>> obj.asyield().eigh(eta=True) # doctest: +SKIP

        See Also
        --------
        asarray : all output as a single array
        asaverage : take the average (with k-weights) of the Brillouin zone
        aslist : all output returned as a Python list
        """

        def _call(self, *args, **kwargs):
            func = getattr(self.parent, self.__attr)
            wrap = kwargs.pop('wrap', _do_nothing)
            eta = tqdm_eta(len(self), self.__class__.__name__ + '.asyield',
                           'k', kwargs.pop('eta', False))
            for k in self:
                yield wrap(func(*args, k=k, **kwargs).astype(dtype, copy=False))
                eta.update()
            eta.close()
        # Set instance __call__
        setattr(self, '__call__', types.MethodType(_call, self))
        return self

    def asaverage(self, dtype=np.float64):
        """ Return `self` with k-averaged quantities

        This forces the `__call__` routine to return a an iterator which may
        yield the quantities calculated.

        Notes
        -----
        All invocations of sub-methods are added these keyword-only arguments:

        eta : bool, optional
           if true a progress-bar is created, default false.
        wrap : callable, optional
           a function that accepts the output of the given routine and post-process
           it. Defaults to ``lambda x: x``.

        Parameters
        ----------
        dtype : numpy.dtype, optional
            the data-type to cast the values to

        Examples
        --------
        >>> obj = BrillouinZone(Hamiltonian) # doctest: +SKIP
        >>> obj.asaverage().DOS(np.linspace(-2, 2, 100)) # doctest: +SKIP

        >>> obj = BrillouinZone(Hamiltonian) # doctest: +SKIP
        >>> obj.asaverage() # doctest: +SKIP
        >>> obj.DOS(np.linspace(-2, 2, 100)) # doctest: +SKIP
        >>> obj.PDOS(np.linspace(-2, 2, 100), eta=True) # doctest: +SKIP

        See Also
        --------
        asarray : all output as a single array
        asyield : all output returned through an iterator
        aslist : all output returned as a Python list
        """

        def _call(self, *args, **kwargs):
            func = getattr(self.parent, self.__attr)
            wrap = kwargs.pop('wrap', _do_nothing)
            eta = tqdm_eta(len(self), self.__class__.__name__ + '.asaverage',
                           'k', kwargs.pop('eta', False))
            w = self.weight.view()
            for i, k in enumerate(self):
                if i == 0:
                    v = wrap(func(*args, k=k, **kwargs)) * w[i]
                else:
                    v += wrap(func(*args, k=k, **kwargs)) * w[i]
                eta.update()
            eta.close()
            return v
        # Set instance __call__
        setattr(self, '__call__', types.MethodType(_call, self))
        return self

    def __call__(self, *args, **kwargs):
        """ Calls the given attribute of the internal object and returns the quantity

        Parameters
        ----------
        *args : optional
            arguments passed to the attribute call, note that an argument `k=k` will be
            added by this routine as a way to loop the k-points.
        **kwargs : optional
            keyword arguments passed to the attribute call, note that the first argument
            will *always* be `k`

        Returns
        -------
        getattr(self, attr)(k, *args, **kwargs) : whatever this returns
        """
        try:
            call = getattr(self, '__call__')
        except Exception:
            raise NotImplementedError("Could not call the object it self")
        return call(*args, **kwargs)

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
            for k in self.k:
                yield k

    __iter__ = iter

    def __len__(self):
        return len(self._k)

    def write(self, sile, *args, **kwargs):
        """ Writes k-points to a `~sisl.io.tableSile`.

        This allows one to pass a `tableSile` or a file-name.
        """
        from sisl.io import tableSile
        kw = np.concatenate((self._k, self._w.reshape(-1, 1)), axis=1)
        if isinstance(sile, tableSile):
            sile.write_data(kw.T, *args, **kwargs)
        else:
            with tableSile(sile, 'w') as fh:
                fh.write_data(kw.T, *args, **kwargs)


class MonkhorstPack(BrillouinZone):
    r""" Create a Monkhorst-Pack grid for the Brillouin zone

    Parameters
    ----------
    parent : object or array_like
       An object with associated `parent.cell` and `parent.rcell` or
       an array of floats which may be turned into a `SuperCell`
    nktp : array_like of ints
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
        super(MonkhorstPack, self).__init__(parent)

        if isinstance(nkpt, Integral):
            nkpt = np.diag([nkpt] * 3)
        elif isinstance(nkpt[0], Integral):
            nkpt = np.diag(nkpt)

        # Now we have a matrix of k-points
        if np.any(nkpt - np.diag(np.diag(nkpt)) != 0):
            raise NotImplementedError(self.__class__.__name__ + " with off-diagonal components is not implemented yet")

        if displacement is None:
            displacement = np.zeros(3, np.float64)
        elif isinstance(displacement, Real):
            displacement = np.zeros(3, np.float64) + displacement

        if size is None:
            size = _a.onesd(3)
        elif isinstance(size, Real):
            size = _a.zerosd(3) + size
        else:
            size = _a.arrayd(size)

        # Retrieve the diagonal number of values
        Dn = np.diag(nkpt).astype(np.int32)
        if np.any(Dn) == 0:
            raise ValueError(self.__class__.__name__ + ' *must* be initialized with '
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

        self._k = _a.emptyd((kw[0][0].size, kw[1][0].size, kw[2][0].size, 3))
        self._w = _a.onesd(self._k.shape[:-1])
        for i in (0, 1, 2):
            k = kw[i][0].reshape(-1, 1, 1)
            w = kw[i][1].reshape(-1, 1, 1)
            self._k[..., i] = np.rollaxis(k, 0, i + 1)
            self._w[...] *= np.rollaxis(w, 0, i + 1)

        del kw
        self._k.shape = (-1, 3)
        self._k = np.where(self._k > .5, self._k - 1, self._k)
        self._w.shape = (-1,)

        # Store information regarding size and diagonal elements
        # This information is basically only necessary when
        # we want to replace special k-points
        self._diag = Dn # vector
        self._displ = displacement # vector
        self._size = size # vector
        self._centered = centered
        self._trs = trs

    def copy(self):
        """ Create a copy of this object """
        bz = self.__class__(self.parent, self._diag, self._displ, self._size, self._centered, self._trs)
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
        k : np.ndarray
           the list of k-points in the Brillouin zone to be sampled
        w : np.ndarray
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

        # We create the full grid, then afterwards we figure out TRS
        n_half = n // 2
        if n % 2 == 1:
            k = _a.aranged(-n_half, n_half + 1) * size / n + displ
        else:
            k = _a.aranged(-n_half, n_half) * size / n + displ
            if not centered:
                # Shift everything by halve the size each occupies
                k += size / (2 * n)

        # Move k to the primitive cell and generate weights
        k = cls.in_primitive(k)
        w = _a.onesd(n) * size / n

        # Check for TRS points
        if trs and np.any(k < 0.):
            # Make all positive to remove the double conting terms
            k_pos = np.abs(k)

            # Sort k-points and weights
            idx = np.argsort(k_pos)

            # Re-arange according to k value
            k_pos = k_pos[idx]
            w = w[idx]

            # Find indices of all equivalent k-points (tolerance of 1e-10 in reciprocal units)
            #  1e-10 ~ 1e10 k-points (no body will do this!)
            idx_same = (np.diff(k_pos) < 1e-10).nonzero()[0]

            # The above algorithm should never create more than two duplicates.
            # Hence we can simply remove all idx_same and double the weight for all
            # idx_same + 1.
            w[idx_same + 1] *= 2
            # Delete the duplicated k-points (they are already sorted)
            k = np.delete(k_pos, idx_same, axis=0)
            w = np.delete(w, idx_same)
        else:
            # Sort them, because it makes more visual sense
            idx = np.argsort(k)
            k = k[idx]
            w = w[idx]

        # Return values
        return k, w

    def replace(self, k, mp):
        r""" Replace a k-point with a new set of k-points from a Monkhorst-Pack grid

        This method tries to replace an area corresponding to `mp.size` around the k-point `k`
        such that the k-points are replaced.
        This enables one to zoom in on specific points in the Brillouin zone for detailed analysis.

        Parameters
        ----------
        k : array_like
           k-point in this object to replace
        mp : MonkhorstPack
           object containing the replacement k-points.

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
        SislError : if the size of the replacement `MonkhorstPack` grid is not compatible with the
                    k-point spacing in this object.
        """
        # First we find all k-points within k +- mp.size
        # Those are the points we wish to remove.
        # Secondly we need to ensure that the k-points we remove are occupying *exactly*
        # the Brillouin zone we wish to replace.
        if not isinstance(mp, MonkhorstPack):
            raise ValueError('Object `mp` is not a MonkhorstPack object')

        # We can easily figure out the BZ that each k-point is averaging
        k_vol = self._size / self._diag
        # Compare against the size of this one
        # Since we can remove more than one k-point, we require that the
        # size of the replacement MP is an integer multiple of the
        # k-point volumes.
        k_int = mp._size / k_vol
        if not np.allclose(np.rint(k_int), k_int):
            raise SislError(self.__class__.__name__ + '.reduce could not replace k-point, BZ '
                            'volume replaced is not equivalent to the inherent k-point volume.')
        k_int = np.rint(k_int)

        # 1. find all k-points
        k = self.in_primitive(k).reshape(1, 3)
        dk = (mp._size / 2).reshape(1, 3)
        # Find all points within [k - dk; k + dk]
        # Since the volume of each k-point is non-zero we know that no k-points will be located
        # on the boundary.
        # This does remove boundary points because we shift everything into the positive
        # plane.
        diff_k = self.in_primitive(self.k % 1. - k % 1.)
        idx = np.logical_and.reduce(np.abs(diff_k) <= dk, axis=1).nonzero()[0]
        if len(idx) == 0:
            raise SislError(self.__class__.__name__ + '.reduce could not find any points to replace.')

        # Now we have the k-points we need to remove
        # Figure out if the total weight is consistent
        total_weight = self.weight[idx].sum()
        replace_weight = mp.weight.sum()
        if abs(total_weight - replace_weight) < 1e-8:
            weight_factor = 1.
        elif abs(total_weight - replace_weight * 2) < 1e-8:
            weight_factor = 2.
            if not self._trs:
                info(self.__class__.__name__ + '.reduce assumes that the replaced k-point has double weights.')
        else:
            print('k-point to replace:')
            print(' ', k.ravel())
            print('delta-k:')
            print(' ', dk.ravel())
            print('Found k-indices that will be replaced:')
            print(' ', idx)
            print('k-points replaced:')
            print(self.k[idx, :])
            raise SislError(self.__class__.__name__ + '.reduce could not assert the weights are consistent during replacement.')

        self._k = np.delete(self._k, idx, axis=0)
        self._w = np.delete(self._w, idx)

        # Append the new k-points and weights
        self._k = np.concatenate((self._k, mp._k), axis=0)
        self._w = np.concatenate((self._w, mp._w * weight_factor))


class BandStructure(BrillouinZone):
    """ Create a path in the Brillouin zone for plotting band-structures etc.

    Parameters
    ----------
    parent : object or array_like
       An object with associated `parentcell` and `parent.rcell` or
       an array of floats which may be turned into a `SuperCell`
    point : array_like of float
       a list of points that are the *corners* of the path
    division : int or array_like of int
       number of divisions in each segment.
       If a single integer is passed it is the total number
       of points on the path (equally separated).
       If it is an array_like input it must have length one
       less than `point`.
    name : array_like of str
       the associated names of the points on the Brillouin Zone path

    Examples
    --------
    >>> sc = SuperCell(10)
    >>> bs = BandStructure(sc, [[0] * 3, [0.5] * 3], 200)
    >>> bs = BandStructure(sc, [[0] * 3, [0.5] * 3, [1.] * 3], 200)
    >>> bs = BandStructure(sc, [[0] * 3, [0.5] * 3, [1.] * 3], 200, ['Gamma', 'M', 'Gamma'])
    """

    def __init__(self, parent, point, division, name=None):
        super(BandStructure, self).__init__(parent)

        # Copy over points
        self.point = _a.arrayd(point)

        # If the array has fewer points we try and determine
        if self.point.shape[1] < 3:
            if self.point.shape[1] != np.sum(self.parent.nsc > 1):
                raise ValueError('Could not determine the non-periodic direction')

            # fix the points where there are no periodicity
            for i in [0, 1, 2]:
                if self.parent.nsc[i] == 1:
                    self.point = np.insert(self.point, i, 0., axis=1)

        # Ensure the shape is correct
        self.point.shape = (-1, 3)

        # Now figure out what to do with the divisions
        if isinstance(division, Integral):

            # Calculate points (we need correct units for distance)
            kpts = [self.tocartesian(pnt) for pnt in self.point]
            if len(kpts) == 2:
                dists = [sum(np.diff(kpts, axis=0) ** 2) ** .5]
            else:
                dists = sum(np.diff(kpts, axis=0)**2, axis=1) ** .5
            dist = sum(dists)

            div = np.floor(dists / dist * division).astype(dtype=np.int32)
            n = sum(div)
            if n < division:
                div[-1] +=1
                n = sum(div)
            while n < division:
                # Get the separation of k-points
                delta = dist / n

                idx = np.argmin(dists - delta * div)
                div[idx] += 1

                n = sum(div)

            division = div[:]

        self.division = _a.arrayi(division)
        self.division.shape = (-1,)

        if name is None:
            self.name = 'ABCDEFGHIJKLMNOPQRSTUVXYZ'[:len(self.point)]
        else:
            self.name = name

        self._k = _a.arrayd([k for k in self])

    def __iter__(self):
        """ Iterate through the path """

        # Calculate points
        dk = np.diff(self.point, axis=0)

        for i in range(len(dk)):

            # Calculate this delta
            if i == len(dk) - 1:
                # to get end-point
                delta = dk[i, :] / (self.division[i] - 1)
            else:
                delta = dk[i, :] / self.division[i]

            for j in range(self.division[i]):
                yield self.point[i] + j * delta

    def lineartick(self):
        """ The tick-marks corresponding to the linear-k values

        Returns
        -------
        linear_k : The positions in reciprocal space determined by the distance between points

        See Also
        --------
        lineark : Routine used to calculate the tick-marks.
        """
        return self.lineark(True)[1:3]

    def lineark(self, ticks=False):
        """ A 1D array which corresponds to the delta-k values of the path

        This is meant for plotting

        Examples
        --------

        >>> p = BandStructure(...) # doctest: +SKIP
        >>> eigs = Hamiltonian.eigh(p) # doctest: +SKIP
        >>> for i in range(len(Hamiltonian)): # doctest: +SKIP
        ...     plt.plot(p.lineark(), eigs[:, i]) # doctest: +SKIP

        >>> p = BandStructure(...) # doctest: +SKIP
        >>> eigs = Hamiltonian.eigh(p) # doctest: +SKIP
        >>> lk, kt, kl = p.lineark(True) # doctest: +SKIP
        >>> plt.xticks(kt, kl) # doctest: +SKIP
        >>> for i in range(len(Hamiltonian)): # doctest: +SKIP
        ...     plt.plot(lk, eigs[:, i]) # doctest: +SKIP

        Parameters
        ----------
        ticks : bool, optional
           if `True` the ticks for the points are also returned

           lk, xticks, label_ticks, lk = BandStructure.lineark(True)

        Returns
        -------
        linear_k : The positions in reciprocal space determined by the distance between points
        k_tick : Linear k-positions of the points, only returned if `ticks` is ``True``
        k_label : Labels at `k_tick`, only returned if `ticks` is ``True``
        """
        # Calculate points
        k = [self.tocartesian(pnt) for pnt in self.point]
        # Get difference between points
        dk = np.diff(k, axis=0)
        # Calculate the cumultative distance between points
        k_len = np.insert(_a.cumsumd((dk ** 2).sum(1) ** .5), 0, 0.)
        xtick = [None] * len(k)
        # Prepare output array
        dK = _a.emptyd(len(self))

        # short-hand
        ls = np.linspace

        xtick = np.insert(_a.cumsumi(self.division), 0, 0)
        for i in range(len(dk)):
            n = self.division[i]
            end = i == len(dk) - 1

            dK[xtick[i]:xtick[i+1]] = ls(k_len[i], k_len[i+1], n, dtype=np.float64, endpoint=end)
        xtick[-1] -= 1

        # Get label tick, in case self.name is a single string 'ABCD'
        label_tick = [a for a in self.name]
        if ticks:
            return dK, dK[xtick], label_tick
        return dK

    def __len__(self):
        return sum(self.division)
