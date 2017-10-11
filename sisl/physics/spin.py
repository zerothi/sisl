from __future__ import print_function, division

import numpy as np

__all__ = ['Spin']


class Spin(object):
    r""" Spin class to determine configurations and spin components.

    The basic class ``Spin`` implements a generic method to determine a spin configuration.

    Its usage can be summarized in these few examples:

    >>> Spin(Spin.UNPOLARIZED) == Spin('unpolarized') == Spin()
    True
    >>> Spin(Spin.POLARIZED) == Spin('polarized') == Spin('p')
    True
    >>> Spin(Spin.NONCOLINEAR, dtype=np.complex128) == Spin('non-colinear') == Spin('nc')
    True
    >>> Spin(Spin.SPINORBIT, dtype=np.complex128) == Spin('spin-orbit') == Spin('so')
    True

    Note that a data-type may be associated with a spin-object. This is not to say
    that the data-type is used in the configuration, but merely that it helps
    any sub-classed or classes who use the spin-object to determine the
    usage of the different spin-components.
    """

    UNPOLARIZED = 0
    POLARIZED = 1
    NONCOLINEAR = 2
    SPINORBIT = 3

    __slots__ = ['_spins', '_kind', '_dtype']

    def __init__(self, kind='', dtype=np.float64):

        if isinstance(kind, Spin):
            self._kind = kind._kind
            self._dtype = kind._dtype
            self._spins = kind._spins
            return

        # Copy data-type
        self._dtype = dtype

        kind = {'unpolarized': Spin.UNPOLARIZED, '': Spin.UNPOLARIZED,
                Spin.UNPOLARIZED: Spin.UNPOLARIZED,
                'polarized': Spin.POLARIZED, 'p': Spin.POLARIZED,
                Spin.POLARIZED: Spin.POLARIZED,
                'non-colinear': Spin.NONCOLINEAR, 'nc': Spin.NONCOLINEAR,
                Spin.NONCOLINEAR: Spin.NONCOLINEAR,
                'spin-orbit': Spin.SPINORBIT, 'so': Spin.SPINORBIT,
                Spin.SPINORBIT: Spin.SPINORBIT}.get(kind)

        # Now assert the checks
        self._kind = kind

        if np.dtype(dtype).kind == 'c':
            spins = {self.UNPOLARIZED: 1,
                     self.POLARIZED: 2,
                     self.NONCOLINEAR: 4,
                     self.SPINORBIT: 4}.get(kind)

        else:
            spins = {self.UNPOLARIZED: 1,
                     self.POLARIZED: 2,
                     self.NONCOLINEAR: 4,
                     self.SPINORBIT: 8}.get(kind)

        self._spins = spins

    def __repr__(self):
        s = self.__class__.__name__
        if self.is_unpolarized:
            return s + '{{unpolarized, kind={}}}'.format(self.dkind)
        if self.is_polarized:
            return s + '{{polarized, kind={}}}'.format(self.dkind)
        if self.is_noncolinear:
            return s + '{{non-colinear, kind={}}}'.format(self.dkind)
        return s + '{{spin-orbit, kind={}}}'.format(self.dkind)

    def copy(self):
        return Spin(self.kind, self.dtype)

    @property
    def dtype(self):
        """ Data-type of the spin configuration """
        return self._dtype

    @property
    def dkind(self):
        """ Data-type kind """
        return np.dtype(self._dtype).kind

    @property
    def spins(self):
        """ Number of spin-components """
        return self._spins

    @property
    def kind(self):
        """ A unique ID for the kind of spin configuration """
        return self._kind

    @property
    def is_unpolarized(self):
        """ True if the configuration is not polarized """
        # Regardless of data-type
        return self.kind == Spin.UNPOLARIZED

    @property
    def is_polarized(self):
        """ True if the configuration is polarized """
        return self.kind == Spin.POLARIZED

    is_colinear = is_polarized

    @property
    def is_noncolinear(self):
        """ True if the configuration non-colinear """
        return self.kind == Spin.NONCOLINEAR

    @property
    def is_spinorbit(self):
        """ True if the configuration is spin-orbit """
        return self.kind == Spin.SPINORBIT

    def __len__(self):
        return self._spins

    # Comparisons
    def __lt__(a, b):
        return a.kind < b.kind

    def __le__(a, b):
        return a.kind <= b.kind

    def __eq__(a, b):
        return a.kind == b.kind

    def __ne__(a, b):
        return not a == b

    def __gt__(a, b):
        return a.kind > b.kind

    def __ge__(a, b):
        return a.kind >= b.kind
