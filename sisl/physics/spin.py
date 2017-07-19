""" Implementation of class to determine spin configurations and spin components.

The basic class ``Spin`` implements a generic method to determine a spin configuration.

Its usage can be summarized in these few examples:

>>> Spin(1) == Spin('non-polarized') == Spin('unpolarized') == Spin('un-polarized')
>>> Spin(2) == Spin('polarized') == Spin('p')
>>> Spin(2, dtype=np.complex128) == Spin('non-colinear') == Spin('nc') == Spin(4)
>>> Spin(4, dtype=np.complex128) == Spin('spin-orbit') == Spin('so') == Spin(8)

Note that a data-type may be associated with a spin-object. This is not to say
that the data-type is used in the configuration, but merely that it helps 
any sub-classed or classes who use the spin-object to determine the 
usage of the different spin-components.
"""
from __future__ import print_function, division

from numbers import Integral

import numpy as np


class Spin(object):
    """ Implementation of the spin configuration space """

    UNPOLARIZED = 0
    POLARIZED = 1
    NONCOLLINEAR = 2
    SPINORBIT = 3

    __slots__ = ['_spin', '_dtype']

    def __init__(self, spin='', dtype=None):

        if isinstance(spin, Spin):
            self._spin = spin._spin
            self._dtype = spin._dtype
            return

        # Determine the spin-configuration
        if dtype is None:
            dtype = np.float64

        # Copy data-type
        self._dtype = dtype

        if np.dtype(dtype).kind == 'c':
            spin = {'unpolarized': 1, '': 1,
                    'polarized': 2, 'p': 2,
                    'non-colinear': 2, 'nc': 2,
                    'spin-orbit': 4, 'so': 4}.get(spin, spin)

        else:
            spin = {'unpolarized': 1, '': 1,
                    'polarized': 2, 'p': 2,
                    'non-colinear': 4, 'nc': 4,
                    'spin-orbit': 8, 'so': 8}.get(spin, spin)

        # Now assert the checks
        self._spin = spin

        if not isinstance(spin, Integral):
            raise ValueError('Could not determine spin-size from input')

        # Perhaps we should add additional checks here to assert that the
        # spin values and data-type makes sense...

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
        return Spin(self.spin, self.dtype)

    @property
    def dtype(self):
        """ Data-type of the spin configuration """
        return self._dtype

    @property
    def dkind(self):
        """ Data-type kind """
        return np.dtype(self._dtype).kind

    @property
    def spin(self):
        """ Number of spin-components """
        return self._spin

    @property
    def kind(self):
        """ A unique ID for the kind of spin configuration """
        if self.is_unpolarized:
            return self.UNPOLARIZED
        if self.is_polarized:
            return self.POLARIZED
        if self.is_noncolinear:
            return self.NONCOLLINEAR
        if self.is_spinorbit:
            return self.SPINORBIT
        raise NotImplementedError

    @property
    def is_unpolarized(self):
        """ True if the configuration is not polarized """
        # Regardless of data-type
        return self.spin == 1

    @property
    def is_polarized(self):
        """ True if the configuration is polarized """
        return self.spin == 2 and self.dkind != 'c'

    is_colinear = is_polarized

    @property
    def is_noncolinear(self):
        """ True if the configuration non-colinear """
        s = self.spin
        k = self.dkind
        return (s == 2 and k == 'c') or (s == 4 and k != 'c')

    @property
    def is_spinorbit(self):
        """ True if the configuration is spin-orbit """
        s = self.spin
        k = self.dkind
        return (s == 4 and k == 'c') or (s == 8 and k != 'c')

    def __len__(self):
        return self.spin

    # Comparison types
    def __lt__(a, b):
        if a.dkind == b.dkind:
            return a.spin < b.spin
        # Explicit checks
        if a.is_unpolarized:
            return not b.is_unpolarized
        elif a.is_polarized:
            return b.is_noncolinear or b.is_spinorbit
        elif a.is_noncolinear:
            return b.is_spinorbit
        # It cannot be less than the other one... spin-orbit is highest
        return False

    def __le__(a, b):
        if a.dkind == b.dkind:
            return a.spin <= b.spin

        if a.is_unpolarized:
            return True
        elif a.is_polarized:
            return not b.is_unpolarized
        elif a.is_noncolinear:
            return b.is_noncolinear or b.is_spinorbit
        return b.is_spinorbit

    def __eq__(a, b):
        if a.dkind == b.dkind:
            return a.spin == b.spin

        if a.is_unpolarized:
            return b.is_unpolarized
        elif a.is_polarized:
            return not b.is_polarized
        elif a.is_noncolinear:
            return b.is_noncolinear
        return b.is_spinorbit

    def __ne__(a, b):
        return not a == b

    def __gt__(a, b):
        return b < a

    def __ge__(a, b):
        return b <= a
