""" Named groups via indices

This module implements the base-class which allows named indices

>>> nidx = NamedIndex('hello', [1, 2])
"""
import numpy as np

from ._indices import indices_only
from ._array import arrayi
from ._help import _str
from .messages import SislError
from .utils.ranges import list2str


class NamedIndex(object):
    __slots__ = ['_name', '_index']

    def __init__(self, name=None, index=None):
        # Initialize lists
        self._name = []
        self._index = []

        if isinstance(name, _str):
            self.add_name(name, index)
        elif not name is None:
            for n, i in zip(name, index):
                self.add_name(n, i)

    def __iter__(self):
        """ Iterate names in the group """
        for name in self._name:
            yield name

    def __len__(self):
        """ Number of uniquely defined names """
        return len(self._name)

    def copy(self):
        """ Create a copy of this """
        return self.__class__(self._name[:], [i.copy() for i in self._index])

    def add_name(self, name, index):
        """ Add a new named group. The indices (`index`) will be associated with the name `name`

        Parameters
        ----------
        name : str
           name of group, must not already exist
        index : array_like of ints
           the indices that has a name associated
        """
        if name in self._name:
            raise SislError(self.__class__.__name__ + '.add_name already contains name {}, please delete group name before adding.'.format(name))
        self._name.append(name)
        self._index.append(arrayi(index).ravel())

    def delete_name(self, name):
        """ Delete an existing named group, if the group does not exist, nothing will happen.

        Parameters
        ----------
        name : str
           name of group to delete
        """
        i = self._name.index(name)
        del self._name[i]
        del self._index[i]

    def __str__(self):
        """ Representation of the object """
        N = len(self)
        if N == 0:
            return NamedIndex.__name__ + '{}'
        s = NamedIndex.__name__ + '{{groups: {0}'.format(N)
        for name, idx in zip(self._name, self._index):
            s += ',\n {0}: [{1}]'.format(name, list2str(idx))
        return s + '\n}'

    def __setitem__(self, name, index):
        """ Equivalent to `add` """
        if isinstance(name, _str):
            self.add_name(name, index)
        else:
            self.add_name(index, name)

    def index(self, name):
        """ Return indices of the group via `name` """
        try:
            i = self._name.index(name)
            return self._index[i]
        except:
            if isinstance(name, _str):
                return None
            return name

    def __getitem__(self, name):
        """ Return indices of the group """
        return self.index(name)

    def __delitem__(self, name):
        """ Delete a named group """
        self.delete_name(name)

    def __contains__(self, name):
        """ Check whether a name exists in this group a named group """
        return name in self._name

    def remove(self, index):
        """ Remove indices from all named index groups

        Parameters
        ----------
        index : array_like of int
           indices to remove
        """
        n = len(self)
        if n == 0:
            return
        index = arrayi(index).ravel()
        for i in range(n):
            print(self._index[i], index)
            idx2 = indices_only(self._index[i], index)
            self._index[i] = np.delete(self._index[i], idx2)
