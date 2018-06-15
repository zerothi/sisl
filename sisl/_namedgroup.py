""" Named groups via indices

This module implements the base-class which allows named indices

>>> grp = Group('hello', [1, 2])
"""
import numpy as np

from ._array import arrayi
from ._help import _str
from .messages import SislError
from .utils.ranges import list2str


class NamedGroup(object):
    __slots__ = ['_grp_names', '_grp_indices']

    def __init__(self, name=None, index=None):
        # Initialize lists
        self._grp_names = []
        self._grp_indices = []

        if isinstance(name, _str):
            self.add(name, index)
        elif not name is None:
            for n, i in zip(name, index):
                self.add(n, i)

    def __iter__(self):
        """ Iterate names in the group """
        for name in self._grp_names:
            yield name

    def __len__(self):
        """ Number of uniquely defined names """
        return len(self._grp_names)

    def add(self, name, index):
        """ Add a new named group. The indices (`index`) will be associated with the name `name`

        Parameters
        ----------
        name : str
           name of group, must not already exist
        index : array_like of ints
           the indices that has a name associated
        """
        if name in self._grp_names:
            raise SislError(self.__class__.__name__ + '.add already contains name {}, please delete group name before adding.'.format(name))
        self._grp_names.append(name)
        self._grp_indices.append(arrayi(index).ravel())

    def delete(self, name):
        """ Delete an existing named group, if the group does not exist, nothing will happen.

        Parameters
        ----------
        name : str
           name of group to delete
        """
        try:
            i = self._grp_names.index(name)
            del self._grp_names[i]
            del self._grp_indices[i]
        except:
            pass

    def __repr__(self):
        """ Representation of the object """
        N = len(self)
        if N == 0:
            return NamedGroup.__name__ + '{}'
        s = NamedGroup.__name__ + '{{groups: {0}'.format(N)
        for name, idx in zip(self._grp_names, self._grp_indices):
            s += ',\n {0}: [{1}]'.format(name, list2str(idx))
        return s + '\n}'

    def __setitem__(self, name, index):
        """ Equivalent to `add` """
        if isinstance(name, _str):
            self.add(name, index)
        else:
            self.add(index, name)

    def group(self, name):
        """ Return indices of the group via `name` """
        try:
            i = self._grp_names.index(name)
            return self._grp_indices[i]
        except:
            if isinstance(name, _str):
                return None
            return name

    def __getitem__(self, name):
        """ Return indices of the group """
        return self.group(name)

    def __delitem__(self, name):
        """ Delete a named group """
        self.delete(name)
