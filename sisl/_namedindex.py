""" Named groups via indices

This module implements the base-class which allows named indices

>>> nidx = NamedIndex('hello', [1, 2])
"""
import numpy as np
from numpy import ndarray, bool_

from ._internal import set_module
from ._indices import indices_only
from ._array import arrayi
from .messages import SislError
from .utils.ranges import list2str


@set_module("sisl")
class NamedIndex:
    __slots__ = ('_name', '_index')

    def __init__(self, name=None, index=None):
        if isinstance(name, dict):
            # Also allow dictionary inputs!
            self.__init__(name.keys(), name.values())
            return

        # Initialize lists
        self._name = []
        self._index = []

        if isinstance(name, str):
            self.add_name(name, index)
        elif not name is None:
            for n, i in zip(name, index):
                self.add_name(n, i)

    @property
    def names(self):
        """ All names contained """
        return self._name

    def clear(self):
        """ Clear all names in this object, no names will exist after this call (in-place) """
        self._name = []
        self._index = []

    def __iter__(self):
        """ Iterate names in the group """
        yield from self._name

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
            raise SislError(f"{self.__class__.__name__}.add_name already contains name {name}, please delete group name before adding.")
        self._name.append(name)
        if isinstance(index, ndarray) and index.dtype == bool_:
            index = np.flatnonzero(index)
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
        """ Representation of object """
        N = len(self)
        if N == 0:
            return self.__class__.__name__ + '{}'
        s = self.__class__.__name__ + f'{{groups: {N}'
        for name, idx in zip(self._name, self._index):
            s += ',\n {}: [{}]'.format(name, list2str(idx))
        return s + '\n}'

    def __setitem__(self, name, index):
        """ Equivalent to `add_name` """
        if isinstance(name, str):
            self.add_name(name, index)
        else:
            self.add_name(index, name)

    def index(self, name):
        """ Return indices of the group via `name` """
        try:
            i = self._name.index(name)
            return self._index[i]
        except:
            if isinstance(name, str):
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

    def merge(self, other, offset=0, duplicate="raise"):
        """ Return a new object which is a merge of self and other

        By default, name conflicts between self and other will raise a ValueError.
        See the `duplicate` parameter for information on how to change this.

        Parameters
        ----------
        other : NamedIndex
            the object to merge names(+indices) with
        offset : int, optional
            `other` will have `offset` added to all indices before merge is done.
        duplicate : {"raise", "union", "left", "right", "omit"}
            Selects the default behaviour in case of name conflict.
            Default ("raise") is to raise a ValueError in the case of conflict.
            If "union", each name will contain indices from both `self` and `other`.
            If "left" or "right", the name from `self` or `other`, respectively,
            will take precedence.
            If "omit", duplicate names are completely omitted.
        """
        new = self.copy()

        set_self = set(self.names)
        set_other = set(other.names)

        # Figure out if there are duplicate names
        intersection = set_self & set_other

        # First add all unique names (equivalent to duplicate == "left")
        for name in other:
            if not name in intersection:
                new[name] = other[name] + offset

        if len(intersection) == 0:
            # If there are no duplicates then we are done, all unique names are already present
            pass

        elif duplicate == "raise":
            raise ValueError("{}.merge has overlapping names without a duplicate handler: {}".format(self.__class__.__name__, list(intersection)))

        elif duplicate == "union":
            # Indices are made a union
            for name in intersection:
                # Clean the name (currently add_name raises an issue if already existing)
                del new[name]

                # Create a union of indices
                new[name] = np.union1d(self[name], other[name] + offset)

        elif duplicate == "left":
            # Indices are chosen from `self`, but since new is already a copy of self
            # we do not have to do anything
            pass

        elif duplicate == "right":
            for name in intersection:
                del new[name]
                new[name] = other[name] + offset

        elif duplicate == "omit":
            for name in intersection:
                del new[name]

        else:
            raise ValueError(f"{self.__class__.__name__}.merge wrong argument: duplicate.")

        return new

    def sub_index(self, index, name=None):
        """ Get a new object with only the indexes in idx.

        Parameters
        ----------
        index : array_like of int
            indices to select
        name : str, Iterable of str, optional
            If given, perform sub only on the indices for `name`. Other names
            are preserved.
        """
        if name is None:
            name = self._name
        elif isinstance(name, str):
            name = [name]

        new_index = []
        for l_name in self:
            if l_name in name:
                new_index.append(np.intersect1d(self[l_name], index))
            else:
                new_index.append(self[l_name].copy())

        return self.__class__(self._name[:], new_index)

    def sub_name(self, names):
        """ Get a new object with only the names in `names`

        Parameters
        ----------
        names : str or iterable of str
            The name(s) which the new object contain.
        """
        if isinstance(names, str):
            names = [names]

        for name in names:
            if name not in self._name:
                raise ValueError(f"{self.__class__.__name__}.sub_name specified name ({name}) is not in object.")

        new_index = []
        for name in self:
            if name in names:
                new_index.append(self[name].copy())

        # IF names and new_index does not have the same length
        # then the constructor will fail!
        return self.__class__(names, new_index)

    def remove_index(self, index):
        """ Remove indices from all named index groups

        Parameters
        ----------
        index : array_like of int
           indices to remove
        """
        new = self.copy()
        index = arrayi(index).ravel()
        for i in range(len(new)):
            idx = new._index[i]
            new._index[i] = np.delete(idx, indices_only(idx, index))

        return new

    def reduce(self):
        """ Removes names from the object which have no index associated (in-place) """
        for i in range(len(self))[::-1]:
            if len(self._index[n]) == 0:
                del self[self.names[i]]
