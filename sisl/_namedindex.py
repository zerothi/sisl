""" Named groups via indices

This module implements the base-class which allows named indices

>>> nidx = NamedIndex('hello', [1, 2])
"""
from collections import Counter, defaultdict
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

    @staticmethod
    def _get_mask_first_uniques(names):
        """ Helper function to determine what names (and idx) to keep/del."""
        uniq_names = set()
        nn = []
        for n in names:
            if n in uniq_names:
                nn.append(False)
            else:
                nn.append(True)
                uniq_names.add(n)
        return nn

    def add(self, other, offset=0, duplicates=None):
        """ Return a new NamedIndex which is a type of union of self and other.
        By default, name conflict between self and other will raise a ValueError.
        See the `duplicates` parameter for information on how to change this.

        Parameters
        ----------
        other : NamedIndex
            The NamedIndex to perform the addition with
        offset : int, optional
            `other` will have `offset` added to all indices.
            Useful for adding geometries.
        duplicates: str, optional, one of "raise", "union", "left", "right", "omit".
            Selects the default behaviour in case of name conflict.
            Default ("raise") is to raise a ValueError in the case of conflict.
            If "union", each name will contain indices from both `self` and `other`.
            If "left" or "right", the name from `self` or `other`, respectively,
            will take precedence.
            If "omit", duplicate names are completely omitted.
        """
        # Add the two NIs directly
        new = self.copy()
        new._name.extend(other._name)
        names = new._name
        new._index.extend(idx.copy() + offset for idx in other._index)

        # Handle duplicates
        if set(self._name).intersection(set(other._name)):
            if duplicates is None or duplicates == "raise":
                raise ValueError(
                    "There are duplicate names and behaviour in this case is "
                    "currently unspecified. Names: {} and {}."
                    "".format(self._name, other._name))
            elif duplicates == "union":
                union = defaultdict(lambda: np.empty((0,), dtype="int32"))
                for name, index in zip(new._name, new._index):
                    union[name] = np.union1d(union[name], index)
                new._name, new._index = zip(*tuple(union.items()))
                return new
            elif duplicates == "omit":
                total_cnt = Counter(names)
                mask = [total_cnt[n] == 1 for n in names]
            elif duplicates == "left":
                mask = NamedIndex._get_mask_first_uniques(names)
            elif duplicates == "right":
                mask = NamedIndex._get_mask_first_uniques(names[::-1])[::-1]
            else:
                raise ValueError("{} is not a valid value for `duplicates`"
                                 "".format(duplicates))
            new._name = [n for i, n in enumerate(new._name) if mask[i]]
            new._index = [n for i, n in enumerate(new._index) if mask[i]]
        return new

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
