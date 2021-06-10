# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from collections.abc import Iterable
import numpy as np


__all__ = ['Variable', 'UpdateVariable', 'read_variable_yaml']


def read_variable_yaml(dict_or_yaml, name=None):
    r""" Reads entries in a yaml file """
    if not isinstance(dict_or_yaml, dict):
        # Then it must be a yaml file
        import yaml
        dict_or_yaml = yaml.load(open(dict_or_yaml, 'r'), Loader=yaml.CLoader)
    if name is not None:
        if name in dict_or_yaml:
            dict_or_yaml = dict_or_yaml[name]
    return dict_or_yaml


class Variable:
    """ A minimization variable with associated name, inital value, and possible bounds.

    Parameters
    ----------
    name : str
       name of variable
    value : float
       initial value of the variable
    bounds : (float, float)
       boundaries of the value
    **attrs : dict, optional
       other attributes that are retrievable
    """

    def __init__(self, name, value, bounds, **attrs):
        self.name = name
        self.bounds = np.array(bounds, np.float64)
        assert self.bounds.size == 2
        assert self.bounds[0] <= self.bounds[1]
        self.value = value
        self.attrs = attrs

    def __str__(self):
        return f"{self.__class__.__name__}{{name: {self.name}, value: {self.value}, bounds: {self.bounds}}}"

    def update(self, value):
        self.value = value

    def _parse_norm(self, norm, with_offset):
        """ Return offset, scale factor """
        if isinstance(norm, str):
            scale = 1.
        elif isinstance(norm, Iterable):
            norm, scale = norm
        else:
            scale = norm
            norm = 'l2'
        if with_offset:
            off = self.bounds[0]
        else:
            off = 0.
        norm = norm.lower()
        if norm in ('none', 'identity'):
            # a norm of none will never scale, nor offset
            return 0., 1.
        elif norm == 'l2':
            return off, scale / (self.bounds[1] - self.bounds[0])
        raise ValueError("norm not found in [none/identity, l2]")

    def normalize(self, value, norm='l2', with_offset=True):
        """ Normalize a value in terms of the norms of this variable

        Parameters
        ----------
        norm : {l2, none/identity} or (str, float) or float
           whether to scale according to bounds or not
           if a scale value (float) is used then that will be the [0, scale] bounds
           of the normalized value, only passing a float is equivalent to ``('l2', scale)``
        with_offset : bool, optional
           whether to offset value by ``self.bounds[0]`` before scaling
        """
        offset, fac = self._parse_norm(norm, with_offset)
        return (value - offset) * fac

    def reverse_normalize(self, value, norm='l2', with_offset=True):
        """ Revert what `normalize` does

        Parameters
        ----------
        norm : {l2, none} or (str, scale) or float
           whether to scale according to bounds or not
           if a scale value is used then that will be the [0, scale] bounds
           of the normalized value, only passing a float is equivalent to ``('l2', scale)``
        with_offset : bool, optional
           whether to offset value by ``self.bounds[0]`` before scaling
        """
        offset, fac = self._parse_norm(norm, with_offset)
        return value / fac + offset

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.name == other.name and self.value == other.value
        return self.name == other


class UpdateVariable(Variable):
    def __init__(self, name, value, bounds, update, **attrs):
        super().__init__(name, value, bounds, **attrs)
        self._update = update

    def update(self, value):
        """ Also run update wrapper call for the new value

        The update routine should have this interface:

        >>> def update(old_value, new_value):
        ...     pass
        """
        old_value = self.value
        super().update(value)
        self._update(old_value, value)
