""" Injection module for adding apply functions for BrillouinZone classes

This module should not expose any methods!
"""
""" Injection module for adding apply functions for BrillouinZone classes

This module should not expose any methods!
"""
from functools import wraps, reduce
import os
import operator as op

import numpy as np
try:
    import pathos as pos
    _has_pathos = True
except ImportError:
    _has_pathos = False
try:
    import xarray
    _has_xarray = True
except ImportError:
    _has_xarray = False

from sisl._dispatcher import ClassDispatcher
from sisl._internal import set_module
from sisl.utils.misc import allow_kwargs
from sisl.oplist import oplist

# Stuff used for patching
from .brillouinzone import BrillouinZone
from ._brillouinzone_apply import (
    BrillouinZoneApply,
    _asoplist
)

# We expose the Apply and ParentApply classes
__all__ = ["BrillouinZonePApply", "BrillouinZoneParentPApply"]


# I do not know what way is the best, we should probably have
_NPROCS = os.environ.get("SISL_NPROCS", None)
if isinstance(_NPROCS, str):
    _NPROCS = int(_NPROCS)


def _apply_str(s):
    def __str__(self):
        return f"PApply{{{s}}}"
    return __str__


@set_module("sisl.physics")
class BrillouinZonePApply(BrillouinZoneApply):
    # this dispatch function will do stuff on the BrillouinZone object
    pass


@set_module("sisl.physics")
class BrillouinZoneParentPApply(BrillouinZonePApply):

    def _parse_kwargs(self, wrap):
        """ Parse kwargs """
        bz = self._obj
        parent = bz.parent
        if wrap is None:
            # we always return a wrap
            def wrap(v, parent=None, k=None, weight=None):
                return v
        else:
            wrap = allow_kwargs("parent", "k", "weight")(wrap)
        return bz, parent, wrap

    def __getattr__(self, key):
        # We need to offload the dispatcher to retrieve
        # methods from the parent object
        # This dispatch will _never_ do anything to the BrillouinZone
        method = getattr(self._obj.parent, key)
        return self.dispatch(method)


@set_module("sisl.physics")
class IteratorPApply(BrillouinZoneParentPApply):
    __str__ = _apply_str("iter")

    def dispatch(self, method):
        """ Dispatch the method by iterating values """
        @wraps(method)
        def func(*args, wrap=None, **kwargs):
            pool = self._attrs["pool"]
            pool.restart()
            bz, parent, wrap = self._parse_kwargs(wrap)
            k = bz.k
            w = bz.weight

            def func(k, w):
                return wrap(method(*args, k=k, **kwargs), parent=parent, k=k, weight=w)

            yield from pool.imap(func, k, w)
            # TODO notify users that this may be bad when used with zip
            # unless this generator is the first argument of zip
            # zip has left-to-right checks of length and stops querying
            # elements as soon as the left-most one stops.
            pool.close()
            pool.join()

        return func


@set_module("sisl.physics")
class SumPApply(IteratorPApply):
    __str__ = _apply_str("sum over k")

    def dispatch(self, method):
        """ Dispatch the method by summing """
        iter_func = super().dispatch(method)

        @wraps(method)
        def func(*args, **kwargs):
            it = iter_func(*args, **kwargs)
            # next will be called before anything else
            return reduce(op.add, it, _asoplist(next(it)))

        return func


@set_module("sisl.physics")
class NonePApply(IteratorPApply):
    __str__ = _apply_str("None")

    def dispatch(self, method):
        """ Dispatch the method by doing nothing (mostly useful if wrapped) """
        iter_func = super().dispatch(method)
        @wraps(method)
        def func(*args, **kwargs):
            for _ in iter_func(*args, **kwargs):
                pass
            return None
        return func


@set_module("sisl.physics")
class ListPApply(IteratorPApply):
    __str__ = _apply_str("list")

    def dispatch(self, method):
        """ Dispatch the method by returning list of values """
        iter_func = super().dispatch(method)
        @wraps(method)
        def func(*args, **kwargs):
            return [v for v in iter_func(*args, **kwargs)]
        return func


@set_module("sisl.physics")
class OpListPApply(IteratorPApply):
    __str__ = _apply_str("oplist")

    def dispatch(self, method):
        """ Dispatch the method by returning oplist of values """
        iter_func = super().dispatch(method)
        @wraps(method)
        def func(*args, **kwargs):
            return oplist(v for v in iter_func(*args, **kwargs))
        return func


@set_module("sisl.physics")
class ArrayPApply(BrillouinZoneParentPApply):
    __str__ = _apply_str("numpy.ndarray")

    def dispatch(self, method):
        """ Dispatch the method by one array """
        @wraps(method)
        def func(*args, wrap=None, **kwargs):
            pool = self._attrs["pool"]
            pool.restart()
            bz, parent, wrap = self._parse_kwargs(wrap)
            k = bz.k
            w = bz.weight

            def func(k, w):
                return wrap(method(*args, k=k, **kwargs), parent=parent, k=k, weight=w)

            it = pool.imap(func, k, w)
            v = next(it)
            # Create full array
            if v.ndim == 0:
                a = np.empty([len(k)], dtype=v.dtype)
            else:
                a = np.empty((len(k), ) + v.shape, dtype=v.dtype)
            a[0] = v

            for i, v in enumerate(it):
                a[i+1] = v
            del v
            pool.close()
            pool.join()
            return a

        return func


@set_module("sisl.physics")
class AveragePApply(BrillouinZoneParentPApply):
    __str__ = _apply_str("average")

    def dispatch(self, method):
        """ Dispatch the method by averaging """
        @wraps(method)
        def func(*args, wrap=None, **kwargs):
            pool = self._attrs["pool"]
            pool.restart()
            bz, parent, wrap = self._parse_kwargs(wrap)
            k = bz.k
            w = bz.weight

            def func(k, w):
                return wrap(method(*args, k=k, **kwargs), parent=parent, k=k, weight=w) * w

            iter_func = pool.uimap(func, k, w)
            avg = reduce(op.add, iter_func, _asoplist(next(iter_func)))
            pool.close()
            pool.join()
            return avg

        return func


@set_module("sisl.physics")
class DataArrayPApply(ArrayPApply):
    __str__ = _apply_str("xarray.DataArray")

    def dispatch(self, method):
        """ Dispatch the method by returning a DataArray """
        # Get data as array
        array_func = super().dispatch(method)

        @wraps(method)
        def func(*args, coords=None, name=method.__name__, **kwargs):
            # xarray specific data (default to function name)
            bz = self._obj

            # retrieve ALL data
            array = array_func(*args, **kwargs)

            # Create coords
            if coords is None:
                coords = [("k", _a.arangei(len(bz)))]
                for i, v in enumerate(array.shape[1:]):
                    coords.append((f"v{i+1}", _a.arangei(v)))
            else:
                coords = list(coords)
                coords.insert(0, ("k", _a.arangei(len(bz))))
                for i in range(1, len(coords)):
                    if isinstance(coords[i], str):
                        coords[i] = (coords[i], _a.arangei(array.shape[i]))
            attrs = {"bz": bz, "parent": bz.parent}

            return xarray.DataArray(array, coords=coords, name=name, attrs=attrs)

        return func


if _has_pathos and not hasattr(BrillouinZone, "papply"):
    # Create pool
    _pool = pos.multiprocessing.ProcessPool(nodes=_NPROCS)
    _pool.close()
    _pool.join()

    BrillouinZone.papply = ClassDispatcher("papply",
                                           obj_getattr=lambda obj, key: getattr(obj.parent, key),
                                           # The rest are attributes
                                           pool=_pool
    )
    # Register dispatched functions
    BrillouinZone.papply.register("iter", IteratorPApply, default=True)
    BrillouinZone.papply.register("average", AveragePApply)
    BrillouinZone.papply.register("sum", SumPApply)
    BrillouinZone.papply.register("array", ArrayPApply)
    BrillouinZone.papply.register("none", NonePApply)
    BrillouinZone.papply.register("list", ListPApply)
    BrillouinZone.papply.register("oplist", OpListPApply)
    if _has_xarray:
        BrillouinZone.papply.register("dataarray", DataArrayPApply)
