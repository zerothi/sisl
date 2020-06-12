""" Injection module for adding apply functions for BrillouinZone classes

This module should not expose any methods!
"""
from functools import wraps, reduce
import operator as op

import numpy as np
try:
    import xarray
    _has_xarray = True
except ImportError:
    _has_xarray = False

from sisl._dispatcher import ClassDispatcher, AbstractDispatch
from sisl._internal import set_module
from sisl.utils.misc import allow_kwargs
from sisl.oplist import oplist
import sisl._array as _a
from sisl.messages import tqdm_eta

# Stuff used for patching
from .brillouinzone import BrillouinZone


# We expose the Apply and ParentApply classes
__all__ = ["BrillouinZoneApply", "BrillouinZoneParentApply"]


def _asoplist(arg):
    if isinstance(arg, tuple):
        return oplist(arg)
    elif isinstance(arg, list) and not isinstance(arg, oplist):
        return oplist(arg)
    return arg


def _apply_str(s):
    def __str__(self):
        return f"Apply{{{s}}}"
    return __str__



@set_module("sisl.physics")
class BrillouinZoneApply(AbstractDispatch):
    # this dispatch function will do stuff on the BrillouinZone object
    pass


@set_module("sisl.physics")
class BrillouinZoneParentApply(BrillouinZoneApply):

    def _parse_kwargs(self, wrap, eta, eta_key):
        """ Parse kwargs """
        bz = self._obj
        parent = bz.parent
        if wrap is None:
            # we always return a wrap
            def wrap(v, parent=None, k=None, weight=None):
                return v
        else:
            wrap = allow_kwargs("parent", "k", "weight")(wrap)
        eta = tqdm_eta(len(bz), f"{bz.__class__.__name__}.{eta_key}", "k", eta)
        return bz, parent, wrap, eta

    def __getattr__(self, key):
        # We need to offload the dispatcher to retrieve
        # methods from the parent object
        # This dispatch will _never_ do anything to the BrillouinZone
        method = getattr(self._obj.parent, key)
        return self.dispatch(method)



@set_module("sisl.physics")
class IteratorApply(BrillouinZoneParentApply):
    __str__ = _apply_str("iter")

    def dispatch(self, method, eta_key="iter"):
        """ Dispatch the method by iterating values """
        @wraps(method)
        def func(*args, wrap=None, eta=False, **kwargs):
            bz, parent, wrap, eta = self._parse_kwargs(wrap, eta, eta_key=eta_key)
            k = bz.k
            w = bz.weight
            for i in range(len(k)):
                yield wrap(method(*args, k=k[i], **kwargs), parent=parent, k=k[i], weight=w[i])
                eta.update()
            eta.close()

        return func


@set_module("sisl.physics")
class SumApply(IteratorApply):
    __str__ = _apply_str("sum over k")

    def dispatch(self, method):
        """ Dispatch the method by summing """
        iter_func = super().dispatch(method, eta_key="sum")

        @wraps(method)
        def func(*args, **kwargs):
            it = iter_func(*args, **kwargs)
            # next will be called before anything else
            return reduce(op.add, it, _asoplist(next(it)))

        return func


@set_module("sisl.physics")
class NoneApply(IteratorApply):
    __str__ = _apply_str("None")

    def dispatch(self, method):
        """ Dispatch the method by doing nothing (mostly useful if wrapped) """
        iter_func = super().dispatch(method, eta_key="none")

        @wraps(method)
        def func(*args, **kwargs):
            for _ in iter_func(*args, **kwargs):
                pass
            return None

        return func


@set_module("sisl.physics")
class ListApply(IteratorApply):
    __str__ = _apply_str("list")

    def dispatch(self, method):
        """ Dispatch the method by returning list of values """
        iter_func = super().dispatch(method, eta_key="list")
        @wraps(method)
        def func(*args, **kwargs):
            return [v for v in iter_func(*args, **kwargs)]
        return func


@set_module("sisl.physics")
class OpListApply(IteratorApply):
    __str__ = _apply_str("oplist")

    def dispatch(self, method):
        """ Dispatch the method by returning oplist of values """
        iter_func = super().dispatch(method, eta_key="oplist")
        @wraps(method)
        def func(*args, **kwargs):
            return oplist(v for v in iter_func(*args, **kwargs))
        return func


@set_module("sisl.physics")
class ArrayApply(BrillouinZoneParentApply):
    __str__ = _apply_str("numpy.ndarray")

    def dispatch(self, method, eta_key="array"):
        """ Dispatch the method by one array """
        @wraps(method)
        def func(*args, wrap=None, eta=False, **kwargs):
            bz, parent, wrap, eta = self._parse_kwargs(wrap, eta, eta_key=eta_key)
            k = bz.k
            w = bz.weight

            # Get first values
            v = wrap(method(*args, k=k[0], **kwargs), parent=parent, k=k[0], weight=w[0])
            eta.update()

            # Create full array
            if v.ndim == 0:
                a = np.empty([len(k)], dtype=v.dtype)
            else:
                a = np.empty((len(k), ) + v.shape, dtype=v.dtype)
            a[0] = v
            del v

            for i in range(1, len(k)):
                a[i] = wrap(method(*args, k=k[i], **kwargs), parent=parent, k=k[i], weight=w[i])
                eta.update()
            eta.close()

            return a
        return func


@set_module("sisl.physics")
class AverageApply(BrillouinZoneParentApply):
    __str__ = _apply_str("average")

    def dispatch(self, method):
        """ Dispatch the method by averaging """
        @wraps(method)
        def func(*args, wrap=None, eta=False, **kwargs):
            bz, parent, wrap, eta = self._parse_kwargs(wrap, eta, eta_key="average")
            # Do actual average
            k = bz.k
            w = bz.weight
            v = _asoplist(wrap(method(*args, k=k[0], **kwargs), parent=parent, k=k[0], weight=w[0])) * w[0]
            eta.update()
            for i in range(1, len(k)):
                v += _asoplist(wrap(method(*args, k=k[i], **kwargs), parent=parent, k=k[i], weight=w[i])) * w[i]
                eta.update()
            eta.close()
            return v

        return func


@set_module("sisl.physics")
class DataArrayApply(ArrayApply):
    __str__ = _apply_str("xarray.DataArray")

    def dispatch(self, method):
        """ Dispatch the method by returning a DataArray """
        # Get data as array
        array_func = super().dispatch(method, eta_key="dataarray")

        @wraps(method)
        def func(*args, coords=None, name=method.__name__, **kwargs):
            # xarray specific data (default to function name)
            bz = self._obj

            # retrieve ALL data
            array = array_func(*args, **kwargs)

            # Create coords
            if coords is None:
                coords = [('k', _a.arangei(len(bz)))]
                for i, v in enumerate(array.shape[1:]):
                    coords.append((f"v{i+1}", _a.arangei(v)))
            else:
                coords = list(coords)
                coords.insert(0, ('k', _a.arangei(len(bz))))
                for i in range(1, len(coords)):
                    if isinstance(coords[i], str):
                        coords[i] = (coords[i], _a.arangei(array.shape[i]))
            attrs = {'bz': bz, 'parent': bz.parent}

            return xarray.DataArray(array, coords=coords, name=name, attrs=attrs)

        return func


# Add dispatcher methods
# Since apply is a built-in, we cannot do "BrillouinZone.assign = ..."
setattr(BrillouinZone, "apply",
        ClassDispatcher("apply",
                        obj_getattr=lambda obj, key: getattr(obj.parent, key)
        )
)
# Register dispatched functions
BrillouinZone.apply.register("iter", IteratorApply, default=True)
BrillouinZone.apply.register("average", AverageApply)
BrillouinZone.apply.register("sum", SumApply)
BrillouinZone.apply.register("array", ArrayApply)
BrillouinZone.apply.register("none", NoneApply)
BrillouinZone.apply.register("list", ListApply)
BrillouinZone.apply.register("oplist", OpListApply)
if _has_xarray:
    BrillouinZone.apply.register("dataarray", DataArrayApply)
