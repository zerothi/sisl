""" Injection module for adding apply functions for BrillouinZone classes

This module should not expose any methods!
"""
from functools import wraps, reduce
from itertools import zip_longest
import operator as op

import numpy as np
try:
    import xarray
    _has_xarray = True
except ImportError:
    _has_xarray = False

from sisl._dispatcher import ClassDispatcher, AbstractDispatch
from sisl._environ import get_environ_variable
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
    if isinstance(arg, (tuple, list)) and not isinstance(arg, oplist):
        return oplist(arg)
    return arg


def _correct_str(orig, insert):
    """ Correct string with `insert` """
    if len(insert) == 0:
        return orig
    i = orig.index("{") + 1
    return f"Apply{{{insert}, {orig[i:]}"


def _pool_procs(pool):
    """
    This is still a bit mysterious to me.

    One have to close/terminate + restart to get a functioning
    pool. Also, the pool's are not existing in a local scope. `pathos`
    have a global list of open pools to not pollute the pools.
    This means that one may accidentially request a pool that the user
    has openened elsewhere.
    """
    if pool is False or pool is None:
        return None
    elif pool is True:
        import pathos as pos
        pool = pos.pools.ProcessPool(nodes=get_environ_variable("SISL_NPROCS"))
    elif isinstance(pool, int):
        import pathos as pos
        pool = pos.pools.ProcessPool(nodes=pool)
    pool.terminate()
    return pool


@set_module("sisl.physics")
class BrillouinZoneApply(AbstractDispatch):
    # this dispatch function will do stuff on the BrillouinZone object
    __slots__ = ()


@set_module("sisl.physics")
class BrillouinZoneParentApply(BrillouinZoneApply):

    def __str__(self, message=''):
        return _correct_str(super().__str__(), message)

    def _parse_kwargs(self, wrap, eta=False, eta_key=""):
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
    def __str__(self, message="iter"):
        return super().__str__(message)

    def dispatch(self, method, eta_key="iter"):
        """ Dispatch the method by iterating values """
        pool = _pool_procs(self._attrs.get("pool", None))
        if pool is None:
            @wraps(method)
            def func(*args, wrap=None, eta=False, **kwargs):
                bz, parent, wrap, eta = self._parse_kwargs(wrap, eta, eta_key=eta_key)
                k = bz.k
                w = bz.weight
                for i in range(len(k)):
                    yield wrap(method(*args, k=k[i], **kwargs), parent=parent, k=k[i], weight=w[i])
                    eta.update()
                eta.close()
        else:
            @wraps(method)
            def func(*args, wrap=None, eta=False, **kwargs):
                pool.restart()
                bz, parent, wrap, _ = self._parse_kwargs(wrap)
                k = bz.k
                w = bz.weight

                def func(k, w):
                    return wrap(method(*args, k=k, **kwargs), parent=parent, k=k, weight=w)

                yield from pool.imap(func, k, w)
                # TODO notify users that this may be bad when used with zip
                # unless this generator is the first argument of zip
                # zip has left-to-right checks of length and stops querying
                # elements as soon as the left-most one stops.
                pool.terminate()

        return func


@set_module("sisl.physics")
class SumApply(IteratorApply):
    def __str__(self, message="sum over k"):
        return super().__str__(message)

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
    def __str__(self, message="None"):
        return super().__str__(message)

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
    def __str__(self, message="list"):
        return super().__str__(message)

    def dispatch(self, method):
        """ Dispatch the method by returning list of values """
        iter_func = super().dispatch(method, eta_key="list")
        if self._attrs.get("unzip", False):
            @wraps(method)
            def func(*args, **kwargs):
                return zip(*(v for v in iter_func(*args, **kwargs)))
        else:
            @wraps(method)
            def func(*args, **kwargs):
                return [v for v in iter_func(*args, **kwargs)]
        return func


@set_module("sisl.physics")
class OpListApply(IteratorApply):
    def __str__(self, message="oplist"):
        return super().__str__(message)

    def dispatch(self, method):
        """ Dispatch the method by returning oplist of values """
        iter_func = super().dispatch(method, eta_key="oplist")
        if self._attrs.get("unzip", False):
            @wraps(method)
            def func(*args, **kwargs):
                return oplist(zip(*(v for v in iter_func(*args, **kwargs))))
        else:
            @wraps(method)
            def func(*args, **kwargs):
                return oplist(v for v in iter_func(*args, **kwargs))
        return func


@set_module("sisl.physics")
class NDArrayApply(BrillouinZoneParentApply):
    def __str__(self, message="ndarray"):
        return super().__str__(message)

    def dispatch(self, method, eta_key="ndarray"):
        """ Dispatch the method by one array """
        pool = _pool_procs(self._attrs.get("pool", None))
        unzip = self._attrs.get("unzip", False)

        def _create_v(nk, v):
            out = np.empty((nk, *v.shape), dtype=v.dtype)
            out[0] = v
            return out

        if pool is None:
            @wraps(method)
            def func(*args, wrap=None, eta=False, **kwargs):
                bz, parent, wrap, eta = self._parse_kwargs(wrap, eta, eta_key=eta_key)
                k = bz.k
                nk = len(k)
                w = bz.weight

                # Get first values
                v = wrap(method(*args, k=k[0], **kwargs), parent=parent, k=k[0], weight=w[0])
                eta.update()

                if unzip:
                    a = tuple(_create_v(nk, vi) for vi in v)
                    for i in range(1, len(k)):
                        v = wrap(method(*args, k=k[i], **kwargs), parent=parent, k=k[i], weight=w[i])
                        for ai, vi in zip(a, v):
                            ai[i] = vi
                        eta.update()
                else:
                    a = _create_v(nk, v)
                    del v
                    for i in range(1, len(k)):
                        a[i] = wrap(method(*args, k=k[i], **kwargs), parent=parent, k=k[i], weight=w[i])
                        eta.update()
                eta.close()

                return a
        else:
            @wraps(method)
            def func(*args, wrap=None, **kwargs):
                pool.restart()
                bz, parent, wrap, _ = self._parse_kwargs(wrap)
                k = bz.k
                nk = len(k)
                w = bz.weight

                def func(k, w):
                    return wrap(method(*args, k=k, **kwargs), parent=parent, k=k, weight=w)

                it = pool.imap(func, k, w)
                v = next(it)

                if unzip:
                    a = tuple(_create_v(nk, vi) for vi in v)
                    for i, v in enumerate(it):
                        i += 1
                        for ai, vi in zip(a, v):
                            ai[i] = vi
                else:
                    a = _create_v(nk, v)
                    for i, v in enumerate(it):
                        a[i+1] = v
                del v
                pool.terminate()
                return a

        return func


@set_module("sisl.physics")
class AverageApply(BrillouinZoneParentApply):
    def __str__(self, message="average"):
        return super().__str__(message)

    def dispatch(self, method):
        """ Dispatch the method by averaging """
        pool = _pool_procs(self._attrs.get("pool", None))
        if pool is None:
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
        else:
            @wraps(method)
            def func(*args, wrap=None, **kwargs):
                pool.restart()
                bz, parent, wrap, _ = self._parse_kwargs(wrap)
                k = bz.k
                w = bz.weight

                def func(k, w):
                    return wrap(method(*args, k=k, **kwargs), parent=parent, k=k, weight=w) * w

                iter_func = pool.uimap(func, k, w)
                avg = reduce(op.add, iter_func, _asoplist(next(iter_func)))
                pool.terminate()
                return avg

        return func


@set_module("sisl.physics")
class XArrayApply(NDArrayApply):
    def __str__(self, message="xarray"):
        return super().__str__(message)

    def dispatch(self, method):
        """ Dispatch the method by returning a DataArray or data-set """

        def _fix_coords_dims(nk, array, coords, dims, prefix="v"):
            if coords is None and dims is None:
                # we need to manually create them
                coords = [('k', _a.arangei(nk))]
                for i, v in enumerate(array.shape[1:]):
                    coords.append((f"{prefix}{i+1}", _a.arangei(v)))
            elif coords is None:
                coords = [('k', _a.arangei(nk))]
                for i, v in enumerate(array.shape[1:]):
                    coords.append((dims[i], _a.arangei(v)))
                # everything is in coords, no need to pass dims
                dims = None
            elif isinstance(coords, dict):
                # ensure coords has "k" as dimensions
                if "k" not in coords:
                    coords["k"] = _a.arangei(nk)
                    dims = list(dims)
                    # ensure we have dims first
                    dims.insert(0, "k")
            else:
                # add "k" correctly
                coords = list(coords)
                coords.insert(0, ('k', _a.arangei(nk)))
                for i in range(1, len(coords)):
                    if isinstance(coords[i], str):
                        coords[i] = (coords[i], _a.arangei(array.shape[i]))
                # ensure dims is not used, everything is in coords
                # and since it is a list, it should also be the correct order
                # TODO add check that dims and coords match in order
                # or convert dims to list and prepend "k"
                dims = None
            return coords, dims

        # Get data as array
        if self._attrs.get("unzip", False):
            array_func = super().dispatch(method, eta_key="dataset")

            @wraps(method)
            def func(*args, coords=(), dims=(), name=method.__name__, **kwargs):
                # xarray specific data (default to function name)
                bz = self._obj
                # retrieve ALL data
                array = array_func(*args, **kwargs)

                def _create_DA(array, coords, dims, name):
                    coords, dims = _fix_coords_dims(len(bz), array, coords, dims,
                                                    prefix=f"{name}.v")
                    return xarray.DataArray(array, coords=coords, dims=dims, name=name)

                if isinstance(name, str):
                    name = [f"{name}{i}" for i in range(len(array))]

                data = {nam: _create_DA(arr, coord, dim, nam)
                        for arr, coord, dim, nam
                        in zip_longest(array, coords, dims, name)
                }

                attrs = {'bz': bz, 'parent': bz.parent}
                return xarray.Dataset(data, attrs=attrs)
        else:
            array_func = super().dispatch(method, eta_key="dataarray")

            @wraps(method)
            def func(*args, coords=None, dims=None, name=method.__name__, **kwargs):
                # xarray specific data (default to function name)
                bz = self._obj
                # retrieve ALL data
                array = array_func(*args, **kwargs)
                coords, dims = _fix_coords_dims(len(bz), array, coords, dims)
                attrs = {'bz': bz, 'parent': bz.parent}
                return xarray.DataArray(array, coords=coords, dims=dims, name=name, attrs=attrs)

        return func

# Add dispatcher methods
# Since apply is a built-in, we cannot do "BrillouinZone.apply = ..."
setattr(BrillouinZone, "apply",
        ClassDispatcher("apply",
                        obj_getattr=lambda obj, key: getattr(obj.parent, key)
        )
)
# Register dispatched functions
BrillouinZone.apply.register("iter", IteratorApply, default=True)
BrillouinZone.apply.register("average", AverageApply)
BrillouinZone.apply.register("sum", SumApply)
BrillouinZone.apply.register("array", NDArrayApply)
BrillouinZone.apply.register("ndarray", NDArrayApply)
BrillouinZone.apply.register("none", NoneApply)
BrillouinZone.apply.register("list", ListApply)
BrillouinZone.apply.register("oplist", OpListApply)
BrillouinZone.apply.register("dataarray", XArrayApply)
BrillouinZone.apply.register("xarray", XArrayApply)
