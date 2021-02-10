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
class ArrayApply(BrillouinZoneParentApply):
    def __str__(self, message="numpy.ndarray"):
        return super().__str__(message)

    def dispatch(self, method, eta_key="array"):
        """ Dispatch the method by one array """
        pool = _pool_procs(self._attrs.get("pool", None))
        if pool is None:
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
        else:
            @wraps(method)
            def func(*args, wrap=None, **kwargs):
                pool.restart()
                bz, parent, wrap, _ = self._parse_kwargs(wrap)
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
class DataArrayApply(ArrayApply):
    def __str__(self, message="xarray.DataArray"):
        return super().__str__(message)

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
BrillouinZone.apply.register("array", ArrayApply)
BrillouinZone.apply.register("none", NoneApply)
BrillouinZone.apply.register("list", ListApply)
BrillouinZone.apply.register("oplist", OpListApply)
BrillouinZone.apply.register("dataarray", DataArrayApply)
