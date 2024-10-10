# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

""" Injection module for adding apply functions for BrillouinZone classes

This module should not expose any methods!
"""
import operator as op
from functools import reduce, wraps
from itertools import zip_longest

import numpy as np
import xarray
from numpy import cross, pi

try:
    import pathos

    _has_pathos = True
except ImportError:
    _has_pathos = False

import sisl._array as _a
from sisl._core.grid import Grid
from sisl._core.lattice import Lattice
from sisl._core.oplist import oplist
from sisl._dispatcher import AbstractDispatch
from sisl._environ import get_environ_variable
from sisl._internal import set_module
from sisl.messages import SislError, progressbar, warn
from sisl.unit import units
from sisl.utils.mathematics import cart2spher
from sisl.utils.misc import allow_kwargs

# Stuff used for patching
from .brillouinzone import BrillouinZone, MonkhorstPack

# We expose the Apply and ParentApply classes
__all__ = ["BrillouinZoneApply", "BrillouinZoneParentApply"]
__all__ += ["MonkhorstPackApply", "MonkhorstPackParentApply"]


def _asoplist(arg):
    if isinstance(arg, (tuple, list)) and not isinstance(arg, oplist):
        return oplist(arg)
    return arg


def _correct_str(orig, insert):
    """Correct string with `insert`"""
    if len(insert) == 0:
        return orig
    i = orig.index("{") + 1
    return f"Apply{{{insert}, {orig[i:]}"


def _pool_procs(pool, size: int):
    """
    This is still a bit mysterious to me.

    One have to close/terminate + restart to get a functioning
    pool. Also, the pool's are not existing in a local scope. `pathos`
    have a global list of open pools to not pollute the pools.
    This means that one may accidentially request a pool that the user
    has openened elsewhere.

    This method is working via 3 different allowed argument types:

    pool = int | bool

        in this case it just uses defaults.

    pool = int | bool, {run}

        here the {run} is the arguments used for the mapping
        tool in the parallel region (i.e. `imap(..., **run)`

    pool = int | bool, {init}, {run}

        here the {init} is the arguments used for the constructor
        of the ProcessPool (except ``nodes``).
        And {run} is equivalent to the before.
    """
    nprocs = get_environ_variable("SISL_NUM_PROCS")
    nchunk = get_environ_variable("SISL_PAR_CHUNKSIZE")

    init = {}
    run = {}

    if isinstance(pool, (tuple, list)):
        run_in = {}
        init_in = {}
        if len(pool) == 1:
            pool = pool[0]
        elif len(pool) == 2:
            pool, run_in = pool
        elif len(pool) == 3:
            pool, init_in, run_in = pool
        else:
            raise ValueError("Too many arguments for pool=, expected a length of <=3")
        init.update(init_in)
        run.update(run_in)

    if pool is None:
        pool = _has_pathos

    if isinstance(pool, bool):
        if pool:
            pool = nprocs
        else:
            pool = 1

    if isinstance(pool, int):
        if pool <= 1:
            return None, run

        ncpus = pool
        import pathos

        pool = pathos.pools.ProcessPool(ncpus=pool, **init)
    else:

        ncpus = None
        try:
            ncpus = pool.ncpus
        except Exception:
            pass

        if ncpus is None:
            try:
                ncpus = pool._processes
            except Exception:
                pass

        # we don't know
        warn(
            f"{__module__} could not determine number of CPUs from the pool, expecting 2"
        )
        ncpus = 2

    if "chunksize" not in run:
        if isinstance(nchunk, float):
            nchunk = max(1, int(size / ncpus * nchunk))
            # Try to make the chunksize as evenly distributed as possible
            tmp = size // nchunk
            nchunk = size // tmp
        run["chunksize"] = nchunk

    # Prepare the pool, just in case it has already been used.
    pool.terminate()
    pool.join()

    return pool, run


@set_module("sisl.physics")
class BrillouinZoneApply(AbstractDispatch):
    # this dispatch function will do stuff on the BrillouinZone object
    pass


@set_module("sisl.physics")
class BrillouinZoneParentApply(BrillouinZoneApply):
    def __str__(self, message=""):
        return _correct_str(super().__str__(), message)

    def _parse_kwargs(self, wrap, eta=None, eta_key=""):
        """Parse kwargs"""
        bz = self._obj
        parent = bz.parent
        if wrap is None:
            # we always return a wrap
            def wrap(v, parent=None, k=None, weight=None):
                return v

        else:
            wrap = allow_kwargs("parent", "k", "weight")(wrap)
        eta = progressbar(len(bz), f"{bz.__class__.__name__}.{eta_key}", "k", eta)
        return bz, parent, wrap, eta

    def __getattr__(self, key):
        # We need to offload the dispatcher to retrieve
        # methods from the parent object
        # This dispatch will _never_ do anything to the BrillouinZone
        method = getattr(self._obj.parent, key)
        return self.dispatch(method)


@set_module("sisl.physics")
class IteratorApply(BrillouinZoneParentApply):
    """Iterator for calculations *while iterating*."""

    def __str__(self, message="iter"):
        return super().__str__(message)

    def dispatch(self, method, eta_key="iter"):
        """Dispatch the method by iterating values"""
        pool, pool_run = _pool_procs(self._attrs.get("pool"), len(self._get_object()))

        if pool is None:

            @wraps(method)
            def func(*args, wrap=None, eta=None, **kwargs):
                bz, parent, wrap, eta = self._parse_kwargs(wrap, eta, eta_key=eta_key)
                k = bz.k
                w = bz.weight
                for i, ki in enumerate(k):
                    yield wrap(
                        method(*args, k=ki, **kwargs),
                        parent=parent,
                        k=ki,
                        weight=w[i],
                    )
                    eta.update()
                eta.close()

        else:

            @wraps(method)
            def func(*args, wrap=None, eta=None, **kwargs):
                pool.restart(True)
                bz, parent, wrap, eta = self._parse_kwargs(wrap, eta, eta_key=eta_key)
                k = bz.k
                w = bz.weight

                def func(k, w):
                    return wrap(
                        method(*args, k=k, **kwargs), parent=parent, k=k, weight=w
                    )

                for ret in pool.imap(func, k, w, **pool_run):
                    eta.update()
                    yield ret

                # TODO notify users that this may be bad when used with zip
                # unless this generator is the first argument of zip
                # zip has left-to-right checks of length and stops querying
                # elements as soon as the left-most one stops.
                pool.close()
                pool.join()
                eta.close()

        return func


@set_module("sisl.physics")
class SumApply(IteratorApply):
    def __str__(self, message="sum over k"):
        return super().__str__(message)

    def dispatch(self, method):
        """Dispatch the method by summing"""
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
        """Dispatch the method by doing nothing (mostly useful if wrapped)"""
        iter_func = super().dispatch(method, eta_key="none")

        @wraps(method)
        def func(*args, **kwargs):
            for _ in iter_func(*args, **kwargs):
                pass

        return func


@set_module("sisl.physics")
class ListApply(IteratorApply):
    def __str__(self, message="list"):
        return super().__str__(message)

    def dispatch(self, method):
        """Dispatch the method by returning list of values"""
        iter_func = super().dispatch(method, eta_key="list")
        if self._attrs.get("zip", self._attrs.get("unzip", False)):

            @wraps(method)
            def func(*args, **kwargs):
                return zip(*(v for v in iter_func(*args, **kwargs)))

        else:

            @wraps(method)
            def func(*args, **kwargs):
                return list(iter_func(*args, **kwargs))

        return func


@set_module("sisl.physics")
class OpListApply(IteratorApply):
    def __str__(self, message="oplist"):
        return super().__str__(message)

    def dispatch(self, method):
        """Dispatch the method by returning oplist of values"""
        iter_func = super().dispatch(method, eta_key="oplist")
        if self._attrs.get("zip", self._attrs.get("unzip", False)):

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
        """Dispatch the method by one array"""
        pool, pool_run = _pool_procs(self._attrs.get("pool"), len(self._get_object()))
        unzip = self._attrs.get("zip", self._attrs.get("unzip", False))

        def _create_v(nk, v):
            out = np.empty((nk, *v.shape), dtype=v.dtype)
            out[0] = v
            return out

        if pool is None:

            @wraps(method)
            def func(*args, wrap=None, eta=None, **kwargs):
                bz, parent, wrap, eta = self._parse_kwargs(wrap, eta, eta_key=eta_key)
                k = bz.k
                nk = len(k)
                w = bz.weight

                # Get first values
                v = wrap(
                    method(*args, k=k[0], **kwargs), parent=parent, k=k[0], weight=w[0]
                )
                eta.update()

                if unzip:
                    a = tuple(_create_v(nk, vi) for vi in v)
                    for i, ki in enumerate(k[1:], 1):
                        v = wrap(
                            method(*args, k=ki, **kwargs),
                            parent=parent,
                            k=ki,
                            weight=w[i],
                        )
                        for ai, vi in zip(a, v):
                            ai[i] = vi
                        eta.update()
                else:
                    a = _create_v(nk, v)
                    del v
                    for i, ki in enumerate(k[1:], 1):
                        a[i] = wrap(
                            method(*args, k=ki, **kwargs),
                            parent=parent,
                            k=ki,
                            weight=w[i],
                        )
                        eta.update()
                eta.close()

                return a

        else:

            @wraps(method)
            def func(*args, wrap=None, eta=None, **kwargs):
                pool.restart(True)
                bz, parent, wrap, eta = self._parse_kwargs(wrap, eta, eta_key=eta_key)
                k = bz.k
                nk = len(k)
                w = bz.weight

                def func(k, w):
                    return wrap(
                        method(*args, k=k, **kwargs), parent=parent, k=k, weight=w
                    )

                it = pool.imap(func, k, w, **pool_run)
                v = next(it)
                eta.update()

                if unzip:
                    a = tuple(_create_v(nk, vi) for vi in v)
                    for i, v in enumerate(it, 1):
                        for ai, vi in zip(a, v):
                            ai[i] = vi
                        eta.update()
                else:
                    a = _create_v(nk, v)
                    for i, v in enumerate(it, 1):
                        a[i] = v
                        eta.update()
                del v
                pool.close()
                pool.join()
                eta.close()
                return a

        return func


@set_module("sisl.physics")
class AverageApply(BrillouinZoneParentApply):
    def __str__(self, message="average"):
        return super().__str__(message)

    def dispatch(self, method):
        """Dispatch the method by averaging"""
        pool, pool_run = _pool_procs(self._attrs.get("pool"), len(self._get_object()))

        if pool is None:

            @wraps(method)
            def func(*args, wrap=None, eta=None, **kwargs):
                bz, parent, wrap, eta = self._parse_kwargs(wrap, eta, eta_key="average")
                # Do actual average
                k = bz.k
                w = bz.weight
                v = (
                    _asoplist(
                        wrap(
                            method(*args, k=k[0], **kwargs),
                            parent=parent,
                            k=k[0],
                            weight=w[0],
                        )
                    )
                    * w[0]
                )
                eta.update()
                for i, ki in enumerate(k[1:], 1):
                    v += (
                        _asoplist(
                            wrap(
                                method(*args, k=ki, **kwargs),
                                parent=parent,
                                k=ki,
                                weight=w[i],
                            )
                        )
                        * w[i]
                    )
                    eta.update()
                eta.close()
                return v

        else:

            @wraps(method)
            def func(*args, wrap=None, eta=None, **kwargs):
                pool.restart(True)

                bz, parent, wrap, eta = self._parse_kwargs(wrap, eta, eta_key="average")
                k = bz.k
                w = bz.weight

                def func(k, w):
                    return w * _asoplist(
                        wrap(method(*args, k=k, **kwargs), parent=parent, k=k, weight=w)
                    )

                iret = pool.imap(func, k, w, **pool_run)
                avg = next(iret)
                eta.update()
                for it in iret:
                    avg += it
                    eta.update()

                pool.close()
                pool.join()

                eta.close()
                return avg

        return func


@set_module("sisl.physics")
class XArrayApply(NDArrayApply):
    def __str__(self, message="xarray"):
        return super().__str__(message)

    def dispatch(self, method):
        """Dispatch the method by returning a DataArray or data-set"""

        def _fix_coords_dims(nk, array, coords, dims, prefix="v"):
            if coords is None and dims is None:
                # we need to manually create them
                coords = [("k", _a.arangei(nk))]
                for i, v in enumerate(array.shape[1:]):
                    coords.append((f"{prefix}{i+1}", _a.arangei(v)))
            elif coords is None:
                coords = [("k", _a.arangei(nk))]
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
                if isinstance(coords, str):
                    coords = [coords]
                coords = list(coords)
                coords.insert(0, ("k", _a.arangei(nk)))
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
        if self._attrs.get("zip", self._attrs.get("unzip", False)):
            array_func = super().dispatch(method, eta_key="dataset")

            @wraps(method)
            def func(*args, coords=(), dims=(), name=method.__name__, **kwargs):
                # xarray specific data (default to function name)
                bz = self._obj
                # retrieve ALL data
                array = array_func(*args, **kwargs)

                def _create_DA(array, coords, dims, name):
                    coords, dims = _fix_coords_dims(
                        len(bz), array, coords, dims, prefix=f"{name}.v"
                    )
                    return xarray.DataArray(array, coords=coords, dims=dims, name=name)

                if isinstance(name, str):
                    name = [f"{name}{i}" for i in range(len(array))]

                data = {
                    nam: _create_DA(arr, coord, dim, nam)
                    for arr, coord, dim, nam in zip_longest(array, coords, dims, name)
                }

                attrs = {"bz": bz, "parent": bz.parent}
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
                attrs = {"bz": bz, "parent": bz.parent}
                return xarray.DataArray(
                    array, coords=coords, dims=dims, name=name, attrs=attrs
                )

        return func


# Register dispatched functions
apply_dispatch = BrillouinZone.apply
apply_dispatch.register("iter", IteratorApply, default=True)
apply_dispatch.register("average", AverageApply)
apply_dispatch.register("sum", SumApply)
apply_dispatch.register("array", NDArrayApply)
apply_dispatch.register("ndarray", NDArrayApply)
apply_dispatch.register("none", NoneApply)
apply_dispatch.register("list", ListApply)
apply_dispatch.register("oplist", OpListApply)
apply_dispatch.register("dataarray", XArrayApply)
apply_dispatch.register("xarray", XArrayApply)

# Remove refernce
del apply_dispatch


@set_module("sisl.physics")
class MonkhorstPackApply(BrillouinZoneApply):
    # this dispatch function will do stuff on the BrillouinZone object
    pass


@set_module("sisl.physics")
class MonkhorstPackParentApply(MonkhorstPackApply):
    def __str__(self, message=""):
        return _correct_str(super().__str__(), message)

    def _parse_kwargs(self, wrap, eta=None, eta_key=""):
        """Parse kwargs"""
        bz = self._obj
        parent = bz.parent
        if wrap is None:
            # we always return a wrap
            def wrap(v, parent=None, k=None, weight=None):
                return v

        else:
            wrap = allow_kwargs("parent", "k", "weight")(wrap)
        eta = progressbar(len(bz), f"{bz.__class__.__name__}.{eta_key}", "k", eta)
        return bz, parent, wrap, eta

    def __getattr__(self, key):
        # We need to offload the dispatcher to retrieve
        # methods from the parent object
        # This dispatch will _never_ do anything to the BrillouinZone
        method = getattr(self._obj.parent, key)
        return self.dispatch(method)


@set_module("sisl.physics")
class GridApply(MonkhorstPackParentApply):
    """Calculate on a Grid

    The calculation of values on a grid requires some careful thought before
    running the calculation as the returned grid may be somewhat difficult
    to comprehend.

    Notes
    -----
    All invocations of sub-methods are added these keyword-only arguments:

    eta : bool, optional
        if true a progress-bar is created, default false.
    wrap : callable, optional
        a function that accepts the output of the given routine and post-process
        it. Defaults to ``lambda x: x``.
    data_axis : int, optional
        the Grid axis to put in the data values in. Has to be specified if the
        subsequent routine calls return more than 1 data-point per k-point.
    grid_unit : {'b', 'Ang', 'Bohr'}, optional
        for 'b' the returned grid will be a cube, otherwise the grid will be the reciprocal lattice
        vectors (for any other value) and in the given reciprocal unit ('Ang' => 1/Ang)

    Examples
    --------
    >>> obj = MonkhorstPack(Hamiltonian, [10, 1, 10])
    >>> grid = obj.asgrid().eigh(data_axis=1)
    """

    def __str__(self, message="grid"):
        return super().__str__(message)

    def dispatch(self, method, eta_key="grid"):
        """Dispatch the method by putting values on the grid"""

        @wraps(method)
        def func(*args, wrap=None, eta=None, **kwargs):
            data_axis = kwargs.pop("data_axis", None)
            grid_unit = kwargs.pop("grid_unit", "b")

            mp, parent, wrap, eta = self._parse_kwargs(wrap, eta, eta_key=eta_key)
            k = mp.k
            w = mp.weight

            # Extract information from the MP grid, these values
            # define the Grid size, etc.
            diag = mp._diag.copy()
            if not np.all(mp._displ == 0):
                raise SislError(
                    f"{mp.__class__.__name__ } requires the displacement to be 0 for all k-points."
                )
            displ = mp._displ.copy()
            size = mp._size.copy()
            steps = size / diag
            if mp._centered:
                offset = np.where(diag % 2 == 0, steps, steps / 2)
            else:
                offset = np.where(diag % 2 == 0, steps / 2, steps)

            # Instead of doing
            #    _in_primitive(k) + 0.5 - offset
            # we can do it here
            #    _in_primitive(k) + offset'
            offset -= 0.5

            # Check the TRS direction
            trs_axis = mp._trs
            _in_primitive = mp.in_primitive
            _rint = np.rint
            _int32 = np.int32

            def k2idx(k):
                # In case TRS is applied two indices may be returned
                return _rint((_in_primitive(k) - offset) / steps).astype(_int32)
                # To find the opposite k-point, do this
                #  idx[i] = [diag[i] - idx[i] - 1, idx[i]
                # with i in [0, 1, 2]

            # Create cell from the reciprocal cell.
            if grid_unit == "b":
                cell = np.diag(mp._size)
            else:
                cell = (
                    parent.lattice.rcell
                    * mp._size.reshape(1, -1)
                    / units("Ang", grid_unit)
                )

            # Find the grid origin
            origin = -(cell * 0.5).sum(0)

            # Calculate first k-point (to get size and dtype)
            v = wrap(
                method(*args, k=k[0], **kwargs), parent=parent, k=k[0], weight=w[0]
            )

            if data_axis is None:
                if v.size != 1:
                    raise SislError(
                        f"{self.__class__.__name__} {func.__name__} requires one value per-kpoint because of the 3D grid values"
                    )

            else:
                # Check the weights
                weights = mp.grid(
                    diag[data_axis],
                    displ[data_axis],
                    size[data_axis],
                    centered=mp._centered,
                    trs=trs_axis == data_axis,
                )[1]

                # Correct the Grid size
                diag[data_axis] = len(v)
                # Create the orthogonal cell direction to ensure it is orthogonal
                # Since array axis is cyclic for negative numbers, we simply do this
                cell[data_axis, :] = cross(
                    cell[data_axis - 1, :], cell[data_axis - 2, :]
                )
                # Check whether we should rotate it
                if cart2spher(cell[data_axis, :])[2] > pi / 4:
                    cell[data_axis, :] *= -1

            # Correct cell for the grid
            if trs_axis >= 0:
                origin[trs_axis] = 0.0
                # Correct offset since we only have the positive halve
                if mp._diag[trs_axis] % 2 == 0 and not mp._centered:
                    offset[trs_axis] = steps[trs_axis] / 2
                else:
                    offset[trs_axis] = 0.0

                # Find number of points
                if trs_axis != data_axis:
                    diag[trs_axis] = len(
                        mp.grid(
                            diag[trs_axis],
                            displ[trs_axis],
                            size[trs_axis],
                            centered=mp._centered,
                            trs=True,
                        )[1]
                    )

            # Create the grid in the reciprocal cell
            lattice = Lattice(cell, origin=origin)
            grid = Grid(diag, lattice=lattice, dtype=v.dtype)
            if data_axis is None:
                grid[k2idx(k[0])] = v
            else:
                idx = k2idx(k[0]).tolist()
                weight = weights[idx[data_axis]]
                idx[data_axis] = slice(None)
                grid[tuple(idx)] = v * weight

            del v

            # Now perform calculation
            eta.update()
            if data_axis is None:
                for i, ki in enumerate(k[1:], 1):
                    grid[k2idx(ki)] = wrap(
                        method(*args, k=ki, **kwargs),
                        parent=parent,
                        k=ki,
                        weight=w[i],
                    )
                    eta.update()
            else:
                for i, ki in enumerate(k[1:], 1):
                    idx = k2idx(ki).tolist()
                    weight = weights[idx[data_axis]]
                    idx[data_axis] = slice(None)
                    grid[tuple(idx)] = (
                        wrap(
                            method(*args, k=ki, **kwargs),
                            parent=parent,
                            k=ki,
                            weight=w[i],
                        )
                        * weight
                    )
                    eta.update()
            eta.close()
            return grid

        return func


# Register dispatched functions
apply_dispatch = MonkhorstPack.apply
apply_dispatch.register("grid", GridApply)

del apply_dispatch
