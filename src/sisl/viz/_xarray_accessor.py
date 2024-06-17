# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""This module creates the sisl accessor in xarray to facilitate operations
on scientifically meaningful indices."""

import functools
import inspect

import xarray as xr

from .figure import get_figure
from .plotters.xarray import draw_xarray_xy
from .processors.atom import reduce_atom_data
from .processors.orbital import reduce_orbital_data, split_orbitals
from .processors.xarray import group_reduce


def wrap_accessor_method(fn):
    @functools.wraps(fn)
    def _method(self, *args, **kwargs):
        return fn(self._obj, *args, **kwargs)

    return _method


def plot_xy(*args, backend: str = "plotly", **kwargs):
    plot_actions = draw_xarray_xy(*args, **kwargs)

    return get_figure(plot_actions=plot_actions, backend=backend)


sig = inspect.signature(draw_xarray_xy)
plot_xy.__signature__ = sig.replace(
    parameters=[
        *sig.parameters.values(),
        inspect.Parameter("backend", inspect.Parameter.KEYWORD_ONLY, default="plotly"),
    ]
)


@xr.register_dataarray_accessor("sisl")
class SislAccessorDataArray:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    group_reduce = wrap_accessor_method(group_reduce)

    reduce_orbitals = wrap_accessor_method(reduce_orbital_data)

    split_orbitals = wrap_accessor_method(split_orbitals)

    reduce_atoms = wrap_accessor_method(reduce_atom_data)

    plot_xy = wrap_accessor_method(plot_xy)


@xr.register_dataset_accessor("sisl")
class SislAccessorDataset:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    group_reduce = wrap_accessor_method(group_reduce)

    reduce_orbitals = wrap_accessor_method(reduce_orbital_data)

    split_orbitals = wrap_accessor_method(split_orbitals)

    reduce_atoms = wrap_accessor_method(reduce_atom_data)

    plot_xy = wrap_accessor_method(plot_xy)
