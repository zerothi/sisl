"""This module creates the sisl accessor in xarray to facilitate operations
on scientifically meaningful indices."""

import functools
import xarray as xr
from sisl.viz.nodes.plots.plot import Plot
from sisl.viz.nodes.plotters.plotter import PlotterNodeXY

from sisl.viz.nodes.processors.xarray import group_reduce
from sisl.viz.nodes.processors.atom import reduce_atom_data
from sisl.viz.nodes.processors.orbital import reduce_orbital_data

def wrap_accessor_method(fn):
    @functools.wraps(fn)
    def _method(self, *args, **kwargs):
        print(args)
        return fn(self._obj, *args, **kwargs)

    return _method

def plot_xy(*args, **kwargs):
    return PlotterNodeXY(*args, **kwargs)
    
plot_xy.__signature__ = PlotterNodeXY.__signature__

plot_xy = Plot.from_func(plot_xy)

@xr.register_dataarray_accessor("sisl")
class SislAccessorDataArray:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    group_reduce = wrap_accessor_method(group_reduce)

    reduce_orbitals = wrap_accessor_method(reduce_orbital_data)

    reduce_atoms = wrap_accessor_method(reduce_atom_data)

    plot_xy = wrap_accessor_method(plot_xy)

@xr.register_dataset_accessor("sisl")
class SislAccessorDataset:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    group_reduce = wrap_accessor_method(group_reduce)

    reduce_orbitals = wrap_accessor_method(reduce_orbital_data)

    reduce_atoms = wrap_accessor_method(reduce_atom_data)

    plot_xy = wrap_accessor_method(plot_xy)