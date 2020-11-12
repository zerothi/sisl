"""
This file defines all the siles that are plotable.

It does so by patching the classes accordingly

In the future, sisl objects will probably be 'plotable' too
"""
from functools import partial
from types import MethodType

import numpy as np

import sisl.io.siesta as siesta
import sisl.io.tbtrans as tbtrans
from sisl.io.sile import get_siles, BaseSile

import sisl
from .plots import *
from .plot import Plot
from .plotutils import get_plot_classes

from .._plotables import register_plotable

__all__= ["register_plotly_plotable"]
# -----------------------------------------------------
#   Let's define the functions that will help us here
# -----------------------------------------------------


def _get_plotting_func(PlotClass, setting_key):
    """
    Generates a plotting function for an object.

    Parameters
    -----------
    PlotClass: child of Plot
        the plot class that you want to use to plot the object.
    setting_key: str
        the setting where the plotable should go

    Returns
    -----------
    function
        a function that accepts the object as first argument and then generates the plot.

        It sends the object to the appropiate setting key. The rest works exactly the same as
        calling the plot class. I.e. you can provide all the extra settings/keywords that you want.  
    """

    def _plot(self, *args, **kwargs):

        return PlotClass(*args, **{setting_key: self, **kwargs})

    _plot.__doc__ = f"""Builds a {PlotClass.__name__} by setting the value of "{setting_key}" to the current object.

    Apart from this specific parameter ,it accepts the same arguments as {PlotClass.__name__}.
    
    Documentation for {PlotClass.__name__}
    -------------
    
    {PlotClass.__doc__}
    """
    return _plot


def register_plotly_plotable(plotable, PlotClass=None, setting_key=None, plotting_func=None,
    name=None, default=False, plot_handler_attr='plot'):
    """
    Makes the sisl.viz module aware of which sisl objects have a plotly representation available.

    Basically, this handles generating the plotting functions from a PlotClass to make the process
    easier and create a consistent behavior that other parts of the framework can rely on (i.e. the GUI).

    Once a plotting function is generated, the rest is handled by `sisl.viz.register_plotable`.

    Parameters
    ------------
    plotable: any
        any class or object that you want to make plotable. Note that, if it's an object, the plotting
        capabilities will be attributed ONLY to that object, not the whole class. You can change this
        behavior by setting the `all_instances` parameter to True.
    PlotClass: child of sisl.Plot, optional
        The class of the Plot that we want this object to use.
    setting_key: str, optional
        The key of the setting where the object must go. This works together with
        the PlotClass parameter.
    name: str, optional
        name that will be used to identify the particular plot function that is being registered.

        E.g.: If name is "nicely", the plotting function will be registered under "obj.plot_nicely()"

        IF THE PLOT CLASS YOU ARE USING IS NOT ALREADY REGISTERED FOR THE PLOTABLE, PLEASE LET THE NAME
        BE HANDLED AUTOMATICALLY UNLESS YOU HAVE A GOOD REASON NOT TO DO SO. This will help keeping consistency
        across the different objects as the name is determined by the plot class that is being used.
    plotting_func: function, optional
        if the PlotClass - setting_key pair does not satisfy your needs, you can pass a more complex function here
        instead.
        It should accept (self, *args, **kwargs) and return a plot object.
    default: boolean, optional 
        whether this way of plotting the class should be the default one.
    plot_handler_attr: str, optional
        the attribute where the plot handler is or should be located in the class that you want to register.
    """

    # If no plotting function is provided, we will try to create one by using the PlotClass
    # and the setting_key that have been provided
    if plotting_func is None:
        plotting_func = _get_plotting_func(PlotClass, setting_key)

    if name is None:
        # We will take the name of the plot class as the name
        name = PlotClass.suffix()

    register_plotable(
        plotable, plotting_func, name=name, engine='plotly', default=default, plot_handler_attr=plot_handler_attr
    )

    # And to help keep track of the plotability we tell to the plot class that
    # it can plot this object, and which setting to use.
    # if PlotClass is not None:
    #     if not hasattr(PlotClass, '_registered_plotables'):
    #         PlotClass._registered_plotables = {}

    #     PlotClass._registered_plotables[plotable] = setting_key

# -----------------------------------------------------
#               Register plotable siles
# -----------------------------------------------------

register = register_plotly_plotable

for GridSile in get_siles(attrs=["read_grid"]):
    register(GridSile, GridPlot, 'grid_file', default=True)

for GeomSile in get_siles(attrs=["read_geometry"]):
    register(GeomSile, GeometryPlot, 'geom_file', default=True)
    register(GeomSile, BondLengthMap, 'geom_file')

for HSile in get_siles(attrs=["read_hamiltonian"]):
    register(HSile, WavefunctionPlot, 'H', default=HSile != siesta.fdfSileSiesta)
    register(HSile, PdosPlot, "H")
    register(HSile, BandsPlot, "H")
    register(HSile, FatbandsPlot, "H")

for cls in get_plot_classes():
    register(siesta.fdfSileSiesta, cls, "root_fdf")

register(siesta.outSileSiesta, ForcesPlot, 'out_file', default=True)

register(siesta.bandsSileSiesta, BandsPlot, 'bands_file', default=True)
register(siesta.bandsSileSiesta, FatbandsPlot, 'bands_file')

register(siesta.pdosSileSiesta, PdosPlot, 'pdos_file', default=True)
register(tbtrans.tbtncSileTBtrans, PdosPlot, 'tbt_out', default=True)

# -----------------------------------------------------
#           Register plotable sisl objects
# -----------------------------------------------------

# Geometry
register(sisl.Geometry, GeometryPlot, 'geometry', default=True)
register(sisl.Geometry, BondLengthMap, 'geometry')

# Grid
register(sisl.Grid, GridPlot, 'grid', default=True)

# Hamiltonian
register(sisl.Hamiltonian, WavefunctionPlot, 'H', default=True)
register(sisl.Hamiltonian, PdosPlot, "H")
register(sisl.Hamiltonian, BandsPlot, "H")
register(sisl.Hamiltonian, FatbandsPlot, "H")

# Band structure
register(sisl.BandStructure, BandsPlot, "band_structure", default=True)
register(sisl.BandStructure, FatbandsPlot, "band_structure")

# Eigenstate
register(sisl.EigenstateElectron, WavefunctionPlot, 'eigenstate', default=True)
