'''
This file defines all the siles that are plotable.

It does so by patching the classes accordingly

In the future, sisl objects will probably be 'plotable' too
'''

import sisl.io.siesta as siesta
from sisl.io.sile import get_siles

import sisl
#from sisl.io.siesta import bandsSileSiesta, pdosSileSiesta, gridncSileSiesta, _gridSileSiesta
from .plots import *
from .plot import Plot


# -----------------------------------------------------
#   Let's define the functions that will help us here
# -----------------------------------------------------

def _plot(self, *args, **kwargs):
    return Plot(self, *args, **kwargs)

def register_plotable_sile(SileClass, PlotClass, setting_key):
    '''
    Makes the sisl.viz module aware of which siles can be plotted and how to do it.

    THE WAY THIS FUNCTION WORKS IS MOST LIKELY TO BE CHANGED, this is just a first
    implementation.
    '''

    SileClass._plot = (PlotClass, setting_key)
    SileClass.plot = _plot

def register_plotable_object(PlotableClass, PlotClass, setting_key):
    '''
    Makes the sisl.viz module aware of which sisl objects can be plotted and how to do it.

    THE WAY THIS FUNCTION WORKS IS MOST LIKELY TO BE CHANGED, this is just a first
    implementation.
    '''

    PlotableClass._plot = (PlotClass, setting_key)
    PlotableClass.plot = _plot

# -----------------------------------------------------
#               Register plotable siles
# -----------------------------------------------------

register_plotable_sile(siesta.bandsSileSiesta, BandsPlot, 'bands_file')

register_plotable_sile(siesta.pdosSileSiesta, PdosPlot, 'pdos_file')

for GridSile in get_siles(attrs=["read_grid"]):
    if GridSile not in [siesta.fdfSileSiesta]:
        register_plotable_sile(GridSile, GridPlot, 'grid_file')

for GeomSile in get_siles(attrs=["read_geometry"]):
    if GeomSile not in [siesta.fdfSileSiesta]:
        register_plotable_sile(GeomSile, GeometryPlot, 'geom_file')

# -----------------------------------------------------
#             Register plotable classes
# -----------------------------------------------------

register_plotable_object(sisl.Geometry, GeometryPlot, 'geom')
    
