'''
This file defines all the siles that are plotable.

It does so by patching the classes accordingly

In the future, sisl objects will probably be 'plotable' too
'''

import sisl.io.siesta as siesta
from sisl.io.sile import get_siles
#from sisl.io.siesta import bandsSileSiesta, pdosSileSiesta, gridncSileSiesta, _gridSileSiesta
from .plots import *

# -----------------------------------------------------
#   Let's define the functions that will help us here
# -----------------------------------------------------

def register_plotable_sile(SileClass, PlotClass, setting_key):
    '''
    Makes the sisl.viz module aware of which siles can be plotted and how to do it.

    THE WAY THIS FUNCTION WORKS IS MOST LIKELY TO BE CHANGED, this is just a first
    implementation.
    '''

    SileClass._plot = (PlotClass, setting_key)

register_plotable_sile(siesta.bandsSileSiesta, BandsPlot, 'bands_file')

register_plotable_sile(siesta.pdosSileSiesta, PdosPlot, 'pdos_file')

for GridSile in get_siles(attrs=["read_grid"]):
    register_plotable_sile(GridSile, GridPlot, 'grid_file')
    