import numpy as np
import itertools

import os
import shutil

import sisl
from ..plot import Plot, MultiplePlot, Animation, PLOTS_CONSTANTS
from ..plotutils import sortOrbitals, initMultiplePlots, copyParams, findFiles, runMultiple, calculateGap
from ..inputFields import InputField, TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput, ProgramaticInput

class GridPlot(Plot):

    '''
    Plot representation of the projected density of states.
    '''

    #Define all the class attributes
    _plotType = "Grid"

    _parameters = (

        ProgramaticInput(
            key="grid", name="Grid",
            default=None,
            help="A sisl.Grid object. If provided, gridFile is ignored."
        ),

        TextInput(
            key="gridFile", name="Path to grid file",
            default=None,
            params={
                "placeholder": "Write the path to your grid file here..."
            }
        ),
        
        DropdownInput(
            key = "axes", name="Axes to display",
            default=[2],
            width = "s100% m50% l90%",
            params={
                'options': [
                    {'label': 'X', 'value': 0},
                    {'label': 'Y', 'value': 1},
                    {'label': 'Z', 'value': 2},
                ],
                'isMulti': True,
                'isSearchable': True,
                'isClearable': False
            },
            help = '''The axis along you want to see the grid, it will be averaged along the other ones '''
        ),

        IntegerInput(
            key="interpX", name="X axis interpolation",
            default=1,
            help="Interpolation factor to make the grid finer on the X axis"
        ),

        IntegerInput(
            key="interpY", name="Y axis interpolation",
            default=1,
            help="Interpolation factor to make the grid finer on the Y axis"
        ),

        IntegerInput(
            key="interpZ", name="Z axis interpolation",
            default=1,
            help="Interpolation factor to make the grid finer on the Z axis"
        )

    )
    
    def _readNoSource(self):

        self.grid = self.setting("grid")
        
        if self.grid is None:
            gridFile = self.setting("gridFile")

            self.grid = sisl.get_sile(gridFile).read_grid()

        return [gridFile]
    
    def _setData(self):

        grid = self.grid
        display_axes = self.setting('axes')

        interpFactors = np.array([ self.setting(key) if ax in display_axes else 1 for ax, key in enumerate(["interpX", "interpY", "interpZ"])], dtype=int)
        
        if (interpFactors != 1).any():
            grid = grid.interp(tuple([int(factor) for factor in grid.shape*interpFactors]))

        for ax in [0,1,2]:
            if ax not in display_axes:
                grid = grid.average(ax)

        #Remove the leftover dimensions
        values = np.squeeze(grid.grid)

        if values.ndim == 1:
            ax = display_axes[0]

            self.data = [{
                'type': 'scatter',
                'mode': 'lines',
                'y': values,
                'x': np.arange(0, grid.cell[ax,ax], grid.dcell[ax,ax]),
            }]

            axesTitles = {'xaxis_title': f'{("X","Y", "Z")[ax]} axis', 'yaxis_title': 'Values' }

        elif values.ndim == 2:

            xaxis = display_axes[1]
            yaxis = display_axes[0]

            self.data = [{
                'type': 'heatmap',
                'z': values,
                'x': np.arange(0, grid.cell[xaxis, xaxis], grid.dcell[xaxis, xaxis]) ,
                'y': np.arange(0, grid.cell[yaxis, yaxis], grid.dcell[yaxis, yaxis]),
            }]

            axesTitles = {'xaxis_title': f'{("X","Y", "Z")[xaxis]} axis', 'yaxis_title': f'{("X","Y", "Z")[yaxis]} axis'}
        elif values.ndim == 3:

            self.data = [{
                'type': 'isosurface',
                'x': np.arange(0, grid.cell[0,0], grid.dcell[0,0]),
                'y': np.arange(0, grid.cell[1,1], grid.dcell[1,1]),
                'z': np.arange(0, grid.cell[2,2], grid.dcell[2,2]),
                'value': values.flatten(),
            }]

            axesTitles = {}

        
        self.updateSettings(updateFig=False, **axesTitles)