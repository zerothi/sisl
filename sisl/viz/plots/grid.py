import os

import numpy as np

import sisl
from ..plot import Plot
from ..input_fields import TextInput, Array1dInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeInput, RangeSlider, QueriesInput, ProgramaticInput

class GridPlot(Plot):

    '''
    Plot representation of the projected density of states.
    '''

    #Define all the class attributes
    _plot_type = "Grid"

    _parameters = (

        ProgramaticInput(
            key="grid", name="Grid",
            default=None,
            help="A sisl.Grid object. If provided, grid_file is ignored."
        ),

        TextInput(
            key="grid_file", name="Path to grid file",
            default=None,
            params={
                "placeholder": "Write the path to your grid file here..."
            }
        ),

        DropdownInput(
            key="represent", name="Representation of the grid",
            default="real",
            width="s100% m50% l90%",
            params={
                'options': [
                    {'label': 'Real part', 'value': "real"},
                    {'label': 'Imaginary part', 'value': 'imag'},
                ],
                'isMulti': False,
                'isSearchable': True,
                'isClearable': False
            },
            help='''The representation of the grid that should be displayed'''
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

        DropdownInput(
            key = "zsmooth", name="2D heatmap smoothing method",
            default=False,
            width = "s100% m50% l90%",
            params={
                'options': [
                    {'label': 'best', 'value': 'best'},
                    {'label': 'fast', 'value': 'fast'},
                    {'label': 'False', 'value': False},
                ],
                'isSearchable': True,
                'isClearable': False
            },
            help = '''Parameter that smoothens how data looks in a heatmap.<br>
            'best' interpolates data, 'fast' interpolates pixels, 'False' displays the data as is.'''
        ),

        Array1dInput(
            key="interp", name="Interpolation",
            default=[1,1,1],
            params={
                'inputType': 'number',
                'shape': (3,),
                'extendable': False,
            },
            help="Interpolation factors to make the grid finer on each axis.<br>See the zsmooth setting for faster smoothing of 2D heatmap."
        ),

        Array1dInput(
            key="sc", name="Supercell",
            default=[1,1,1],
            params={
                'inputType': 'number',
                'shape': (3,),
                'extendable': False,
            },
        ),

        RangeSlider(
            key="xRange", name="X range",
            default=None,
            params={
                "min": 0
            },
            help="Range where the X is displayed. Should be inside the unit cell, otherwise it will fail.",
        ),

        RangeSlider(
            key="yRange", name="Y range",
            default=None,
            params={
                "min": 0
            },
            help="Range where the Y is displayed. Should be inside the unit cell, otherwise it will fail.",
        ),

        RangeSlider(
            key="zRange", name="Z range",
            default=None,
            params={
                "min": 0
            },
            help="Range where the Z is displayed. Should be inside the unit cell, otherwise it will fail.",
        ),

        RangeInput(
            key="crange", name="Colorbar range",
            default=[None, None],
            help="The range of values that the colorbar must enclose. This controls saturation and hides below threshold values."
        ),

        IntegerInput(
            key="cmid", name="Colorbar center",
            default=0
        ),

        TextInput(
            key="colorscale", name="Color scale",
            default=None
        ),

        RangeInput(
            key="iso_vals", name="Min and max isosurfaces",
            default=None
        ),

        IntegerInput(
            key="surface_count", name="Number of surfaces",
            default=2,
        ),

        DropdownInput(
            key="type3D", name="Type of 3D representation",
            default=None,
            params={
                'options': [
                    {'label': 'Isosurface', 'value': 'isosurface'},
                    {'label': 'Volume', 'value': 'volume'},
                ],
                'isSearchable': True,
                'isClearable': True,
                'isMulti': True
            },
            help='''This controls how the 3D data is displayed. 
            'volume' displays different layers with different levels of opacity so that there is more sensation of depth.
            'isosurface' displays only isosurfaces and nothing inbetween them. For plotting grids with positive and negative
            values, you should use 'isosurface' or two different 'volume' plots. 
            If not provided, the plot will decide for you based on the above mentioned fact'''
        ),

        DropdownInput(
            key="opacityscale", name="Opacity scale for 3D volume",
            default="uniform",
            params={
                'options': [
                    {'label': 'Uniform', 'value': 'uniform'},
                    {'label': 'Min', 'value': 'min'},
                    {'label': 'Max', 'value': 'max'},
                    {'label': 'Extremes', 'value': 'extremes'},
                ],
                'isSearchable': True,
                'isClearable': False,
                'isMulti': True
            },
            help='''Controls how the opacity changes through layers. 
            See https://plotly.com/python/3d-volume-plots/ for a display of the different possibilities'''
        ),

        FloatInput(
            key='surface_opacity', name="Surface opacity",
            default=1,
            params={'min': 0, 'max': 1, 'step': 0.1},
            help='''The opacity of the isosurfaces drawn by 3d plots from 0 (transparent) to 1 (opaque).'''
        )
    )

    def _after_init(self):

        self._add_shortcuts()
    
    def _read_nosource(self):

        self.grid = self.setting("grid")
        
        if self.grid is None:
            grid_file = self.setting("grid_file")

            self.grid = self.get_sile(grid_file).read_grid()
    
    def _after_read(self):

        #Inform of the new available ranges
        range_keys = ("xRange", "yRange", "zRange")

        for ax, key in enumerate(range_keys):
            self.modify_param(key, "inputField.params.max", self.grid.cell[ax,ax] )
            self.get_param(key, justDict=False).update_marks()

    def _set_data(self):

        grid = self.grid
        display_axes = self.setting('axes')
        sc = self.setting("sc")

        # Get only the part of the grid that we need
        range_keys = ("xRange", "yRange", "zRange")
        ax_ranges = [self.setting(key) for ax, key in enumerate(range_keys)]

        for ax, ax_range in enumerate(ax_ranges):
            if ax_range is not None:
                #Build an array with the limits
                lims = np.zeros((2,3))
                lims[:, ax] = ax_range
                
                #Get the indices of those points
                indices = np.array([grid.index(lim) for lim in lims], dtype=int)

                #And finally get the subpart of the grid
                grid = grid.sub(np.arange(indices[0,ax], indices[1,ax] + 1), ax)

        interpFactors = np.array([ factor if ax in display_axes else 1 for ax, factor in enumerate(self.setting("interp"))], dtype=int)

        interpolate = (interpFactors != 1).any()

        for ax in [0,1,2]:
            if ax not in display_axes:
                grid = grid.average(ax)

                if interpolate:
                    #Duplicate this axis so that interpolate can work
                    grid.grid = np.concatenate((grid.grid, grid.grid), axis=2)
        
        if interpolate:
            grid = grid.interp([factor for factor in grid.shape*interpFactors])
        
            for ax in [0,1,2]:
                if ax not in display_axes:
                    grid = grid.average(ax)

        # Choose the representation of the grid that we want to display
        representation = self.setting("represent")
        if representation == 'real':
            values = grid.grid.real
        elif representation == 'imag':
            values = grid.grid.imag

        #Remove the leftover dimensions
        values = np.squeeze(values)

        # Choose which plotting function we need to use
        if values.ndim == 1:
            plot_func = self._plot1D
        elif values.ndim == 2:
            plot_func = self._plot2D
        elif values.ndim == 3:
            plot_func == self._plot3D
        
        # Use it
        plot_func(grid, values, display_axes, sc)

    def _plot1D(self, grid, values, display_axes, sc):

        ax = display_axes[0]

        if sc[ax] > 1:
            values = np.tile(values, sc[ax])

        self.data = [{
            'type': 'scatter',
            'mode': 'lines',
            'y': values,
            'x': np.arange(0, sc[ax]*grid.cell[ax,ax], grid.dcell[ax,ax]),
            'name': os.path.basename(self.setting("grid_file") or "")
        }]

        axes_titles = {'xaxis_title': f'{("X","Y", "Z")[ax]} axis (Ang)', 'yaxis_title': 'Values' }

        self.update_settings(update_fig=False, **axes_titles, no_log=True)
    
    def _plot2D(self, grid, values, display_axes, sc):

        xaxis = display_axes[0]
        yaxis = display_axes[1]

        if xaxis < yaxis:
            values = values.T
        
        values = np.tile(values, (sc[yaxis], sc[xaxis]) )

        crange = self.setting('crange')
        if crange is None:
            crange = [None, None]
        cmin, cmax = crange

        self.data = [{
            'type': 'heatmap',
            'z': values,
            'x': np.arange(0, sc[xaxis]*grid.cell[xaxis, xaxis], grid.dcell[xaxis, xaxis]),
            'y': np.arange(0, sc[yaxis]*grid.cell[yaxis, yaxis], grid.dcell[yaxis, yaxis]),
            'zsmooth': self.setting('zsmooth'),
            'zmin': cmin,
            'zmax': cmax,
            'zmid': self.setting('cmid'),
            'colorscale': self.setting('colorscale')
        }]

        axes_titles = {'xaxis_title': f'{("X","Y", "Z")[xaxis]} axis (Ang)', 'yaxis_title': f'{("X","Y", "Z")[yaxis]} axis (Ang)'}

        self.update_settings(update_fig=False, **axes_titles, no_log=True)
        
    def _plot3D(self, grid, values, display_axes, sc):

        # The minimum and maximum values might be needed at some places
        minval, maxval = np.min(values), np.max(values)

        crange = self.setting('crange')
        if crange is None:
            crange = [None, None]
        cmin, cmax = crange

        isovals = self.setting('iso_vals')
        if isovals is None:
            isovals = [minval + (maxval-minval)*0.3,
                        maxval - (maxval-minval)*0.3]
        isomin, isomax = isovals

        X, Y, Z = np.meshgrid(*[np.linspace(0, grid.cell[i, i], shape) for i, shape in enumerate(grid.shape)])
        
        type3D = self.setting('type3D')
        if type3D is None:
            if np.min(values)*np.max(values) < 0:
                # Then we have mixed positive and negative values, use isosurface
                type3D = 'isosurface'
            else:
                type3D = 'volume'

        if type3D == 'volume':
            plot_func = self._plot3D_volume
        elif type3D == 'isosurface':
            plot_func = self._plot3D_isosurface
        
        plot_func(X, Y, Z, values, cmin, cmax, isomin, isomax)
        
        self.layout.scene = {'aspectmode': 'data'}

    def _plot3D_volume(self, X, Y, Z, values, cmin, cmax, isomin, isomax):

        self.data = [{
            'type': 'volume',
            'x': X.ravel(),
            'y': Y.ravel(),
            'z': Z.ravel(),
            'value': np.rollaxis(values, 1).ravel(),
            'cmin': cmin,
            'cmid': self.setting('cmid'),
            'cmax': cmax,
            'isomin': isomin,
            'isomax': isomax,
            'surface_count': self.setting('surface_count'),
            'opacity': self.setting('surface_opacity'),
            'autocolorscale': False,
            'colorscale': self.setting('colorscale'),
            'opacityscale': self.setting('opacityscale'),
            'caps': {'x_show': False, 'y_show': False, 'z_show': False},
        }]
    
    def _plot3D_isosurface(self, X, Y, Z, values, cmin, cmax, isomin, isomax):

        self.data = [{
            'type': 'isosurface',
            'x': X.ravel(),
            'y': Y.ravel(),
            'z': Z.ravel(),
            'value': np.rollaxis(values, 1).ravel(),
            'cmin': cmin,
            'cmid': self.setting('cmid'),
            'cmax': cmax,
            'isomin': isomin,
            'isomax': isomax,
            'opacity': self.setting('surface_opacity'),
            'surface_count': self.setting('surface_count'),
            'colorscale': self.setting('colorscale'),
            'caps': {'x_show': False, 'y_show': False, 'z_show': False},
        }]

    def _after_get_figure(self):

        # If we are plotting the 2D version, use a 1:1 ratio
        if len(self.setting("axes")) == 2:
            self.figure.layout.yaxis.scaleanchor = "x"
            self.figure.layout.yaxis.scaleratio = 1
 
    def _add_shortcuts(self):

        axes = self.get_param("axes")["inputField.params.options"]

        for ax in axes:

            ax_name = ax["label"]
            ax_val = ax["value"]

            self.add_shortcut(f'{ax_name.lower()}+enter', f"Show {ax_name} axis", self.update_settings, axes=[ax_val])

            self.add_shortcut(f'{ax_name.lower()} {ax_name.lower()}', f"Duplicate {ax_name} axis", self.tile, 2, ax_val)

            self.add_shortcut(f'{ax_name.lower()}+-', f"Substract a unit cell along {ax_name}", self.tighten, 1, ax_val)

            self.add_shortcut(f'{ax_name.lower()}++', f"Add a unit cell along {ax_name}", self.tighten, -1, ax_val)
        
        for xaxis in axes:
            xaxis_name = xaxis["label"]
            for yaxis in [ax for ax in axes if ax != xaxis]:
                yaxis_name = yaxis["label"]
                self.add_shortcut(
                    f'{xaxis_name.lower()}+{yaxis_name.lower()}', f"Show {xaxis_name} and {yaxis_name} axes",
                    self.update_settings, axes=[xaxis["value"], yaxis['value']]
                )

    def tighten(self, steps, ax):
        '''
        Makes the supercell tighter by a number of unit cells

        Parameters
        ---------
        steps: int or array-like
            Number of unit cells that you want to substract.
            If there are not enough unit cells to substract, one unit cell will remain.

            If you provide multiple steps, it needs to match the number of axes provided.
        ax: int or array-like
            Axis along which to tighten the supercell.

            If you provide multiple axes, the number of different steps must match the number of axes or be a single int.
        '''
        
        if isinstance(ax, int):
            ax = [ax]
        if isinstance(steps, int):
            steps = [steps]*len(ax)
        
        sc = [*self.setting("sc")]

        for a, step in zip(ax, steps):
            sc[a] = max(1, sc[a]-step)

        return self.update_settings(sc=sc)

    def tile(self, tiles, ax):
        '''
        Tile a given axis to display more unit cells in the plot

        Parameters
        ----------
        tiles: int or array-like
            factor by which the supercell will be multiplied along axes `ax`.
            
            If you provide multiple tiles, it needs to match the number of axes provided.
        ax: int or array-like
            axis that you want to tile.

            If you provide multiple axes, the number of different tiles must match the number of axes or be a single int.
        '''

        if isinstance(ax, int):
            ax = [ax]
        if isinstance(tiles, int):
            tiles = [tiles]*len(ax)
        
        sc = [*self.setting("sc")]

        for a, tile in zip(ax, tiles):
            sc[a] *= tile

        return self.update_settings(sc=sc)

    def scan(self, along=None, start=None, stop=None, steps=None, animation_kwargs=None, **kwargs):
        '''
        Returns an animation containing multiple frames scaning along an axis.

        Parameters
        -----------
        along: int
            the axis along which the scan is performed. If not provided, it will scan along the axes that are not displayed.
        start: float
            the starting value for the scan (in Angstrom).
            Make sure this value is inside the range of the unit cell, otherwise it will fail.
        stop: float
            the last value of the scan (in Angstrom).
            Make sure this value is inside the range of the unit cell, otherwise it will fail.
        steps: int or float
            If it's an integer:
                the number of steps that you want the scan to consist of.
            If it's a float:
                the division between steps in Angstrom.
            
            Note that the grid is only stored once, so having a big number of steps is not that big of a deal.
        animation_kwargs: dict, optional
            dictionary whose keys and values are directly passed to the animated method as kwargs and therefore
            end up being passed to animation initialization.
        **kwargs:
            the rest of settings that you want to apply to overwrite the existing ones.

            This settings apply to each plot and go directly to their initialization.

        Returns
        ----------
        scan: sisl Animation
            An animation representation of the scan
        '''

        # If no axis is provided, let's get the first one that is not displayed
        if along is None:
            displayed = self.setting('axes')
            along = [ax["value"] for ax in self.get_param('axes')['inputField.params.options'] if ax["value"] not in displayed][0]
        
        # If no steps is provided, we will do 1 Angstrom steps
        if steps is None:
            steps = 1.0
        
        # We get the key that needs to be animated (we will divide the full range in frames)
        range_key = ["xRange", "yRange", "zRange"][along]

        # Get the full range
        if start is not None and stop is not None:
            along_range = [start,stop]
        else:
            along_range = self.setting(range_key)
            if along_range is None:
                range_param = self.get_param(range_key)
                along_range = [range_param[f"inputField.params.{lim}"] for lim in ["min", "max"]]
            if start is not None:
                along_range[0] = start
            if stop is not None:
                along_range[1] = stop
        
        # Divide it in steps
        if isinstance(steps, int):
            step = (along_range[1] - along_range[0])/steps
        elif isinstance(steps, float):
            step = steps
            steps = (along_range[1] - along_range[0])/step
        steps_range = np.linspace(*along_range, steps)[:-1]

        # Generate the plot using self as a template so that plots don't need
        # to read data, just process it and show it differently.
        # (If each plot read the grid, the memory requirements would be HUGE)
        scan = GridPlot.animated(
            {
                range_key: [[minVal, minVal + step] for minVal in steps_range]
            },
            plot_template=self,
            fixed={**{key: val for key, val in self.settings.items() if key != range_key}, **kwargs},
            frameNames=[ f'{step:2f}' for step in steps_range],
            **(animation_kwargs or {})
        )

        scan.layout = self.layout

        return scan
