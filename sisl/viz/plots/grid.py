import os

import numpy as np
import plotly.graph_objects as go

import sisl
from ..plot import Plot
from ..input_fields import TextInput, FilePathInput, Array1dInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeInput, RangeSlider, QueriesInput, ProgramaticInput

class GridPlot(Plot):
    '''
    Versatile visualization tool for any kind of grid.

    Parameters
    -------------
    grid: None, optional
        A sisl.Grid object. If provided, grid_file is ignored.
    grid_file: str, optional
    
    represent: None, optional
        The representation of the grid that should be displayed
    transforms: None, optional
        The representation of the grid that should be displayed
    axes: None, optional
        The axis along you want to see the grid, it will be averaged along
        the other ones
    zsmooth: None, optional
        Parameter that smoothens how data looks in a heatmap.
        'best' interpolates data, 'fast' interpolates pixels, 'False'
        displays the data as is.
    interp: array-like, optional
        Interpolation factors to make the grid finer on each axis.See the
        zsmooth setting for faster smoothing of 2D heatmap.
    sc: array-like, optional
    
    offset: array-like, optional
        The offset of the grid along each axis. This is important if you are
        planning to match this grid with other geometry related plots.
    cut_vacuum: bool, optional
        Whether the vacuum should not be taken into account for displaying
        the grid.             This is essential especially in 3D
        representations, since plotly needs to calculate the
        isosurfaces of the grid.
    trace_name: str, optional
        The name that the trace will show in the legend. Good when merging
        with other plots to be able to toggle the trace in the legend
    xRange: array-like of shape (2,), optional
        Range where the X is displayed. Should be inside the unit cell,
        otherwise it will fail.
    yRange: array-like of shape (2,), optional
        Range where the Y is displayed. Should be inside the unit cell,
        otherwise it will fail.
    zRange: array-like of shape (2,), optional
        Range where the Z is displayed. Should be inside the unit cell,
        otherwise it will fail.
    crange: array-like of shape (2,), optional
        The range of values that the colorbar must enclose. This controls
        saturation and hides below threshold values.
    cmid: int, optional
    
    colorscale: str, optional
    
    iso_vals: array-like of shape (2,), optional
    
    surface_count: int, optional
    
    type3D: None, optional
        This controls how the 3D data is displayed.              'volume'
        displays different layers with different levels of opacity so that
        there is more sensation of depth.             'isosurface' displays
        only isosurfaces and nothing inbetween them. For plotting grids with
        positive and negative             values, you should use 'isosurface'
        or two different 'volume' plots.              If not provided, the
        plot will decide for you based on the above mentioned fact
    opacityscale: None, optional
        Controls how the opacity changes through layers.              See
        https://plotly.com/python/3d-volume-plots/ for a display of the
        different possibilities
    surface_opacity: float, optional
        The opacity of the isosurfaces drawn by 3d plots from 0 (transparent)
        to 1 (opaque).
    reading_order: None, optional
        Order in which the plot tries to read the data it needs.
    root_fdf: str, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    '''

    #Define all the class attributes
    _plot_type = "Grid"

    _parameters = (

        ProgramaticInput(
            key="grid", name="Grid",
            default=None,
            help="A sisl.Grid object. If provided, grid_file is ignored."
        ),

        FilePathInput(
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
            key="transforms", name="Grid transforms",
            default=[],
            width="s100% m50% l90%",
            params={
                'options': [
                    {'label': 'Squared', 'value': 'squared'},
                    {'label': 'Absolute', 'value': 'abs'},
                ],
                'isMulti': False,
                'isSearchable': True,
                'isClearable': True
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

        Array1dInput(
            key="offset", name="Grid offset",
            default=[0,0,0],
            params={
                'inputType': 'number',
                'shape': (3,),
                'extendable': False,
            },
            help='''The offset of the grid along each axis. This is important if you are planning to match this grid with other geometry related plots.'''
        ),

        SwitchInput(
            key="cut_vacuum", name="Cut vacuum",
            default=True,
            help='''Whether the vacuum should not be taken into account for displaying the grid.
            This is essential especially in 3D representations, since plotly needs to calculate the
            isosurfaces of the grid.'''
        ),

        TextInput(
            key="trace_name", name="Trace name",
            default=None,
            params={
                "placeholder": "Give a name to the trace..."
            },
            help='''The name that the trace will show in the legend. Good when merging with other plots to be able to toggle the trace in the legend'''
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
            default=None
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
            self.get_param(key, as_dict=False).update_marks()

    def _set_data(self):

        grid = self.grid

        transforms = self.setting('transforms')
        for transform in transforms:
            grid = self._transform_grid(grid, transform)

        cut_vacuum = self.setting('cut_vacuum')
        if cut_vacuum and getattr(grid, "geom", None):
            grid, lims = self._cut_vacuum(grid)
            self.grid_offset = lims[0]
        else:
            self.grid_offset = [0,0,0]

        display_axes = self.setting('axes')
        sc = self.setting("sc")
        name = self.setting("trace_name")
        if name is None:
            name = os.path.basename(self.setting("grid_file") or "")

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
            plot_func = self._plot3D
        
        # Use it
        plot_func(grid, values, display_axes, sc, name, showlegend=bool(name))

    def _get_ax_range(self, grid, ax, sc):

        ax_range = self.setting(["xRange", "yRange", "zRange"][ax])
        grid_offset = self.grid_offset + self.setting("offset")

        if ax_range is not None:
            offset = ax_range[0]
        else:
            offset = 0

        return np.arange(0, sc[ax]*grid.cell[ax, ax], grid.dcell[ax, ax]) + offset + grid_offset[ax]

    @staticmethod
    def _cut_vacuum(grid):

        if not hasattr(grid, "geom"):
            raise Exception("The grid does not have an associated geometry, and therefore we can not calculate where the vacuum is.")

        geom = grid.geom
        maxR = geom.maxR()

        # Calculate the limits based on the positions of the atoms and the maxR of
        # the geometry.
        lims = np.array([geom.xyz.min(axis=0) - maxR, geom.xyz.max(axis=0) + maxR])

        # Get the limits in indexes
        i_lims = grid.index(lims)

        # Make sure that the limits don't go outside the cell
        i_lims[i_lims < 0] = 0
        for ax in (0, 1, 2):
            i_lims[1, ax] = min(i_lims[1, ax], grid.shape[ax])

        # Actually "cut" the grid
        for ax, (min_i, max_i) in enumerate(i_lims.T):
            grid = grid.sub(range(min_i, max_i), ax)
        
        # Return the cut grid, but also the limits that have been applied
        # The user can access the offset of the grid at lims[0]
        return grid, grid.index2xyz(i_lims)

    @staticmethod
    def _transform_grid(grid, transform):

        if transform == 'abs':
            grid = abs(grid)
        elif transform == 'squared':
            grid.grid = grid.grid ** 2

        return grid

    def _plot1D(self, grid, values, display_axes, sc, name, **kwargs):

        ax = display_axes[0]

        if sc[ax] > 1:
            values = np.tile(values, sc[ax])

        self.data = [{
            'type': 'scatter',
            'mode': 'lines',
            'y': values,
            'x': self._get_ax_range(grid, ax, sc),
            'name': name,
            **kwargs 
        }]

        axes_titles = {'xaxis_title': f'{("X","Y", "Z")[ax]} axis (Ang)', 'yaxis_title': 'Values' }

        self.update_layout(**axes_titles)
    
    def _plot2D(self, grid, values, display_axes, sc, name, **kwargs):

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
            'name': name,
            'z': values,
            'x': self._get_ax_range(grid, xaxis, sc),
            'y': self._get_ax_range(grid, yaxis, sc),
            'zsmooth': self.setting('zsmooth'),
            'zmin': cmin,
            'zmax': cmax,
            'zmid': self.setting('cmid'),
            'colorscale': self.setting('colorscale'),
            **kwargs
        }]

        axes_titles = {'xaxis_title': f'{("X","Y", "Z")[xaxis]} axis (Ang)', 'yaxis_title': f'{("X","Y", "Z")[yaxis]} axis (Ang)'}

        self.update_layout(**axes_titles)
        
    def _plot3D(self, grid, values, display_axes, sc, name, **kwargs):

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

        X, Y, Z = np.meshgrid(*[self._get_ax_range(grid, i, sc) for i, shape in enumerate(grid.shape)])
        
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
        
        plot_func(X, Y, Z, values, cmin, cmax, isomin, isomax, name, **kwargs)
        
        self.layout.scene = {'aspectmode': 'data'}

    def _plot3D_volume(self, X, Y, Z, values, cmin, cmax, isomin, isomax, name, **kwargs):

        self.data = [{
            'type': 'volume',
            'name': name,
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
            **kwargs
        }]
    
    def _plot3D_isosurface(self, X, Y, Z, values, cmin, cmax, isomin, isomax, name, **kwargs):

        self.data = [{
            'type': 'isosurface',
            'name': name,
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
            **kwargs
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

    def scan(self, along=None, start=None, stop=None, steps=None, breakpoints=None, mode="moving_slice", animation_kwargs=None, **kwargs):
        '''
        Returns an animation containing multiple frames scaning along an axis.

        Parameters
        -----------
        along: int, optional
            the axis along which the scan is performed. If not provided, it will scan along the axes that are not displayed.
        start: float, optional
            the starting value for the scan (in Angstrom).
            Make sure this value is inside the range of the unit cell, otherwise it will fail.
        stop: float, optional
            the last value of the scan (in Angstrom).
            Make sure this value is inside the range of the unit cell, otherwise it will fail.
        steps: int or float, optional
            If it's an integer:
                the number of steps that you want the scan to consist of.
            If it's a float:
                the division between steps in Angstrom.
            
            Note that the grid is only stored once, so having a big number of steps is not that big of a deal.
        breakpoints: array-like, optional
            the discrete points of the scan. To be used if you don't want regular steps.
            If the last step is exactly the length of the cell, it will be moved one dcell back to avoid errors.
        mode: {"moving_slice", "as_is"}, optional
            the type of scan you want to see.
            "moving_slice" renders a volumetric scan where a slice moves through the grid.
            "as_is" renders each part of the scan as an animation frame.
            (therefore, "as_is" SUPPORTS SCANNING 1D, 2D AND 3D REPRESENTATIONS OF THE GRID, e.g. display the volume data for different ranges of z)
        animation_kwargs: dict, optional
            dictionary whose keys and values are directly passed to the animated method as kwargs and therefore
            end up being passed to animation initialization.
        **kwargs:
            the rest of settings that you want to apply to overwrite the existing ones.

            This settings apply to each plot and go directly to their initialization.

        Returns
        ----------
        sisl.viz.Animation
            An animation representation of the scan
        '''

        # If no axis is provided, let's get the first one that is not displayed
        if along is None:
            displayed = self.setting('axes')
            not_displayed = [ax["value"] for ax in self.get_param('axes')['inputField.params.options'] if ax["value"] not in displayed]
            along = not_displayed[0] if not_displayed else 2

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
        
        if breakpoints is None:
            if steps is None:
                steps = 1.0
            # Divide it in steps
            if isinstance(steps, int):
                step = (along_range[1] - along_range[0])/steps
            elif isinstance(steps, float):
                step = steps
                steps = (along_range[1] - along_range[0])/step

            # np.linspace will use the last point as a step (and we don't want it)
            # therefore we will add an extra step
            breakpoints = np.linspace(*along_range, steps + 1)
        
        if breakpoints[-1] == self.grid.cell[along, along]:
            breakpoints[-1] = self.grid.cell[along, along] - self.grid.dcell[along, along]
        
        if mode == "moving_slice":
            return self._moving_slice_scan(along, breakpoints)
        elif mode == "as_is":
            return self._asis_scan(range_key, breakpoints, animation_kwargs=animation_kwargs, **kwargs)

    def _asis_scan(self, range_key, breakpoints, animation_kwargs=None, **kwargs):

        '''
        Returns an animation containing multiple frames scaning along an axis.

        Parameters
        -----------
        range_key: {'xRange', 'yRange', 'zRange'}
            the key of the setting that is to be animated through the scan.
        breakpoints: array-like
            the discrete points of the scan
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

        # To keep the same iso_vals along all the animation in case it is a scan of 3D frames
        if getattr(self.data[0], "isomin", None):
            isovals = [self.data[0].isomin, self.data[0].isomax]
        else:
            isovals = self.setting("iso_vals")

        # Generate the plot using self as a template so that plots don't need
        # to read data, just process it and show it differently.
        # (If each plot read the grid, the memory requirements would be HUGE)
        scan = GridPlot.animated(
            {
                range_key: [[bp, breakpoints[i+1]] for i, bp in enumerate(breakpoints[:-1])]
            },
            plot_template=self,
            fixed={**{key: val for key, val in self.settings.items() if key != range_key}, "iso_vals": isovals, **kwargs},
            frameNames=[ f'{bp:2f}' for bp in breakpoints],
            **(animation_kwargs or {})
        )

        # Set all frames to the same colorscale
        cmin = 10**6; cmax = -10**6
        for scan_im in scan:
            c = getattr(scan_im.data[0], "value", scan_im.data[0].z)
            cmin = min(cmin, np.min(c))
            cmax = max(cmax, np.max(c))
        for scan_im in scan:
            scan_im.update_settings(crange=[cmin, cmax])
        scan.get_figure()

        scan.layout = self.layout

        return scan

    def _moving_slice_scan(self, along, breakpoints):

        ax = along
        displayed_axes = [ i for i in range(3) if i != ax]
        shape = np.array(self.grid.shape)[displayed_axes]
        cmin = np.min(self.grid.grid)
        cmax = np.max(self.grid.grid)
        x_ax , y_ax = displayed_axes
        x = np.linspace(0, self.grid.cell[x_ax, x_ax], self.grid.shape[x_ax])
        y = np.linspace(0, self.grid.cell[y_ax, y_ax], self.grid.shape[y_ax])

        fig = go.Figure(frames=[go.Frame(data=go.Surface(
            x=x, y=y,
            z=bp * np.ones(shape),
            surfacecolor=np.squeeze(self.grid.cross_section(self.grid.index(bp, ax), ax).grid),
            cmin=cmin, cmax=cmax,
            ),
            name=f'{bp:.2f}'
            )
            for bp in breakpoints])

        # Add data to be displayed before animation starts
        fig.add_traces(fig.frames[0].data)

        def frame_args(duration):
            return {
                    "frame": {"duration": duration},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": duration, "easing": "linear"},
                }

        sliders = [
                    {
                        "pad": {"b": 10, "t": 60},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": [
                            {
                                "args": [[f.name], frame_args(0)],
                                "label": str(k),
                                "method": "animate",
                            }
                            for k, f in enumerate(fig.frames)
                        ],
                    }
                ]
        
        def ax_title(ax): return f'{["X", "Y", "Z"][ax]} axis (Ang)'

        # Layout
        fig.update_layout(
                title=f'Grid scan along {["X", "Y", "Z"][ax]} axis',
                width=600,
                height=600,
                scene=dict(
                            xaxis=dict(title=ax_title(x_ax)),
                            yaxis=dict(title=ax_title(y_ax)),
                            zaxis=dict(range=[0, self.grid.cell[ax,ax]], autorange=False, title=ax_title(ax)),
                            aspectratio=dict(x=1, y=1, z=1),
                            ),
                updatemenus = [
                    {
                        "buttons": [
                            {
                                "args": [None, frame_args(50)],
                                "label": "&#9654;", # play symbol
                                "method": "animate",
                            },
                            {
                                "args": [[None], frame_args(0)],
                                "label": "&#9724;", # pause symbol
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 70},
                        "type": "buttons",
                        "x": 0.1,
                        "y": 0,
                    }
                ],
                sliders=sliders
        )

        return fig

# class GridSlice(Plot):

#     _parameters = (
#         ProgramaticInput(key="grid_slice", name="Grid slice",
#             default=None, 
#         ),

#         DropdownInput(
#             key="ax", name="Axis",
#             default=2,
#             params={
#                 "options":[{"label": ax, "value": ax} for ax in [0,1,2]]
#             }   
#         ),

#         IntegerInput(
#             key="index", name="Index",
#             default=0,
#         ),

#         FloatInput(key="cmin", name="cmin",
#             default=0
#         ),

#         FloatInput(key="cmax", name="cmax",
#             default=None
#         )
#     )

#     def _set_data(self):

#         self.data = [{
#             z=self.grid.index2xyz([0,0,i])[ax] * np.ones(shape),
#             surfacecolor=self.grid.cross_section(i, ax).grid,
#             cmin=cmin, cmax=cmax,
#         }]

