from collections import defaultdict
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import affine_transform

import sisl
from sisl._supercell import cell_invert
from sisl import _array as _a
from ..plot import Plot, entry_point
from ..input_fields import (
    TextInput, SileInput, Array1DInput, SwitchInput,
    ColorPicker, DropdownInput, CreatableDropdown, IntegerInput, FloatInput, RangeInput, RangeSlider,
    QueriesInput, ProgramaticInput, PlotableInput, SislObjectInput, PlotableInput, SpinSelect
)


class GridPlot(Plot):
    """
    Versatile visualization tool for any kind of grid.

    Parameters
    -------------
    grid: Grid, optional
        A sisl.Grid object. If provided, grid_file is ignored.
    grid_file: Sile, optional
        a sile that can read grids, e.g. `_gridSileSiesta` or `chgSileVASP` and friends.
    represent:  optional
        The representation of the grid that should be displayed
    transforms:  optional
        Transformations to apply to the whole grid.             It can be a
        function, or a string that represents the path             to a
        function (e.g. "scipy.exp"). If a string that is a single
        word is provided, numpy will be assumed to be the module (e.g.
        "square" will be converted into "np.square").              Note that
        transformations will be applied in the order provided. Some
        transforms might not be necessarily commutable (e.g. "abs" and
        "cos").
    transform_bc: optional
        Transform the boundary conditions.
    axes:  optional
        The axis along you want to see the grid, it will be averaged along
        the other ones
    plot_geom: bool, optional
        If True the geometry associated to the grid will also be plotted
    geom_kwargs: dict, optional
        Extra arguments that are passed to geom.plot() if plot_geom is set to
        True
    zsmooth:  optional
        Parameter that smoothens how data looks in a heatmap.
        'best' interpolates data, 'fast' interpolates pixels, 'False'
        displays the data as is.
    interp: array-like, optional
        Interpolation factors to make the grid finer on each axis.See the
        zsmooth setting for faster smoothing of 2D heatmap.
    nsc: array-like, optional
        number of times the grid should be repeated
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
    x_range: array-like of shape (2,), optional
        Range where the X is displayed. Should be inside the unit cell,
        otherwise it will fail.
    y_range: array-like of shape (2,), optional
        Range where the Y is displayed. Should be inside the unit cell,
        otherwise it will fail.
    z_range: array-like of shape (2,), optional
        Range where the Z is displayed. Should be inside the unit cell,
        otherwise it will fail.
    crange: array-like of shape (2,), optional
        The range of values that the colorbar must enclose. This controls
        saturation and hides below threshold values.
    cmid: int, optional
        The value to set at the center of the colorbar. If not provided, the
        color range is used
    colorscale: str, optional
        A valid plotly colorscale. See https://plotly.com/python/colorscales/
    isos: array-like of dict, optional
        The isovalues that you want to represent.             The way they
        will be represented are of course dependant on the type of
        representation:                 - 1D representations: Just a scatter
        mark                 - 2D representations: A contour (i.e. a line)
        - 3D representations: A surface                Each item is a dict.
        Structure of the expected dicts:{         'name': The name of the iso
        query. Note that you can use $isoval$ as a template to indicate where
        the isoval should go.         'val':          'frac': If val is not
        provided, this is used to calculate where the isosurface should be
        drawn.                     It calculates them from the minimum and
        maximum values of the grid like so:                     If iso_frac =
        0.3:                     (min_value-----
        ISOVALUE(30%)-----------max_value)                     Therefore, it
        should be a number between 0 and 1.
        'step_size': The step size to use to calculate the isosurface in case
        it's a 3D representation                     A bigger step-size can
        speed up the process dramatically, specially the rendering part
        and the resolution may still be more than satisfactory (try to use
        step_size=2). For very big                     grids your computer
        may not even be able to render very fine surfaces, so it's worth
        keeping                     this setting in mind.         'color':
        'opacity':  }
    root_fdf: fdfSileSiesta, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    """

    #Define all the class attributes
    _plot_type = "Grid"

    _update_methods = {
        "read_data": [],
        "set_data": ["_plot1D", "_plot2D", "_plot3D"],
        "get_figure": []
    }

    _parameters = (

        PlotableInput(
            key="grid", name="Grid",
            dtype=sisl.Grid,
            default=None,
            help="A sisl.Grid object. If provided, grid_file is ignored."
        ),

        SileInput(
            key="grid_file", name="Path to grid file",
            required_attrs=["read_grid"],
            default=None,
            params={
                "placeholder": "Write the path to your grid file here..."
            },
            help="A filename that can be return a Grid through `read_grid`."
        ),

        DropdownInput(
            key="represent", name="Representation of the grid",
            default="real",
            width="s100% m50% l90%",
            params={
                'options': [
                    {'label': 'Real part', 'value': "real"},
                    {'label': 'Imaginary part', 'value': 'imag'},
                    {'label': 'Complex modulus', 'value': "mod"},
                    {'label': 'Phase (in rad)', 'value': 'rad_phase'},
                    {'label': 'Phase (in deg)', 'value': 'deg_phase'},
                ],
                'isMulti': False,
                'isSearchable': True,
                'isClearable': False
            },
            help="""The representation of the grid that should be displayed"""
        ),

        CreatableDropdown(
            key="transforms", name="Grid transforms",
            default=[],
            width="s100% m50% l90%",
            params={
                'options': [
                    {'label': 'Square', 'value': 'square'},
                    {'label': 'Absolute', 'value': 'abs'},
                ],
                'isMulti': True,
                'isSearchable': True,
                'isClearable': True
            },
            help="""Transformations to apply to the whole grid.
            It can be a function, or a string that represents the path
            to a function (e.g. "scipy.exp"). If a string that is a single
            word is provided, numpy will be assumed to be the module (e.g.
            "square" will be converted into "np.square"). 
            Note that transformations will be applied in the order provided. Some
            transforms might not be necessarily commutable (e.g. "abs" and "cos")."""
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
            help = """The axis along you want to see the grid, it will be averaged along the other ones """
        ),

        SwitchInput(key='plot_geom', name='Plot geometry',
            default=False,
            help="""If True the geometry associated to the grid will also be plotted"""
        ),

        ProgramaticInput(key='geom_kwargs', name='Geometry plot extra arguments',
            default={},
            dtype=dict,
            help="""Extra arguments that are passed to geom.plot() if plot_geom is set to True"""
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
            help = """Parameter that smoothens how data looks in a heatmap.<br>
            'best' interpolates data, 'fast' interpolates pixels, 'False' displays the data as is."""
        ),

        Array1DInput(
            key="interp", name="Interpolation",
            default=[1, 1, 1],
            params={
                'inputType': 'number',
                'shape': (3,),
                'extendable': False,
            },
            help="Interpolation factors to make the grid finer on each axis.<br>See the zsmooth setting for faster smoothing of 2D heatmap."
        ),

        DropdownInput(key="transform_bc", name="Transform boundary conditions",
            default="constant",
            params={
                'options': [
                    {'label': 'constant', 'value': 'constant'},
                    {'label': 'wrap', 'value': 'wrap'},
                ],
            },
            help="""
            The boundary conditions when a cell transform is applied to the grid. Cell transforms are only
            applied when the grid's cell doesn't follow the cartesian coordinates and the requested display is 2D.
            """
        ),

        Array1DInput(
            key="nsc", name="Supercell",
            default=[1, 1, 1],
            params={
                'inputType': 'number',
                'shape': (3,),
                'extendable': False,
            },
            help="Number of times the grid should be repeated"
        ),

        Array1DInput(
            key="offset", name="Grid offset",
            default=[0, 0, 0],
            params={
                'inputType': 'number',
                'shape': (3,),
                'extendable': False,
            },
            help="""The offset of the grid along each axis. This is important if you are planning to match this grid with other geometry related plots."""
        ),

        SwitchInput(
            key="cut_vacuum", name="Cut vacuum",
            default=False,
            help="""Whether the vacuum should not be taken into account for displaying the grid.
            This is essential especially in 3D representations, since plotly needs to calculate the
            isosurfaces of the grid."""
        ),

        TextInput(
            key="trace_name", name="Trace name",
            default=None,
            params={
                "placeholder": "Give a name to the trace..."
            },
            help="""The name that the trace will show in the legend. Good when merging with other plots to be able to toggle the trace in the legend"""
        ),

        RangeSlider(
            key="x_range", name="X range",
            default=None,
            params={
                "min": 0
            },
            help="Range where the X is displayed. Should be inside the unit cell, otherwise it will fail.",
        ),

        RangeSlider(
            key="y_range", name="Y range",
            default=None,
            params={
                "min": 0
            },
            help="Range where the Y is displayed. Should be inside the unit cell, otherwise it will fail.",
        ),

        RangeSlider(
            key="z_range", name="Z range",
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
            default=None,
            help="""The value to set at the center of the colorbar. If not provided, the color range is used"""
        ),

        TextInput(
            key="colorscale", name="Color scale",
            default=None,
            help="""A valid plotly colorscale. See https://plotly.com/python/colorscales/"""
        ),

        QueriesInput(key = "isos", name = "Isosurfaces / contours",
            default = [],
            help = """The isovalues that you want to represent.
            The way they will be represented is of course dependant on the type of representation:
                - 2D representations: A contour (i.e. a line)
                - 3D representations: A surface
            """,
            queryForm = [

                TextInput(
                    key="name", name="Name",
                    default="Iso=$isoval$",
                    width="s100% m50% l20%",
                    params={
                        "placeholder": "Name of the isovalue..."
                    },
                    help="The name of the iso query. Note that you can use $isoval$ as a template to indicate where the isoval should go."
                ),

                FloatInput(
                    key="val", name="Value",
                    default=None,
                    help="The iso value. If not provided, it will be infered from `frac`"
                ),

                FloatInput(
                    key="frac", name="Fraction",
                    default=0.3,
                    params={
                        "min": 0,
                        "max": 1,
                        "step": 0.05
                    },
                    help="""If val is not provided, this is used to calculate where the isosurface should be drawn.
                    It calculates them from the minimum and maximum values of the grid like so:
                    If iso_frac = 0.3:
                    (min_value-----ISOVALUE(30%)-----------max_value)
                    Therefore, it should be a number between 0 and 1.
                    """
                ),

                IntegerInput(
                    key="step_size", name="Step size",
                    default=1,
                    help="""The step size to use to calculate the isosurface in case it's a 3D representation
                    A bigger step-size can speed up the process dramatically, specially the rendering part
                    and the resolution may still be more than satisfactory (try to use step_size=2). For very big
                    grids your computer may not even be able to render very fine surfaces, so it's worth keeping
                    this setting in mind."""
                ),

                ColorPicker(
                    key="color", name="Color",
                    default=None,
                    help="The color of the surface/contour."
                ),

                FloatInput(
                    key="opacity", name="Opacity",
                    default=1,
                    params={
                        "min": 0,
                        "max": 1,
                        "step": 0.1
                    },
                    help="Opacity of the surface/contour. Between 0 (transparent) and 1 (opaque)."
                )

            ]
        ),

    )

    def _after_init(self):

        self.offsets = defaultdict(lambda: _a.arrayd([0, 0, 0]))

        self._add_shortcuts()

    @entry_point('grid')
    def _read_nosource(self, grid):
        """
        Reads the grid directly from a sisl grid.
        """
        self.grid = grid

        if self.grid is None:
            raise ValueError("grid was not set")

    @entry_point('grid file')
    def _read_grid_file(self, grid_file):
        """
        Reads the grid from any sile that implements `read_grid`.
        """
        self.grid = self.get_sile(grid_file).read_grid()

    def _after_read(self):

        #Inform of the new available ranges
        range_keys = ("x_range", "y_range", "z_range")

        for ax, key in enumerate(range_keys):
            self.modify_param(key, "inputField.params.max", self.grid.cell[ax, ax])
            self.get_param(key, as_dict=False).update_marks()

    def _set_data(self, axes, nsc, interp, trace_name, transforms, represent, cut_vacuum, grid_file,
        x_range, y_range, z_range, plot_geom, geom_kwargs, transform_bc):

        grid = self.grid

        is_skewed_2d = not grid.sc.is_cartesian() and len(axes) == 2
        if is_skewed_2d or len(axes) == 3:
            # We will tile the grid now, as at the moment there's no good way to tile it afterwards
            # Note that this means extra computation, as we are transforming (skewed_2d) or calculating
            # the isosurfaces (3d) using more than one unit cell (FIND SMARTER WAYS!)
            for ax, reps in enumerate(nsc):
                grid = grid.tile(reps, ax)
            # We have already tiled the grid, so we will set nsc to [1,1,1]
            nsc = [1, 1, 1]

        if is_skewed_2d:
            # If the grid doesn't follow the cartesian axes and we want to display it in 2D,
            # we will transform it to make our lives easier.
            grid, self.offsets["cell_transform"] = self._transform_grid_cell(
                grid, mode=transform_bc, output_shape=(np.array(interp)*grid.shape).astype(int), cval=np.nan
            )
            # The interpolation has already happened, so just set it to [1,1,1] for the rest of the method
            interp = [1, 1, 1]

        for transform in transforms:
            grid = self._transform_grid(grid, transform)

        if trace_name is None and grid_file:
            trace_name = grid_file.name

        # Get only the part of the grid that we need
        ax_ranges = [x_range, y_range, z_range]

        for ax, ax_range in enumerate(ax_ranges):
            if ax_range is not None:
                #Build an array with the limits
                lims = np.zeros((2, 3))
                lims[:, ax] = ax_range

                #Get the indices of those points
                indices = np.array([grid.index(lim) for lim in lims], dtype=int)

                #And finally get the subpart of the grid
                grid = grid.sub(np.arange(indices[0, ax], indices[1, ax] + 1), ax)

        if cut_vacuum and getattr(grid, "geometry", None):
            grid, lims = self._cut_vacuum(grid)
            self.offsets["vacuum"] = lims[0]
        else:
            self.offsets["vacuum"] = [0, 0, 0]

        interp_factors = np.array([factor if ax in axes else 1 for ax, factor in enumerate(interp)], dtype=int)

        interpolate = (interp_factors != 1).any()

        if interpolate:
            grid = grid.interp((np.array(interp_factors)*grid.shape).astype(int))

        for ax in [0, 1, 2]:
            if ax not in axes:
                grid = grid.average(ax)

        # Choose the representation of the grid that we want to display
        if represent == 'real':
            values = grid.grid.real
        elif represent == 'imag':
            values = grid.grid.imag
        elif represent == 'mod':
            values = np.absolute(grid.grid)
        elif represent.endswith("phase"):
            values = np.angle(grid.grid, deg=represent.startswith("deg"))

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
        plot_func(grid, values, axes, nsc, trace_name, showlegend=bool(trace_name) or values.ndim == 3)

        # Add also the geometry if the user requested it
        # This should probably not work like this. It should make use
        # of MultiplePlot somehow. The problem is that right now, the bonds
        # are calculated each time this method is called, for example
        if plot_geom:
            geom = getattr(self.grid, 'geometry', None)
            if geom is None:
                print('You asked to plot the geometry, but the grid does not contain any geometry')
            else:
                geom_plot = geom.plot(**{'axes': axes, "nsc": self.get_setting("nsc"), **geom_kwargs})

                self.add_traces(geom_plot.data)

        self.update_layout(legend_orientation='h')

    def _get_ax_range(self, grid, ax, nsc):

        offset = self._get_offset(ax)

        ax_vals = np.arange(0, nsc[ax]*grid.cell[ax, ax], grid.dcell[ax, ax]) + offset

        if len(ax_vals) == grid.shape[ax] + 1:
            ax_vals = ax_vals[:-1]

        return ax_vals

    def _get_offset(self, ax, offset, x_range, y_range, z_range):

        ax_range = [x_range, y_range, z_range][ax]
        grid_offset = _a.asarrayd(offset) + self.offsets["vacuum"] + self.offsets["cell_transform"]

        if ax_range is not None:
            offset = ax_range[0]
        else:
            offset = 0

        return offset + grid_offset[ax]

    def _get_offsets(self, display_axes=[0, 1, 2]):
        return np.array([self._get_offset(ax) for ax in display_axes])

    @staticmethod
    def _cut_vacuum(grid):

        if not hasattr(grid, "geometry"):
            raise ValueError("The grid does not have an associated geometry, and therefore we can not calculate where the vacuum is.")

        geom = grid.geometry
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

        if isinstance(transform, str):

            # Since this may come from the GUI, there might be extra spaces
            transform = transform.strip()

            # If is a string with no dots, we will assume it is a numpy function
            if len(transform.split(".")) == 1:
                transform = f"numpy.{transform}"

        return grid.apply(transform)

    def _plot1D(self, grid, values, display_axes, nsc, name, **kwargs):
        """Takes care of plotting the grid in 1D"""
        ax = display_axes[0]

        if nsc[ax] > 1:
            values = np.tile(values, nsc[ax])

        self.data = [{
            'type': 'scatter',
            'mode': 'lines',
            'y': values,
            'x': self._get_ax_range(grid, ax, nsc),
            'name': name,
            **kwargs
        }]

        axes_titles = {'xaxis_title': f'{("X","Y", "Z")[ax]} axis [Ang]', 'yaxis_title': 'Values'}

        self.update_layout(**axes_titles)

    def _plot2D(self, grid, values, display_axes, nsc, name, crange, cmid, colorscale, zsmooth, isos, **kwargs):
        """
        Takes care of plotting the grid in 2D

        - It uses plotly's heatmap trace.
        - It also draws contours for the isovalues.
        """
        from skimage.measure import find_contours
        xaxis = display_axes[0]
        yaxis = display_axes[1]

        if xaxis < yaxis:
            values = values.T

        values = np.tile(values, (nsc[yaxis], nsc[xaxis]))

        if crange is None:
            crange = [None, None]
        cmin, cmax = crange

        if cmid is None and cmin is None and cmax is None:
            real_vals = values[~np.isnan(values)]
            if np.any(real_vals > 0) and np.any(real_vals < 0):
                cmid = 0

        xs = self._get_ax_range(grid, xaxis, nsc)
        ys = self._get_ax_range(grid, yaxis, nsc)

        is_cartesian = grid.sc.is_cartesian()

        self.add_trace({
            'type': 'heatmap',
            'name': name,
            'z': values,
            'x': xs,
            'y': ys,
            'zsmooth': zsmooth,
            'zmin': cmin,
            'zmax': cmax,
            'zmid': cmid,
            'colorscale': colorscale,
            **kwargs
        })

        # Draw the contours (if any)
        if len(isos) > 0:
            offsets = self._get_offsets(display_axes)
            isos_param = self.get_param("isos")
            minval = np.nanmin(values)
            maxval = np.nanmax(values)

        for iso in isos:

            iso = isos_param.complete_query(iso)

            # Infer the iso value either from val or from frac
            isoval = iso.get("val")
            if isoval is None:
                frac = iso.get("frac")
                if frac is None:
                    raise ValueError(f"You are providing an iso query without 'val' and 'frac'. There's no way to know the isovalue!\nquery: {iso}")
                isoval = minval + (maxval-minval)*frac

            # Find contours at a constant value of 0.8
            contours = find_contours(values, isoval)

            contour_xs = []
            contour_ys = []
            for contour in contours:
                # Swap the first and second columns so that we have [x,y] for each
                # contour point (instead of [row, col], which means [y, x])
                contour_coords = contour[:, [1, 0]]
                # Then convert from indices to
                contour_coords = contour_coords.dot(grid.dcell[display_axes])[:, display_axes] + offsets
                contour_xs = [*contour_xs, None, *contour_coords[:, 0]]
                contour_ys = [*contour_ys, None, *contour_coords[:, 1]]

            color = iso.get("color")
            self.add_scatter(
                x=contour_xs, y=contour_ys,
                marker_color=color, line_color=color,
                opacity=iso.get("opacity"),
                name=iso.get("name", "").replace("$isoval$", str(isoval))
            )

        axes_titles = {'xaxis_title': f'{("X","Y", "Z")[xaxis]} axis [Ang]', 'yaxis_title': f'{("X","Y", "Z")[yaxis]} axis [Ang]'}

        self.update_layout(**axes_titles)

    def _plot_2D_carpet(self, grid, values, xaxis, yaxis):
        """
        CURRENTLY NOT USED, but kept here just in case it is needed in the future

        It was supposed to display skewed grids in 2D, but it has some limitations
        (see https://github.com/zerothi/sisl/pull/268#issuecomment-702758586). In these cases,
        the grid is first transformed to cartesian coordinates and then plotted in a regular map
        instead of using the Carpet trace.
        """

        minval, maxval = values.min(), values.max()

        values = values.T

        x, y = np.mgrid[:values.shape[0], :values.shape[1]]
        x, y = x * grid.dcell[xaxis, xaxis] + y * grid.dcell[yaxis, xaxis], x * grid.dcell[xaxis, yaxis] + y * grid.dcell[yaxis, yaxis]

        self.figure.add_trace(go.Contourcarpet(
            z = values,
            contours = dict(
                start = minval,
                end = maxval,
                size = (maxval - minval) / 40,
                showlines=False
            ),
        ))

        self.figure.add_trace(go.Carpet(
            a = np.arange(values.shape[1]),
            b = np.arange(values.shape[0]),
            x = x,
            y = y,
            aaxis = dict(
                showgrid=False,
                showline=False,
                showticklabels="none"
            ),
            baxis = dict(
                showgrid=False,
                showline=False,
                showticklabels="none"
            ),
        ))

    @staticmethod
    def _transform_grid_cell(grid, cell=np.eye(3), output_shape=None, mode="constant", order=1, **kwargs):
        """
        Applies a linear transformation to the grid to get it relative to arbitrary cell.

        This method can be used, for example to get the values of the grid with respect to
        the standard basis, so that you can easily visualize it or overlap it with other grids
        (e.g. to perform integrals).

        Parameters
        -----------
        cell: array-like of shape (3,3)
            these cell represent the directions that you want to use as references for
            the new grid.

            The length of the axes does not have any effect! They will be rescaled to create
            the minimum bounding box necessary to accomodate the unit cell.
        output_shape: array-like of int of shape (3,), optional
            the shape of the final output. If not provided, the current shape of the grid
            will be used. 

            Notice however that if the transformation applies a big shear to the image (grid)
            you will probably need to have a bigger output_shape.
        mode: str, optional
            determines how to handle borders. See scipy docs for more info on the possible values.
        order : int 0-5, optional
            the order of the spline interpolation to calculate the values (since we are applying
            a transformation, we don't actually have values for the new locations and we need to
            interpolate them)
            1 means linear, 2 quadratic, etc...
        **kwargs:
            the rest of keyword arguments are passed directly to `scipy.ndimage.affine_transform`

        See also
        ----------
        scipy.ndimage.affine_transform : method used to apply the linear transformation.
        """
        # Take the current shape of the grid if no output shape was provided
        if output_shape is None:
            output_shape = grid.shape

        # Get the current cell in coordinates of the destination axes
        inv_cell = cell_invert(cell).T
        projected_cell = grid.cell.dot(inv_cell)

        # From that, infere how long will the bounding box of the cell be
        lengths = abs(projected_cell).sum(axis=0)

        # Create the transformation matrix. Since we want to control the shape
        # of the output, we can not use grid.dcell directly, we need to modify it.
        scales = output_shape / lengths
        forward_t = (grid.dcell.dot(inv_cell)*scales).T

        # Scipy's affine transform asks for the inverse transformation matrix, to
        # map from output pixels to input pixels. By taking the inverse of our
        # transformation matrix, we get exactly that.
        tr = cell_invert(forward_t).T

        # Calculate the offset of the image so that all points of the grid "fall" inside
        # the output array.
        # For this we just calculate the centers of the input and output images
        center_input = 0.5 * (_a.asarrayd(grid.shape) - 1)
        center_output = 0.5 * (_a.asarrayd(output_shape) - 1)

        # And then make sure that the input center that is interpolated from the output
        # falls in the actual input's center
        offset = center_input - tr.dot(center_output)

        # We pass all the parameters to scipy's affine_transform
        transformed_image = affine_transform(grid.grid, tr, order=1, offset=offset,
            output_shape=output_shape, mode=mode, **kwargs)

        # Create a new grid with the new shape and the new cell (notice how the cell
        # is rescaled from the input cell to fit the actual coordinates of the system)
        new_grid = grid.__class__((1, 1, 1), sc=cell*lengths.reshape(3, 1))
        new_grid.grid = transformed_image

        # Find the offset between the origin before and after the transformation
        return new_grid, new_grid.dcell.dot(forward_t.dot(offset))

    def _plot3D(self, grid, values, display_axes, nsc, name, isos, **kwargs):
        """Takes care of plotting the grid in 3D"""
        # The minimum and maximum values might be needed at some places
        minval, maxval = np.min(values), np.max(values)

        # Get the isos input field. It will be used to get the default fraction
        # value and to complete queries
        isos_param = self.get_param("isos")

        # If there are no iso queries, we are going to create 2 isosurfaces.
        if len(isos) == 0 and maxval != minval:

            default_iso_frac = isos_param["frac"].default

            # If the default frac is 0.3, they will be displayed at 0.3 and 0.7
            isos = [
                {"frac": default_iso_frac},
                {"frac": 1-default_iso_frac}
            ]

        # Go through each iso query to draw the isosurface
        for iso in isos:

            iso = isos_param.complete_query(iso)

            if not iso.get("active", True):
                continue

            # Infer the iso value either from val or from frac
            isoval = iso.get("val")
            if isoval is None:
                frac = iso.get("frac")
                if frac is None:
                    raise ValueError(f"You are providing an iso query without 'val' and 'frac'. There's no way to know the isovalue!\nquery: {iso}")
                isoval = minval + (maxval-minval)*frac

            # Calculate the isosurface
            vertices, faces, normals, intensities = grid.isosurface(isoval, iso.get("step_size", 1))

            vertices = vertices + self._get_offsets()

            # Create the mesh trace and add it to the plot
            x, y, z = vertices.T
            I, J, K = faces.T

            self.add_trace(go.Mesh3d(x=x,
                        y=y,
                        z=z,
                        i=I,
                        j=J,
                        k=K,
                        color=iso.get("color"),
                        opacity=iso.get("opacity"),
                        name=iso.get("name", "").replace("$isoval$", str(isoval)),
                        **kwargs
            ))

        self.layout.scene = {'aspectmode': 'data'}

    def _after_get_figure(self, axes):

        # If we are plotting the 2D version, use a 1:1 ratio
        if len(axes) == 2:
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
        """
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
        """
        if isinstance(ax, int):
            ax = [ax]
        if isinstance(steps, int):
            steps = [steps]*len(ax)

        nsc = list(self.get_setting("nsc"))

        for a, step in zip(ax, steps):
            nsc[a] = max(1, nsc[a]-step)

        return self.update_settings(nsc=nsc)

    def tile(self, tiles, ax):
        """
        Tile a given axis to display more unit cells in the plot

        Parameters
        ----------
        tiles: int or array-like
            factor by which the supercell will be multiplied along axes `ax`.

            If you provide multiple tiles, it needs to match the number of axes provided.
        ax: int or array-like
            axis that you want to tile.

            If you provide multiple axes, the number of different tiles must match the number of axes or be a single int.
        """
        if isinstance(ax, int):
            ax = [ax]
        if isinstance(tiles, int):
            tiles = [tiles]*len(ax)

        nsc = [*self.get_setting("nsc")]

        for a, tile in zip(ax, tiles):
            nsc[a] *= tile

        return self.update_settings(nsc=nsc)

    def scan(self, along=None, start=None, stop=None, step=None, num=None, breakpoints=None, mode="moving_slice", animation_kwargs=None, **kwargs):
        """
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
        step: float, optional
            the distance between steps in Angstrom.

            If not provided and `num` is also not provided, it will default to 1 Ang.
        num: int , optional
            the number of steps that you want the scan to consist of.

            If `step` is passed, this argument is ignored.

            Note that the grid is only stored once, so having a big number of steps is not that big of a deal.
        breakpoints: array-like, optional
            the discrete points of the scan. To be used if you don't want regular steps.
            If the last step is exactly the length of the cell, it will be moved one dcell back to avoid errors.

            Note that if this parameter is passed, both `step` and `num` are ignored.
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
        -------
        sisl.viz.plotly.Animation
            An animation representation of the scan
        """
        # Do some checks on the args provided
        if sum(1 for arg in (step, num, breakpoints) if arg is not None) > 1:
            raise ValueError(f"Only one of ('step', 'num', 'breakpoints') should be passed.")

        # If no axis is provided, let's get the first one that is not displayed
        if along is None:
            displayed = self.get_setting('axes')
            not_displayed = [ax["value"] for ax in self.get_param('axes')['inputField.params.options'] if ax["value"] not in displayed]
            along = not_displayed[0] if not_displayed else 2

        # We get the key that needs to be animated (we will divide the full range in frames)
        range_key = ["x_range", "y_range", "z_range"][along]

        # Get the full range
        if start is not None and stop is not None:
            along_range = [start, stop]
        else:
            along_range = self.get_setting(range_key)
            if along_range is None:
                range_param = self.get_param(range_key)
                along_range = [range_param[f"inputField.params.{lim}"] for lim in ["min", "max"]]
            if start is not None:
                along_range[0] = start
            if stop is not None:
                along_range[1] = stop

        if breakpoints is None:
            if step is None and num is None:
                step = 1.0
            if step is None:
                step = (along_range[1] - along_range[0]) / num
            else:
                num = (along_range[1] - along_range[0]) // step

            # np.linspace will use the last point as a step (and we don't want it)
            # therefore we will add an extra step
            breakpoints = np.linspace(*along_range, int(num) + 1)

        if breakpoints[-1] == self.grid.cell[along, along]:
            breakpoints[-1] = self.grid.cell[along, along] - self.grid.dcell[along, along]

        if mode == "moving_slice":
            return self._moving_slice_scan(along, breakpoints)
        elif mode == "as_is":
            return self._asis_scan(range_key, breakpoints, animation_kwargs=animation_kwargs, **kwargs)

    def _asis_scan(self, range_key, breakpoints, animation_kwargs=None, **kwargs):
        """
        Returns an animation containing multiple frames scaning along an axis.

        Parameters
        -----------
        range_key: {'x_range', 'y_range', 'z_range'}
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
        """
        # Generate the plot using self as a template so that plots don't need
        # to read data, just process it and show it differently.
        # (If each plot read the grid, the memory requirements would be HUGE)
        scan = self.animated(
            {
                range_key: [[bp, breakpoints[i+1]] for i, bp in enumerate(breakpoints[:-1])]
            },
            plot_template=self,
            fixed={**{key: val for key, val in self.settings.items() if key != range_key}, **kwargs},
            frame_names=[f'{bp:2f}' for bp in breakpoints],
            **(animation_kwargs or {})
        )

        # Set all frames to the same colorscale, if it's a 2d representation
        if len(self.get_setting("axes")) == 2:
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
        displayed_axes = [i for i in range(3) if i != ax]
        shape = np.array(self.grid.shape)[displayed_axes]
        cmin = np.min(self.grid.grid)
        cmax = np.max(self.grid.grid)
        x_ax, y_ax = displayed_axes
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

        def ax_title(ax): return f'{["X", "Y", "Z"][ax]} axis [Ang]'

        # Layout
        fig.update_layout(
                title=f'Grid scan along {["X", "Y", "Z"][ax]} axis',
                width=600,
                height=600,
                scene=dict(
                            xaxis=dict(title=ax_title(x_ax)),
                            yaxis=dict(title=ax_title(y_ax)),
                            zaxis=dict(range=[0, self.grid.cell[ax, ax]], autorange=False, title=ax_title(ax)),
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


class WavefunctionPlot(GridPlot):
    """
    An extension of GridPlot specifically tailored for plotting wavefunctions

    Parameters
    -----------
    eigenstate: EigenstateElectron, optional
        The eigenstate that contains the coefficients of the wavefunction.
        Note that an eigenstate can contain coefficients for multiple states.
    geometry: Geometry, optional
        Necessary to generate the grid and to plot the wavefunctions, since
        the basis orbitals are needed.             If you provide a
        hamiltonian, the geometry is probably inside the hamiltonian, so you
        don't need to provide it.             However, this field is
        compulsory if you are providing the eigenstate directly.
    k: array-like, optional
        If the eigenstates need to be calculated from a hamiltonian, the k
        point for which you want them to be calculated
    grid_prec: float, optional
        The spacing between points of the grid where the wavefunction will be
        projected (in Ang).             If you are plotting a 3D
        representation, take into account that a very fine and big grid could
        result in             your computer crashing on render. If it's the
        first time you are using this function,             assess the
        capabilities of your computer by first using a low-precision grid and
        increase             it gradually.
    i: int, optional
        The index of the wavefunction
    grid: Grid, optional
        A sisl.Grid object. If provided, grid_file is ignored.
    grid_file: Sile, optional
        a sile that can read grids, e.g. `_gridSileSiesta` or `chgSileVASP` and friends.
    represent:  optional
        The representation of the grid that should be displayed
    transforms:  optional
        Transformations to apply to the whole grid.             It can be a
        function, or a string that represents the path             to a
        function (e.g. "scipy.exp"). If a string that is a single
        word is provided, numpy will be assumed to be the module (e.g.
        "square" will be converted into "np.square").              Note that
        transformations will be applied in the order provided. Some
        transforms might not be necessarily commutable (e.g. "abs" and
        "cos").
    transform_bc: optional
        Transform the boundary conditions.
    spin : int, optional
        which spin-component to plot, default to the first spin-component.
        Only used in polarized wavefunctions.
    axes:  optional
        The axis along you want to see the grid, it will be averaged along
        the other ones
    plot_geom: bool, optional
        If True the geometry associated to the grid will also be plotted
    geom_kwargs: dict, optional
        Extra arguments that are passed to geom.plot() if plot_geom is set to
        True
    zsmooth:  optional
        Parameter that smoothens how data looks in a heatmap.
        'best' interpolates data, 'fast' interpolates pixels, 'False'
        displays the data as is.
    interp: array-like, optional
        Interpolation factors to make the grid finer on each axis.See the
        zsmooth setting for faster smoothing of 2D heatmap.
    nsc: array-like, optional
        number of times the geometry should be repeated
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
    x_range: array-like of shape (2,), optional
        Range where the X is displayed. Should be inside the unit cell,
        otherwise it will fail.
    y_range: array-like of shape (2,), optional
        Range where the Y is displayed. Should be inside the unit cell,
        otherwise it will fail.
    z_range: array-like of shape (2,), optional
        Range where the Z is displayed. Should be inside the unit cell,
        otherwise it will fail.
    crange: array-like of shape (2,), optional
        The range of values that the colorbar must enclose. This controls
        saturation and hides below threshold values.
    cmid: int, optional
        The value to set at the center of the colorbar. If not provided, the
        color range is used
    colorscale: str, optional
        A valid plotly colorscale. See https://plotly.com/python/colorscales/
    iso_vals: array-like of shape (2,), optional
        The minimum and maximum values of the isosurfaces to be displayed.
        If not provided, iso_frac will be used to calculate these values
        (which is more versatile).
    iso_frac: float, optional
        If iso_vals is not provided, this value is used to calculate where
        the isosurfaces are drawn.             It calculates them from the
        minimum and maximum values of the grid like so:             If
        iso_frac = 0.3:             (min_value----30%-----ISOMIN----------
        ISOMAX---30%-----max_value)             Therefore, it should be a
        number between 0 and 0.5.
    surface_count: int, optional
        The number of surfaces between the lower and the upper limits of
        iso_vals
    type3D:  optional
        This controls how the 3D data is displayed.              'volume'
        displays different layers with different levels of opacity so that
        there is more sensation of depth.             'isosurface' displays
        only isosurfaces and nothing inbetween them. For plotting grids with
        positive and negative             values, you should use 'isosurface'
        or two different 'volume' plots.              If not provided, the
        plot will decide for you based on the above mentioned fact
    opacityscale:  optional
        Controls how the opacity changes through layers.              See
        https://plotly.com/python/3d-volume-plots/ for a display of the
        different possibilities
    surface_opacity: float, optional
        The opacity of the isosurfaces drawn by 3d plots from 0 (transparent)
        to 1 (opaque).
    root_fdf: fdfSileSiesta, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    """

    _plot_type = 'Wavefunction'

    _parameters = (

        PlotableInput(key="eigenstate", name="Electron eigenstate",
            default=None,
            dtype=sisl.EigenstateElectron,
            help="""The eigenstate that contains the coefficients of the wavefunction.
            Note that an eigenstate can contain coefficients for multiple states.
            """
        ),

        SislObjectInput(key='geometry', name='Geometry',
            default=None,
            dtype=sisl.Geometry,
            help="""Necessary to generate the grid and to plot the wavefunctions, since the basis orbitals are needed.
            If you provide a hamiltonian, the geometry is probably inside the hamiltonian, so you don't need to provide it.
            However, this field is compulsory if you are providing the eigenstate directly."""
        ),

        Array1DInput(key='k', name='K point',
            default=(0, 0, 0),
            help="""If the eigenstates need to be calculated from a hamiltonian, the k point for which you want them to be calculated"""
        ),

        SpinSelect(key='spin', name="Spin",
            default=0,
            help="""The spin component where the eigenstate should be calculated.
            Only meaningful if the state needs to be calculated from the hamiltonian.""",
            only_if_polarized=True,
        ),

        FloatInput(key='grid_prec', name='Grid precision',
            default=0.2,
            help="""The spacing between points of the grid where the wavefunction will be projected (in Ang).
            If you are plotting a 3D representation, take into account that a very fine and big grid could result in
            your computer crashing on render. If it's the first time you are using this function,
            assess the capabilities of your computer by first using a low-precision grid and increase
            it gradually.
            """
        ),

        IntegerInput(key='i', name='Wavefunction index',
            default=0,
            help="The index of the wavefunction"
        ),

    )

    _overwrite_defaults = {
        'axes': [0, 1, 2],
        'plot_geom': True,
    }

    @entry_point('eigenstate')
    def _read_nosource(self, eigenstate):
        """
        Uses an already calculated Eigenstate object to generate the wavefunctions.
        """
        if eigenstate is None:
            raise ValueError('No eigenstate was provided')

        self.eigenstate = eigenstate

    @entry_point('hamiltonian')
    def _read_from_H(self, k, spin):
        """
        Calculates the eigenstates from a Hamiltonian and then generates the wavefunctions.
        """
        self.setup_hamiltonian()

        self.eigenstate = self.H.eigenstate(k, spin=spin[0])

    def _after_read(self):
        # Just avoid here GridPlot's _after_grid. Note that we are
        # calling it later in _set_data
        pass

    def _set_data(self, i, geometry, grid, k, grid_prec):

        if geometry is not None:
            self.geometry = geometry
        if getattr(self, 'geometry', None) is None:
            raise ValueError('No geometry was provided and we need it the basis orbitals to build the wavefunctions from the coefficients!')

        if grid is None:
            dtype = np.float64 if (np.array(k) == 0).all() else np.complex128
            self.grid = sisl.Grid(grid_prec, geometry=self.geometry, dtype=dtype)

        # GridPlot's after_read basically sets the x_range, y_range and z_range options
        # which need to know what the grid is, that's why we are calling it here
        super()._after_read()

        self.eigenstate[i].wavefunction(self.grid)

        super()._set_data()

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
