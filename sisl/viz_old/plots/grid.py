# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from collections import defaultdict
from sisl.viz.plots.geometry import GeometryPlot
import numpy as np
from scipy.ndimage import affine_transform

import sisl
from sisl.messages import warn
from sisl._supercell import cell_invert
from sisl import _array as _a
from ..plot import Plot, entry_point
from ..input_fields import (
    TextInput, SileInput, Array1DInput, BoolInput,
    ColorInput, OptionsInput, CreatableOptionsInput, IntegerInput, FloatInput, RangeInput, RangeSliderInput,
    QueriesInput, ProgramaticInput, PlotableInput, SislObjectInput, PlotableInput, SpinSelect,
    GeomAxisSelect
)


class GridPlot(Plot):
    """
    Versatile visualization tool for any kind of grid.

    Parameters
    ------------
    grid: Grid, optional
        A sisl.Grid object. If provided, grid_file is ignored.
    grid_file: cubeSile or rhoSileSiesta or ldosSileSiesta or rhoinitSileSiesta or rhoxcSileSiesta or drhoSileSiesta or baderSileSiesta or iorhoSileSiesta or totalrhoSileSiesta or stsSileSiesta or stmldosSileSiesta or hartreeSileSiesta or neutralatomhartreeSileSiesta or totalhartreeSileSiesta or gridncSileSiesta or ncSileSiesta or fdfSileSiesta or tsvncSileSiesta or chgSileVASP or locpotSileVASP, optional
        A filename that can be return a Grid through `read_grid`.
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
    axes:  optional
        The axis along you want to see the grid, it will be reduced along the
        other ones, according to the the `reduce_method` setting.
    zsmooth:  optional
        Parameter that smoothens how data looks in a heatmap.
        'best' interpolates data, 'fast' interpolates pixels, 'False'
        displays the data as is.
    interp: array-like, optional
        Interpolation factors to make the grid finer on each axis.See the
        zsmooth setting for faster smoothing of 2D heatmap.
    transform_bc:  optional
        The boundary conditions when a cell transform is applied to the grid.
        Cell transforms are only             applied when the grid's cell
        doesn't follow the cartesian coordinates and the requested display is
        2D or 1D.
    nsc: array-like, optional
        Number of times the grid should be repeated
    offset: array-like, optional
        The offset of the grid along each axis. This is important if you are
        planning to match this grid with other geometry related plots.
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
    reduce_method:  optional
        The method used to reduce the dimensions that will not be displayed
        in the plot.
    isos: array-like of dict, optional
        The isovalues that you want to represent.             The way they
        will be represented is of course dependant on the type of
        representation:                 - 2D representations: A contour (i.e.
        a line)                 - 3D representations: A surface
        Each item is a dict.    Structure of the dict: {         'name': The
        name of the iso query. Note that you can use $isoval$ as a template
        to indicate where the isoval should go.         'val': The iso value.
        If not provided, it will be infered from `frac`         'frac': If
        val is not provided, this is used to calculate where the isosurface
        should be drawn.                     It calculates them from the
        minimum and maximum values of the grid like so:
        If iso_frac = 0.3:                     (min_value-----
        ISOVALUE(30%)-----------max_value)                     Therefore, it
        should be a number between 0 and 1.
        'step_size': The step size to use to calculate the isosurface in case
        it's a 3D representation                     A bigger step-size can
        speed up the process dramatically, specially the rendering part
        and the resolution may still be more than satisfactory (try to use
        step_size=2). For very big                     grids your computer
        may not even be able to render very fine surfaces, so it's worth
        keeping                     this setting in mind.         'color':
        The color of the surface/contour.         'opacity': Opacity of the
        surface/contour. Between 0 (transparent) and 1 (opaque). }
    plot_geom: bool, optional
        If True the geometry associated to the grid will also be plotted
    geom_kwargs: dict, optional
        Extra arguments that are passed to geom.plot() if plot_geom is set to
        True
    root_fdf: fdfSileSiesta, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    entry_points_order: array-like, optional
        Order with which entry points will be attempted.
    backend:  optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    """

    # Define all the class attributes
    _plot_type = "Grid"

    _update_methods = {
        "read_data": [],
        "set_data": ["_prepare1D", "_prepare2D", "_prepare3D"],
        "get_figure": []
    }

    _param_groups = (
        {
            "key": "grid_shape",
            "name": "Grid shape",
            "icon": "image_aspect_ratio",
            "description": "Settings related to the shape of the grid, including it's dimensionality and how it is reduced if needed."
        },

        {
            "key": "grid_values",
            "name": "Grid values",
            "icon": "image",
            "description": "Settings related to the values of the grid. They involve both how they are processed and displayed"
        },
    )

    _parameters = (

        PlotableInput(
            key="grid", name="Grid",
            dtype=sisl.Grid,
            default=None,
            group="dataread",
            help="A sisl.Grid object. If provided, grid_file is ignored."
        ),

        SileInput(
            key="grid_file", name="Path to grid file",
            required_attrs=["read_grid"],
            default=None,
            params={
                "placeholder": "Write the path to your grid file here..."
            },
            group="dataread",
            help="A filename that can be return a Grid through `read_grid`."
        ),

        OptionsInput(
            key="represent", name="Representation of the grid",
            default="real",
            params={
                'options': [
                    {'label': 'Real part', 'value': "real"},
                    {'label': 'Imaginary part', 'value': 'imag'},
                    {'label': 'Complex modulus', 'value': "mod"},
                    {'label': 'Phase (in rad)', 'value': 'phase'},
                    {'label': 'Phase (in deg)', 'value': 'deg_phase'},
                ],
                'isMulti': False,
                'isSearchable': True,
                'isClearable': False
            },
            group="grid_values",
            help="""The representation of the grid that should be displayed"""
        ),

        CreatableOptionsInput(
            key="transforms", name="Grid transforms",
            default=[],
            params={
                'options': [
                    {'label': 'Square', 'value': 'square'},
                    {'label': 'Absolute', 'value': 'abs'},
                ],
                'isMulti': True,
                'isSearchable': True,
                'isClearable': True
            },
            group="grid_values",
            help="""Transformations to apply to the whole grid.
            It can be a function, or a string that represents the path
            to a function (e.g. "scipy.exp"). If a string that is a single
            word is provided, numpy will be assumed to be the module (e.g.
            "square" will be converted into "np.square"). 
            Note that transformations will be applied in the order provided. Some
            transforms might not be necessarily commutable (e.g. "abs" and "cos")."""
        ),

        GeomAxisSelect(
            key = "axes", name="Axes to display",
            default=["z"],
            group="grid_shape",
            help = """The axis along you want to see the grid, it will be reduced along the other ones, according to the the `reduce_method` setting."""
        ),

        OptionsInput(
            key = "zsmooth", name="2D heatmap smoothing method",
            default=False,
            params={
                'options': [
                    {'label': 'best', 'value': 'best'},
                    {'label': 'fast', 'value': 'fast'},
                    {'label': 'False', 'value': False},
                ],
                'isSearchable': True,
                'isClearable': False
            },
            group="grid_values",
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
            group="grid_shape",
            help="Interpolation factors to make the grid finer on each axis.<br>See the zsmooth setting for faster smoothing of 2D heatmap."
        ),

        CreatableOptionsInput(key="transform_bc", name="Transform boundary conditions",
            default="wrap",
            params={
                'options': [
                    {'label': 'constant', 'value': 'constant'},
                    {'label': 'wrap', 'value': 'wrap'},
                ],
            },
            group="grid_values",
            help="""The boundary conditions when a cell transform is applied to the grid. Cell transforms are only
            applied when the grid's cell doesn't follow the cartesian coordinates and the requested display is 2D or 1D.
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
            group="grid_shape",
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

        TextInput(
            key="trace_name", name="Trace name",
            default=None,
            params={
                "placeholder": "Give a name to the trace..."
            },
            help="""The name that the trace will show in the legend. Good when merging with other plots to be able to toggle the trace in the legend"""
        ),

        RangeSliderInput(
            key="x_range", name="X range",
            default=None,
            params={
                "min": 0
            },
            group="grid_shape",
            help="Range where the X is displayed. Should be inside the unit cell, otherwise it will fail.",
        ),

        RangeSliderInput(
            key="y_range", name="Y range",
            default=None,
            params={
                "min": 0
            },
            group="grid_shape",
            help="Range where the Y is displayed. Should be inside the unit cell, otherwise it will fail.",
        ),

        RangeSliderInput(
            key="z_range", name="Z range",
            default=None,
            params={
                "min": 0
            },
            group="grid_shape",
            help="Range where the Z is displayed. Should be inside the unit cell, otherwise it will fail.",
        ),

        RangeInput(
            key="crange", name="Colorbar range",
            default=[None, None],
            group="grid_values",
            help="The range of values that the colorbar must enclose. This controls saturation and hides below threshold values."
        ),

        IntegerInput(
            key="cmid", name="Colorbar center",
            default=None,
            group="grid_values",
            help="""The value to set at the center of the colorbar. If not provided, the color range is used"""
        ),

        TextInput(
            key="colorscale", name="Color scale",
            default=None,
            group="grid_values",
            help="""A valid plotly colorscale. See https://plotly.com/python/colorscales/"""
        ),

        OptionsInput(key="reduce_method", name="Reduce method",
            default="average",
            params={
                'options': [
                    {'label': 'average', 'value': 'average'},
                    {'label': 'sum', 'value': 'sum'},
                ],
            },
            group="grid_values",
            help="""The method used to reduce the dimensions that will not be displayed in the plot."""
        ),

        QueriesInput(key = "isos", name = "Isosurfaces / contours",
            default = [],
            group="grid_values",
            help = """The isovalues that you want to represent.
            The way they will be represented is of course dependant on the type of representation:
                - 2D representations: A contour (i.e. a line)
                - 3D representations: A surface
            """,
            queryForm = [

                TextInput(
                    key="name", name="Name",
                    default="Iso=$isoval$",
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

                ColorInput(
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

        BoolInput(key='plot_geom', name='Plot geometry',
            default=False,
            help="""If True the geometry associated to the grid will also be plotted"""
        ),

        ProgramaticInput(key='geom_kwargs', name='Geometry plot extra arguments',
            default={},
            dtype=dict,
            help="""Extra arguments that are passed to geom.plot() if plot_geom is set to True"""
        ),

    )

    def _after_init(self):

        self.offsets = defaultdict(lambda: _a.arrayd([0, 0, 0]))

        self._add_shortcuts()

    @entry_point('grid', 0)
    def _read_nosource(self, grid):
        """
        Reads the grid directly from a sisl grid.
        """
        self.grid = grid

        if self.grid is None:
            raise ValueError("grid was not set")

    @entry_point('grid file', 1)
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

    def _infer_grid_axes(self, axes, cell, tol=1e-3):
        """Returns which are the lattice vectors that correspond to each cartesian direction"""
        grid_axes = []
        for ax in axes:
            if ax in ("x", "y", "z"):
                coord_index = "xyz".index(ax)
                lattice_vecs = np.where(cell[:, coord_index] > tol)[0]
                if lattice_vecs.shape[0] != 1:
                    raise ValueError(f"There are {lattice_vecs.shape[0]} lattice vectors that contribute to the {'xyz'[coord_index]} coordinate.")
                grid_axes.append(lattice_vecs[0])
            else:
                grid_axes.append("abc".index(ax))

        return grid_axes

    def _is_cartesian_unordered(self, cell, tol=1e-3):
        """Whether a cell has cartesian axes as lattice vectors, regardless of their order.

        Parameters
        -----------
        cell: np.array of shape (3, 3)
            The cell that you want to check.
        tol: float, optional
            Threshold value to consider a component of the cell nonzero.
        """
        bigger_than_tol = abs(cell) > tol
        return bigger_than_tol.sum() == 3 and bigger_than_tol.any(axis=0).all() and bigger_than_tol.any(axis=1).all()

    def _is_1D_cartesian(self, cell, coord_ax, tol=1e-3):
        """Whether a cell contains only one vector that contributes only to a given coordinate.

        That is, one vector follows the direction of the cartesian axis and the other vectors don't
        have any component in that direction.

        Parameters
        -----------
        cell: np.array of shape (3, 3)
            The cell that you want to check.
        coord_ax: {"x", "y", "z"}
            The cartesian axis that you are looking for in the cell.
        tol: float, optional
            Threshold value to consider a component of the cell nonzero.
        """
        coord_index = "xyz".index(coord_ax)
        lattice_vecs = np.where(cell[:, coord_index] > tol)[0]

        is_1D_cartesian = lattice_vecs.shape[0] == 1
        return is_1D_cartesian and (cell[lattice_vecs[0]] > tol).sum() == 1

    def _set_data(self, axes, nsc, interp, trace_name, transforms, represent, grid_file,
        x_range, y_range, z_range, plot_geom, geom_kwargs, transform_bc, reduce_method):

        if trace_name is None and grid_file:
            trace_name = grid_file.name

        grid = self.grid.copy()

        self._ndim = len(axes)
        self.offsets["origin"] = grid.origin

        # Choose the representation of the grid that we want to display
        grid.grid = self._get_representation(grid, represent)

        # We will tile the grid now, as at the moment there's no good way to tile it afterwards
        # Note that this means extra computation, as we are transforming (skewed_2d) or calculating
        # the isosurfaces (3d) using more than one unit cell (FIND SMARTER WAYS!)
        for ax, reps in enumerate(nsc):
            grid = grid.tile(reps, ax)

        # Determine whether we should transform the grid to cartesian axes. This will be needed
        # if the grid is skewed. However, it is never needed for the 3D representation, since we
        # compute the coordinates of each point in the isosurface, and we don't need to reduce the
        # grid.
        should_orthogonalize = ~self._is_cartesian_unordered(grid.cell) and self._ndim < 3
        # We also don't need to orthogonalize if cartesian coordinates are not requested
        # (this would mean that axes is a combination of "a", "b" and "c")
        should_orthogonalize = should_orthogonalize and bool(set(axes).intersection(["x", "y", "z"]))

        if should_orthogonalize and self._ndim == 1:
            # In 1D representations, even if the cell is skewed, we might not need to transform.
            # An example of a cell that we don't need to transform is:
            # a = [1, 1, 0], b = [1, -1, 0], c = [0, 0, 1]
            # If the user wants to display the values on the z coordinate, we can safely reduce the
            # first two axes, as they don't contribute in the Z direction. Also, it is required that
            # "c" doesn't contribute to any of the other two directions.
            should_orthogonalize &= not self._is_1D_cartesian(grid.cell, axes[0])

        if should_orthogonalize:
            grid, self.offsets["cell_transform"] = self._transform_grid_cell(
                grid, mode=transform_bc, output_shape=(np.array(interp)*grid.shape).astype(int), cval=np.nan
            )
            # The interpolation has already happened, so just set it to [1,1,1] for the rest of the method
            interp = [1, 1, 1]

            # Now the grid axes correspond to the cartesian coordinates.
            grid_axes = [{"x": 0, "y": 1, "z": 2}[ax] for ax in axes]
        elif self._ndim < 3:
            # If we are not transforming the grid, we need to get the axes of the grid that contribute to the
            # directions we have to plot.
            grid_axes = self._infer_grid_axes(axes, grid.cell)
        elif self._ndim == 3:
            grid_axes = [0, 1, 2]

        # Apply all transforms requested by the user
        for transform in transforms:
            grid = self._transform_grid(grid, transform)

        # Get only the part of the grid that we need
        ax_ranges = [x_range, y_range, z_range]
        for ax, ax_range in enumerate(ax_ranges):
            if ax_range is not None:
                # Build an array with the limits
                lims = np.zeros((2, 3))
                # If the cell was transformed, then we need to modify
                # the range to get what the user wants.
                lims[:, ax] = ax_range + self.offsets["cell_transform"][ax] - self.offsets["origin"][ax]

                # Get the indices of those points
                indices = np.array([grid.index(lim) for lim in lims], dtype=int)

                # And finally get the subpart of the grid
                grid = grid.sub(np.arange(indices[0, ax], indices[1, ax] + 1), ax)

        # Reduce the dimensions that are not going to be displayed
        for ax in [0, 1, 2]:
            if ax not in grid_axes:
                grid = getattr(grid, reduce_method)(ax)

        # Interpolate the grid to a different shape, if needed
        interp_factors = np.array([factor if ax in grid_axes else 1 for ax, factor in enumerate(interp)], dtype=int)
        interpolate = (interp_factors != 1).any()
        if interpolate:
            grid = grid.interp((np.array(interp_factors)*grid.shape).astype(int))

        # Remove the leftover dimensions
        values = np.squeeze(grid.grid)

        # Choose which function we need to use to prepare the data
        prepare_func = getattr(self, f"_prepare{self._ndim}D")

        # Use it
        backend_info = prepare_func(grid, values, axes, grid_axes, nsc, trace_name, showlegend=bool(trace_name) or values.ndim == 3)

        backend_info["ndim"] = self._ndim

        # Add also the geometry if the user requested it
        # This should probably not work like this. It should make use
        # of MultiplePlot somehow. The problem is that right now, the bonds
        # are calculated each time this method is called, for example
        geom_plot = None
        if plot_geom:
            geom = getattr(self.grid, 'geometry', None)
            if geom is None:
                warn('You asked to plot the geometry, but the grid does not contain any geometry')
            else:
                geom_plot = geom.plot(**{'axes': axes, "nsc": self.get_setting("nsc"), **geom_kwargs})

        backend_info["geom_plot"] = geom_plot

        # Define the axes titles
        backend_info["axes_titles"] = {
            f"{ax_name}axis": GeometryPlot._get_ax_title(ax) for ax_name, ax in zip(("x", "y", "z"), axes)
        }
        if self._ndim == 1:
            backend_info["axes_titles"]["yaxis"] = "Values"

        return backend_info

    def _get_ax_range(self, grid, ax, nsc):
        if isinstance(ax, int) or ax in ("a", "b", "c"):
            ax = {"a": 0, "b": 1, "c": 2}.get(ax, ax)
            ax_vals = np.linspace(0, nsc[ax], grid.shape[ax])
        else:
            offset = self._get_offset(grid, ax)

            ax = {"x": 0, "y": 1, "z": 2}[ax]

            ax_vals = np.arange(0, grid.cell[ax, ax], grid.dcell[ax, ax]) + offset

            if len(ax_vals) == grid.shape[ax] + 1:
                ax_vals = ax_vals[:-1]

        return ax_vals

    def _get_offset(self, grid, ax, offset, x_range, y_range, z_range):
        if isinstance(ax, int) or ax in ("a", "b", "c"):
            return 0
        else:
            coord_range = {"x": x_range, "y": y_range, "z": z_range}[ax]
            grid_offset =  _a.asarrayd(offset) + self.offsets["vacuum"]

            coord_index = "xyz".index(ax)
            # Now let's get the offset due to the minimum value of the axis range
            if coord_range is not None:
                offset = coord_range[0]
            else:
                # If a range was specified, the cell_transform and origo offsets were applied
                # when subbing the grid. Otherwise they have not been applied yet.
                offset = self.offsets["cell_transform"][coord_index] + self.offsets["origin"][coord_index]

            return offset + grid_offset[coord_index]

    def _get_offsets(self, grid, display_axes=[0, 1, 2]):
        return np.array([self._get_offset(grid, ax) for ax in display_axes])

    @staticmethod
    def _transform_grid(grid, transform):

        if isinstance(transform, str):

            # Since this may come from the GUI, there might be extra spaces
            transform = transform.strip()

            # If is a string with no dots, we will assume it is a numpy function
            if len(transform.split(".")) == 1:
                transform = f"numpy.{transform}"

        return grid.apply(transform)

    @staticmethod
    def _get_representation(grid, represent):
        """Returns a representation of the grid

        Parameters
        ------------
        grid: sisl.Grid
            the grid for which we want return
        represent: {"real", "imag", "mod", "phase", "deg_phase", "rad_phase"}
            the type of representation. "phase" is equivalent to "rad_phase"

        Returns
        ------------
        np.ndarray of shape = grid.shape
        """
        if represent == 'real':
            values = grid.grid.real
        elif represent == 'imag':
            values = grid.grid.imag
        elif represent == 'mod':
            values = np.absolute(grid.grid)
        elif represent in ['phase', 'rad_phase', 'deg_phase']:
            values = np.angle(grid.grid, deg=represent.startswith("deg"))
        else:
            raise ValueError(f"'{represent}' is not a valid value for the `represent` argument")

        return values

    def _prepare1D(self, grid, values, display_axes, grid_axes, nsc, name, **kwargs):
        """Takes care of preparing the values to plot in 1D"""
        display_ax = display_axes[0]

        return {"ax": display_ax, "values": values, "ax_range": self._get_ax_range(grid, display_ax, nsc), "name": name}

    def _prepare2D(self, grid, values, display_axes, grid_axes, nsc, name, crange, cmid, colorscale, zsmooth, isos, **kwargs):
        """Takes care of preparing the values to plot in 2D"""
        from skimage.measure import find_contours
        xaxis = display_axes[0]
        yaxis = display_axes[1]

        if grid_axes[0] < grid_axes[1]:
            values = values.T

        if crange is None:
            crange = [None, None]
        cmin, cmax = crange

        if cmid is None and cmin is None and cmax is None:
            real_vals = values[~np.isnan(values)]
            if np.any(real_vals > 0) and np.any(real_vals < 0):
                cmid = 0

        xs = self._get_ax_range(grid, xaxis, nsc)
        ys = self._get_ax_range(grid, yaxis, nsc)

        # Draw the contours (if any)
        if len(isos) > 0:
            offsets = self._get_offsets(grid, display_axes)
            isos_param = self.get_param("isos")
            minval = np.nanmin(values)
            maxval = np.nanmax(values)

        if set(display_axes).intersection(["x", "y", "z"]):
            coord_indices = ["xyz".index(ax) for ax in display_axes]

            def _indices_to_2Dspace(contour_coords):
                return contour_coords.dot(grid.dcell[grid_axes, :])[:, coord_indices]
        else:
            def _indices_to_2Dspace(contour_coords):
                return contour_coords / (np.array(grid.shape) / nsc)[grid_axes]

        isos_to_draw = []
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
                # Then convert from indices to coordinates in the 2D space
                contour_coords = _indices_to_2Dspace(contour_coords) + offsets
                contour_xs = [*contour_xs, None, *contour_coords[:, 0]]
                contour_ys = [*contour_ys, None, *contour_coords[:, 1]]

            # Add the information about this isoline to the list of isolines
            isos_to_draw.append({
                "x": contour_xs, "y": contour_ys,
                "color": iso.get("color"), "opacity": iso.get("opacity"),
                "name": iso.get("name", "").replace("$isoval$", f"{isoval:.4f}")
            })

        return {
            "values": values, "x": xs, "y": ys, "zsmooth": zsmooth,
            "xaxis": xaxis, "yaxis": yaxis,
            "cmin": cmin, "cmax": cmax, "cmid": cmid, "colorscale": colorscale,
            "name": name, "contours": isos_to_draw
        }

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

    def _prepare3D(self, grid, values, display_axes, grid_axes, nsc, name, isos, **kwargs):
        """Takes care of preparing the values to plot in 3D"""
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

        isos_to_draw = []
        # Go through each iso query to prepare the isosurface
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

            vertices = vertices + self._get_offsets(grid) + self.offsets["origin"]

            # Add all the isosurface info to the list that will be passed to the drawer
            isos_to_draw.append({
                "vertices": vertices, "faces": faces,
                "color": iso.get("color"), "opacity": iso.get("opacity"),
                "name": iso.get("name", "").replace("$isoval$", f"{isoval:.4f}")
            })

        return {"isosurfaces": isos_to_draw}

    def _add_shortcuts(self):

        axes = ["x", "y", "z"]

        for ax in axes:

            self.add_shortcut(f'{ax.lower()}+enter', f"Show {ax} axis", self.update_settings, axes=[ax])

            self.add_shortcut(f'{ax.lower()} {ax.lower()}', f"Duplicate {ax} axis", self.tile, 2, ax)

            self.add_shortcut(f'{ax.lower()}+-', f"Substract a unit cell along {ax}", self.tighten, 1, ax)

            self.add_shortcut(f'{ax.lower()}++', f"Add a unit cell along {ax}", self.tighten, -1, ax)

        for xaxis in axes:
            for yaxis in [ax for ax in axes if ax != xaxis]:
                self.add_shortcut(
                    f'{xaxis.lower()}+{yaxis.lower()}', f"Show {xaxis} and {yaxis} axes",
                    self.update_settings, axes=[xaxis, yaxis]
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

    def scan(self, along, start=None, stop=None, step=None, num=None, breakpoints=None, mode="moving_slice", animation_kwargs=None, **kwargs):
        """
        Returns an animation containing multiple frames scaning along an axis.

        Parameters
        -----------
        along: {"x", "y", "z"}
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

        axes = self.get_setting('axes')
        if mode == "as_is" and set(axes) - set(["x", "y", "z"]):
            raise ValueError("To perform a scan, the axes need to be cartesian. Please set the axes to a combination of 'x', 'y' and 'z'.")

        if self.grid.sc.is_cartesian():
            grid = self.grid
        else:
            transform_bc = kwargs.pop("transform_bc", self.get_setting("transform_bc"))
            grid, transform_offset = self._transform_grid_cell(
                self.grid, mode=transform_bc, output_shape=self.grid.shape, cval=np.nan
            )

            kwargs["offset"] = transform_offset + kwargs.get("offset", self.get_setting("offset"))

        # We get the key that needs to be animated (we will divide the full range in frames)
        range_key = f"{along}_range"
        along_i = {"x": 0, "y": 1, "z": 2}[along]

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

        if breakpoints[-1] == grid.cell[along_i, along_i]:
            breakpoints[-1] = grid.cell[along_i, along_i] - grid.dcell[along_i, along_i]

        if mode == "moving_slice":
            return self._moving_slice_scan(grid, along_i, breakpoints)
        elif mode == "as_is":
            return self._asis_scan(grid, range_key, breakpoints, animation_kwargs=animation_kwargs, **kwargs)

    def _asis_scan(self, grid, range_key, breakpoints, animation_kwargs=None, **kwargs):
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
            fixed={**{key: val for key, val in self.settings.items() if key != range_key}, **kwargs, "grid": grid},
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

    def _moving_slice_scan(self, grid, along_i, breakpoints):
        import plotly.graph_objs as go
        ax = along_i
        displayed_axes = [i for i in range(3) if i != ax]
        shape = np.array(grid.shape)[displayed_axes]
        cmin = np.min(grid.grid)
        cmax = np.max(grid.grid)
        x_ax, y_ax = displayed_axes
        x = np.linspace(0, grid.cell[x_ax, x_ax], grid.shape[x_ax])
        y = np.linspace(0, grid.cell[y_ax, y_ax], grid.shape[y_ax])

        fig = go.Figure(frames=[go.Frame(data=go.Surface(
            x=x, y=y,
            z=(bp * np.ones(shape)).T,
            surfacecolor=np.squeeze(grid.cross_section(grid.index(bp, ax), ax).grid).T,
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
                            zaxis=dict(autorange=True, title=ax_title(ax)),
                            aspectmode="data",
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

        # We need to add an invisible trace so that the z axis stays with the correct range
        fig.add_trace({"type": "scatter3d", "mode": "markers", "marker_size": 0.001, "x": [0, 0], "y": [0, 0], "z": [0, grid.cell[ax, ax]]})

        return fig


class WavefunctionPlot(GridPlot):
    """
    An extension of GridPlot specifically tailored for plotting wavefunctions

    Parameters
    -----------
    eigenstate: EigenstateElectron, optional
        The eigenstate that contains the coefficients of the wavefunction.
        Note that an eigenstate can contain coefficients for multiple states.
    wfsx_file: wfsxSileSiesta, optional
        Siesta WFSX file to directly read the coefficients from.
        If the root_fdf file is provided but the wfsx one isn't, we will try
        to find it             as SystemLabel.WFSX.
    geometry: Geometry, optional
        Necessary to generate the grid and to plot the wavefunctions, since
        the basis orbitals are needed.             If you provide a
        hamiltonian, the geometry is probably inside the hamiltonian, so you
        don't need to provide it.             However, this field is
        compulsory if you are providing the eigenstate directly.
    k: array-like, optional
        If the eigenstates need to be calculated from a hamiltonian, the k
        point for which you want them to be calculated
    spin:  optional
        The spin component where the eigenstate should be calculated.
        Only meaningful if the state needs to be calculated from the
        hamiltonian.
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
    grid_file: cubeSile or rhoSileSiesta or ldosSileSiesta or rhoinitSileSiesta or rhoxcSileSiesta or drhoSileSiesta or baderSileSiesta or iorhoSileSiesta or totalrhoSileSiesta or stsSileSiesta or stmldosSileSiesta or hartreeSileSiesta or neutralatomhartreeSileSiesta or totalhartreeSileSiesta or gridncSileSiesta or ncSileSiesta or fdfSileSiesta or tsvncSileSiesta or chgSileVASP or locpotSileVASP, optional
        A filename that can be return a Grid through `read_grid`.
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
    axes:  optional
        The axis along you want to see the grid, it will be reduced along the
        other ones, according to the the `reduce_method` setting.
    zsmooth:  optional
        Parameter that smoothens how data looks in a heatmap.
        'best' interpolates data, 'fast' interpolates pixels, 'False'
        displays the data as is.
    interp: array-like, optional
        Interpolation factors to make the grid finer on each axis.See the
        zsmooth setting for faster smoothing of 2D heatmap.
    transform_bc:  optional
        The boundary conditions when a cell transform is applied to the grid.
        Cell transforms are only             applied when the grid's cell
        doesn't follow the cartesian coordinates and the requested display is
        2D or 1D.
    nsc: array-like, optional
        Number of times the grid should be repeated
    offset: array-like, optional
        The offset of the grid along each axis. This is important if you are
        planning to match this grid with other geometry related plots.
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
    reduce_method:  optional
        The method used to reduce the dimensions that will not be displayed
        in the plot.
    isos: array-like of dict, optional
        The isovalues that you want to represent.             The way they
        will be represented is of course dependant on the type of
        representation:                 - 2D representations: A contour (i.e.
        a line)                 - 3D representations: A surface
        Each item is a dict.    Structure of the dict: {         'name': The
        name of the iso query. Note that you can use $isoval$ as a template
        to indicate where the isoval should go.         'val': The iso value.
        If not provided, it will be infered from `frac`         'frac': If
        val is not provided, this is used to calculate where the isosurface
        should be drawn.                     It calculates them from the
        minimum and maximum values of the grid like so:
        If iso_frac = 0.3:                     (min_value-----
        ISOVALUE(30%)-----------max_value)                     Therefore, it
        should be a number between 0 and 1.
        'step_size': The step size to use to calculate the isosurface in case
        it's a 3D representation                     A bigger step-size can
        speed up the process dramatically, specially the rendering part
        and the resolution may still be more than satisfactory (try to use
        step_size=2). For very big                     grids your computer
        may not even be able to render very fine surfaces, so it's worth
        keeping                     this setting in mind.         'color':
        The color of the surface/contour.         'opacity': Opacity of the
        surface/contour. Between 0 (transparent) and 1 (opaque). }
    plot_geom: bool, optional
        If True the geometry associated to the grid will also be plotted
    geom_kwargs: dict, optional
        Extra arguments that are passed to geom.plot() if plot_geom is set to
        True
    root_fdf: fdfSileSiesta, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    entry_points_order: array-like, optional
        Order with which entry points will be attempted.
    backend:  optional
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

        SileInput(key='wfsx_file', name='Path to WFSX file',
            dtype=sisl.io.siesta.wfsxSileSiesta,
            default=None,
            help="""Siesta WFSX file to directly read the coefficients from.
            If the root_fdf file is provided but the wfsx one isn't, we will try to find it
            as SystemLabel.WFSX.
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
        'axes': "xyz",
        'plot_geom': True
    }

    @entry_point('eigenstate', 0)
    def _read_nosource(self, eigenstate):
        """
        Uses an already calculated Eigenstate object to generate the wavefunctions.
        """
        if eigenstate is None:
            raise ValueError('No eigenstate was provided')

        self.eigenstate = eigenstate

    @entry_point('wfsx file', 1)
    def _read_from_WFSX_file(self, wfsx_file, k, spin, root_fdf):
        """Reads the wavefunction coefficients from a SIESTA WFSX file"""
        # Try to read the geometry
        fdf = self.get_sile(root_fdf or "root_fdf")
        if fdf is None:
            raise ValueError("The setting 'root_fdf' needs to point to an fdf file with a geometry")
        geometry = fdf.read_geometry(output=True)

        # Get the WFSX file. If not provided, it is inferred from the fdf.
        wfsx = self.get_sile(wfsx_file or "wfsx_file")
        if not wfsx.file.is_file():
            raise ValueError(f"File '{wfsx.file}' does not exist.")

        sizes = wfsx.read_sizes()
        H = sisl.Hamiltonian(geometry, dim=sizes.nspin)

        wfsx = sisl.get_sile(wfsx.file, parent=H)

        # Try to find the eigenstate that we need
        self.eigenstate = wfsx.read_eigenstate(k=k, spin=spin[0])
        if self.eigenstate is None:
            # We have not found it.
            raise ValueError(f"A state with k={k} was not found in file {wfsx.file}.")

    @entry_point('hamiltonian', 2)
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

    def _get_eigenstate(self, i):

        if "index" in self.eigenstate.info:
            wf_i = np.nonzero(self.eigenstate.info["index"] == i)[0]
            if len(wf_i) == 0:
                raise ValueError(f"Wavefunction with index {i} is not present in the eigenstate. Available indices: {self.eigenstate.info['index']}."
                    f"Entry point used: {self.source._name}")
            wf_i = wf_i[0]
        else:
            max_index = len(self.eigenstate)
            if i > max_index:
                raise ValueError(f"Wavefunction with index {i} is not present in the eigenstate. Available range: [0, {max_index}]."
                    f"Entry point used: {self.source._name}")
            wf_i = i

        return self.eigenstate[wf_i]

    def _set_data(self, i, geometry, grid, k, grid_prec, nsc):

        if geometry is not None:
            self.geometry = geometry
        elif isinstance(self.eigenstate.parent, sisl.Geometry):
            self.geometry = self.eigenstate.parent
        else:
            self.geometry = getattr(self.eigenstate.parent, "geometry", None)
        if self.geometry is None:
            raise ValueError('No geometry was provided and we need it the basis orbitals to build the wavefunctions from the coefficients!')

        # Get the spin class for which the eigenstate was calculated.
        spin = sisl.Spin()
        if self.eigenstate.parent is not None:
            spin = getattr(self.eigenstate.parent, "spin", None)

        # Check that number of orbitals match
        no = self.eigenstate.shape[1] * (1 if spin.is_diagonal else 2)
        if self.geometry.no != no:
            raise ValueError(f"Number of orbitals in the state ({no}) and the geometry ({self.geometry.no}) don't match."
                " This is most likely because the geometry doesn't contain the appropiate basis.")

        # Move all atoms inside the unit cell, otherwise the wavefunction is not
        # properly displayed.
        self.geometry = self.geometry.copy()
        self.geometry.xyz = (self.geometry.fxyz % 1).dot(self.geometry.cell)

        # If we are calculating the wavefunction for any point other than gamma,
        # the periodicity of the WF will be bigger than the cell. Therefore, if
        # the user wants to see more than the unit cell, we need to generate the
        # wavefunction for all the supercell. Here we intercept the `nsc` setting
        # with this objective.
        tiled_geometry = self.geometry
        nsc = list(nsc)
        for ax, sc_i in enumerate(nsc):
            if k[ax] != 0:
                tiled_geometry = tiled_geometry.tile(sc_i, ax)
                nsc[ax] = 1

        is_gamma = (np.array(k) == 0).all()
        if grid is None:
            dtype = np.float64 if is_gamma else np.complex128
            self.grid = sisl.Grid(grid_prec, geometry=tiled_geometry, dtype=dtype)
            grid = self.grid

        # GridPlot's after_read basically sets the x_range, y_range and z_range options
        # which need to know what the grid is, that's why we are calling it here
        super()._after_read()

        # Get the particular WF that we want from the eigenstate object
        wf_state = self._get_eigenstate(i)

        # Ensure we are dealing with the R gauge
        wf_state.change_gauge('R')

        # Finally, insert the wavefunction values into the grid.
        sisl.physics.electron.wavefunction(
            wf_state.state, grid, geometry=tiled_geometry,
            k=k, spinor=0, spin=spin
        )

        return super()._set_data(nsc=nsc, trace_name=f"WF {i} ({wf_state.eig[0]:.2f} eV)")

GridPlot.backends.register_child(WavefunctionPlot.backends)
