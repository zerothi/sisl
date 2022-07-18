from abc import abstractmethod
import functools
import itertools
from sisl.messages import info
import numpy as np
from sisl.viz.nodes.canvas.canvas import CanvasNode
from xarray import Dataset, DataArray

from sisl.viz.nodes.processors.grid import get_isos

from ..node import Node, NodeMeta

class PlotterMeta(NodeMeta):
    def __getattr__(self, key):
        if hasattr(self._canvas_cls, key):
            def get_plotter(draw=True, **kwargs):
                return PlotterAction(method=key, draw=draw, **kwargs)
                
            return get_plotter
        else:
            raise AttributeError(f"{self} has no attribute {key}")
    
    def __dir__(self):
        return dir(self._canvas_cls)

class PlotterNode(Node, metaclass=PlotterMeta):
    """Base plotter class that all backends should inherit from.

    It contains all the plotting actions that should be supported by a canvas.

    This class contains various methods that need to be implemented by its subclasses.

    Methods that MUST be implemented are marked as abstract methods, therefore you won't
    even be able to use the class if you don't implement them. On the other hand, there are
    methods that are not absolutely essential to the general workings of the framework. 
    These are written in this class to raise a NotImplementedError. Therefore, the backend 
    will be instantiable but errors may happen during the plotting process.

    Below are all methods that need to be implemented by...

    (1) the generic backend of the framework:
        - `clear`, MUST
        - `draw_on`, optional (highly recommended, otherwise no multiple plot functionality)
        - `draw_line`, optional (highly recommended for 2D)
        - `draw_scatter`, optional (highly recommended for 2D)
        - `draw_line3D`, optional
        - `draw_scatter3D`, optional
        - `draw_arrows3D`, optional
        - `show`, optional

    (2) specific backend of a plot:
        - `draw`, MUST

    Also, you probably need to write an `__init__` method to initialize the state of the plot.
    Usually drawing methods will add to the state and finally on `show` you display the full
    plot.
    """

    _canvas_cls = CanvasNode

    def __init_subclass__(cls):

        def _get(self, *args, **kwargs):
            self._actions = []
            self.draw(*args, **kwargs)
            return self._actions

        cls._get = functools.wraps(cls.draw)(cls._get)

        return super().__init_subclass__()

    def _get(self, *args, **kwargs):
        self._actions = []
        self.draw(*args, **kwargs)
        return self._actions

    def __dir__(self):
        return dir(self._canvas_cls)
    
    def __getattr__(self, key):
        if not key[0] == "_" and hasattr(self._canvas_cls, key):
            def _action(*args, **kwargs):
                self._actions.append({
                    "method": key, "args": args, "kwargs": kwargs
                })
            
            return _action

        raise AttributeError(f"{self.__class__.__name__} has no attribute {key}, and the reference canvas ({self._canvas_cls.__name__}) does not have it either.")
    
    @abstractmethod
    def draw(self, *args, **kwargs) -> None:
        """Draw the plot. Should be implemented by the child classes."""
        raise NotImplementedError


class PlotterAction(PlotterNode):
    """A plotter that only contains a single action.

    This is useful if you simply want to draw a particular thing.
    """
    def draw(self, method, draw=True, **kwargs):
        if draw:
            getattr(self, method)(**kwargs)


class CompositePlotterNode(PlotterNode):
    
    def _get(self, *plotters, composite_method="multiple"):
        
        return {
            "composite_method": composite_method,
            "plot_actions": plotters,
        }


class PlotterXArray(PlotterNode):

    def _process_data(self, data, x=None, y=None, z=False, style={}):
        axes = {"x": x, "y": y}
        if z is not False:
            axes["z"] = z
        
        ndim = len(axes)

        # Normalize data to a Dataset
        if isinstance(data, DataArray):
            if np.all([ax is None for ax in axes.values()]):
                raise ValueError("You have to provide either x or y (or z if it is not False) (one needs to be the fixed variable).")
            axes = {k: v or data.name for k, v in axes.items()}
            data = data.to_dataset()
        else:
            if np.any([ax is None for ax in axes.values()]):
                raise ValueError("Since you provided a Dataset, you have to provide both x and y (and z if it is not False).")
        
        data_axis = None
        fixed_axes = {}
        # Check, for each axis, if it is uni dimensional (in which case we add it to the fixed axes dictionary)
        # or it contains more than one dimension, in which case we set it as the data axis
        for k in axes:
            if axes[k] in data.coords or (axes[k] in data and data[axes[k]].ndim == 1):
                if len(fixed_axes) < ndim - 1:
                    fixed_axes[k] = axes[k]
                else:
                    data_axis = k
            else:
                data_axis = k

        # Transpose the data so that the fixed axes are first.
        last_dims = []
        for ax_key, fixed_axis in fixed_axes.items():
            if fixed_axis not in data.dims:
                # This means that the fixed axis is a variable, which should contain only one dimension
                last_dim = data[fixed_axis].dims[-1]
            else:
                last_dim = fixed_axis
            last_dims.append(last_dim)
        last_dims = np.unique(last_dims)
        data = data.transpose(..., *last_dims)

        data_var = axes[data_axis]

        style_dims = set()
        for key, value in style.items():
            if value in data:
                style_dims = style_dims.union(set(data[value].dims))
            
        extra_style_dims = style_dims - set(data[data_var].dims)
        if extra_style_dims:
            data = data.stack(extra_style_dim=extra_style_dims).transpose('extra_style_dim', ...)

        if len(data[data_var].shape) == 1:
            data = data.expand_dims(dim={"fake_dim": [0]}, axis=0)
        # We have to flatten all the dimensions that will not be represented as an axis,
        # since we will just iterate over them.
        dims_to_stack = data[data_var].dims[:-len(last_dims)]
        data = data.stack(iterate_dim=dims_to_stack).transpose("iterate_dim", ...)

        styles = {}
        for key, value in style.items():
            if value in data:
                styles[key] = data[value]
            else:
                styles[key] = None

        plot_data = data[axes[data_axis]]

        fixed_coords = {}
        for ax_key, fixed_axis in fixed_axes.items():
            fixed_coord = data[fixed_axis] 
            if "iterate_dim" in fixed_coord.dims:
                # This is if fixed_coord was a variable of the dataset, which possibly has
                # gotten the extra iterate_dim added.
                fixed_coord = fixed_coord.isel(iterate_dim=0)
            fixed_coords[ax_key] = fixed_coord

        if self._debug_show_variables:
            info(f"{self} variables: \n\t- Fixed: {fixed_axes}\n\t- Data axis: {data_axis}\n\t")

        return plot_data, fixed_coords, styles, data_axis, axes


class PlotterNodeXY(PlotterXArray):

    _debug_show_variables = False

    def draw(self, data, x=None, y=None, z=False, color="color", width="width", opacity="opacity", colorscale=None, what="line", 
        set_axrange=False, set_axequal=False):
        """Draws

        Parameters
        ----------
        what: {"line", "scatter", "balls", "area_line"}
        """
        plot_data, fixed_coords, styles, data_axis, axes = self._process_data(
            data, x=x, y=y, z=z, style={"color": color, "width": width, "opacity": opacity}
        )

        self.draw_lines(data=plot_data, style=styles, fixed_coords=fixed_coords, data_axis=data_axis, colorscale=colorscale, what=what)

        if set_axequal:
            self.set_axes_equal()

        # Set axis range
        for key, coord_key in axes.items():
            ax = data[coord_key]
            title = ax.name
            units = ax.attrs.get("units")
            if units:
                title += f" [{units}]"

            axis = {"title": title}

            if set_axrange:
                axis["range"] = (float(ax.min()), float(ax.max()))
            
            axis.update(ax.attrs.get("axis", {}))

            self.set_axis(key, **axis)

    def draw_lines(self, data, style, fixed_coords, data_axis, colorscale, what):
        # Get the lines styles
        lines_style = {}
        extra_style_dims = False
        for key in ("color", "width", "opacity"):
            lines_style[key] = style.get(key)

            if lines_style[key] is not None:
                extra_style_dims = extra_style_dims or "extra_style_dim" in lines_style[key].dims
            # If some style is constant, just repeat it.
            if lines_style[key] is None or "iterate_dim" not in lines_style[key].dims:
                lines_style[key] = itertools.repeat(lines_style[key])

        # If we have to draw multicolored lines, we have to initialize a color axis and
        # use a special drawing function.
        line_kwargs = {}
        if isinstance(lines_style['color'], itertools.repeat):
            color_value = next(lines_style['color'])
        else:
            color_value = lines_style['color']

        if isinstance(color_value, DataArray) and (data.dims[-1] in color_value.dims):
            color = color_value
            if color.dtype in (int, float):
                self.init_coloraxis(color.name, color.values.min(), color.values.max(), colorscale)
                line_kwargs = {'coloraxis': color.name}
            drawing_function_name = f"draw_multicolor_{what}"
        else:
            drawing_function_name = f"draw_{what}"

        # Check if we have to use a 3D function
        if len(fixed_coords) == 2:
            self.init_3D()
            drawing_function_name += "_3D"
            
        _drawing_function = getattr(self, drawing_function_name)
        if what in ("scatter", "balls"):
            def drawing_function(*args, **kwargs):
                marker = kwargs.pop("line")
                marker['size'] = marker.pop("width")
                return _drawing_function(*args, marker=marker, **kwargs)
        else:
            drawing_function = _drawing_function

        # Define the iterator over lines, containing both values and styles
        iterator = zip(data,
            lines_style['color'], lines_style['width'], lines_style['opacity'],
        )

        fixed_coords_values = {k: arr.values for k, arr in fixed_coords.items()}

        # Now just iterate over each line and plot it.
        for values, *styles in iterator:
            names = values.iterate_dim.values[()]
            if len(names) == 1:
                name = str(names[0])
            else:
                name = str(names)

            parsed_styles = []
            for style in styles:
                if style is not None:
                    style = style.values
                    if style.ndim == 0:
                        style = style[()]
                parsed_styles.append(style)

            line_color, line_width, line_opacity = parsed_styles
            line_style = {"color": line_color, "width": line_width, "opacity": line_opacity}
            line = {**line_style, **line_kwargs}

            coords = {
                data_axis: values,
                **fixed_coords_values,
            }

            if not extra_style_dims:
                drawing_function(**coords, line=line, name=name)
            else:
                for k, v in line_style.items():
                    if v is None or v.ndim == 0:
                        line_style[k] = itertools.repeat(v)

                for l_color, l_width, l_opacity in zip(line_style['color'], line_style['width'], line_style['opacity']):
                    line_style = {"color": l_color, "width": l_width, "opacity": l_opacity}
                    drawing_function(**coords, line=line_style, name=name)


class PlotterNodeGrid(PlotterXArray):
    
    def draw(self, data, isos=[]):
        
        ndim = data.ndim

        if ndim == 2:
            transposed = data.transpose("y", "x")

            self.draw_heatmap(transposed.values, x=data.x, y=data.y, name="HEAT", zsmooth="best")

            dx = data.x[1] - data.x[0]
            dy = data.y[1] - data.y[0]

            iso_lines = get_isos(transposed, isos)
            for iso_line in iso_lines:
                iso_line['line'] = {
                    "color": iso_line.pop("color", None),
                    "opacity": iso_line.pop("opacity", None),
                    "width": iso_line.pop("width", None),
                    **iso_line.get("line", {})
                }
                self.draw_line(**iso_line)
        elif ndim == 3:
            isosurfaces = get_isos(data, isos)
            
            for isosurface in isosurfaces:
                self.draw_mesh_3D(**isosurface)
        
        
        self.set_axes_equal()