import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.pyplot import Normalize
import numpy as np


from sisl.messages import warn
from .canvas import CanvasNode

class MatplotlibCanvas(CanvasNode):
    """Generic backend for the matplotlib framework.

    On initialization, `matplotlib.pyplot.subplots` is called and the figure and and
    axes obtained are stored under `self.figure` and `self.axeses`, respectively.
    If an attribute is not found on the backend, it is looked for
    in the axes.

    On initialization, we also take the class attribute `_axes_defaults` (a dictionary) 
    and run `self.axes.update` with those parameters. Therefore this parameter can be used
    to provide default parameters for the axes.
    """

    _axes_defaults = {}

    def _init_figure(self, *args, **kwargs):
        self.figure, self.axes = self._init_plt_figure()
        self._init_axes()

    # def draw_on(self, axes, axes_indices=None):
    #     """Draws this plot in a different figure.

    #     Parameters
    #     -----------
    #     axes: Plot, PlotlyBackend or matplotlib.axes.Axes
    #         The axes to draw this plot in.
    #     """
    #     if isinstance(axes, Plot):
    #         axes = axes._backend.axes
    #     elif isinstance(axes, MatplotlibBackend):
    #         axes = axes.axes

    #     if axes_indices is not None:
    #         axes = axes[axes_indices]

    #     if not isinstance(axes, Axes):
    #         raise TypeError(f"{self.__class__.__name__} was provided a {axes.__class__.__name__} to draw on.")

    #     self_axes = self.axes
    #     self.axes = axes
    #     self._init_axes()
    #     self._plot.get_figure(backend=self._backend_name, clear_fig=False)
    #     self.axes = self_axes

    def _init_plt_figure(self):
        """Initializes the matplotlib figure and axes

        Returns
        --------
        Figure:
            the matplotlib figure of this plot.
        Axes:
            the matplotlib axes of this plot.
        """
        return plt.subplots()

    def _init_axes(self):
        """Does some initial modification on the axes."""
        self.axes.update(self._axes_defaults)

    def __getattr__(self, key):
        if key != "axes":
            return getattr(self.axes, key)
        raise AttributeError(key)

    def clear(self, layout=False):
        """ Clears the plot canvas so that data can be reset

        Parameters
        --------
        layout: boolean, optional
            whether layout should also be deleted
        """
        if layout:
            self.axes.clear()

        for artist in self.axes.lines + self.axes.collections:
            artist.remove()

        return self

    def get_ipywidget(self):
        return self.figure

    def show(self, *args, **kwargs):
        return self.figure.show(*args, **kwargs)

    def draw_line(self, x, y, name=None, line={}, marker={}, text=None, **kwargs):
        marker_format = marker.get("symbol", "o") if marker else None
        marker_color = marker.get("color")
        return self.axes.plot(
            x, y, color=line.get("color"), linewidth=line.get("width", 1), 
            marker=marker_format, markersize=marker.get("size"), markerfacecolor=marker_color, markeredgecolor=marker_color,
            label=name
        )

    def draw_multicolor_line(self, x, y, name=None, line={}, marker={}, text=None, **kwargs):
        # This is heavily based on
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc_kwargs = {}
        coloraxis = line.get("coloraxis")
        if coloraxis is not None:
            coloraxis = self._coloraxes.get(coloraxis)
            lc_kwargs["cmap"] = coloraxis.get("colorscale")
            if coloraxis.get("cmin") is not None:
                lc_kwargs["norm"] = Normalize(coloraxis['cmin'], coloraxis['cmax'])

        lc = LineCollection(segments, **lc_kwargs)

        # Set the values used for colormapping
        lc.set_array(line.get("color"))
        lc.set_linewidth(line.get("width", 1))

        self.axes.add_collection(lc)

        #self._colorbar = self.axes.add_collection(lc)

    def draw_area_line(self, x, y, line={}, name=None, **kwargs):
        spacing = line.get('width', 1) / 2
        
        self.axes.fill_between(
            x, y + spacing, y - spacing,
            color=line.get('color'), label=name
        )

    def draw_scatter(self, x, y, name=None, marker={}, text=None, zorder=2, **kwargs):
        try:
            return self.axes.scatter(x, y, c=marker.get("color"), s=marker.get("size", 1), cmap=marker.get("colorscale"), alpha=marker.get("opacity"), label=name, zorder=zorder, **kwargs)
        except TypeError as e:
            if str(e) == "alpha must be a float or None":
                warn(f"Your matplotlib version doesn't support multiple opacity values, please upgrade to >=3.4 if you want to use opacity.")
                return self.axes.scatter(x, y, c=marker.get("color"), s=marker.get("size", 1), cmap=marker.get("colorscale"), label=name, zorder=zorder, **kwargs)
            else:
                raise e
    
    def draw_multicolor_scatter(self, *args, **kwargs):
        marker = {**kwargs.pop("marker",{})}
        coloraxis = marker.get("coloraxis")
        if coloraxis is not None:
            coloraxis = self._coloraxes.get(coloraxis)
            marker["colorscale"] = coloraxis.get("colorscale")
        return super().draw_multicolor_scatter(*args, marker=marker, **kwargs)
    
    def draw_heatmap(self, values, x=None, y=None, name=None, zsmooth=False, coloraxis=None):

        extent = None
        if x is not None and y is not None:
            extent = [x[0], x[-1], y[0], y[-1]]

        self.axes.imshow(
            values, 
            #cmap=backend_info["colorscale"], vmin=backend_info["cmin"], vmax=backend_info["cmax"],
            label=name, extent=extent,
            origin="lower"
        )
    
    def set_axis(self, axis, range=None, title="", tickvals=None, ticktext=None, showgrid=False, **kwargs):
        if range is not None:
            updater = getattr(self.axes, f'set_{axis}lim')
            updater(*range)

        if title:
            updater = getattr(self.axes, f'set_{axis}label')
            updater(title)
        
        if tickvals is not None:
            updater = getattr(self.axes, f'set_{axis}ticks')
            updater(ticks=tickvals, labels=ticktext)

        self.axes.grid(visible=showgrid, axis=axis)
    
    def set_axes_equal(self):
        self.axes.axis("equal")