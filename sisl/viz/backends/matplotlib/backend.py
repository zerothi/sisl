import itertools
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from ..templates.backend import Backend, MultiplePlotBackend, SubPlotsBackend
from ...plot import Plot, SubPlots, MultiplePlot

class MatplotlibBackend(Backend):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.figure, self.axes = plt.subplots()
        self._init_axes()

    def draw_on(self, axes, axes_indices=None):
        """Draws this plot in a different figure.

        Parameters
        -----------
        axes: Plot, PlotlyBackend or matplotlib.axes.Axes
            The axes to draw this plot in.
        """
        if isinstance(axes, Plot):
            axes = axes._backend.axes
        elif isinstance(axes, MatplotlibBackend):
            axes = axes.axes

        if axes_indices is not None:
            axes = axes[axes_indices]

        if not isinstance(axes, Axes):
            raise TypeError(f"{self.__class__.__name__} was provided a {axes.__class__.__name__} to draw on.")
        
        self_axes = self.axes
        self.axes = axes
        self._init_axes()
        self._plot.get_figure(backend=self._backend_name, clear_fig=False)
        self.axes = self_axes
    
    def _init_axes(self):
        self.axes.update(self._axes_defaults)

    def __getattr__(self, key):
        if key != "figure":
            return getattr(self.figure, key)
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

    def show(self):
        return self.figure.show()

    # Methods for testing
    def _test_number_of_items_drawn(self):
        return len(self.axes.lines + self.axes.collections)
    
    def draw_line(self, x, y, name, line, marker={}, text=None, **kwargs):
        return self.axes.plot(x, y, color=line.get("color"), linewidth=line.get("width", 1), markersize=marker.get("size"), label=name)

    def draw_scatter(self, x, y, name, marker={}, text=None, **kwargs):
        return self.axes.scatter(x, y, c=marker.get("color"), s=marker.get("size", 1), cmap=marker.get("colorscale"), label=name, **kwargs)

class MatplotlibMultiplePlotBackend(MatplotlibBackend, MultiplePlotBackend):
    pass

class MatplotlibSubPlotsBackend(MatplotlibMultiplePlotBackend, SubPlotsBackend):

    def draw(self, backend_info):
        childs = backend_info["child_plots"]
        rows, cols = backend_info["rows"], backend_info["cols"]

        self.figure, self.axes = plt.subplots(rows, cols)

        # Normalize the axes array to have two dimensions
        if rows == 1 and cols == 1:
            self.axes = np.array([[self.axes]])
        elif rows == 1:
            self.axes = np.expand_dims(self.axes, axis=0)
        elif cols == 1:
            self.axes = np.expand_dims(self.axes, axis=1)
            
        indices = itertools.product(range(rows), range(cols))
        # Start assigning each plot to a position of the layout
        for (row, col) , child in zip(indices, childs):
            self.draw_other_plot(child, axes_indices=(row, col))

MultiplePlot.backends.register("matplotlib", MatplotlibMultiplePlotBackend)
SubPlots.backends.register("matplotlib", MatplotlibSubPlotsBackend)