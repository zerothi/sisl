import itertools
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .._plot_drawers import Drawer
from ...plot import SubPlots, MultiplePlot, Animation

class MatplotlibDrawer(Drawer):
    
    _ax_defaults = {}

    def __init__(self):
        self.figure, self.ax = plt.subplots()
        self._init_ax()
    
    def _init_ax(self):
        self.ax.update(self._ax_defaults)

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
            self.ax.clear()

        for artist in self.ax.lines + self.ax.collections:
            artist.remove()

        return self

    def show(self):
        return self.figure.show()

class MatplotlibMultiplePlotDrawer(MatplotlibDrawer):

    def draw(self, drawer_info, childs):

        # Start assigning each plot to a position of the layout
        for child in childs:
            self._draw_child_in_ax(child, self.ax)
            
    def _draw_child_in_ax(self, child, ax):
        child_ax = child._drawer.ax
        child._drawer.ax = ax
        child._init_ax()
        child.get_figure(clear_fig=False)
        child._drawer.ax = child_ax

class MatplotlibSubplotsDrawer(MatplotlibMultiplePlotDrawer):

    def draw_subplots(self, drawer_info, rows, cols, childs, **make_subplots_kwargs):

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
            self._draw_child_in_ax(child, self.axes[row, col])

MultiplePlot._drawers.register("matplotlib", MatplotlibMultiplePlotDrawer)
SubPlots._drawers.register("matplotlib", MatplotlibSubplotsDrawer)