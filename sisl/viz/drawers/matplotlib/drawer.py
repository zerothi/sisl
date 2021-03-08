import matplotlib.pyplot as plt

from .._plot_drawers import Drawer

class MatplotlibDrawer(Drawer):
    
    _ax_defaults = {}

    def __init__(self):
        self.figure, self.ax = plt.subplots()
        self.ax.update(self._ax_defaults)

    def __getattr__(self, key):
        if key != "figure":
            return getattr(self.figure, key)
        raise AttributeError(key)

    def clear(self, frames=True, layout=False):
        """ Clears the plot canvas so that data can be reset

        Parameters
        --------
        frames: boolean, optional
            whether frames should also be deleted
        layout: boolean, optional
            whether layout should also be deleted
        """
        return self
        self.figure.data = []

        if frames:
            self.figure.frames = []

        if layout:
            self.figure.layout = {}

        return self

    # def get_ipywidget(self):
    #     return go.FigureWidget(self.figure, )