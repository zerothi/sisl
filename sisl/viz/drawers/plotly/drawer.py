from .._plot_drawers import Drawer
import plotly
import plotly.graph_objects as go

class PlotlyDrawer(Drawer):

    _layout_defaults = {}

    def __init__(self):
        self.figure = go.Figure()
        self.update_layout(**self._layout_defaults)

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
        self.figure.data = []

        if frames:
            self.figure.frames = []

        if layout:
            self.figure.layout = {}

        return self

    def get_ipywidget(self):
        # Update the title of the plot if there is none
        if not self.figure.layout["title"]:
            self.update_layout(title = '{} {}'.format(getattr(self, "struct", ""), self.plot_name()))
        return go.FigureWidget(self.figure, )

