from .canvas import *
from .plotly import *
from .matplotlib import *

@CanvasNode.from_func
def Canvas(plot_actions=[], backend="plotly", **kwargs):
    """ Generic canvas that selects the specific canvas based on the backend argument."""

    if backend == "matplotlib":
        canvas_cls = MatplotlibCanvas
    elif backend == "plotly":
        canvas_cls = PlotlyCanvas
    else:
        raise ValueError(f"'backend' must be one of plotly or matplotlib, but was {backend}")

    return canvas_cls(plot_actions=plot_actions, **kwargs)

def _canvas_getattr(self, key):
    return getattr(self.get(), key)

Canvas.__getattr__ = _canvas_getattr