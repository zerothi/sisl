import inspect
import functools
import types

from ..workflow import Workflow
from ..canvas import Canvas

class Plot(Workflow):

    def __init_subclass__(cls):
        # If this is just a subclass of Plot that is not meant to be ran, continue 
        if not hasattr(cls, "_workflow"):
            return super().__init_subclass__()

        work_func = cls._workflow

        # Otherwise, wrap the workflow run to provide a canvas
        @functools.wraps(work_func)
        def work_func_with_canvas(**kwargs):
            backend = kwargs.pop("backend", "plotly")
            plot_actions = work_func(**kwargs)
            return Canvas(plot_actions=plot_actions, backend=backend)

        # Update the signature of the wrapper to account for the extra inputs
        # added.
        sig = inspect.signature(work_func_with_canvas)
        wrap_sig = sig.replace(
            parameters=(
                *sig.parameters.values(),
                inspect.Parameter("backend", kind=inspect.Parameter.KEYWORD_ONLY, default="plotly"),
            )
        )
        work_func_with_canvas.__signature__ = wrap_sig

        cls._workflow = work_func_with_canvas

        return super().__init_subclass__()

    @classmethod
    def from_plotter(cls, plotter):
        ...

    def __getattr__(self, key):
        return getattr(self._canvas, key)

    @property
    def _canvas(self):
        return self._final_node