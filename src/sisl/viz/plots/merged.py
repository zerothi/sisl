from typing import Literal, Optional

from ..figure import Figure, get_figure
from ..plot import Plot
from ..plotters.plot_actions import combined


def merge_plots(*figures: Figure, 
    composite_method: Optional[Literal["multiple", "subplots", "multiple_x", "multiple_y", "animation"]] = "multiple", 
    backend: Literal["plotly", "matplotlib", "py3dmol", "blender"] = "plotly",
    **kwargs
) -> Figure:
    """Combines multiple plots into a single figure.

    Parameters
    ----------
    *figures : Figure
        The figures (or plots) to combine.
    composite_method : {"multiple", "subplots", "multiple_x", "multiple_y", "animation", None}, optional
        The method to use to combine the plots. None is the same as multiple.
    backend : {"plotly", "matplotlib", "py3dmol", "blender"}, optional
        The backend to use for the merged figure.
    **kwargs
        Additional arguments that will be passed to the `_init_figure_*` method of the Figure class.
        The arguments accepted here are basically backend specific, but for subplots all backends should
        support `rows` and `cols` to specify the number of rows and columns of the subplots, and `arrange`
        which controls the arrangement ("rows", "cols" or "square").
    """

    plot_actions = combined(
        *[fig.plot_actions for fig in figures],
        composite_method=composite_method,
        **kwargs
    )

    return get_figure(plot_actions=plot_actions, backend=backend)
