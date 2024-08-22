# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Literal, Optional

from ..figure import Figure, get_figure
from ..plotters.plot_actions import combined


def merge_plots(
    *figures: Figure,
    composite_method: Optional[
        Literal["multiple", "subplots", "multiple_x", "multiple_y", "animation"]
    ] = "multiple",
    backend: Literal["plotly", "matplotlib", "py3dmol", "blender"] = "plotly",
    **kwargs,
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
        **kwargs,
    )

    return get_figure(plot_actions=plot_actions, backend=backend)


def subplots(
    *figures: Figure,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    arrange: Literal["rows", "cols", "square"] = "rows",
    backend: Literal["plotly", "matplotlib", "py3dmol", "blender"] = "plotly",
    **kwargs,
) -> Figure:
    """Combines multiple plots into a single figure using subplots.

    Parameters
    ----------
    *figures : Figure
        The figures (or plots) to combine.
    rows : int, optional
        The number of rows in the subplots grid.
        If not specified, it will be inferred automatically based on the number of columns and the number of plots.
        If neither `rows` nor `cols` are specified,
        the `arrange` parameter will be used to determine the number of rows and columns.
    cols : int, optional
        The number of columns in the subplots grid.
        If not specified, it will be inferred automatically based on the number of rows and the number of plots.
        If neither `rows` nor `cols` are specified,
        the `arrange` parameter will be used to determine the number of rows and columns.
    arrange : {"rows", "cols", "square"}, optional
        Determines number of rows and columns if neither ``rows`` nor ``cols`` are specified.
        If ``arrange`` is "rows", the number of rows will be equal to the number of plots.
        If ``arrange`` is "cols", the number of columns will be equal to the number of plots.
        If ``arrange`` is "square", the number of rows and columns will be equal to the square root of the number of plots,
    backend : {"plotly", "matplotlib", "py3dmol", "blender"}, optional
        The backend to use for the merged figure.
    **kwargs
        Additional arguments that will be passed to the `_init_figure_*` method of the Figure class.
        The arguments accepted here are basically backend specific, but for subplots all backends should
        support `rows` and `cols` to specify the number of rows and columns of the subplots, and `arrange`
        which controls the arrangement ("rows", "cols" or "square").

    See Also
    --------
    merge_plots : The function that is called to merge the plots.
    """

    return merge_plots(
        *figures,
        composite_method="subplots",
        backend=backend,
        arrange=arrange,
        rows=rows,
        cols=cols,
        **kwargs,
    )


def animation(
    *figures: Figure,
    frame_duration: float = 500,
    interpolated_frames: int = 5,
    backend: Literal["plotly", "matplotlib", "py3dmol", "blender"] = "plotly",
    **kwargs,
) -> Figure:
    """Combines multiple plots into a single figure using an animation.

    Parameters
    ----------
    *figures : Figure
        The figures (or plots) to combine.
    frame_duration : float, optional
        Number of milliseconds each frame should be displayed.
    interpolated_frames : int, optional
        Number of interpolated frames to add between each frame.
        This only works with the blender backend.
    backend : {"plotly", "matplotlib", "py3dmol", "blender"}, optional
        The backend to use for the merged figure.
    **kwargs
        Additional arguments that will be passed to the `_init_figure_*` method of the Figure class.
        The arguments accepted here are basically backend specific, but for subplots all backends should
        support `rows` and `cols` to specify the number of rows and columns of the subplots, and `arrange`
        which controls the arrangement ("rows", "cols" or "square").

    See Also
    --------
    merge_plots : The function that is called to merge the plots.
    """

    return merge_plots(
        *figures,
        frame_duration=frame_duration,
        interpolated_frames=interpolated_frames,
        composite_method="animation",
        backend=backend,
        **kwargs,
    )
