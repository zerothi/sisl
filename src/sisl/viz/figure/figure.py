# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections import ChainMap
from typing import Any, Literal, Optional

import numpy as np

from sisl.messages import warn
from sisl.viz.plotutils import values_to_colors

BACKENDS = {}


class Figure:
    """Base figure class that all backends should inherit from.

    It contains all the plotting actions that should be supported by a figure.

    A subclass for a specific backend should implement as many methods as possible
    from the ones where Figure returns NotImplementedError.
    Other methods are optional because Figure contains a default implementation
    using other methods, which should work for most backends.

    To create a new backend, one might take the PlotlyFigure as a template.
    """

    _coloraxes: dict = {}
    _multi_axes: dict = {}

    _rows: Optional[int] = None
    _cols: Optional[int] = None

    # The composite mode of the plot
    _composite_mode = 0
    # Here are the different composite methods that can be used.
    _NONE = 0
    _SAME_AXES = 1
    _MULTIAXIS = 2
    _SUBPLOTS = 3
    _ANIMATION = 4

    plot_actions: list = []

    def __init__(self, plot_actions, *args, **kwargs):
        self.plot_actions = plot_actions
        self._build(plot_actions, *args, **kwargs)

    def _build(self, plot_actions, *args, **kwargs):
        plot_actions = self._sanitize_plot_actions(plot_actions)

        self._coloraxes = {}
        self._multi_axes = {}

        fig = self.init_figure(
            composite_method=plot_actions["composite_method"],
            plot_actions=plot_actions["plot_actions"],
            init_kwargs=plot_actions["init_kwargs"],
        )

        for section_actions in self._composite_iter(
            self._composite_mode, plot_actions["plot_actions"]
        ):
            for action in section_actions:
                getattr(self, action["method"])(
                    *action.get("args", ()), **action.get("kwargs", {})
                )

        return fig

    @classmethod
    def fig_has_attr(cls, key: str) -> bool:
        """Whether the figure that this class generates has a given attribute.

        Parameters
        -----------
        key
            the attribute to check for.
        """
        return False

    @staticmethod
    def _sanitize_plot_actions(plot_actions):
        def _flatten(plot_actions, out, level=0, root_i=0):
            for i, section_actions in enumerate(plot_actions):
                if level == 0:
                    out.append([])
                    root_i = i

                if isinstance(section_actions, dict):
                    _flatten(
                        section_actions["plot_actions"], out, level + 1, root_i=root_i
                    )
                else:
                    # If it's a plot object, we need to extract the plot_actions
                    out[root_i].extend(section_actions)

        if isinstance(plot_actions, dict):
            composite_method = plot_actions.get("composite_method")
            init_kwargs = plot_actions.get("init_kwargs", {})
            out = []
            _flatten(plot_actions["plot_actions"], out)
            plot_actions = out
        else:
            composite_method = None
            plot_actions = [plot_actions]
            init_kwargs = {}

        return {
            "composite_method": composite_method,
            "plot_actions": plot_actions,
            "init_kwargs": init_kwargs,
        }

    def init_figure(
        self,
        composite_method: Literal[
            None,
            "same_axes",
            "multiple",
            "multiple_x",
            "multiple_y",
            "subplots",
            "animation",
        ] = None,
        plot_actions=(),
        init_kwargs: dict[str, Any] = {},
    ):
        if composite_method is None:
            self._composite_mode = self._NONE
            return self._init_figure(**init_kwargs)
        elif composite_method == "same_axes":
            self._composite_mode = self._SAME_AXES
            return self._init_figure_same_axes(**init_kwargs)
        elif composite_method.startswith("multiple"):
            # This could be multiple
            self._composite_mode = self._MULTIAXIS
            multi_axes = [ax for ax in "xy" if ax in composite_method[8:]]
            return self._init_figure_multiple_axes(
                multi_axes, plot_actions, **init_kwargs
            )
        elif composite_method == "animation":
            self._composite_mode = self._ANIMATION
            return self._init_figure_animated(n=len(plot_actions), **init_kwargs)
        elif composite_method == "subplots":
            self._composite_mode = self._SUBPLOTS
            self._rows, self._cols = self._subplots_rows_and_cols(
                len(plot_actions),
                rows=init_kwargs.get("rows"),
                cols=init_kwargs.get("cols"),
                arrange=init_kwargs.pop("arrange", "rows"),
            )
            init_kwargs = ChainMap(
                {"rows": self._rows, "cols": self._cols}, init_kwargs
            )
            return self._init_figure_subplots(**init_kwargs)
        else:
            raise ValueError(f"Unknown composite method '{composite_method}'")

    def _init_figure(self, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a _init_figure method."
        )

    def _init_figure_same_axes(self, *args, **kwargs):
        return self._init_figure(*args, **kwargs)

    def _init_figure_multiple_axes(self, multi_axes, plot_actions, **kwargs):
        figure = self._init_figure()

        if len(multi_axes) > 2:
            raise ValueError(
                f"{self.__class__.__name__} doesn't support more than one multiple axes."
            )

        for axis in multi_axes:
            self._multi_axes[axis] = self._init_multiaxis(axis, len(plot_actions))

        return figure

    def _init_multiaxis(self, axis, n):
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a _init_multiaxis method."
        )

    def _init_figure_animated(self, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a _init_figure_animated method."
        )

    def _init_figure_subplots(self, rows, cols, **kwargs):
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a _init_figure_subplots method."
        )

    def _subplots_rows_and_cols(
        self,
        n: int,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        arrange: Literal["rows", "cols", "square"] = "rows",
    ) -> tuple[int, int]:
        """Returns the number of rows and columns for a subplot grid."""
        if rows is None and cols is None:
            if arrange == "rows":
                rows = n
                cols = 1
            elif arrange == "cols":
                cols = n
                rows = 1
            elif arrange == "square":
                cols = n**0.5
                rows = n**0.5
                # we will correct so it *fits*, always have more columns
                rows, cols = int(rows), int(cols)
                cols = n // rows + min(1, n % rows)
        elif rows is None and cols is not None:
            # ensure it is large enough by adding 1 if they don't add up
            rows = n // cols + min(1, n % cols)
        elif cols is None and rows is not None:
            # ensure it is large enough by adding 1 if they don't add up
            cols = n // rows + min(1, n % rows)

        rows, cols = int(rows), int(cols)

        if cols * rows < n:
            warn(
                f"requested {n} subplots on a {rows}x{cols} grid layout. {n - cols*rows} plots will be missing."
            )

        return rows, cols

    def _composite_iter(self, mode, plot_actions):
        if mode == self._NONE:
            return plot_actions
        elif mode == self._SAME_AXES:
            return self._iter_same_axes(plot_actions)
        elif mode == self._MULTIAXIS:
            return self._iter_multiaxis(plot_actions)
        elif mode == self._SUBPLOTS:
            return self._iter_subplots(plot_actions)
        elif mode == self._ANIMATION:
            return self._iter_animation(plot_actions)
        else:
            raise ValueError(f"Unknown composite mode '{mode}'")

    def _iter_same_axes(self, plot_actions):
        return plot_actions

    def _iter_multiaxis(self, plot_actions):
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a _iter_multiaxis method."
        )

    def _iter_subplots(self, plot_actions):
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a _iter_subplots method."
        )

    def _iter_animation(self, plot_actions):
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a _iter_animation method."
        )

    def clear(self):
        """Clears the figure so that we can draw again."""
        pass

    def show(self):
        pass

    def init_3D(self):
        """Called if functions that draw in 3D are going to be called."""
        return

    def init_coloraxis(
        self,
        name,
        cmin=None,
        cmax=None,
        cmid=None,
        colorscale=None,
        showscale=True,
        **kwargs,
    ):
        """Initializes a color axis to be used by the drawing functions"""
        self._coloraxes[name] = {
            "cmin": cmin,
            "cmax": cmax,
            "cmid": cmid,
            "colorscale": colorscale,
            "showscale": showscale,
            **kwargs,
        }

    def draw_line(
        self,
        x,
        y,
        name=None,
        line={},
        marker={},
        text=None,
        row=None,
        col=None,
        **kwargs,
    ):
        """Draws a line satisfying the specifications

        Parameters
        -----------
        x: array-like
            the coordinates of the points along the X axis.
        y: array-like
            the coordinates of the points along the Y axis.
        name: str, optional
            the name of the line
        line: dict, optional
            specifications for the line style, following plotly standards. The backend
            should at least be able to implement `line["color"]` and `line["width"]`
        marker: dict, optional
            specifications for the markers style, following plotly standards. The backend
            should at least be able to implement `marker["color"]` and `marker["size"]`
        text: str, optional
            contains the text asigned to each marker. On plotly this is seen on hover,
            other options could be annotating. However, it is not necessary that this
            argument is supported.
        row: int, optional
            If the figure contains subplots, the row where to draw.
        col: int, optional
            If the figure contains subplots, the column where to draw.
        **kwargs:
            should allow other keyword arguments to be passed directly to the creation of
            the line. This will of course be framework specific
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a draw_line method."
        )

    def draw_multicolor_line(self, *args, line={}, row=None, col=None, **kwargs):
        """By default, multicoloured lines are drawn simply by drawing scatter points."""
        marker = {
            **kwargs.pop("marker", {}),
            "color": line.get("color"),
            "size": line.get("width"),
            "opacity": line.get("opacity"),
            "coloraxis": line.get("coloraxis"),
        }
        self.draw_multicolor_scatter(*args, marker=marker, row=row, col=col, **kwargs)

    def draw_multisize_line(self, *args, line={}, row=None, col=None, **kwargs):
        """By default, multisized lines are drawn simple by drawing scatter points."""
        marker = {
            **kwargs.pop("marker", {}),
            "color": line.get("color"),
            "size": line.get("width"),
            "opacity": line.get("opacity"),
            "coloraxis": line.get("coloraxis"),
        }
        self.draw_multisize_scatter(*args, marker=marker, row=row, col=col, **kwargs)

    def draw_area_line(
        self,
        x,
        y,
        name=None,
        line={},
        text=None,
        dependent_axis=None,
        row=None,
        col=None,
        **kwargs,
    ):
        """Same as draw line, but to draw a line with an area. This is for example used to draw fatbands.

        Parameters
        -----------
        x: array-like
            the coordinates of the points along the X axis.
        y: array-like
            the coordinates of the points along the Y axis.
        name: str, optional
            the name of the scatter
        line: dict, optional
            specifications for the line style, following plotly standards. The backend
            should at least be able to implement `line["color"]` and `line["width"]`, but
            it is very advisable that it supports also `line["opacity"]`.
        text: str, optional
            contains the text asigned to each marker. On plotly this is seen on hover,
            other options could be annotating. However, it is not necessary that this
            argument is supported.
        dependent_axis: str, optional
            The axis that contains the dependent variable. This is important because
            the area is drawn in parallel to that axis.
        row: int, optional
            If the figure contains subplots, the row where to draw.
        col: int, optional
            If the figure contains subplots, the column where to draw.
        **kwargs:
            should allow other keyword arguments to be passed directly to the creation of
            the scatter. This will of course be framework specific
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a draw_area_line method."
        )

    def draw_multicolor_area_line(
        self,
        x,
        y,
        name=None,
        line={},
        text=None,
        dependent_axis=None,
        row=None,
        col=None,
        **kwargs,
    ):
        """Draw a line with an area with multiple colours.

        Parameters
        -----------
        x: array-like
            the coordinates of the points along the X axis.
        y: array-like
            the coordinates of the points along the Y axis.
        name: str, optional
            the name of the scatter
        line: dict, optional
            specifications for the line style, following plotly standards. The backend
            should at least be able to implement `line["color"]` and `line["width"]`, but
            it is very advisable that it supports also `line["opacity"]`.
        text: str, optional
            contains the text asigned to each marker. On plotly this is seen on hover,
            other options could be annotating. However, it is not necessary that this
            argument is supported.
        dependent_axis: str, optional
            The axis that contains the dependent variable. This is important because
            the area is drawn in parallel to that axis.
        row: int, optional
            If the figure contains subplots, the row where to draw.
        col: int, optional
            If the figure contains subplots, the column where to draw.
        **kwargs:
            should allow other keyword arguments to be passed directly to the creation of
            the scatter. This will of course be framework specific
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a draw_multicolor_area_line method."
        )

    def draw_multisize_area_line(
        self,
        x,
        y,
        name=None,
        line={},
        text=None,
        dependent_axis=None,
        row=None,
        col=None,
        **kwargs,
    ):
        """Draw a line with an area with multiple colours.

        This is already usually supported by the normal draw_area_line.

        Parameters
        -----------
        x: array-like
            the coordinates of the points along the X axis.
        y: array-like
            the coordinates of the points along the Y axis.
        name: str, optional
            the name of the scatter
        line: dict, optional
            specifications for the line style, following plotly standards. The backend
            should at least be able to implement `line["color"]` and `line["width"]`, but
            it is very advisable that it supports also `line["opacity"]`.
        text: str, optional
            contains the text asigned to each marker. On plotly this is seen on hover,
            other options could be annotating. However, it is not necessary that this
            argument is supported.
        dependent_axis: str, optional
            The axis that contains the dependent variable. This is important because
            the area is drawn in parallel to that axis.
        row: int, optional
            If the figure contains subplots, the row where to draw.
        col: int, optional
            If the figure contains subplots, the column where to draw.
        **kwargs:
            should allow other keyword arguments to be passed directly to the creation of
            the scatter. This will of course be framework specific
        """
        # Usually, multisized area lines are already supported.
        return self.draw_area_line(
            x,
            y,
            name=name,
            line=line,
            text=text,
            dependent_axis=dependent_axis,
            row=row,
            col=col,
            **kwargs,
        )

    def draw_scatter(
        self, x, y, name=None, marker={}, text=None, row=None, col=None, **kwargs
    ):
        """Draws a scatter satisfying the specifications

        Parameters
        -----------
        x: array-like
            the coordinates of the points along the X axis.
        y: array-like
            the coordinates of the points along the Y axis.
        name: str, optional
            the name of the scatter
        marker: dict, optional
            specifications for the markers style, following plotly standards. The backend
            should at least be able to implement `marker["color"]` and `marker["size"]`, but
            it is very advisable that it supports also `marker["opacity"]` and `marker["colorscale"]`
        text: str, optional
            contains the text asigned to each marker. On plotly this is seen on hover,
            other options could be annotating. However, it is not necessary that this
            argument is supported.
        row: int, optional
            If the figure contains subplots, the row where to draw.
        col: int, optional
            If the figure contains subplots, the column where to draw.
        **kwargs:
            should allow other keyword arguments to be passed directly to the creation of
            the scatter. This will of course be framework specific
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a draw_scatter method."
        )

    def draw_multicolor_scatter(self, *args, **kwargs):
        """Draws a multicoloured scatter.

        Usually the normal scatter can already support this.
        """
        # Usually, multicoloured scatter plots are already supported.
        return self.draw_scatter(*args, **kwargs)

    def draw_multisize_scatter(self, *args, **kwargs):
        """Draws a multisized scatter.

        Usually the normal scatter can already support this.
        """
        # Usually, multisized scatter plots are already supported.
        return self.draw_scatter(*args, **kwargs)

    def draw_arrows(
        self,
        x,
        y,
        dxy,
        arrowhead_scale=0.2,
        arrowhead_angle=20,
        scale: float = 1,
        annotate: bool = False,
        row=None,
        col=None,
        **kwargs,
    ):
        """Draws multiple arrows using the generic draw_line method.

        Parameters
        -----------
        xy: np.ndarray of shape (n_arrows, 2)
            the positions where the atoms start.
        dxy: np.ndarray of shape (n_arrows, 2)
            the arrow vector.
        arrow_head_scale: float, optional
            how big is the arrow head in comparison to the arrow vector.
        arrowhead_angle: angle
            the angle that the arrow head forms with the direction of the arrow (in degrees).
        scale: float, optional
            multiplying factor to display the arrows. It does not affect the underlying data,
            therefore if the data is somehow displayed it should be without the scale factor.
        annotate:
            whether to annotate the arrows with the vector they represent.
        row: int, optional
            If the figure contains subplots, the row where to draw.
        col: int, optional
            If the figure contains subplots, the column where to draw.
        """
        # Make sure we are treating with numpy arrays
        xy = np.array([x, y]).T
        dxy = np.array(dxy) * scale

        # Get the destination of the arrows
        final_xy = xy + dxy

        # Convert from degrees to radians.
        arrowhead_angle = np.radians(arrowhead_angle)

        # Get the rotation matrices to get the tips of the arrowheads
        rot_matrix = np.array(
            [
                [np.cos(arrowhead_angle), -np.sin(arrowhead_angle)],
                [np.sin(arrowhead_angle), np.cos(arrowhead_angle)],
            ]
        )
        inv_rot = np.linalg.inv(rot_matrix)

        # Calculate the tips of the arrow heads
        arrowhead_tips1 = final_xy - (dxy * arrowhead_scale).dot(rot_matrix)
        arrowhead_tips2 = final_xy - (dxy * arrowhead_scale).dot(inv_rot)

        # Now build an array with all the information to draw the arrows
        # This has shape (n_arrows * 7, 2). The information to draw an arrow
        # occupies 7 rows and the columns are the x and y coordinates.
        arrows = np.empty((xy.shape[0] * 7, xy.shape[1]), dtype=np.float64)

        arrows[0::7] = xy
        arrows[1::7] = final_xy
        arrows[2::7] = np.nan
        arrows[3::7] = arrowhead_tips1
        arrows[4::7] = final_xy
        arrows[5::7] = arrowhead_tips2
        arrows[6::7] = np.nan

        #
        hovertext = np.tile(dxy / scale, 7).reshape(dxy.shape[0] * 7, -1)

        if annotate:
            # Add text annotations just at the tip of the arrows.
            annotate_text = np.full((arrows.shape[0],), "", dtype=object)
            annotate_text[4::7] = [str(xy / scale) for xy in dxy]
            kwargs["text"] = list(annotate_text)

        return self.draw_line(
            arrows[:, 0],
            arrows[:, 1],
            hovertext=list(hovertext),
            row=row,
            col=col,
            **kwargs,
        )

    def draw_line_3D(
        self,
        x,
        y,
        z,
        name=None,
        line={},
        marker={},
        text=None,
        row=None,
        col=None,
        **kwargs,
    ):
        """Draws a 3D line satisfying the specifications.

        Parameters
        -----------
        x: array-like
            the coordinates of the points along the X axis.
        y: array-like
            the coordinates of the points along the Y axis.
        z: array-like
            the coordinates of the points along the Z axis.
        name: str, optional
            the name of the line
        line: dict, optional
            specifications for the line style, following plotly standards. The backend
            should at least be able to implement `line["color"]` and `line["width"]`
        marker: dict, optional
            specifications for the markers style, following plotly standards. The backend
            should at least be able to implement `marker["color"]` and `marker["size"]`
        text: str, optional
            contains the text asigned to each marker. On plotly this is seen on hover,
            other options could be annotating. However, it is not necessary that this
            argument is supported.
        row: int, optional
            If the figure contains subplots, the row where to draw.
        col: int, optional
            If the figure contains subplots, the column where to draw.
        **kwargs:
            should allow other keyword arguments to be passed directly to the creation of
            the line. This will of course be framework specific
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a draw_line_3D method."
        )

    def draw_multicolor_line_3D(self, *args, **kwargs):
        """Draws a multicoloured 3D line."""
        self.draw_line_3D(*args, **kwargs)

    def draw_multisize_line_3D(self, *args, **kwargs):
        """Draws a multisized 3D line."""
        self.draw_line_3D(*args, **kwargs)

    def draw_scatter_3D(
        self, x, y, z, name=None, marker={}, text=None, row=None, col=None, **kwargs
    ):
        """Draws a 3D scatter satisfying the specifications

        Parameters
        -----------
        x: array-like
            the coordinates of the points along the X axis.
        y: array-like
            the coordinates of the points along the Y axis.
        z: array-like
            the coordinates of the points along the Z axis.
        name: str, optional
            the name of the scatter
        marker: dict, optional
            specifications for the markers style, following plotly standards. The backend
            should at least be able to implement `marker["color"]` and `marker["size"]`
        text: str, optional
            contains the text asigned to each marker. On plotly this is seen on hover,
            other options could be annotating. However, it is not necessary that this
            argument is supported.
        row: int, optional
            If the figure contains subplots, the row where to draw.
        col: int, optional
            If the figure contains subplots, the column where to draw.
        **kwargs:
            should allow other keyword arguments to be passed directly to the creation of
            the scatter. This will of course be framework specific
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a draw_scatter_3D method."
        )

    def draw_multicolor_scatter_3D(self, *args, **kwargs):
        """Draws a multicoloured 3D scatter.

        Usually the normal 3D scatter can already support this.
        """
        # Usually, multicoloured scatter plots are already supported.
        return self.draw_scatter_3D(*args, **kwargs)

    def draw_multisize_scatter_3D(self, *args, **kwargs):
        """Draws a multisized 3D scatter.

        Usually the normal 3D scatter can already support this.
        """
        # Usually, multisized scatter plots are already supported.
        return self.draw_scatter_3D(*args, **kwargs)

    def draw_balls_3D(
        self, x, y, z, name=None, markers={}, row=None, col=None, **kwargs
    ):
        """Draws points as 3D spheres."""
        return NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a draw_balls_3D method."
        )

    def draw_multicolor_balls_3D(
        self, x, y, z, name=None, marker={}, row=None, col=None, **kwargs
    ):
        """Draws points as 3D spheres with different colours.

        If marker_color is an array of numbers, a coloraxis is created and values are converted to rgb.
        """

        kwargs["marker"] = marker.copy()

        if "color" in marker and np.array(marker["color"]).dtype in (int, float):
            coloraxis = kwargs["marker"]["coloraxis"]
            coloraxis = self._coloraxes[coloraxis]

            kwargs["marker"]["color"] = values_to_colors(
                kwargs["marker"]["color"], coloraxis["colorscale"] or "viridis"
            )

        return self.draw_balls_3D(x, y, z, name=name, row=row, col=col, **kwargs)

    def draw_multisize_balls_3D(
        self, x, y, z, name=None, marker={}, row=None, col=None, **kwargs
    ):
        """Draws points as 3D spheres with different sizes.

        Usually supported by the normal draw_balls_3D
        """
        return self.draw_balls_3D(
            x, y, z, name=name, marker=marker, row=row, col=col, **kwargs
        )

    def draw_arrows_3D(
        self,
        x,
        y,
        z,
        dxyz,
        arrowhead_scale=0.3,
        arrowhead_angle=15,
        scale: float = 1,
        row=None,
        col=None,
        **kwargs,
    ):
        """Draws multiple 3D arrows using the generic draw_line_3D method.

        Parameters
        -----------
        x: np.ndarray of shape (n_arrows, )
            the X coordinates of the arrow's origin.
        y: np.ndarray of shape (n_arrows, )
            the Y coordinates of the arrow's origin.
        z: np.ndarray of shape (n_arrows, )
            the Z coordinates of the arrow's origin.
        dxyz: np.ndarray of shape (n_arrows, 2)
            the arrow vector.
        arrow_head_scale: float, optional
            how big is the arrow head in comparison to the arrow vector.
        arrowhead_angle: angle
            the angle that the arrow head forms with the direction of the arrow (in degrees).
        scale: float, optional
            multiplying factor to display the arrows. It does not affect the underlying data,
            therefore if the data is somehow displayed it should be without the scale factor.
        row: int, optional
            If the figure contains subplots, the row where to draw.
        col: int, optional
            If the figure contains subplots, the column where to draw.
        """
        # Make sure we are dealing with numpy arrays
        xyz = np.array([x, y, z]).T
        dxyz = np.array(dxyz) * scale

        # Get the destination of the arrows
        final_xyz = xyz + dxyz

        # Convert from degrees to radians.
        arrowhead_angle = np.radians(arrowhead_angle)

        # Calculate the arrowhead positions. This is a bit more complex than the 2D case,
        # since there's no unique plane to rotate all vectors.
        # First, we get a unitary vector that is perpendicular to the direction of the arrow in xy.
        dxy_norm = np.linalg.norm(dxyz[:, :2], axis=1)
        # Some vectors might be only in the Z direction, which will result in dxy_norm being 0.
        # We avoid problems by dividinc
        dx_p = np.divide(
            dxyz[:, 1],
            dxy_norm,
            where=dxy_norm != 0,
            out=np.zeros(dxyz.shape[0], dtype=np.float64),
        )
        dy_p = np.divide(
            -dxyz[:, 0],
            dxy_norm,
            where=dxy_norm != 0,
            out=np.ones(dxyz.shape[0], dtype=np.float64),
        )

        # And then we build the rotation matrices. Since each arrow needs a unique rotation matrix,
        # we will have n 3x3 matrices, where n is the number of arrows, for each arrowhead tip.
        c = np.cos(arrowhead_angle)
        s = np.sin(arrowhead_angle)

        # Rotation matrix to build the first arrowhead tip positions.
        rot_matrices = np.array(
            [
                [c + (dx_p**2) * (1 - c), dx_p * dy_p * (1 - c), dy_p * s],
                [dy_p * dx_p * (1 - c), c + (dy_p**2) * (1 - c), -dx_p * s],
                [-dy_p * s, dx_p * s, np.full_like(dx_p, c)],
            ]
        )

        # The opposite rotation matrix, to get the other arrowhead's tip positions.
        inv_rots = rot_matrices.copy()
        inv_rots[[0, 1, 2, 2], [2, 2, 0, 1]] *= -1

        # Calculate the tips of the arrow heads.
        arrowhead_tips1 = final_xyz - np.einsum(
            "ij...,...j->...i", rot_matrices, dxyz * arrowhead_scale
        )
        arrowhead_tips2 = final_xyz - np.einsum(
            "ij...,...j->...i", inv_rots, dxyz * arrowhead_scale
        )

        # Now build an array with all the information to draw the arrows
        # This has shape (n_arrows * 7, 3). The information to draw an arrow
        # occupies 7 rows and the columns are the x and y coordinates.
        arrows = np.empty((xyz.shape[0] * 7, 3))

        arrows[0::7] = xyz
        arrows[1::7] = final_xyz
        arrows[2::7] = np.nan
        arrows[3::7] = arrowhead_tips1
        arrows[4::7] = final_xyz
        arrows[5::7] = arrowhead_tips2
        arrows[6::7] = np.nan

        return self.draw_line_3D(
            arrows[:, 0], arrows[:, 1], arrows[:, 2], row=row, col=col, **kwargs
        )

    def draw_heatmap(
        self,
        values,
        x=None,
        y=None,
        name=None,
        zsmooth=False,
        coloraxis=None,
        opacity=None,
        textformat=None,
        textfont={},
        row=None,
        col=None,
        **kwargs,
    ):
        """Draws a heatmap following the specifications."""
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a draw_heatmap method."
        )

    def draw_mesh_3D(
        self,
        vertices,
        faces,
        color=None,
        opacity=None,
        name=None,
        row=None,
        col=None,
        **kwargs,
    ):
        """Draws a 3D mesh following the specifications."""
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a draw_mesh_3D method."
        )

    def set_axis(self, **kwargs):
        """Sets the axis parameters.

        The specification for the axes is exactly the plotly one. This is to have a good
        reference for consistency. Other frameworks should translate the calls to their
        functionality.
        """

    def set_axes_equal(self):
        """Sets the axes equal."""
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't implement a set_axes_equal method."
        )

    def to(self, key: str):
        """Converts the figure to another backend.

        Parameters
        -----------
        key: str
            the backend to convert to.
        """
        return BACKENDS[key](self.plot_actions)


def get_figure(backend: str, plot_actions, *args, **kwargs) -> Figure:
    """Get a figure object.

    Parameters
    ----------
    backend : {"plotly", "matplotlib", "py3dmol", "blender"}
        the backend to use
    plot_actions : list of callable
        the plot actions to perform
    *args, **kwargs
        passed to the figure constructor
    """
    return BACKENDS[backend](plot_actions, *args, **kwargs)
