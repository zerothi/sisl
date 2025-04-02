# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import itertools
from collections.abc import Sequence
from numbers import Number
from typing import Optional

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

from ..processors.coords import sphere
from .figure import Figure

# Special plotly templates for sisl
pio.templates["sisl"] = go.layout.Template(
    layout={
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        **{
            f"{ax}_{key}": val
            for ax, (key, val) in itertools.product(
                ("xaxis", "yaxis"),
                (
                    ("visible", True),
                    ("showline", True),
                    ("linewidth", 1),
                    ("mirror", True),
                    ("color", "black"),
                    ("showgrid", False),
                    ("gridcolor", "#ccc"),
                    ("gridwidth", 1),
                    ("zeroline", False),
                    ("zerolinecolor", "#ccc"),
                    ("zerolinewidth", 1),
                    ("ticks", "outside"),
                    ("ticklen", 5),
                    ("ticksuffix", " "),
                ),
            )
        },
        "hovermode": "closest",
        "scene": {
            **{
                f"{ax}_{key}": val
                for ax, (key, val) in itertools.product(
                    ("xaxis", "yaxis", "zaxis"),
                    (
                        ("visible", True),
                        ("showline", True),
                        ("linewidth", 1),
                        ("mirror", True),
                        ("color", "black"),
                        ("showgrid", False),
                        ("gridcolor", "#ccc"),
                        ("gridwidth", 1),
                        ("zeroline", False),
                        ("zerolinecolor", "#ccc"),
                        ("zerolinewidth", 1),
                        ("ticks", "outside"),
                        ("ticklen", 5),
                        ("ticksuffix", " "),
                    ),
                )
            },
        },
        # "editrevision": True
        # "title": {"xref": "paper", "x": 0.5, "text": "Whhhhhhhat up", "pad": {"b": 0}}
    },
)

pio.templates["sisl_dark"] = go.layout.Template(
    layout={
        "plot_bgcolor": "black",
        "paper_bgcolor": "black",
        **{
            f"{ax}_{key}": val
            for ax, (key, val) in itertools.product(
                ("xaxis", "yaxis"),
                (
                    ("visible", True),
                    ("showline", True),
                    ("linewidth", 1),
                    ("mirror", True),
                    ("color", "white"),
                    ("showgrid", False),
                    ("gridcolor", "#ccc"),
                    ("gridwidth", 1),
                    ("zeroline", False),
                    ("zerolinecolor", "#ccc"),
                    ("zerolinewidth", 1),
                    ("ticks", "outside"),
                    ("ticklen", 5),
                    ("ticksuffix", " "),
                ),
            )
        },
        "font": {"color": "white"},
        "hovermode": "closest",
        "scene": {
            **{
                f"{ax}_{key}": val
                for ax, (key, val) in itertools.product(
                    ("xaxis", "yaxis", "zaxis"),
                    (
                        ("visible", True),
                        ("showline", True),
                        ("linewidth", 1),
                        ("mirror", True),
                        ("color", "white"),
                        ("showgrid", False),
                        ("gridcolor", "#ccc"),
                        ("gridwidth", 1),
                        ("zeroline", False),
                        ("zerolinecolor", "#ccc"),
                        ("zerolinewidth", 1),
                        ("ticks", "outside"),
                        ("ticklen", 5),
                        ("ticksuffix", " "),
                    ),
                )
            },
        },
        # "editrevision": True
        # "title": {"xref": "paper", "x": 0.5, "text": "Whhhhhhhat up", "pad": {"b": 0}}
    },
)

# This will be the default one for sisl plots
# Maybe we should pass it explicitly to plots instead of making it the default
# so that it doesn't affect plots outside sisl.
pio.templates.default = "sisl"


class PlotlyFigure(Figure):
    """Generic canvas for the plotly framework.

    On initialization, a plotly.graph_objs.Figure object is created and stored
    under `self.figure`. If an attribute is not found on the backend, it is looked for
    in the figure. Therefore, you can apply all the methods that are appliable to a plotly
    figure!

    On initialization, we also take the class attribute `_layout_defaults` (a dictionary)
    and run `update_layout` with those parameters.
    """

    _multi_axis = None

    _layout_defaults = {}

    figure: Optional[go.Figure] = None

    def _init_figure(self, *args, **kwargs):
        self.figure = go.Figure()
        self.update_layout(**self._layout_defaults)
        return self

    def _init_figure_subplots(self, rows, cols, **kwargs):
        figure = self._init_figure()

        figure.set_subplots(
            **{
                "rows": rows,
                "cols": cols,
                **kwargs,
            }
        )

        return figure

    def _iter_subplots(self, plot_actions):
        it = zip(itertools.product(range(self._rows), range(self._cols)), plot_actions)

        # Start assigning each plot to a position of the layout
        for i, ((row, col), section_actions) in enumerate(it):
            row_col_kwargs = {"row": row + 1, "col": col + 1}
            active_axes = {
                ax: f"{ax}axis" if row == 0 and col == 0 else f"{ax}axis{i + 1}"
                for ax in "xyz"
            }

            sanitized_section_actions = []
            for action in section_actions:
                action_name = action["method"]
                if action_name.startswith("draw_"):
                    action = {
                        **action,
                        "kwargs": {**action.get("kwargs", {}), **row_col_kwargs},
                    }
                    action["kwargs"]["meta"] = {
                        **action["kwargs"].get("meta", {}),
                        "i_plot": i,
                    }
                elif action_name.startswith("set_ax"):
                    action = {
                        **action,
                        "kwargs": {
                            **action.get("kwargs", {}),
                            "_active_axes": active_axes,
                        },
                    }

                sanitized_section_actions.append(action)

            yield sanitized_section_actions

    def _init_multiaxis(self, axis, n):
        axes = [f"{axis}{i + 1}" if i > 0 else axis for i in range(n)]
        layout_axes = [
            f"{axis}axis{i + 1}" if i > 0 else f"{axis}axis" for i in range(n)
        ]
        if axis == "x":
            sides = ["bottom", "top"]
        elif axis == "y":
            sides = ["left", "right"]
        else:
            raise ValueError(f"Multiple axis are only supported for 'x' or 'y'")

        layout_updates = {}
        for ax, side in zip(layout_axes, itertools.cycle(sides)):
            layout_updates[ax] = {"side": side, "overlaying": axis}
        layout_updates[f"{axis}axis"]["overlaying"] = None
        self.update_layout(**layout_updates)

        return layout_axes

    def _iter_multiaxis(self, plot_actions):
        for i, section_actions in enumerate(plot_actions):
            active_axes = {ax: v[i] for ax, v in self._multi_axes.items()}
            active_axes_kwargs = {
                f"{ax}axis": v.replace("axis", "") for ax, v in active_axes.items()
            }

            sanitized_section_actions = []
            for action in section_actions:
                action_name = action["method"]
                if action_name.startswith("draw_"):
                    action = {
                        **action,
                        "kwargs": {**action.get("kwargs", {}), **active_axes_kwargs},
                    }
                    action["kwargs"]["meta"] = {
                        **action["kwargs"].get("meta", {}),
                        "i_plot": i,
                    }
                elif action_name.startswith("set_ax"):
                    action = {
                        **action,
                        "kwargs": {
                            **action.get("kwargs", {}),
                            "_active_axes": active_axes,
                        },
                    }

                sanitized_section_actions.append(action)

            yield sanitized_section_actions

    def _iter_same_axes(self, plot_actions):
        for i, section_actions in enumerate(plot_actions):
            sanitized_section_actions = []
            for action in section_actions:
                action_name = action["method"]
                if action_name.startswith("draw_"):
                    action = {**action, "kwargs": action.get("kwargs", {})}
                    action["kwargs"]["meta"] = {
                        **action["kwargs"].get("meta", {}),
                        "i_plot": i,
                    }

                sanitized_section_actions.append(action)

            yield sanitized_section_actions

    def _init_figure_animated(
        self,
        frame_names: Optional[Sequence[str]] = None,
        frame_duration: int = 500,
        transition: int = 300,
        redraw: bool = False,
        **kwargs,
    ):
        self._animation_settings = {
            "frame_names": frame_names,
            "frame_duration": frame_duration,
            "transition": transition,
            "redraw": redraw,
        }
        self._animate_frame_names = frame_names
        self._animate_init_kwargs = kwargs
        return self._init_figure(**kwargs)

    def _iter_animation(self, plot_actions):
        frame_duration = self._animation_settings["frame_duration"]
        transition = self._animation_settings["transition"]
        redraw = self._animation_settings["redraw"]

        frame_names = self._animation_settings["frame_names"]
        if frame_names is None:
            frame_names = [i for i in range(len(plot_actions))]

        frame_names = [str(name) for name in frame_names]

        frames = []
        for i, section_actions in enumerate(plot_actions):
            sanitized_section_actions = []
            for action in section_actions:
                action_name = action["method"]
                if action_name.startswith("draw_"):
                    action = {**action, "kwargs": action.get("kwargs", {})}
                    action["kwargs"]["meta"] = {
                        **action["kwargs"].get("meta", {}),
                        "i_plot": i,
                    }
                sanitized_section_actions.append(action)

            yield sanitized_section_actions

            # Create a frame and append it
            frames.append(
                go.Frame(
                    name=frame_names[i],
                    data=self.figure.data,
                    layout=self.figure.layout,
                )
            )

            # Reinit the figure
            self._init_figure(**self._animate_init_kwargs)

        self.figure.update(data=frames[0].data, frames=frames)

        slider_steps = [
            {
                "args": [
                    [frame["name"]],
                    {
                        "frame": {"duration": int(frame_duration), "redraw": redraw},
                        "mode": "immediate",
                        "transition": {"duration": transition},
                    },
                ],
                "label": frame["name"],
                "method": "animate",
            }
            for frame in self.figure.frames
        ]

        slider = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                # "prefix": "Bands file:",
                "visible": True,
                "xanchor": "right",
            },
            # "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": slider_steps,
        }

        # Buttons to play and pause the animation
        updatemenus = [
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "▶",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": int(frame_duration),
                                    "redraw": redraw,
                                },
                                "fromcurrent": True,
                                "transition": {"duration": 100},
                            },
                        ],
                    },
                    {
                        "label": "⏸",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0},
                                "redraw": redraw,
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ]

        self.update_layout(sliders=[slider], updatemenus=updatemenus)

    @classmethod
    def fig_has_attr(cls, key: str) -> bool:
        return hasattr(go.Figure, key)

    def __getattr__(self, key):
        if key != "figure":
            return getattr(self.figure, key)
        raise AttributeError(key)

    def show(self, *args, **kwargs):
        return self.figure.show(*args, **kwargs)

    def clear(self, frames=True, layout=False):
        """Clears the plot canvas so that data can be reset

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

    # --------------------------------
    #  METHODS TO STANDARIZE BACKENDS
    # --------------------------------
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
        if len(self._coloraxes) == 0:
            kwargs["ax_name"] = "coloraxis"
        else:
            kwargs["ax_name"] = f"coloraxis{len(self._coloraxes) + 1}"

        super().init_coloraxis(name, cmin, cmax, cmid, colorscale, showscale, **kwargs)

        ax_name = kwargs["ax_name"]
        self.update_layout(
            **{
                ax_name: {
                    "colorscale": colorscale,
                    "cmin": cmin,
                    "cmax": cmax,
                    "cmid": cmid,
                    "showscale": showscale,
                }
            }
        )

    def _get_coloraxis_name(self, coloraxis: Optional[str]):
        if coloraxis in self._coloraxes:
            return self._coloraxes[coloraxis]["ax_name"]
        else:
            return coloraxis

    def _handle_multicolor_scatter(self, marker, scatter_kwargs):
        if "coloraxis" in marker:
            marker = marker.copy()
            coloraxis = marker["coloraxis"]

            if coloraxis is not None:
                scatter_kwargs["hovertemplate"] = (
                    "x: %{x:.2f}<br>y: %{y:.2f}<br>"
                    + coloraxis
                    + ": %{marker.color:.2f}"
                )
                marker["coloraxis"] = self._get_coloraxis_name(coloraxis)

        return marker

    def draw_line(self, x, y, name=None, line={}, row=None, col=None, **kwargs):
        """Draws a line in the current plot."""
        opacity = kwargs.get("opacity", line.get("opacity", 1))

        # Define the mode of the scatter trace. If markers or text are passed,
        # we enforce the mode to show them.
        mode = kwargs.pop("mode", "lines")
        if kwargs.get("marker") and "markers" not in mode:
            mode += "+markers"
        if kwargs.get("text") and "text" not in mode:
            mode += "+text"

        # Finally, we add the trace.
        self.add_trace(
            {
                "type": "scatter",
                "x": x,
                "y": y,
                "mode": mode,
                "name": name,
                "line": {k: v for k, v in line.items() if k != "opacity"},
                "opacity": opacity,
                "meta": kwargs.pop("meta", {}),
                **kwargs,
            },
            row=row,
            col=col,
        )

    def draw_multicolor_line(self, *args, **kwargs):
        kwargs["marker_line_width"] = 0

        super().draw_multicolor_line(*args, **kwargs)

    def draw_multisize_line(self, *args, **kwargs):
        kwargs["marker_line_width"] = 0

        super().draw_multisize_line(*args, **kwargs)

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
        def _sanitize_prop(prop, start, end):
            if isinstance(prop, (Number, str)) or prop is None:
                return prop
            else:
                return prop[start:end]

        x = np.asarray(x)
        y = np.asarray(y)

        nan_indices = [*np.where(np.isnan(y))[0], len(y)]
        chunk_start = 0
        for chunk_end in nan_indices:
            chunk_x = x[chunk_start:chunk_end]
            chunk_y = y[chunk_start:chunk_end]

            width = _sanitize_prop(line.get("width"), chunk_start, chunk_end)
            if width is None:
                width = 1
            chunk_spacing = width / 2

            if dependent_axis is None:
                # We draw the area line using the perpendicular direction to the line, because we don't know which
                # direction should we draw it in.
                normal = np.array([-np.gradient(y), np.gradient(x)]).T
                norms = normal / np.linalg.norm(normal, axis=1).reshape(-1, 1)

                trace_x = [
                    *(chunk_x + norms[:, 0] * chunk_spacing),
                    *reversed(chunk_x - norms[:, 0] * chunk_spacing),
                ]
                trace_y = [
                    *(chunk_y + norms[:, 1] * chunk_spacing),
                    *reversed(chunk_y - norms[:, 1] * chunk_spacing),
                ]
            elif dependent_axis == "y":
                trace_x = [*chunk_x, *reversed(chunk_x)]
                trace_y = [
                    *(chunk_y + chunk_spacing),
                    *reversed(chunk_y - chunk_spacing),
                ]
            elif dependent_axis == "x":
                trace_x = [
                    *(chunk_x + chunk_spacing),
                    *reversed(chunk_x - chunk_spacing),
                ]
                trace_y = [*chunk_y, *reversed(chunk_y)]
            else:
                raise ValueError(f"Invalid dependent axis: {dependent_axis}")

            self.add_trace(
                {
                    "type": "scatter",
                    "mode": "lines",
                    "x": trace_x,
                    "y": trace_y,
                    "line": {
                        "width": 0,
                        "color": _sanitize_prop(
                            line.get("color"), chunk_start, chunk_end
                        ),
                    },
                    "name": name,
                    "legendgroup": name,
                    "showlegend": (
                        kwargs.pop("showlegend", None) if chunk_start == 0 else False
                    ),
                    "fill": "toself",
                    "opacity": _sanitize_prop(
                        line.get("opacity"), chunk_start, chunk_end
                    ),
                    "meta": kwargs.pop("meta", {}),
                },
                row=row,
                col=col,
            )

            chunk_start = chunk_end + 1

    def draw_scatter(self, x, y, name=None, marker={}, **kwargs):
        marker = {k: v for k, v in marker.items() if k != "dash"}
        self.draw_line(x, y, name, marker=marker, mode="markers", **kwargs)

    def draw_multicolor_scatter(self, *args, **kwargs):
        kwargs["marker"] = self._handle_multicolor_scatter(kwargs["marker"], kwargs)

        super().draw_multicolor_scatter(*args, **kwargs)

    def draw_line_3D(self, x, y, z, **kwargs):
        self.draw_line(x, y, type="scatter3d", z=z, **kwargs)

    def draw_multicolor_line_3D(self, x, y, z, **kwargs):
        kwargs["line"] = self._handle_multicolor_scatter(kwargs["line"], kwargs)

        super().draw_multicolor_line_3D(x, y, z, **kwargs)

    def draw_scatter_3D(self, *args, marker={}, **kwargs):
        marker = {k: v for k, v in marker.items() if k != "dash"}
        self.draw_line_3D(*args, mode="markers", marker=marker, **kwargs)

    def draw_multicolor_scatter_3D(self, *args, **kwargs):
        kwargs["marker"] = self._handle_multicolor_scatter(kwargs["marker"], kwargs)

        super().draw_multicolor_scatter_3D(*args, **kwargs)

    def draw_balls_3D(self, x, y, z, name=None, marker={}, **kwargs):
        style = {}
        for k in ("size", "color", "opacity"):
            val = marker.get(k)

            if isinstance(val, (str, int, float)):
                val = itertools.repeat(val)

            style[k] = val

        iterator = enumerate(
            zip(
                np.array(x),
                np.array(y),
                np.array(z),
                style["size"],
                style["color"],
                style["opacity"],
            )
        )

        meta = kwargs.pop("meta", {})
        showlegend = True
        for i, (sp_x, sp_y, sp_z, sp_size, sp_color, sp_opacity) in iterator:
            self.draw_ball_3D(
                xyz=[sp_x, sp_y, sp_z],
                size=sp_size,
                color=sp_color,
                opacity=sp_opacity,
                name=name,
                legendgroup=name,
                showlegend=showlegend,
                meta={**meta, f"{name}_i": i},
            )
            showlegend = False

        return

    def draw_ball_3D(
        self,
        xyz,
        size,
        color="gray",
        name=None,
        vertices=15,
        row=None,
        col=None,
        **kwargs,
    ):
        self.add_trace(
            {
                "type": "mesh3d",
                **{
                    key: val
                    for key, val in sphere(
                        center=xyz, r=size, vertices=vertices
                    ).items()
                },
                "alphahull": 0,
                "color": color,
                "showscale": False,
                "name": name,
                "meta": {
                    "position": "({:.2f}, {:.2f}, {:.2f})".format(*xyz),
                    "meta": kwargs.pop("meta", {}),
                },
                "hovertemplate": "%{meta.position}",
                **kwargs,
            },
            row=None,
            col=None,
        )

    def draw_arrows_3D(
        self,
        x,
        y,
        z,
        dxyz,
        arrowhead_angle=20,
        arrowhead_scale=0.3,
        scale: float = 1,
        row=None,
        col=None,
        **kwargs,
    ):
        """Draws 3D arrows in plotly using a combination of a scatter3D and a Cone trace."""
        # Make sure we are dealing with numpy arrays
        xyz = np.array([x, y, z]).T
        dxyz = np.array(dxyz) * scale

        final_xyz = xyz + dxyz

        line = kwargs.get("line", {}).copy()
        color = line.get("color")
        if color is None:
            color = "red"
        line["color"] = color
        # 3D lines don't support opacity
        line.pop("opacity", None)

        name = kwargs.get("name", "Arrows")

        arrows_coords = np.empty((xyz.shape[0] * 3, 3), dtype=np.float64)

        arrows_coords[0::3] = xyz
        arrows_coords[1::3] = final_xyz
        arrows_coords[2::3] = np.nan

        conebase_xyz = xyz + (1 - arrowhead_scale) * dxyz

        rows_cols = {}
        if row is not None:
            rows_cols["rows"] = [row, row]
        if col is not None:
            rows_cols["cols"] = [col, col]

        meta = kwargs.pop("meta", {})

        self.figure.add_traces(
            [
                {
                    "x": arrows_coords[:, 0],
                    "y": arrows_coords[:, 1],
                    "z": arrows_coords[:, 2],
                    "mode": "lines",
                    "type": "scatter3d",
                    "hoverinfo": "none",
                    "line": line,
                    "legendgroup": name,
                    "name": f"{name} lines",
                    "showlegend": False,
                    "meta": meta,
                },
                {
                    "type": "cone",
                    "x": conebase_xyz[:, 0],
                    "y": conebase_xyz[:, 1],
                    "z": conebase_xyz[:, 2],
                    "u": arrowhead_scale * dxyz[:, 0],
                    "v": arrowhead_scale * dxyz[:, 1],
                    "w": arrowhead_scale * dxyz[:, 2],
                    "hovertemplate": "[%{u}, %{v}, %{w}]",
                    "sizemode": "absolute",
                    "sizeref": arrowhead_scale * np.linalg.norm(dxyz, axis=1).max() / 2,
                    "colorscale": [[0, color], [1, color]],
                    "showscale": False,
                    "legendgroup": name,
                    "name": name,
                    "showlegend": True,
                    "meta": meta,
                },
            ],
            **rows_cols,
        )

    def draw_heatmap(
        self,
        values,
        x=None,
        y=None,
        name=None,
        zsmooth=False,
        coloraxis=None,
        textformat=None,
        row=None,
        col=None,
        **kwargs,
    ):
        if textformat is not None:
            # If the user wants a custom color, we must define the text strings to be empty
            # for NaN values. If there is not custom color, plotly handles this for us by setting
            # the text color to the same as the background for those values so that they are not
            # visible.
            if "color" in kwargs.get("textfont", {}) and np.any(np.isnan(values)):
                to_string = np.vectorize(
                    lambda x: "" if np.isnan(x) else f"{x:{textformat}}"
                )
                kwargs = {
                    "text": to_string(values),
                    "texttemplate": "%{text}",
                    **kwargs,
                }
            else:
                kwargs = {
                    "texttemplate": "%{z:" + textformat + "}",
                    **kwargs,
                }

        self.add_trace(
            {
                "type": "heatmap",
                "z": values,
                "x": x,
                "y": y,
                "name": name,
                "zsmooth": zsmooth,
                "coloraxis": self._get_coloraxis_name(coloraxis),
                "meta": kwargs.pop("meta", {}),
                **kwargs,
            },
            row=row,
            col=col,
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
        x, y, z = vertices.T
        I, J, K = faces.T

        self.add_trace(
            dict(
                type="mesh3d",
                x=x,
                y=y,
                z=z,
                i=I,
                j=J,
                k=K,
                color=color,
                opacity=opacity,
                name=name,
                showlegend=True,
                meta=kwargs.pop("meta", {}),
                **kwargs,
            ),
            row=row,
            col=col,
        )

    def set_axis(self, axis, _active_axes={}, **kwargs):
        if axis in _active_axes:
            ax_name = _active_axes[axis]
        else:
            ax_name = f"{axis}axis"

        updates = {}
        if ax_name.endswith("axis"):
            scene_updates = {**kwargs}
            scene_updates.pop("constrain", None)
            updates = {f"scene_{ax_name}": scene_updates}
        if axis != "z":
            updates.update({ax_name: kwargs})

        self.update_layout(**updates)

    def set_axes_equal(self, _active_axes={}):
        x_axis = _active_axes.get("x", "xaxis")
        y_axis = _active_axes.get("y", "yaxis").replace("axis", "")

        self.update_layout({x_axis: {"scaleanchor": y_axis, "scaleratio": 1}})
        self.update_layout(scene_aspectmode="data")
