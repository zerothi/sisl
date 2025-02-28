# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import itertools
import math
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.pyplot import Normalize
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

from sisl.messages import warn

from .figure import Figure


class MatplotlibFigure(Figure):
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

    figure: Optional[plt.Figure] = None
    axes: Optional[plt.Axes] = None

    def _init_figure(self, *args, **kwargs):
        self.figure, self.axes = self._init_plt_figure()
        self._init_axes()
        return self.figure

    # def draw_on(self, axes, axes_indices=None):
    #     """Draws this plot in a different figure.

    #     Parameters
    #     -----------
    #     axes: Plot, PlotlyBackend or matplotlib.axes.Axes
    #         The axes to draw this plot in.
    #     """
    #     if isinstance(axes, Plot):
    #         axes = axes._backend.axes
    #     elif isinstance(axes, MatplotlibBackend):
    #         axes = axes.axes

    #     if axes_indices is not None:
    #         axes = axes[axes_indices]

    #     if not isinstance(axes, Axes):
    #         raise TypeError(f"{self.__class__.__name__} was provided a {axes.__class__.__name__} to draw on.")

    #     self_axes = self.axes
    #     self.axes = axes
    #     self._init_axes()
    #     self._plot.get_figure(backend=self._backend_name, clear_fig=False)
    #     self.axes = self_axes

    def _init_plt_figure(self):
        """Initializes the matplotlib figure and axes

        Returns
        --------
        Figure:
            the matplotlib figure of this plot.
        Axes:
            the matplotlib axes of this plot.
        """
        return plt.subplots()

    def _init_axes(self):
        """Does some initial modification on the axes."""
        self.axes.update(self._axes_defaults)

    def _init_figure_subplots(self, rows, cols, **kwargs):
        self.figure, self.axes = plt.subplots(rows, cols, **kwargs)

        # Normalize the axes array to have two dimensions
        if rows == 1 and cols == 1:
            self.axes = np.array([[self.axes]])
        elif rows == 1:
            self.axes = np.expand_dims(self.axes, axis=0)
        elif cols == 1:
            self.axes = np.expand_dims(self.axes, axis=1)

        return self.figure

    def _get_subplot_axes(self, row=None, col=None) -> plt.Axes:
        if row is None or col is None:
            # This is not a subplot
            return self.axes
        # Otherwise, it is indeed a subplot, so we get the axes
        return self.axes[row, col]

    def _iter_subplots(self, plot_actions):
        it = zip(itertools.product(range(self._rows), range(self._cols)), plot_actions)

        # Start assigning each plot to a position of the layout
        for i, ((row, col), section_actions) in enumerate(it):
            row_col_kwargs = {"row": row, "col": col}
            # active_axes = {
            #     ax: f"{ax}axis" if row == 0 and col == 0 else f"{ax}axis{i + 1}"
            #     for ax in "xyz"
            # }

            sanitized_section_actions = []
            for action in section_actions:
                action_name = action["method"]
                if action_name.startswith("draw_"):
                    action = {
                        **action,
                        "kwargs": {**action.get("kwargs", {}), **row_col_kwargs},
                    }
                elif action_name.startswith("set_ax"):
                    action = {
                        **action,
                        "kwargs": {**action.get("kwargs", {}), **row_col_kwargs},
                    }

                sanitized_section_actions.append(action)

            yield sanitized_section_actions

    def _init_figure_multiple_axes(self, multi_axes, plot_actions, **kwargs):
        if len(multi_axes) > 1:
            self.figure = plt.figure()
            self.axes = self.figure.add_axes(
                [0.15, 0.1, 0.65, 0.8], axes_class=HostAxes
            )
            self._init_axes()

            multi_axis = "xy"
        else:
            self.figure = self._init_figure()
            multi_axis = multi_axes[0]

        self._multi_axes[multi_axis] = self._init_multiaxis(
            multi_axis, len(plot_actions)
        )

        return self.figure

    def _init_multiaxis(self, axis, n):
        axes = [self.axes]
        for i in range(n - 1):
            if axis == "x":
                axes.append(self.axes.twiny())
            elif axis == "y":
                axes.append(self.axes.twinx())
            elif axis == "xy":
                new_axes = ParasiteAxes(self.axes, visible=True)

                new_axes.axis["right"].set_visible(True)
                new_axes.axis["right"].major_ticklabels.set_visible(True)
                new_axes.axis["right"].label.set_visible(True)
                new_axes.axis["top"].set_visible(True)
                new_axes.axis["top"].major_ticklabels.set_visible(True)
                new_axes.axis["top"].label.set_visible(True)

                self.axes.axis["right"].set_visible(False)
                self.axes.axis["top"].set_visible(False)

                self.axes.parasites.append(new_axes)
                axes.append(new_axes)

        return axes

    def _iter_multiaxis(self, plot_actions):
        multi_axis = list(self._multi_axes)[0]
        for i, section_actions in enumerate(plot_actions):
            axes = self._multi_axes[multi_axis][i]

            sanitized_section_actions = []
            for action in section_actions:
                action_name = action["method"]
                if action_name.startswith("draw_"):
                    action = {
                        **action,
                        "kwargs": {**action.get("kwargs", {}), "_axes": axes},
                    }
                elif action_name == "set_axis":
                    action = {
                        **action,
                        "kwargs": {**action.get("kwargs", {}), "_axes": axes},
                    }

                sanitized_section_actions.append(action)

            yield sanitized_section_actions

    @classmethod
    def fig_has_attr(cls, key: str) -> bool:
        return hasattr(plt.Axes, key) or hasattr(plt.Figure, key)

    def __getattr__(self, key):
        if key != "axes":
            if hasattr(self.axes, key):
                return getattr(self.axes, key)
            elif key != "figure" and hasattr(self.figure, key):
                return getattr(self.figure, key)
        raise AttributeError(key)

    def clear(self, layout=False):
        """Clears the plot canvas so that data can be reset

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

    def get_ipywidget(self):
        return self.figure

    def show(self, *args, **kwargs):
        return self.figure.show(*args, **kwargs)

    def _plotly_dash_to_matplotlib(self, dash: str) -> str:
        """Converts a plotly line_dash specification to a matplotlib linestyle."""
        return {
            "dash": "dashed",
            "dot": "dotted",
            None: "solid",
        }.get(dash, dash)

    def _sanitize_colorscale(self, colorscale):
        """Makes sure that a colorscale is either a string or a colormap."""
        if isinstance(colorscale, str):
            return colorscale
        elif isinstance(colorscale, list):

            def _sanitize_scale_item(item):
                # Plotly uses rgb colors as a string like "rgb(r,g,b)",
                # while matplotlib uses tuples
                # Also plotly's range goes from 0 to 255 while matplotlib's goes from 0 to 1
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    return (item[0], _sanitize_scale_item(item[1]))
                elif isinstance(item, str) and item.startswith("rgb("):
                    return tuple(float(x) / 255 for x in item[4:-1].split(","))

            colorscale = [_sanitize_scale_item(item) for item in colorscale]

            return matplotlib.colors.LinearSegmentedColormap.from_list(
                "custom", colorscale
            )
        else:
            return colorscale

    def draw_line(
        self,
        x,
        y,
        name=None,
        line={},
        marker={},
        text=None,
        showlegend=True,
        row=None,
        col=None,
        _axes=None,
        **kwargs,
    ):
        marker_format = marker.get("symbol", "o") if marker else None
        marker_color = marker.get("color")

        axes = _axes or self._get_subplot_axes(row=row, col=col)

        # Matplotlib doesn't show lines on the legend if their name starts
        # with an underscore, so prepend the name with "_" if showlegend is False.
        name = name if showlegend else f"_{name}"

        return axes.plot(
            x,
            y,
            color=line.get("color"),
            linewidth=line.get("width", 1),
            linestyle=self._plotly_dash_to_matplotlib(line.get("dash", "solid")),
            marker=marker_format,
            markersize=marker.get("size"),
            markerfacecolor=marker_color,
            markeredgecolor=marker_color,
            label=name,
        )

    def draw_multicolor_line(
        self,
        x,
        y,
        name=None,
        line={},
        marker={},
        text=None,
        showlegend=True,
        row=None,
        col=None,
        _axes=None,
        **kwargs,
    ):
        # This is heavily based on
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

        color = line.get("color")

        # Matplotlib doesn't show lines on the legend if their name starts
        # with an underscore, so prepend the name with "_" if showlegend is False.
        name = name if showlegend else f"_{name}"

        if not np.issubdtype(np.array(color).dtype, np.number):
            return self.draw_multicolor_scatter(
                x,
                y,
                name=name,
                marker=line,
                text=text,
                row=row,
                col=col,
                _axes=_axes,
                **kwargs,
            )

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc_kwargs = {}
        coloraxis = line.get("coloraxis")
        if coloraxis is not None:
            coloraxis = self._coloraxes.get(coloraxis)
            lc_kwargs["cmap"] = coloraxis.get("colorscale")
            if coloraxis.get("cmin") is not None:
                lc_kwargs["norm"] = Normalize(coloraxis["cmin"], coloraxis["cmax"])

        lc = LineCollection(segments, **lc_kwargs)

        # Set the values used for colormapping
        lc.set_array(line.get("color"))
        lc.set_linewidth(line.get("width", 1))
        lc.set_linestyle(self._plotly_dash_to_matplotlib(line.get("dash", "solid")))

        axes = _axes or self._get_subplot_axes(row=row, col=col)

        axes.add_collection(lc)

        # self._colorbar = axes.add_collection(lc)

    def draw_multisize_line(
        self,
        x,
        y,
        name=None,
        line={},
        marker={},
        text=None,
        row=None,
        col=None,
        _axes=None,
        **kwargs,
    ):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments)

        # Set the values used for colormapping
        lc.set_linewidth(line.get("width", 1))
        lc.set_linestyle(self._plotly_dash_to_matplotlib(line.get("dash", "solid")))
        lc.set_color(line.get("color"))
        lc.set_alpha(line.get("opacity"))

        axes = _axes or self._get_subplot_axes(row=row, col=col)

        axes.add_collection(lc)

    def draw_area_line(
        self,
        x,
        y,
        line={},
        name=None,
        showlegend=True,
        dependent_axis=None,
        row=None,
        col=None,
        _axes=None,
        **kwargs,
    ):
        width = line.get("width")
        if width is None:
            width = 1
        spacing = width / 2

        axes = _axes or self._get_subplot_axes(row=row, col=col)

        # Matplotlib doesn't show lines on the legend if their name starts
        # with an underscore, so prepend the name with "_" if showlegend is False.
        name = name if showlegend else f"_{name}"

        if dependent_axis in ("y", None):
            axes.fill_between(
                x,
                y + spacing,
                y - spacing,
                color=line.get("color"),
                label=name,
                alpha=line.get("opacity"),
            )
        elif dependent_axis == "x":
            axes.fill_betweenx(
                y,
                x + spacing,
                x - spacing,
                color=line.get("color"),
                label=name,
                alpha=line.get("opacity"),
            )
        else:
            raise ValueError(
                f"dependent_axis must be one of 'x', 'y', or None, but was {dependent_axis}"
            )

    def draw_scatter(
        self,
        x,
        y,
        name=None,
        marker={},
        text=None,
        zorder=2,
        showlegend=True,
        row=None,
        col=None,
        _axes=None,
        meta={},
        legendgroup=None,
        **kwargs,
    ):
        axes = _axes or self._get_subplot_axes(row=row, col=col)

        # Matplotlib doesn't show lines on the legend if their name starts
        # with an underscore, so prepend the name with "_" if showlegend is False.
        name = name if showlegend else f"_{name}"

        try:
            return axes.scatter(
                x,
                y,
                c=marker.get("color"),
                s=marker.get("size", 1),
                linewidths=marker.get("line_width", 1),
                edgecolors=marker.get("line_color"),
                cmap=self._sanitize_colorscale(marker.get("colorscale")),
                alpha=marker.get("opacity"),
                label=name,
                zorder=zorder,
                **kwargs,
            )
        except TypeError as e:
            if str(e) == "alpha must be a float or None":
                warn(
                    f"Your matplotlib version doesn't support multiple opacity values, please upgrade to >=3.4 if you want to use opacity."
                )
                return axes.scatter(
                    x,
                    y,
                    c=marker.get("color"),
                    s=marker.get("size", 1),
                    linewidths=marker.get("line_width", 1),
                    edgecolors=marker.get("line_color"),
                    cmap=self._sanitize_colorscale(marker.get("colorscale")),
                    label=name,
                    zorder=zorder,
                    **kwargs,
                )
            else:
                raise e

    def draw_multicolor_scatter(self, *args, **kwargs):
        marker = {**kwargs.pop("marker", {})}
        coloraxis = marker.get("coloraxis")
        if coloraxis is not None:
            coloraxis = self._coloraxes.get(coloraxis)
            marker["colorscale"] = coloraxis.get("colorscale")
        return super().draw_multicolor_scatter(*args, marker=marker, **kwargs)

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
        _axes=None,
        **kwargs,
    ):
        extent = None
        if x is not None and y is not None:
            extent = [x[0], x[-1], y[0], y[-1]]

        axes = _axes or self._get_subplot_axes(row=row, col=col)

        coloraxis = self._coloraxes.get(coloraxis, {})
        colorscale = self._sanitize_colorscale(coloraxis.get("colorscale"))
        vmin = coloraxis.get("cmin")
        vmax = coloraxis.get("cmax")

        im = axes.imshow(
            values,
            cmap=colorscale,
            vmin=vmin,
            vmax=vmax,
            label=name,
            extent=extent,
            origin="lower",
            alpha=opacity,
        )

        if textformat is not None:
            self._annotate_heatmap(
                im,
                data=values,
                valfmt="{x:" + textformat + "}",
                cmap=matplotlib.colormaps.get_cmap(colorscale),
                **textfont,
            )

    def _annotate_heatmap(
        self,
        im,
        cmap,
        data=None,
        valfmt="{x:.2f}",
        textcolors=("black", "white"),
        **textkw,
    ):
        """A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A pair of colors.  The first is used for values below a threshold,
            the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(
            horizontalalignment="center",
            verticalalignment="center",
        )
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        def color_to_textcolor(rgb):
            r, g, b = rgb
            r *= 255
            g *= 255
            b *= 255

            hsp = math.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
            if hsp > 127.5:
                return textcolors[0]
            else:
                return textcolors[1]

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isnan(data[i, j]):
                    continue

                if "color" not in textkw:
                    rgb = cmap(im.norm(data[i, j]))[:-1]
                    kw.update(color=color_to_textcolor(rgb))

                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)

        return texts

    def set_axis(
        self,
        axis,
        range=None,
        title="",
        tickvals=None,
        ticktext=None,
        showgrid=False,
        row=None,
        col=None,
        _axes=None,
        **kwargs,
    ):
        axes = _axes or self._get_subplot_axes(row=row, col=col)

        if range is not None:
            updater = getattr(axes, f"set_{axis}lim")
            updater(*range)

        if title:
            updater = getattr(axes, f"set_{axis}label")
            updater(title)

        if tickvals is not None:
            updater = getattr(axes, f"set_{axis}ticks")
            updater(ticks=tickvals, labels=ticktext)

        axes.grid(visible=showgrid, axis=axis)

    def set_axes_equal(self, row=None, col=None, _axes=None):
        axes = _axes or self._get_subplot_axes(row=row, col=col)
        axes.axis("equal")
