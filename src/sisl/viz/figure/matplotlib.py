import itertools

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
        self.figure, self.axes = plt.subplots(rows, cols)

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
                action_name = action['method']
                if action_name.startswith("draw_"):
                    action = {**action, "kwargs": {**action.get("kwargs", {}), **row_col_kwargs}}
                elif action_name.startswith("set_ax"):
                     action = {**action, "kwargs": {**action.get("kwargs", {}), **row_col_kwargs}}
                
                sanitized_section_actions.append(action)

            yield sanitized_section_actions


    def _init_figure_multiple_axes(self, multi_axes, plot_actions, **kwargs):
        
        if len(multi_axes) > 1:
            self.figure = plt.figure()
            self.axes = self.figure.add_axes([0.15, 0.1, 0.65, 0.8], axes_class=HostAxes)
            self._init_axes()

            multi_axis = "xy"
        else:
            self.figure = self._init_figure()
            multi_axis = multi_axes[0]

        self._multi_axes[multi_axis] = self._init_multiaxis(multi_axis, len(plot_actions))

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
                action_name = action['method']
                if action_name.startswith("draw_"):
                    action = {**action, "kwargs": {**action.get("kwargs", {}), "_axes": axes}}
                elif action_name == "set_axis":
                    action = {**action, "kwargs": {**action.get("kwargs", {}), "_axes": axes}}
                
                sanitized_section_actions.append(action)

            yield sanitized_section_actions

    def __getattr__(self, key):
        if key != "axes":
            return getattr(self.axes, key)
        raise AttributeError(key)

    def clear(self, layout=False):
        """ Clears the plot canvas so that data can be reset

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

    def draw_line(self, x, y, name=None, line={}, marker={}, text=None, row=None, col=None, _axes=None, **kwargs):
        marker_format = marker.get("symbol", "o") if marker else None
        marker_color = marker.get("color")

        axes = _axes or self._get_subplot_axes(row=row, col=col)

        return axes.plot(
            x, y, color=line.get("color"), linewidth=line.get("width", 1), 
            marker=marker_format, markersize=marker.get("size"), markerfacecolor=marker_color, markeredgecolor=marker_color,
            label=name
        )

    def draw_multicolor_line(self, x, y, name=None, line={}, marker={}, text=None, row=None, col=None, _axes=None, **kwargs):
        # This is heavily based on
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

        color = line.get("color")

        if not np.issubdtype(np.array(color).dtype, np.number):
            return self.draw_multicolor_scatter(x, y, name=name, marker=line, text=text, row=row, col=col, _axes=_axes, **kwargs)

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc_kwargs = {}
        coloraxis = line.get("coloraxis")
        if coloraxis is not None:
            coloraxis = self._coloraxes.get(coloraxis)
            lc_kwargs["cmap"] = coloraxis.get("colorscale")
            if coloraxis.get("cmin") is not None:
                lc_kwargs["norm"] = Normalize(coloraxis['cmin'], coloraxis['cmax'])

        lc = LineCollection(segments, **lc_kwargs)

        # Set the values used for colormapping
        lc.set_array(line.get("color"))
        lc.set_linewidth(line.get("width", 1))

        axes = _axes or self._get_subplot_axes(row=row, col=col)

        axes.add_collection(lc)

        #self._colorbar = axes.add_collection(lc)

    def draw_multisize_line(self, x, y, name=None, line={}, marker={}, text=None, row=None, col=None, _axes=None, **kwargs):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments)

        # Set the values used for colormapping
        lc.set_linewidth(line.get("width", 1))

        axes = _axes or self._get_subplot_axes(row=row, col=col)

        axes.add_collection(lc)

    def draw_area_line(self, x, y, line={}, name=None, dependent_axis=None, row=None, col=None, _axes=None, **kwargs):

        width = line.get('width')
        if width is None:
            width = 1
        spacing = width / 2

        axes = _axes or self._get_subplot_axes(row=row, col=col)

        if dependent_axis in ("y", None):
            axes.fill_between(
                x, y + spacing, y - spacing,
                color=line.get('color'), label=name
            )
        elif dependent_axis == "x":
            axes.fill_betweenx(
                y, x + spacing, x - spacing,
                color=line.get('color'), label=name
            )
        else:
            raise ValueError(f"dependent_axis must be one of 'x', 'y', or None, but was {dependent_axis}")

    def draw_scatter(self, x, y, name=None, marker={}, text=None, zorder=2, row=None, col=None, _axes=None, meta={}, **kwargs):
        axes = _axes or self._get_subplot_axes(row=row, col=col)
        try:
            return axes.scatter(x, y, c=marker.get("color"), s=marker.get("size", 1), cmap=marker.get("colorscale"), alpha=marker.get("opacity"), label=name, zorder=zorder, **kwargs)
        except TypeError as e:
            if str(e) == "alpha must be a float or None":
                warn(f"Your matplotlib version doesn't support multiple opacity values, please upgrade to >=3.4 if you want to use opacity.")
                return axes.scatter(x, y, c=marker.get("color"), s=marker.get("size", 1), cmap=marker.get("colorscale"), label=name, zorder=zorder, **kwargs)
            else:
                raise e
    
    def draw_multicolor_scatter(self, *args, **kwargs):
        marker = {**kwargs.pop("marker",{})}
        coloraxis = marker.get("coloraxis")
        if coloraxis is not None:
            coloraxis = self._coloraxes.get(coloraxis)
            marker["colorscale"] = coloraxis.get("colorscale")
        return super().draw_multicolor_scatter(*args, marker=marker, **kwargs)
    
    def draw_heatmap(self, values, x=None, y=None, name=None, zsmooth=False, coloraxis=None, row=None, col=None, _axes=None, **kwargs):

        extent = None
        if x is not None and y is not None:
            extent = [x[0], x[-1], y[0], y[-1]]

        axes = _axes or self._get_subplot_axes(row=row, col=col)

        coloraxis = self._coloraxes.get(coloraxis, {})
        colorscale = coloraxis.get("colorscale")
        vmin = coloraxis.get("cmin")
        vmax = coloraxis.get("cmax")

        axes.imshow(
            values, 
            cmap=colorscale, 
            vmin=vmin, vmax=vmax,
            label=name, extent=extent,
            origin="lower"
        )
    
    def set_axis(self, axis, range=None, title="", tickvals=None, ticktext=None, showgrid=False, row=None, col=None, _axes=None, **kwargs):
        axes = _axes or self._get_subplot_axes(row=row, col=col)

        if range is not None:
            updater = getattr(axes, f'set_{axis}lim')
            updater(*range)

        if title:
            updater = getattr(axes, f'set_{axis}label')
            updater(title)
        
        if tickvals is not None:
            updater = getattr(axes, f'set_{axis}ticks')
            updater(ticks=tickvals, labels=ticktext)

        axes.grid(visible=showgrid, axis=axis)
    
    def set_axes_equal(self, row=None, col=None, _axes=None):
        axes = _axes or self._get_subplot_axes(row=row, col=col)
        axes.axis("equal")