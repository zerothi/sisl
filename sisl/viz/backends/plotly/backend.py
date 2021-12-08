# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from collections import defaultdict
import itertools
from functools import partial
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..templates.backend import Backend, MultiplePlotBackend, SubPlotsBackend, AnimationBackend
from ...plot import Plot, SubPlots, MultiplePlot, Animation


class PlotlyBackend(Backend):
    """Generic backend for the plotly framework.

    On initialization, a plotly.graph_objs.Figure object is created and stored 
    under `self.figure`. If an attribute is not found on the backend, it is looked for
    in the figure. Therefore, you can apply all the methods that are appliable to a plotly
    figure!

    On initialization, we also take the class attribute `_layout_defaults` (a dictionary) 
    and run `update_layout` with those parameters.
    """

    _layout_defaults = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.figure = go.Figure()
        self.update_layout(**self._layout_defaults)

    def __getattr__(self, key):
        if key != "figure":
            return getattr(self.figure, key)
        raise AttributeError(key)

    def show(self, *args, **kwargs):
        return self.figure.show(*args, **kwargs)

    def draw_on(self, figure):
        """Draws this plot in a different figure.

        Parameters
        -----------
        figure: Plot, PlotlyBackend or plotly.graph_objs.Figure
            The figure to draw this plot in.
        """
        if isinstance(figure, Plot):
            figure = figure._backend.figure
        elif isinstance(figure, PlotlyBackend):
            figure = figure.figure

        if not isinstance(figure, go.Figure):
            raise TypeError(f"{self.__class__.__name__} was provided a {figure.__class__.__name__} to draw on.")

        self_fig = self.figure
        self.figure = figure
        self._plot.get_figure(backend=self._backend_name, clear_fig=False)
        self.figure = self_fig

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
        return go.FigureWidget(self.figure, )

    def _update_ipywidget(self, fig_widget):
        """ Updates a figure widget so that it is in sync with this plot's data

        Parameters
        ----------
        fig_widget: plotly.graph_objs.FigureWidget
            The figure widget that we need to extend.
        """
        fig_widget.data = []
        fig_widget.add_traces(self.data)
        fig_widget.layout = self.layout
        fig_widget.update(frames=self.frames)

    #-------------------------------------------
    #       PLOT MANIPULATION METHODS
    #-------------------------------------------

    def group_legend(self, by=None, names=None, show_all=False, extra_updates=None, **kwargs):
        """ Joins plot traces in groups in the legend

        As the result of this method, plot traces end up with a legendgroup attribute.
        You can use that for selecting traces in further processing of your plot.

        This also provides the ability to toggle the whole group from the legend, which is nice.

        Parameters
        ---------
        by: str or function, optional
            it defines what are the criteria to group the traces.

            If it's a string:
                It is the name of the trace attribute. Remember that plotly allows you to
                lookup for nested attributes using underscores. E.g: "line_color" gets {line: {color: THIS VALUE}}
            If it's a function:
                It will recieve each trace and needs to decide which group to put it in by returning the group value.
                Note that the value will also be used as the group name if `names` is not provided, so you can save yourself
                some code and directly return the group's name.
            If not provided:
                All traces will be put in the same group
        names: array-like, dict or function, optional
            it defines what the names of the generated groups will be.

            If it's an array:
                When a new group is found, the name will be taken from this array (order can be very arbitrary)
            If it's a dict:
                When a new group is found, the value of the group will be used as a key to get the name from this dictionary.
                If the key is not found, the name will just be the value.
                E.g.: If grouping by `line_color` and `blue` is found, the name will be `names.get('blue', 'blue')`
            If it's a function:
                It will recieve the group value and the trace and needs to return the name of the TRACE.
                NOTE: If `show_all` is set to `True` all traces will appear in the legend, so it would be nice
                to give them different names. Otherwise, you can just return the group's name.
                If you provided a grouping function and `show_all` is False you don't need this, as you can return
                directly the group name from there.
            If not provided:
                the values will be used as names.
        show_all: boolean, optional
            whether all the items of the group should be displayed in the legend.
            If `False`, only one item per group will be displayed.
            If `True`, all the items of the group will be displayed.
        extra_updates: dict, optional
            A dict stating extra updates that you want to do for each group.

            E.g.: `{"blue": {"line_width": 4}}`

            would also convert the lines with a group VALUE (not name) of "blue" to a width of 4.

            This is just for convenience so that you can run other methods after this one.
            Note that you can always do something like this by doing

            ```
            plot.update_traces(
                selector={"line_width": "blue"}, # Selects the traces that you will update
                line_width=4,
            ) 
            ```

            If you use a function to return the group values, there is probably no point on using this
            argument. Since you recieve the trace, you can run `trace.update(...)` inside your function.
        **kwargs:
            like extra_updates but they are passed to all groups without distinction
        """
        unique_values = []

        # Normalize the "by" parameter to a function
        if by is None:
            if show_all:
                name = names[0] if names is not None else "Group"
                self.figure.update_traces(showlegend=True, legendgroup=name, name=name)
                return self
            else:
                func = lambda trace: 0
        if isinstance(by, str):
            def func(trace):
                try:
                    return trace[by]
                except Exception:
                    return None
        else:
            func = by

        # Normalize also the names parameter to a function
        if names is None:
            def get_name(val, trace):
                return str(val) if not show_all else f'{val}: {trace.name}'
        elif callable(names):
            get_name = names
        elif isinstance(names, dict):
            def get_name(val, trace):
                name = names.get(val, val)
                return str(name) if not show_all else f'{name}: {trace.name}'
        else:
            def get_name(val, trace):
                name = names[len(unique_values) - 1]
                return str(name) if not show_all else f'{name}: {trace.name}'

        # And finally normalize the extra updates
        if extra_updates is None:
            get_extra_updates = lambda *args, **kwargs: {}
        elif isinstance(extra_updates, dict):
            get_extra_updates = lambda val, trace: extra_updates.get(val, {})
        elif callable(extra_updates):
            get_extra_updates = extra_updates

        # Build the function that will apply the change
        def check_and_apply(trace):

            val = func(trace)

            if isinstance(val, np.ndarray):
                val =  val.tolist()
            if isinstance(val, list):
                val = ", ".join([str(item) for item in val])

            if val in unique_values:
                showlegend = show_all
            else:
                unique_values.append(val)
                showlegend = True

            customdata = trace.customdata if trace.customdata is not None else [{}]

            trace.update(
                showlegend=showlegend,
                legendgroup=str(val),
                name=get_name(val, trace=trace),
                customdata=[{**customdata[0], "name": trace.name}, *customdata[1:]],
                **get_extra_updates(val, trace=trace),
                **kwargs
            )

        # And finally apply all the changes
        self.figure.for_each_trace(
            lambda trace: check_and_apply(trace)
        )

        return self

    def ungroup_legend(self):
        """ Ungroups traces if a legend contains groups """
        self.figure.for_each_trace(
            lambda trace: trace.update(
                legendgroup=None,
                showlegend=True,
                name=trace.customdata[0]["name"]
            )
        )

        return self

    def normalize(self, min_val=0, max_val=1, axis="y", **kwargs):
        """ Normalizes traces to a given range along an axis

        Parameters
        -----------
        min_val: float, optional
            The lower bound of the range.
        max_val: float, optional
            The upper part of the range
        axis: {"x", "y", "z"}, optional
            The axis along which we want to normalize.
        **kwargs:
            keyword arguments that are passed directly to plotly's Figure `for_each_trace`
            method. You can check its documentation. One important thing is that you can pass a
            'selector', which will choose if the trace is updated or not. 
        """
        from ...plotutils import normalize_trace

        self.for_each_trace(partial(normalize_trace, min_val=min_val, max_val=max_val, axis=axis), **kwargs)

        return self

    def swap_axes(self, ax1='x', ax2='y', **kwargs):
        """ Swaps two axes in the plot

        Parameters
        -----------
        ax1, ax2: str, {'x', 'x*', 'y', 'y*', 'z', 'z*'}
            The names of the axes that you want to swap. 
        **kwargs:
            keyword arguments that are passed directly to plotly's Figure `for_each_trace`
            method. You can check its documentation. One important thing is that you can pass a
            'selector', which will choose if the trace is updated or not. 
        """
        from ...plotutils import swap_trace_axes
        # Swap the traces
        self.for_each_trace(partial(swap_trace_axes, ax1=ax1, ax2=ax2), **kwargs)

        # Try to also swap the axes
        try:
            self.update_layout({
                f'{ax1}axis': self.layout[f'{ax2}axis'].to_plotly_json(),
                f'{ax2}axis': self.layout[f'{ax1}axis'].to_plotly_json(),
            }, overwrite=True)
        except:
            pass

        return self

    def shift(self, shift, axis="y", **kwargs):
        """ Shifts the traces of the plot by a given value in the given axis

        Parameters
        -----------
        shift: float or array-like
            If it's a float, it will be a solid shift (i.e. all points moved equally).
            If it's an array, an element-wise sum will be performed
        axis: {"x","y","z"}, optional
            The axis along which we want to shift the traces.
        **kwargs:
            keyword arguments that are passed directly to plotly's Figure `for_each_trace`
            method. You can check its documentation. One important thing is that you can pass a
            'selector', which will choose if the trace is updated or not. 
        """
        from ...plotutils import shift_trace

        self.for_each_trace(partial(shift_trace, shift=shift, axis=axis), **kwargs)

        return self

    # -----------------------------
    #      SOME OTHER METHODS
    # -----------------------------

    def to_chart_studio(self, *args, **kwargs):
        """ Sends the plot to chart studio if it is possible

        For it to work, the user should have their credentials correctly set up.

        It is a shortcut for chart_studio.plotly.plot(self.figure, ...etc) so you can pass any extra arguments as if
        you were using `py.plot`
        """
        import chart_studio.plotly as py

        return py.plot(self.figure, *args, **kwargs)

    # -----------------------------
    #      METHODS FOR TESTING
    # -----------------------------

    def _test_number_of_items_drawn(self):
        return len(self.figure.data)

    # --------------------------------
    #  METHODS TO STANDARIZE BACKENDS
    # --------------------------------

    def draw_line(self, x, y, name=None, line={}, **kwargs):
        """Draws a line in the current plot."""
        opacity = kwargs.get("opacity", line.get("opacity", 1))
        self.add_trace({
            'type': 'scatter',
            'x': x,
            'y': y,
            'mode': 'lines',
            'name': name,
            'line': {k: v for k, v in line.items() if k != "opacity"},
            'opacity': opacity,
            **kwargs,
        })

    def draw_scatter(self, x, y, name=None, marker={}, **kwargs):
        self.draw_line(x, y, name, marker=marker, mode="markers", **kwargs)

    def draw_line3D(self, x, y, z, **kwargs):
        self.draw_line(x, y, type="scatter3d", z=z, **kwargs)

    def draw_scatter3D(self, *args, **kwargs):
        self.draw_line3D(*args, mode="markers", **kwargs)

    def draw_arrows3D(self, xyz, dxyz, arrowhead_angle=20, arrowhead_scale=0.3, **kwargs):
        """Draws 3D arrows in plotly using a combination of a scatter3D and a Cone trace."""
        final_xyz = xyz + dxyz

        color = kwargs.get("line", {}).get("color")
        if color is None:
            color = "red"

        name = kwargs.get("name", "Arrows")

        arrows_coords = np.empty((xyz.shape[0]*3, 3), dtype=np.float64)

        arrows_coords[0::3] = xyz
        arrows_coords[1::3] = final_xyz
        arrows_coords[2::3] = np.nan

        conebase_xyz = xyz + (1 - arrowhead_scale) * dxyz

        self.figure.add_traces([{
            "x": arrows_coords[:, 0],
            "y": arrows_coords[:, 1],
            "z": arrows_coords[:, 2],
            "mode": "lines",
            "type": "scatter3d",
            "hoverinfo": "none",
            "line": {**kwargs.get("line"), "color": color, },
            "legendgroup": name,
            "name": f"{name} lines",
            "showlegend": False,
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
        }])


class PlotlyMultiplePlotBackend(PlotlyBackend, MultiplePlotBackend):
    pass


class PlotlySubPlotsBackend(PlotlyBackend, SubPlotsBackend):

    def draw(self, backend_info):
        children = backend_info["children"]
        rows, cols = backend_info["rows"], backend_info["cols"]

        # Check if all childplots have the same xaxis or yaxis titles.
        axes_titles = defaultdict(list)
        for child_plot in children:
            axes_titles["x"].append(child_plot.layout.xaxis.title.text)
            axes_titles["y"].append(child_plot.layout.yaxis.title.text)

        # If so, we will set the subplots figure x_title and/or y_title so that it looks cleaner.
        # See how we remove the titles from the axis layout below when we allocate each plot.
        axes_titles = {f"{key}_title": val[0] for key, val in axes_titles.items() if len(set(val)) == 1}

        self.figure = make_subplots(**{
            "rows": rows, "cols": cols,
            **axes_titles,
            **backend_info["make_subplots_kwargs"]
        })

        # Start assigning each plot to a position of the layout
        for (row, col), plot in zip(itertools.product(range(1, rows + 1), range(1, cols + 1)), children):

            ntraces = len(plot.data)

            self.add_traces(plot.data, rows=[row]*ntraces, cols=[col]*ntraces)

            for ax in "x", "y":
                ax_layout = getattr(plot.layout, f"{ax}axis").to_plotly_json()

                # If we have set a global title for this axis, just remove it from the plot
                if axes_titles.get(f"{ax}_title"):
                    ax_layout["title"] = None

                update_axis = getattr(self, f"update_{ax}axes")

                update_axis(ax_layout, row=row, col=col)

        # Since we have directly copied the layouts of the child plots, there may be some references
        # between axes that we need to fix. E.g.: if yaxis was set to follow xaxis in the second child plot,
        # since the second child plot is put in (xaxes2, yaxes2) the reference will be now to the first child
        # plot xaxis, not itself. This is best understood by printing the figure of a subplot :)
        new_layouts = {}
        for ax, layout in self.figure.layout.to_plotly_json().items():
            if "axis" in ax:
                ax_name, ax_num = ax.split("axis")

                # Go over all possible problematic keys
                for key in ["anchor", "scaleanchor"]:
                    val = layout.get(key)
                    if val in ["x", "y"]:
                        layout[key] = f"{val}{ax_num}"

                new_layouts[ax] = layout

        self.update_layout(**new_layouts)


class PlotlyAnimationBackend(PlotlyBackend, AnimationBackend):

    def draw(self, backend_info):
        children = backend_info["children"]
        frame_names = backend_info["frame_names"]
        frames_layout = self._build_frames(children, None, frame_names)
        self.update_layout(**frames_layout)

    def _build_frames(self, children, ani_method, frame_names):
        """ Builds the frames of the plotly figure from the child plots' data

        It actually sets the frames of the figure.

        Returns
        -----------
        dict
            keys and values that need to be added to the layout
            in order for frames to work.
        """
        if ani_method is None:
            same_traces = np.unique(
                [len(plot.data) for plot in children]
            ).shape[0] == 1

            ani_method = "animate" if same_traces else "update"

        # Choose the method that we need to run in order to get the figure
        if ani_method == "animate":
            figure_builder = self._figure_animate_method
        elif ani_method == "update":
            figure_builder = self._figure_update_method

        steps, updatemenus = figure_builder(children, frame_names)

        frames_layout = {

            "sliders": [
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        #"prefix": "Bands file:",
                        "visible": True,
                        "xanchor": "right"
                    },
                    #"transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": steps
                }
            ],

            "updatemenus": updatemenus
        }

        return frames_layout

    def _figure_update_method(self, children, frame_names):
        """
        In the update method, we give all the traces to data, and we are just going to toggle
        their visibility depending on which 'frame' needs to be displayed.
        """
        # Add all the traces
        for i, (frame_name, plot) in enumerate(zip(frame_names, children)):

            visible = i == 0

            self.add_traces([{
                **trace.to_plotly_json(),
                'customdata': [{'frame': frame_name, "iFrame": i}],
                'visible': visible
            } for trace in plot.data])

        # Generate the steps
        steps = []
        for i, frame_name in enumerate(frame_names):

            steps.append({
                "label": frame_name,
                "method": "restyle",
                "args": [{"visible": [trace.customdata[0]["iFrame"] == i for trace in self.data]}]
            })

        # WE SHOULD DEFINE PLAY AND PAUSE BUTTONS TO BE RENDERED IN JUPYTER'S NOTEBOOK HERE
        # IT IS IMPOSSIBLE TO PASS CONDITIONS TO DECIDE WHAT TO DISPLAY USING PLOTLY JSON
        self.animate_widgets = []

        return steps, []

    def _figure_animate_method(self, children, frame_names):
        """
        In the animate method, we explicitly define frames, And the transition from one to the other
        will be animated
        """
        # Here are some things that were settings
        frame_duration = 500
        redraw = True

        # Data will actually only be the first frame
        self.figure.update(data=children[0].data)

        frames = []

        maxN = np.max([len(plot.data) for plot in children])
        for frame_name, plot in zip(frame_names, children):

            data = plot.data
            nTraces = len(data)
            if nTraces < maxN:
                nAddTraces = maxN - nTraces
                data = [
                    *data, *np.full(nAddTraces, {"type": "scatter", "x":  [0], "y": [0], "visible": False})]

            frames = [
                *frames, {'name': frame_name, 'data': data, "layout": plot.get_settings_group("layout")}]

        self.figure.update(frames=frames)

        steps = [
            {"args": [
            [frame["name"]],
            {"frame": {"duration": int(frame_duration), "redraw": redraw},
            "mode": "immediate",
            "transition": {"duration": 300}}
        ],
            "label": frame["name"],
            "method": "animate"} for frame in self.figure.frames
        ]

        updatemenus = [

            {'type': 'buttons',
            'buttons': [
                {
                    'label': '▶',
                    'method': 'animate',
                    'args': [None, {"frame": {"duration": int(frame_duration), "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 100,
                                                                        "easing": "quadratic-in-out"}}],
                },

                {
                    'label': '⏸',
                    'method': 'animate',
                    'args': [[None], {"frame": {"duration": 0}, "redraw": True,
                                    'mode': 'immediate',
                                    "transition": {"duration": 0}}],
                }
            ]}
        ]

        return steps, updatemenus

Animation.backends.register("plotly", PlotlyAnimationBackend)
MultiplePlot.backends.register("plotly", PlotlyMultiplePlotBackend)
SubPlots.backends.register("plotly", PlotlySubPlotsBackend)
