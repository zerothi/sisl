import itertools
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np

from sisl.viz.nodes.processors.coords import sphere

from .canvas import CanvasNode


class PlotlyCanvas(CanvasNode):
    """Generic canvas for the plotly framework.

    On initialization, a plotly.graph_objs.Figure object is created and stored 
    under `self.figure`. If an attribute is not found on the backend, it is looked for
    in the figure. Therefore, you can apply all the methods that are appliable to a plotly
    figure!

    On initialization, we also take the class attribute `_layout_defaults` (a dictionary) 
    and run `update_layout` with those parameters.
    """

    _layout_defaults = {}

    def _init_figure(self, *args, **kwargs):
        self.figure = go.Figure()
        self.update_layout(**self._layout_defaults)
        return self

    def _init_figure_multiple(self, *args, **kwargs):
        return self._init_figure(*args, **kwargs)

    def __getattr__(self, key):
        if key != "figure":
            return getattr(self.figure, key)
        raise AttributeError(key)

    def show(self, *args, **kwargs):
        return self.figure.show(*args, **kwargs)

    # def draw_on(self, figure):
    #     """Draws this plot in a different figure.

    #     Parameters
    #     -----------
    #     figure: Plot, PlotlyBackend or plotly.graph_objs.Figure
    #         The figure to draw this plot in.
    #     """
    #     if isinstance(figure, Plot):
    #         figure = figure._backend.figure
    #     elif isinstance(figure, PlotlyBackend):
    #         figure = figure.figure

    #     if not isinstance(figure, go.Figure):
    #         raise TypeError(f"{self.__class__.__name__} was provided a {figure.__class__.__name__} to draw on.")

    #     self_fig = self.figure
    #     self.figure = figure
    #     self._plot.get_figure(backend=self._backend_name, clear_fig=False)
    #     self.figure = self_fig

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

    # --------------------------------
    #  METHODS TO STANDARIZE BACKENDS
    # --------------------------------
    def init_coloraxis(self, name, cmin, cmax, colorscale, **kwargs):
        if len(self._coloraxes) == 0:
            kwargs['ax_name'] = "coloraxis"
        else:
            kwargs['ax_name'] = f'coloraxis{len(self._coloraxes)}'

        super().init_coloraxis(name, cmin, cmax, colorscale, **kwargs)
        
        ax_name = kwargs['ax_name']
        self.update_layout(**{ax_name: {"colorscale": colorscale, "cmin": cmin, "cmax": cmax}})

    def _handle_multicolor_scatter(self, marker, scatter_kwargs):

        if 'coloraxis' in marker:
            marker = marker.copy()
            coloraxis = marker['coloraxis']

            scatter_kwargs['hovertemplate'] = "x: %{x:.2f}<br>y: %{y:.2f}<br>" + coloraxis + ": %{marker.color:.2f}"
            marker['coloraxis'] = self._coloraxes[coloraxis]['ax_name']
        
        return marker

    def draw_line(self, x, y, name=None, line={}, **kwargs):
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
        self.add_trace({
            'type': 'scatter',
            'x': x,
            'y': y,
            'mode': mode,
            'name': name,
            'line': {k: v for k, v in line.items() if k != "opacity"},
            'opacity': opacity,
            **kwargs,
        })

    def draw_multicolor_line(self, *args, **kwargs):
        kwargs['marker_line_width'] = 0

        super().draw_multicolor_line(*args, **kwargs)

    def draw_area_line(self, x, y, name=None, line={}, text=None, **kwargs):

        chunk_x = x
        chunk_y = y
        chunk_spacing = line.get('width', 1) / 2

        self.add_trace({
            "type": "scatter",
            "mode": "lines",
            "x": [*chunk_x, *reversed(chunk_x)],
            "y": [*(chunk_y + chunk_spacing), *reversed(chunk_y - chunk_spacing)],
            "line": {"width": 0, "color": line.get('color')},
            #"showlegend": is_group_first and i_chunk == 0,
            "name": name,
            "legendgroup": name,
            "fill": "toself"
        })

    def draw_scatter(self, x, y, name=None, marker={}, **kwargs):
        self.draw_line(x, y, name, marker=marker, mode="markers", **kwargs)
    
    def draw_multicolor_scatter(self, *args, **kwargs):
        
        kwargs['marker'] = self._handle_multicolor_scatter(kwargs['marker'], kwargs)

        super().draw_multicolor_scatter(*args, **kwargs)

    def draw_line_3D(self, x, y, z, **kwargs):
        self.draw_line(x, y, type="scatter3d", z=z, **kwargs)

    def draw_multicolor_line_3D(self, x, y, z, **kwargs):
        kwargs['line'] = self._handle_multicolor_scatter(kwargs['line'], kwargs)

        super().draw_multicolor_line_3D(x, y, z, **kwargs)

    def draw_scatter_3D(self, *args, **kwargs):
        self.draw_line_3D(*args, mode="markers", **kwargs)
    
    def draw_multicolor_scatter_3D(self, *args, **kwargs):
        
        kwargs['marker'] = self._handle_multicolor_scatter(kwargs['marker'], kwargs)

        super().draw_multicolor_scatter_3D(*args, **kwargs)

    def draw_balls_3D(self, x, y, z, name=None, marker={}, **kwargs):

        style = {}
        for k in ("size", "color", "opacity"):
            val = marker.get(k)

            if isinstance(val, (str, int, float)):
                val = itertools.repeat(val)

            style[k] = val

        iterator = enumerate(zip(np.array(x), np.array(y), np.array(z), style["size"], style["color"], style["opacity"]))

        showlegend = True
        for i, (sp_x, sp_y, sp_z, sp_size, sp_color, sp_opacity) in iterator:
            self.draw_ball_3D(
                xyz=[sp_x, sp_y, sp_z], 
                size=sp_size, color=sp_color, opacity=sp_opacity,
                name=f"{name}_{i}",
                legendgroup=name, showlegend=showlegend
            )
            showlegend = False

        return

    def draw_ball_3D(self, xyz, size, color="gray", name=None, vertices=15, **kwargs):
        sphere(center=xyz, r=0.3, vertices=5)
        self.add_trace({
            'type': 'mesh3d',
            **{key: np.ravel(val) for key, val in sphere(center=xyz, r=size, vertices=vertices).items()},
            'alphahull': 0,
            'color': color,
            'showscale': False,
            'name': name,
            'meta': ['({:.2f}, {:.2f}, {:.2f})'.format(*xyz)],
            'hovertemplate': '%{meta[0]}',
            **kwargs
        })
    
    def draw_arrows_3D(self, xyz, dxyz, arrowhead_angle=20, arrowhead_scale=0.3, **kwargs):
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

    def draw_heatmap(self, values, x=None, y=None, name=None, zsmooth=False, coloraxis=None):

        self.add_trace({
            'type': 'heatmap', 'z': values,
            'x': x, 'y': y,
            'name': name,
            'zsmooth': zsmooth,
            # 'zmin': backend_info["cmin"],
            # 'zmax': backend_info["cmax"],
            # 'zmid': backend_info["cmid"],
            # 'colorscale': backend_info["colorscale"],
            # **kwargs
        })

    def draw_mesh_3D(self, vertices, faces, color=None, opacity=None, name=None, **kwargs):

        x, y, z = vertices.T
        I, J, K = faces.T

        self.add_trace(dict(
            type="mesh3d",
            x=x, y=y, z=z,
            i=I, j=J, k=K,
            color=color,
            opacity=opacity,
            name=name,
            showlegend=True,
            **kwargs
        ))

    def set_axis(self, axis, **kwargs):
        ax_name = f"{axis}axis"
        updates = {f"scene_{ax_name}": kwargs}
        if axis != "z":
            updates.update({ax_name: kwargs})
        self.update_layout(**updates)

    def set_axes_equal(self):
        self.update_layout(xaxis_scaleanchor="y", xaxis_scaleratio=1)
        self.update_layout(scene_aspectmode="data")


pio.templates["sisl"] = go.layout.Template(
    layout={
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        **{f"{ax}_{key}": val for ax, (key, val) in itertools.product(
            ("xaxis", "yaxis"),
            (("visible", True), ("showline", True), ("linewidth", 1), ("mirror", True),
             ("color", "black"), ("showgrid", False), ("gridcolor", "#ccc"), ("gridwidth", 1),
             ("zeroline", False), ("zerolinecolor", "#ccc"), ("zerolinewidth", 1),
             ("ticks", "outside"), ("ticklen", 5), ("ticksuffix", " "))
        )},
        "hovermode": "closest",
        "scene": {
            **{f"{ax}_{key}": val for ax, (key, val) in itertools.product(
                ("xaxis", "yaxis", "zaxis"),
                (("visible", True), ("showline", True), ("linewidth", 1), ("mirror", True),
                 ("color", "black"), ("showgrid",
                                      False), ("gridcolor", "#ccc"), ("gridwidth", 1),
                    ("zeroline", False), ("zerolinecolor",
                                          "#ccc"), ("zerolinewidth", 1),
                    ("ticks", "outside"), ("ticklen", 5), ("ticksuffix", " "))
            )},
        }
        #"editrevision": True
        #"title": {"xref": "paper", "x": 0.5, "text": "Whhhhhhhat up", "pad": {"b": 0}}
    },
)

pio.templates["sisl_dark"] = go.layout.Template(
    layout={
        "plot_bgcolor": "black",
        "paper_bgcolor": "black",
        **{f"{ax}_{key}": val for ax, (key, val) in itertools.product(
            ("xaxis", "yaxis"),
            (("visible", True), ("showline", True), ("linewidth", 1), ("mirror", True),
             ("color", "white"), ("showgrid",
                                  False), ("gridcolor", "#ccc"), ("gridwidth", 1),
             ("zeroline", False), ("zerolinecolor", "#ccc"), ("zerolinewidth", 1),
             ("ticks", "outside"), ("ticklen", 5), ("ticksuffix", " "))
        )},
        "font": {'color': 'white'},
        "hovermode": "closest",
        "scene": {
            **{f"{ax}_{key}": val for ax, (key, val) in itertools.product(
                ("xaxis", "yaxis", "zaxis"),
                (("visible", True), ("showline", True), ("linewidth", 1), ("mirror", True),
                 ("color", "white"), ("showgrid",
                                      False), ("gridcolor", "#ccc"), ("gridwidth", 1),
                    ("zeroline", False), ("zerolinecolor",
                                          "#ccc"), ("zerolinewidth", 1),
                    ("ticks", "outside"), ("ticklen", 5), ("ticksuffix", " "))
            )},
        }
        #"editrevision": True
        #"title": {"xref": "paper", "x": 0.5, "text": "Whhhhhhhat up", "pad": {"b": 0}}
    },
)

# This will be the default one for the sisl.viz.plotly module
pio.templates.default = "sisl"