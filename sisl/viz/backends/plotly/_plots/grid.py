import plotly.graph_objects as go

from ....plots import GridPlot
from ..backend import PlotlyBackend
from ...templates import GridBackend


class PlotlyGridBackend(PlotlyBackend, GridBackend):

    def draw_1D(self, backend_info, **kwargs):

        self.add_trace({
            'type': 'scatter',
            'mode': 'lines',
            'y': backend_info["values"],
            'x': backend_info["ax_range"],
            'name': backend_info["name"],
            **kwargs
        })

        axes_titles = {'xaxis_title': f'{("X","Y", "Z")[backend_info["ax"]]} axis [Ang]', 'yaxis_title': 'Values'}

        self.update_layout(**axes_titles)

    def draw_2D(self, backend_info, **kwargs):

        # Draw the heatmap
        self.add_trace({
            'type': 'heatmap',
            'name': backend_info["name"],
            'z': backend_info["values"],
            'x': backend_info["x"],
            'y': backend_info["y"],
            'zsmooth': backend_info["zsmooth"],
            'zmin': backend_info["cmin"],
            'zmax': backend_info["cmax"],
            'zmid': backend_info["cmid"],
            'colorscale': backend_info["colorscale"],
            **kwargs
        })

        # Draw the isocontours
        for contour in backend_info["contours"]:
            self.add_scatter(
                x=contour["x"], y=contour["y"],
                marker_color=contour["color"], line_color=contour["color"],
                opacity=contour["opacity"],
                name=contour["name"]
            )

        axes_titles = {f'{ax}_title': f'{("X","Y", "Z")[backend_info[ax]]} axis [Ang]' for ax in ("xaxis", "yaxis")}

        self.update_layout(**axes_titles)

        self.figure.layout.yaxis.scaleanchor = "x"
        self.figure.layout.yaxis.scaleratio = 1

    def draw_3D(self, backend_info, **kwargs):

        for isosurf in backend_info["isosurfaces"]:

            x, y, z = isosurf["vertices"].T
            I, J, K = isosurf["faces"].T

            self.add_trace(go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                color=isosurf["color"],
                opacity=isosurf["opacity"],
                name=isosurf["name"],
                **kwargs
            ))

        self.layout.scene = {'aspectmode': 'data'}

    def _after_get_figure(self):
        self.update_layout(legend_orientation='h')

GridPlot._backends.register("plotly", PlotlyGridBackend)
