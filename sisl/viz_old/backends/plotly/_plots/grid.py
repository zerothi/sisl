# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import plotly.graph_objects as go

from ....plots.grid import GridPlot
from ..backend import PlotlyBackend
from ...templates import GridBackend


class PlotlyGridBackend(PlotlyBackend, GridBackend):

    def draw_1D(self, backend_info, **kwargs):
        self.figure.layout.yaxis.scaleanchor = None
        self.figure.layout.yaxis.scaleratio = None

        super().draw_1D(backend_info, **kwargs)

        self.update_layout(**{f"{k}_title": v for k, v in backend_info["axes_titles"].items()})

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

        self.update_layout(**{f"{k}_title": v for k, v in backend_info["axes_titles"].items()})

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
                showlegend=True,
                **kwargs
            ))

        self.layout.scene = {'aspectmode': 'data'}
        self.update_layout(**{f"scene_{k}_title": v for k, v in backend_info["axes_titles"].items()})

    def _after_get_figure(self):
        self.update_layout(legend_orientation='h')

GridPlot.backends.register("plotly", PlotlyGridBackend)
