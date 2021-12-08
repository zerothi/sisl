# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from ....plots import GeometryPlot
from ..backend import PlotlyBackend
from ...templates import GeometryBackend


class PlotlyGeometryBackend(PlotlyBackend, GeometryBackend):

    _layout_defaults = {
        'xaxis_showgrid': False,
        'xaxis_zeroline': False,
        'yaxis_showgrid': False,
        'yaxis_zeroline': False,
    }

    def draw_1D(self, backend_info, **kwargs):
        super().draw_1D(backend_info, **kwargs)

        self.update_layout(**{f"{k}_title": v for k, v in backend_info["axes_titles"].items()})

    def draw_2D(self, backend_info, **kwargs):
        super().draw_2D(backend_info, **kwargs)

        self.update_layout(**{f"{k}_title": v for k, v in backend_info["axes_titles"].items()})

        self.layout.yaxis.scaleanchor = "x"
        self.layout.yaxis.scaleratio = 1

    def draw_3D(self, backend_info):
        self._one_atom_trace = False

        super().draw_3D(backend_info)

        self.layout.scene.aspectmode = 'data'

    def _draw_bonds_3D(self, *args, line={}, bonds_labels=None, x_labels=None, y_labels=None, z_labels=None, **kwargs):
        if "hoverinfo" not in kwargs:
            kwargs["hoverinfo"] = None
        super()._draw_bonds_3D(*args, line=line, **kwargs)

        if bonds_labels:
            self.add_trace({
                'type': 'scatter3d', 'mode': 'markers',
                'x': x_labels, 'y': y_labels, 'z': z_labels,
                'text': bonds_labels, 'hoverinfo': 'text',
                'marker': {'size': line["width"]*3, "color": "rgba(255,255,255,0)"},
                "showlegend": False
            })

    def _draw_single_atom_3D(self, xyz, size, color="gray", name=None, group="Atoms", vertices=15, **kwargs):

        self.add_trace({
            'type': 'mesh3d',
            **{key: np.ravel(val) for key, val in GeometryPlot._sphere(xyz, size, vertices=vertices).items()},
            'showlegend': not self._one_atom_trace,
            'alphahull': 0,
            'color': color,
            'showscale': False,
            'legendgroup': group,
            'name': name,
            'meta': ['({:.2f}, {:.2f}, {:.2f})'.format(*xyz)],
            'hovertemplate': '%{meta[0]}',
            **kwargs
        })

        self._one_atom_trace = True

    def _draw_single_bond_3D(self, *args, group=None, showlegend=False, line_kwargs={}, **kwargs):
        kwargs["legendgroup"] = group
        kwargs["showlegend"] = showlegend
        super()._draw_single_bond_3D(*args, **kwargs)

    def _draw_cell_3D_axes(self, cell, geometry, **kwargs):
        return super()._draw_cell_3D_axes(cell, geometry, mode="lines+markers", **kwargs)

GeometryPlot.backends.register("plotly", PlotlyGeometryBackend)
