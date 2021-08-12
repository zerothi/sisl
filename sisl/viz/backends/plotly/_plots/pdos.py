from ....plots import PdosPlot
from ..backend import PlotlyBackend
from ...templates import PdosBackend


class PlotlyPDOSBackend(PlotlyBackend, PdosBackend):

    _layout_defaults = {
        'xaxis_title': 'Density of states [1/eV]',
        'xaxis_mirror': True,
        'yaxis_mirror': True,
        'yaxis_title': 'Energy [eV]',
        'showlegend': True
    }

    def draw_PDOS_lines(self, drawer_info):
        super().draw_PDOS_lines(drawer_info)

        Es = drawer_info["Es"]
        self.update_layout(yaxis_range=[min(Es), max(Es)])


PdosPlot._backends.register("plotly", PlotlyPDOSBackend)