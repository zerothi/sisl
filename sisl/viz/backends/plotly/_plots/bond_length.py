from ....plots import BondLengthMap
from .geometry import PlotlyGeometryBackend
from ...templates import BondLengthMapBackend


class PlotlyBondLengthMapBackend(PlotlyGeometryBackend, BondLengthMapBackend):
    
    def draw_2D(self, backend_info, **kwargs):
        super().draw_2D(backend_info, **kwargs)
        self._setup_coloraxis(backend_info)
    
    def draw_3D(self, backend_info, **kwargs):
        super().draw_3D(backend_info, **kwargs)
        self._setup_coloraxis(backend_info)

    def _setup_coloraxis(self, backend_info):
        if "bonds_coloraxis" in backend_info:
            self.update_layout(coloraxis=backend_info["bonds_coloraxis"])

        self.update_layout(legend_orientation='h')

BondLengthMap._backends.register("plotly", PlotlyBondLengthMapBackend)