from abc import abstractmethod
from ..backend import Backend

from ....plots import PdosPlot

class PdosBackend(Backend):
    
    def draw_PDOS_lines(self, drawer_info):

        lines = drawer_info["PDOS_values"]
        Es = drawer_info["Es"]

        for name, values in lines.items():
            self.draw_PDOS_line(Es, values, drawer_info["request_metadata"][name], name)

    def draw_PDOS_line(self, Es, values, request_metadata, name):

        line_style = request_metadata["style"]["line"]

        self.draw_line(x=values, y=Es, name=name, line=line_style)

PdosPlot._backends.register_template(PdosBackend)