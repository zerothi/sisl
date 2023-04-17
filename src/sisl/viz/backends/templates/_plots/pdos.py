# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from abc import abstractmethod
from ..backend import Backend

from ....plots import PdosPlot


class PdosBackend(Backend):
    """It draws the PDOS values provided by a `PdosPlot`

    The workflow implemented by it is as follows:
        for line in PDOS_lines:
            `self.draw_PDOS_line()`, generic implementation that calls `self._draw_line`.
    """

    def draw(self, backend_info):
        self.draw_PDOS_lines(backend_info)

    def draw_PDOS_lines(self, backend_info):
        lines = backend_info["PDOS_values"]
        Es = backend_info["Es"]

        for name, values in lines.items():
            self.draw_PDOS_line(Es, values, backend_info["request_metadata"][name], name)

    def draw_PDOS_line(self, Es, values, request_metadata, name):
        line_style = request_metadata["style"]["line"]

        self.draw_line(x=values, y=Es, name=name, line=line_style)

PdosPlot.backends.register_template(PdosBackend)
