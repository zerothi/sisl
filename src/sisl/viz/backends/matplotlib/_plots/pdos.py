# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ....plots import PdosPlot
from ..backend import MatplotlibBackend
from ...templates import PdosBackend


class MatplotlibPDOSBackend(MatplotlibBackend, PdosBackend):

    _axes_defaults = {
        'xlabel': 'Density of states [1/eV]',
        'ylabel': 'Energy [eV]'
    }

    def draw_PDOS_lines(self, backend_info):
        super().draw_PDOS_lines(backend_info)

        Es = backend_info["Es"]
        self.axes.set_ylim(min(Es), max(Es))


PdosPlot.backends.register("matplotlib", MatplotlibPDOSBackend)
