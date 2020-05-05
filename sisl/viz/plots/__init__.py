from .bands import BandsPlot
from .fatbands import FatbandsPlot
from .pdos import PdosPlot
from .ldos import LDOSmap
from .bond_length import BondLengthMap
from .forces import ForcesPlot
from .grid import GridPlot
from .geometry import GeometryPlot

from .._user_customs import get_user_plots as _get_user_plots

_user_plots = _get_user_plots()

for PlotClass in _user_plots:
    locals()[PlotClass.__name__] = PlotClass