'''

Tests specific functionality of the bands plot.

Different inputs are tested (siesta .bands and sisl Hamiltonian).

'''

from pandas import DataFrame
import numpy as np

import sisl
from sisl.viz.plots import PdosPlot

# ------------------------------------------------------------
#         Build a generic tester for the bands plot
# ------------------------------------------------------------

class PdosPlotTester:
    pass


# ------------------------------------------------------------
#       Test the pdos plot reading from siesta .PDOS
# ------------------------------------------------------------

pdos_file = "/home/pfebrer/webDevelopement/sislGUI/sisl/sisl/viz/Tutorials/files/SrTiO3.PDOS"

# class TestBandsSiestaOutput(BandsPlotTester):

#     plot = BandsPlot(bands_file=bands_file)
#     bands_shape = (150, 1, 72)
#     ticktext = ('Gamma', 'X', 'M', 'Gamma', 'R', 'X')
#     tickvals = [0.0, 0.429132, 0.858265, 1.465149, 2.208428, 2.815313]
#     gap = 1.677

# ------------------------------------------------------------
#     Test the bands plot reading from a sisl Hamiltonian
# ------------------------------------------------------------

# gr = sisl.geom.graphene()
# H = sisl.Hamiltonian(gr)
# H.construct([(0.1, 1.44), (0, -2.7)])
# bz = sisl.BandStructure(H, [[0,0,0], [2/3, 1/3, 0], [1/2, 0, 0]], 9, ["Gamma", "M", "K"])

# class TestBandsSislHamiltonian(BandsPlotTester):

#     plot = BandsPlot(H=H, band_structure=bz)
#     bands_shape = (9, 1, 2)
#     gap = 0
#     ticktext = ["Gamma", "M", "K"]
#     tickvals = [0., 1.70309799, 2.55464699]
    