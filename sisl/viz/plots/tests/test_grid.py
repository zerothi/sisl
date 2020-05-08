'''

Tests specific functionality of the grid plot.

Different inputs are tested (siesta .RHO and sisl Hamiltonian).

'''

import numpy as np

import sisl
from sisl.viz.plots import GridPlot
from sisl.viz import Animation

# ------------------------------------------------------------
#         Build a generic tester for the bands plot
# ------------------------------------------------------------

class GridPlotTester:

    plot = None
    grid_shape = []

    def test_grid(self):

        grid = self.plot.grid

        assert isinstance(grid, sisl.Grid)
        assert grid.shape == self.grid_shape

    def test_scan(self):

        scanned = self.plot.scan(0, steps=2)

        assert isinstance(scanned, Animation)
        assert scanned.frames
    
    def test_request_management(self):

        plot = self.plot

        plot.update_settings(requests=[])
        assert len(plot.data) == 0

        sel_species = self.species[0]
        plot.add_request({"species": [sel_species]})
        assert len(plot.data) == 1

        # Try to split this request in multiple ones
        plot.split_requests(0, on="orbitals")
        species_no = len(plot.df[ plot.df["Species"] == sel_species ]["Orbital name"].unique())
        assert len(plot.data) == species_no

        # Then try to merge
        if species_no >= 2:
            plot.merge_requests(species_no - 1, species_no - 2)
            assert len(plot.data) == species_no - 1
        
        # And try to remove one request
        prev = len(plot.data)
        assert len(plot.remove_requests(0).data) == prev - 1

# ------------------------------------------------------------
#       Test the grid plot reading from siesta .RHO
# ------------------------------------------------------------

pdos_file = "/home/pfebrer/webDevelopement/sislGUI/sisl/sisl/viz/Tutorials/files/SrTiO3.RHO"

class TestPDOSSiestaOutput(PdosPlotTester):

    plot = GridPlot(grid_file=grid_file)
    na = 5
    no = 72
    n_spin = 1
    species = ('Sr', 'Ti', 'O')

# ------------------------------------------------------------
#     Test the PDOS plot reading from a sisl Hamiltonian
# ------------------------------------------------------------

gr = sisl.geom.graphene()
H = sisl.Hamiltonian(gr)
H.construct([(0.1, 1.44), (0, -2.7)])

class TestPDOSSislHamiltonian(PdosPlotTester):

    plot = PdosPlot(H=H, Erange=[-5,5])
    na = 2
    no = 2
    n_spin = 1
    species = ('C',)
    