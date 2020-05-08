'''

Tests specific functionality of the PDOS plot.

Different inputs are tested (siesta .PDOS and sisl Hamiltonian).

'''

from pandas import DataFrame
import numpy as np
from sisl.viz.plots.tests.get_files import from_files

import sisl
from sisl.viz.plots import PdosPlot

# ------------------------------------------------------------
#         Build a generic tester for the bands plot
# ------------------------------------------------------------

class PdosPlotTester:

    plot = None
    na = 0
    no = 0
    n_spin = 1
    species = []

    def test_dataframe(self):

        df = self.plot.df
        geom = self.plot.geom

        assert isinstance(df, DataFrame)
        assert isinstance(geom, sisl.Geometry)

        # Check if we read the correct number of atoms
        assert len(df["iAtom"].unique()) == self.na == geom.na
        # Check if we have the correct number of orbitals
        assert df.shape[0] == self.no == geom.no
        # Check that the species have been correctly read
        assert set(df["Species"].unique()) == set(self.species) == set([at.symbol for at in geom.atoms.atom])

    def test_splitDOS(self):

        split_DOS = self.plot.split_DOS

        unique_orbs = self.plot.df["Orbital name"].unique()

        expected_splits = {
            "species": (len(self.species), self.species[0]),
            "atoms": (self.na, 1),
            "orbitals":(len(unique_orbs), unique_orbs[0]),
            "spin": (self.n_spin, None)
        }

        print(self.plot.split_DOS().setting("requests"))

        # Test all splittings
        for on, (n, toggle_val) in expected_splits.items():
            err_message = f'Error splitting DOS based on {on}'
            assert len(split_DOS(on=on).data) == n, err_message
            if toggle_val is not None:
                assert len(split_DOS(on=on, only=[toggle_val]).data) == 1 , err_message
                assert len(split_DOS(on=on, exclude=[toggle_val]).data) == n - 1, err_message
    
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
#       Test the pdos plot reading from siesta .PDOS
# ------------------------------------------------------------

pdos_file = from_files("SrTiO3.PDOS")

class TestPDOSSiestaOutput(PdosPlotTester):

    plot = PdosPlot(pdos_file=pdos_file)
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
    