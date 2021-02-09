"""

Tests specific functionality of the PDOS plot.

Different inputs are tested (siesta .PDOS and sisl Hamiltonian).

"""
import os.path as osp
from functools import partial
import pytest
from xarray import DataArray
import numpy as np

import sisl
from sisl.viz.plotly import PdosPlot
from sisl.viz.plotly.plots.tests.conftest import PlotTester


pytestmark = [pytest.mark.viz, pytest.mark.plotly]
_dir = osp.join('sisl', 'io', 'siesta')


@pytest.fixture(params=[True, False], ids=["inplace_split", "method_splitting"])
def inplace_split(request):
    return request.param


class PdosPlotTester(PlotTester):

    _required_attrs = [
        "na", # int, number of atoms in the geometry
        "no", # int, number of orbitals in the geometry
        "n_spin", # int, number of spin components for the PDOS
        "species", # array-like of str. The names of the species.
    ]

    def test_dataarray(self):

        PDOS = self.plot.PDOS
        geom = self.plot.geometry

        assert isinstance(PDOS, DataArray)
        assert isinstance(geom, sisl.Geometry)

        # Check if we have the correct number of orbitals
        assert len(PDOS.orb) == self.no == geom.no

    def test_splitDOS(self, inplace_split):

        if inplace_split:
            def split_DOS(on, **kwargs):
                return self.plot.update_settings(requests=[{"split_on": on, **kwargs}])
        else:
            split_DOS = self.plot.split_DOS

        unique_orbs = self.plot.get_param('requests')['orbitals'].options

        expected_splits = {
            "species": (len(self.species), self.species[0]),
            "atoms": (self.na, 1),
            "orbitals": (len(unique_orbs), unique_orbs[0]),
            "spin": (self.n_spin, None),
        }

        # Test all splittings
        for on, (n, toggle_val) in expected_splits.items():
            err_message = f'Error splitting DOS based on {on}'
            assert len(split_DOS(on=on).data) == n, err_message
            if toggle_val is not None and not inplace_split:
                assert len(split_DOS(on=on, only=[toggle_val]).data) == 1, err_message
                assert len(split_DOS(on=on, exclude=[toggle_val]).data) == n - 1, err_message

    def test_composite_splitting(self, inplace_split):

        if inplace_split:
            def split_DOS(on, **kwargs):
                return self.plot.update_settings(requests=[{"split_on": on, **kwargs}])
        else:
            split_DOS = self.plot.split_DOS

        split_DOS(on="species+orbitals", name="This is $species")

        first_trace = self.plot.data[0]
        assert "This is " in first_trace.name, "Composite splitting not working"
        assert "species" not in first_trace.name, "Name templating not working in composite splitting"
        assert "orbitals=" in first_trace.name, "Name templating not working in composite splitting"

    def test_request_splitting(self, inplace_split):

        # Here we are just checking that, when splitting a request
        # the plot understands that it has constrains
        self.plot.update_settings(requests=[{"atoms": 0}])
        prev_len = len(self.plot.data)

        # Even if there are more atoms, the plot should understand
        # that it is constrained to the values of the current request

        if inplace_split:
            self.plot.update_settings(requests=[{"atoms": 0, "split_on": "atoms"}])
        else:
            self.plot.split_requests(0, on="atoms")

        assert len(self.plot.data) == prev_len

    def test_request_management(self):

        plot = self.plot

        plot.update_settings(requests=[])
        assert len(plot.data) == 0

        sel_species = self.species[0]
        plot.add_request({"species": [sel_species]})
        assert len(plot.data) == 1

        # Try to split this request in multiple ones
        plot.split_requests(0, on="orbitals")
        species_no = len(self.plot.geometry.atoms[sel_species].orbitals)
        assert len(plot.data) == species_no

        # Then try to merge
        # if species_no >= 2:
        #     plot.merge_requests(species_no - 1, species_no - 2)
        #     assert len(plot.data) == species_no - 1

        # And try to remove one request
        prev = len(plot.data)
        assert len(plot.remove_requests(0).data) == prev - 1

pdos_plots = {}

# ---- For a siesta PDOS file

pdos_plots["siesta_PDOS_file"] = {
    "plot_file": osp.join(_dir, "SrTiO3.PDOS"),
    "na": 5,
    "no": 72,
    "n_spin": 1,
    "species": ('Sr', 'Ti', 'O')
}

pdos_plots["siesta_PDOS_file_polarized"] = {
    "plot_file": osp.join(_dir, "SrTiO3_polarized.PDOS"),
    "na": 5,
    "no": 72,
    "n_spin": 2,
    "species": ('Sr', 'Ti', 'O')
}

pdos_plots["siesta_PDOS_file_noncollinear"] = {
    "plot_file": osp.join(_dir, "SrTiO3_noncollinear.PDOS"),
    "na": 5,
    "no": 72,
    "n_spin": 4,
    "species": ('Sr', 'Ti', 'O')
}

# ---- From a hamiltonian generated in sisl


gr = sisl.geom.graphene()
H = sisl.Hamiltonian(gr)
H.construct([(0.1, 1.44), (0, -2.7)])

pdos_plots["sisl_H"] = {
    "init_func": partial(H.plot.pdos, Erange=[-5, 5]),
    "na": 2,
    "no": 2,
    "n_spin": 1,
    "species": ('C',)
}


class TestPDOSPlot(PdosPlotTester):
    run_for = pdos_plots
