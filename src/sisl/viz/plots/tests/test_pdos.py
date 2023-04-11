# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

Tests specific functionality of the PDOS plot.

Different inputs are tested (siesta .PDOS and sisl Hamiltonian).

"""
from functools import partial
import pytest

import sisl
from sisl.viz.plots.tests.conftest import _TestPlot


pytestmark = [pytest.mark.viz, pytest.mark.plotly]


@pytest.fixture(params=[True, False], ids=["inplace_split", "method_splitting"])
def inplace_split(request):
    return request.param


class TestPdosPlot(_TestPlot):

    _required_attrs = [
        "na", # int, number of atoms in the geometry
        "no", # int, number of orbitals in the geometry
        "n_spin", # int, number of spin components for the PDOS
        "species", # array-like of str. The names of the species.
    ]

    @pytest.fixture(scope="class", params=[
        # From a siesta PDOS file
        "siesta_PDOS_file_unpolarized", "siesta_PDOS_file_polarized", "siesta_PDOS_file_noncollinear",
        # From a sisl hamiltonian
        "sisl_H_unpolarized", "sisl_H_polarized", "sisl_H_noncolinear", "sisl_H_spinorbit",
        # From a WFSX file and the overlap matrix
        "wfsx_file"

    ])
    def init_func_and_attrs(self, request, siesta_test_files):
        name = request.param

        if name.startswith("siesta_PDOS_file"):

            spin_type = name.split("_")[-1]

            n_spin, filename = {
                "unpolarized": (1, "SrTiO3.PDOS"),
                "polarized": (2, "SrTiO3_polarized.PDOS"),
                "noncollinear": (4, "SrTiO3_noncollinear.PDOS")
            }[spin_type]

            init_func = sisl.get_sile(siesta_test_files(filename)).plot
            attrs = {
                "na": 5,
                "no": 72,
                "n_spin": n_spin,
                "species": ('Sr', 'Ti', 'O')
            }
        elif name.startswith("sisl_H"):
            gr = sisl.geom.graphene()
            H = sisl.Hamiltonian(gr)
            H.construct([(0.1, 1.44), (0, -2.7)])

            spin_type = name.split("_")[-1]

            n_spin, H = {
                "unpolarized": (1, H),
                "polarized": (2, H.transform(spin=sisl.Spin.POLARIZED)),
                "noncolinear": (4, H.transform(spin=sisl.Spin.NONCOLINEAR)),
                "spinorbit": (4, H.transform(spin=sisl.Spin.SPINORBIT))
            }[spin_type]

            init_func = partial(H.plot.pdos, Erange=[-5, 5])
            attrs = {
                "na": 2,
                "no": 2,
                "n_spin": n_spin,
                "species": ('C',)
            }
        elif name == "wfsx_file":
            # From a siesta .WFSX file
            # Since there is no hamiltonian for bi2se3_3ql.fdf, we create a dummy one
            wfsx = sisl.get_sile(siesta_test_files("bi2se3_3ql.bands.WFSX"))

            geometry = sisl.get_sile(siesta_test_files("bi2se3_3ql.fdf")).read_geometry()
            geometry = sisl.Geometry(geometry.xyz, atoms=wfsx.read_basis())

            H = sisl.Hamiltonian(geometry, dim=4)

            init_func = partial(
                H.plot.pdos, wfsx_file=wfsx,
                entry_points_order=["wfsx file"]
            )

            attrs = {
                "na": 15,
                "no": 195,
                "n_spin": 4,
                "species": ('Bi', 'Se')
            }

        return init_func, attrs

    @pytest.fixture(scope="class", params=[None, *sisl.viz.PdosPlot.get_class_param("backend").options])
    def backend(self, request):
        return request.param

    def test_dataarray(self, plot, test_attrs):
        pytest.importorskip("xarray")
        from xarray import DataArray

        PDOS = plot.PDOS
        geom = plot.geometry

        assert isinstance(PDOS, DataArray)
        assert isinstance(geom, sisl.Geometry)

        # Check if we have the correct number of orbitals
        assert len(PDOS.orb) == test_attrs["no"] == geom.no

    def test_request_PDOS(self, plot):
        total_DOS = plot._get_request_PDOS({})

        assert total_DOS.ndim == 1
        assert total_DOS.shape == (plot.PDOS.E.shape)

    def test_splitDOS(self, plot, test_attrs, inplace_split):
        if inplace_split:
            def split_DOS(on, **kwargs):
                return plot.update_settings(requests=[{"split_on": on, **kwargs}])
        else:
            split_DOS = plot.split_DOS

        unique_orbs = plot.get_param('requests')['orbitals'].options

        expected_splits = {
            "species": (len(test_attrs["species"]), test_attrs["species"][0]),
            "atoms": (test_attrs["na"], 1),
            "orbitals": (len(unique_orbs), unique_orbs[0]),
            "spin": (test_attrs["n_spin"], None),
        }

        # Test all splittings
        for on, (n, toggle_val) in expected_splits.items():
            err_message = f'Error splitting DOS based on {on}'
            assert len(split_DOS(on=on)._for_backend["PDOS_values"]) == n, err_message
            if toggle_val is not None and not inplace_split:
                assert len(split_DOS(on=on, only=[toggle_val])._for_backend["PDOS_values"]) == 1, err_message
                assert len(split_DOS(on=on, exclude=[toggle_val])._for_backend["PDOS_values"]) == n - 1, err_message

    def test_composite_splitting(self, plot, inplace_split):

        if inplace_split:
            def split_DOS(on, **kwargs):
                return plot.update_settings(requests=[{"split_on": on, **kwargs}])
        else:
            split_DOS = plot.split_DOS

        split_DOS(on="species+orbitals", name="This is $species")

        first_trace_name = list(plot._for_backend["PDOS_values"].keys())[0]
        assert "This is " in first_trace_name, "Composite splitting not working"
        assert "species" not in first_trace_name, "Name templating not working in composite splitting"
        assert "orbitals=" in first_trace_name, "Name templating not working in composite splitting"

    @pytest.mark.parametrize("request_atoms", [0, {"index": 0}])
    def test_request_splitting(self, plot, inplace_split, request_atoms):

        # Here we are just checking that, when splitting a request
        # the plot understands that it has constrains
        plot.update_settings(requests=[{"atoms": request_atoms}])
        prev_len = len(plot._for_backend["PDOS_values"])

        # Even if there are more atoms, the plot should understand
        # that it is constrained to the values of the current request

        if inplace_split:
            plot.update_settings(requests=[{"atoms": request_atoms, "split_on": "atoms"}])
        else:
            plot.split_requests(0, on="atoms")

        assert len(plot._for_backend["PDOS_values"]) == prev_len

    def test_request_management(self, plot, test_attrs):

        plot.update_settings(requests=[])
        assert len(plot._for_backend["PDOS_values"]) == 0

        sel_species = test_attrs["species"][0]
        plot.add_request({"species": [sel_species]})
        assert len(plot._for_backend["PDOS_values"]) == 1

        # Try to split this request in multiple ones
        plot.split_requests(0, on="orbitals")
        # Get the number of orbitals with unique names
        species_no = len(set(orb.name() for orb in plot.geometry.atoms[sel_species].orbitals))
        assert len(plot._for_backend["PDOS_values"]) == species_no

        # Then try to merge
        # if species_no >= 2:
        #     plot.merge_requests(species_no - 1, species_no - 2)
        #     assert len(plot.data) == species_no - 1

        # And try to remove one request
        prev = len(plot._for_backend["PDOS_values"])
        assert len(plot.remove_requests(0)._for_backend["PDOS_values"]) == prev - 1
