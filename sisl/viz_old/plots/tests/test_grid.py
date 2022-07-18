# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

Tests specific functionality of the grid plot.

Different inputs are tested (siesta .RHO and sisl Hamiltonian).

"""
from typing import ChainMap

import pytest
import numpy as np
import os.path as osp

import sisl
from sisl.viz import GridPlot, Animation
from sisl.viz.plots.tests.conftest import _TestPlot


pytestmark = [pytest.mark.viz, pytest.mark.plotly]

try:
    import skimage
    skip_skimage = pytest.mark.skipif(False, reason="scikit-image (skimage) not available")
except ImportError:
    skip_skimage = pytest.mark.skipif(True, reason="scikit-image (skimage) not available")

try:
    import plotly
    skip_plotly = pytest.mark.skipif(False, reason="plotly not available")
except ImportError:
    skip_plotly = pytest.mark.skipif(True, reason="plotly not available")


class TestGridPlot(_TestPlot):

    _required_attrs = [
        "grid_shape" # Tuple indicating the grid shape
    ]

    @pytest.fixture(scope="class", params=["siesta_RHO", "VASP CHGCAR", "complex_grid"])
    def init_func_and_attrs(self, request, siesta_test_files, vasp_test_files):
        name = request.param

        if name == "siesta_RHO":
            init_func = sisl.get_sile(siesta_test_files("SrTiO3.RHO")).plot
            attrs = {"grid_shape": (48, 48, 48)}
        if name == "VASP CHGCAR":
            init_func = sisl.get_sile(vasp_test_files(osp.join("graphene", "CHGCAR"))).plot.grid
            attrs = {"grid_shape": (24, 24, 100)}
        elif name == "complex_grid":
            complex_grid_shape = (8, 10, 10)
            np.random.seed(1)
            values = np.random.random(complex_grid_shape).astype(np.complex128) + np.random.random(complex_grid_shape) * 1j
            complex_grid = sisl.Grid(complex_grid_shape, sc=1)
            complex_grid.grid = values

            init_func = complex_grid.plot
            attrs = {"grid_shape": complex_grid_shape}

        return init_func, attrs

    @pytest.fixture(scope="class", params=[None, *sisl.viz.GridPlot.get_class_param("backend").options])
    def backend(self, request):
        return request.param

    @pytest.fixture(scope="function", params=["imag", "mod", "rad_phase", "deg_phase", "real"])
    def grid_representation(self, request, plot):
        """
        Fixture that returns all possible grid representations of the grid that we are testing.

        To be used only in methods of test classes. 

        Returns
        -----------
        sisl.Grid:
            a new grid that contains the specific representation of the plot's grid
        str:
            the name of the representation that we are returning
        """

        representation = request.param

        # Copy the plot's grid
        new_grid = plot.grid.copy()

        # Substitute the values by the appropiate representations
        new_grid.grid = GridPlot._get_representation(new_grid, representation)

        return (new_grid, representation)

    @pytest.fixture(scope="class", params=[1, 2, 3])
    def ndim(self, request, backend):
        if backend == "matplotlib" and request.param == 3:
            pytest.skip("Matplotlib 3D representations are not available yet")
        if request.param > 1:
            pytest.importorskip("skimage")
        return request.param

    @pytest.fixture(scope="class", params=["cartesian", "lattice"])
    def axes(self, request, ndim):
        if request.param == "cartesian":
            return {1: "x", 2: "xy", 3: "xyz"}[ndim]
        elif request.param == "lattice":
            # We don't test the 3D case because it doesn't work
            if ndim == 3:
                pytest.skip("3D view doesn't support fractional coordinates")
            return {1: "a", 2: "ab"}[ndim]

    @pytest.fixture(scope="class")
    def lattice_axes(self, ndim):
        return {1: [0], 2: [0, 1], 3: [0, 1, 2]}[ndim]

    @pytest.fixture(params=["Unit cell", "supercell"])
    def nsc(self, request):
        return {"Unit cell": [1, 1, 1], "supercell": [2, 2, 2]}[request.param]

    def _get_plotted_values(self, plot):

        ndim = len(plot.get_setting("axes"))
        if ndim < 3:
            values = plot._for_backend["values"]
            if ndim == 2:
                values = values.T
            return values
        elif ndim == 3:
            return plot._for_backend["isosurfaces"][0]["vertices"]

    def test_values(self, plot, ndim, axes, nsc):
        plot.update_settings(axes=axes, nsc=nsc)

        if ndim < 3:
            assert "values" in plot._for_backend
            assert plot._for_backend["values"].ndim == ndim

        elif ndim == 3:
            plot.update_settings(isos=[])
            assert "isosurfaces" in plot._for_backend

            assert len(plot._for_backend["isosurfaces"]) == 2

            for iso in plot._for_backend["isosurfaces"]:
                assert set(("vertices", "faces", "color", "opacity", "name")) == set(iso)
                assert iso["vertices"].shape[1] == 3
                assert iso["faces"].shape[1] == 3

    def test_ax_ranges(self, plot, axes, ndim, nsc):
        if ndim == 3:
            return

        plot.update_settings(axes=axes, nsc=nsc)
        values = plot._for_backend["values"]

        if ndim == 1:
            assert values.shape == plot._for_backend["ax_range"].shape
        if ndim == 2:
            assert (values.shape[1], ) == plot._for_backend["x"].shape
            assert (values.shape[0], ) == plot._for_backend["y"].shape

        plot.update_settings(nsc=[1, 1, 1])

    def test_representation(self, plot, lattice_axes, grid_representation):

        kwargs = {"isos": [], "reduce_method": "average"}

        ndim = len(lattice_axes)
        if ndim == 3:
            kwargs["isos"] = [{"frac": 0.5}]

        new_grid, representation = grid_representation

        if new_grid.grid.min() == new_grid.grid.max() and ndim == 3:
            return

        plot.update_settings(axes=lattice_axes, represent=representation, nsc=[1, 1, 1], **kwargs)
        new_plot = new_grid.plot(**ChainMap(plot.settings, dict(axes=lattice_axes, represent="real", grid_file=None)))

        assert np.allclose(
            self._get_plotted_values(plot), self._get_plotted_values(plot=new_plot)
        ), f"'{representation}' representation of the {ndim}D plot is not correct"

    def test_grid(self, plot, test_attrs):
        grid = plot.grid

        assert isinstance(grid, sisl.Grid)
        assert grid.shape == test_attrs["grid_shape"]

    @skip_skimage
    @skip_plotly
    def test_scan(self, plot, backend):
        import plotly.graph_objs as go
        plot.update_settings(axes="xy")
        # AS_IS SCAN
        # Provide number of steps
        if backend == "plotly":
            scanned = plot.scan("z", num=2, mode="as_is")
            assert isinstance(scanned, Animation)
            assert len(scanned.frames) == 2

            # Provide step in Ang
            step = plot.grid.cell[2, 2]/2
            scanned = plot.scan(along="z", step=step, mode="as_is")
            assert len(scanned.frames) == 2

            # Provide breakpoints
            breakpoints = [plot.grid.cell[2, 2]*frac for frac in [1/3, 2/3, 3/3]]
            scanned = plot.scan(along="z", breakpoints=breakpoints, mode="as_is")
            assert len(scanned.frames) == 2

            # Check that it doesn't accept step and breakpoints at the same time
            with pytest.raises(ValueError):
                plot.scan(along="z", step=4.5, breakpoints=breakpoints, mode="as_is")

        # 3D SCAN
        breakpoints = [plot.grid.cell[0, 0]*frac for frac in [1/3, 2/3, 3/3]]
        scanned = plot.scan(along="z", mode="moving_slice", breakpoints=breakpoints)

        assert isinstance(scanned, go.Figure)
        assert len(scanned.frames) == 3 # One cross section for each breakpoint

    @skip_skimage
    def test_supercell(self, plot):
        plot.update_settings(axes=[0, 1], interp=[1, 1, 1], nsc=[1, 1, 1])

        # Check that the shapes for the unit cell are right
        uc_shape = plot._for_backend["values"].shape
        assert uc_shape == (plot.grid.shape[1], plot.grid.shape[0])

        # Check that the supercell is displayed
        plot.update_settings(nsc=[2, 1, 1])
        sc_shape = plot._for_backend["values"].shape
        assert sc_shape[1] == 2*uc_shape[1]
        assert sc_shape[0] == uc_shape[0]

        plot.update_settings(nsc=[1, 1, 1])

    @pytest.mark.parametrize("reduce_method", ["sum", "average"])
    def test_reduce_method(self, plot, reduce_method, lattice_axes, grid_representation):
        new_grid, representation = grid_representation

        # If this is a 3D plot, no dimension is reduced, therefore it makes no sense
        if len(lattice_axes) == 3:
            return

        numpy_func = getattr(np, reduce_method)

        plot.update_settings(axes=lattice_axes, reduce_method=reduce_method, represent=representation, transforms=[])

        assert np.allclose(
            self._get_plotted_values(plot), numpy_func(new_grid.grid, axis=tuple(ax for ax in [0, 1, 2] if ax not in lattice_axes))
        )

    def test_transforms(self, plot, lattice_axes, grid_representation):

        if len(lattice_axes) == 3:
            return

        new_grid, representation = grid_representation

        plot.update_settings(axes=lattice_axes, reduce_method="average", transforms=["cos"], represent=representation, nsc=[1, 1, 1])

        # Check that transforms = ["cos"] applies np.cos
        assert np.allclose(
            self._get_plotted_values(plot), np.cos(new_grid.grid).mean(axis=tuple(ax for ax in [0, 1, 2] if ax not in lattice_axes))
        )
