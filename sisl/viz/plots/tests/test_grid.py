# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

Tests specific functionality of the grid plot.

Different inputs are tested (siesta .RHO and sisl Hamiltonian).

"""
from functools import reduce
import os.path as osp
import pytest
import numpy as np
import plotly.graph_objs as go

import sisl
from sisl.viz import GridPlot, Animation
from sisl.viz.plots.tests.conftest import _TestPlot


pytestmark = [pytest.mark.viz, pytest.mark.plotly]

try:
    import skimage
    skip_skimage = pytest.mark.skipif(False, reason="scikit-image (skimage) not available")
except ImportError:
    skip_skimage = pytest.mark.skipif(True, reason="scikit-image (skimage) not available")


@pytest.fixture(params=[1, 2, 3])
def ndim(request):
    if request.param > 1:
        pytest.importorskip("skimage")
    return request.param

@pytest.fixture()
def axes(ndim):
    return {1: [0], 2: [0, 1], 3: [0, 1, 2]}[ndim]


class TestGridPlot(_TestPlot):

    _required_attrs = [
        "grid_shape" # Tuple indicating the grid shape
    ]

    @pytest.fixture(scope="class", params=["siesta_RHO", "complex_grid"])
    def init_func_and_attrs(self, request, siesta_test_files):
        name = request.param

        if name == "siesta_RHO":
            init_func = sisl.get_sile(siesta_test_files("SrTiO3.RHO")).plot
            attrs = {"grid_shape": (48, 48, 48)}
        elif name == "complex_grid":
            complex_grid_shape = (8, 10, 10)
            values = np.random.random(complex_grid_shape).astype(np.complex128) + np.random.random(complex_grid_shape) * 1j
            complex_grid = sisl.Grid(complex_grid_shape, sc=1)
            complex_grid.grid = values
            
            init_func = complex_grid.plot
            attrs = {"grid_shape": complex_grid_shape}

        return init_func, attrs

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

    def _get_trace_class(self, ndim):
        return [go.Scatter, go.Heatmap, go.Mesh3d][ndim - 1]

    def _get_plotted_values(self, plot):

        ndim = len(plot.get_setting("axes"))
        if ndim == 1:
            return plot.figure.data[0].y
        elif ndim == 2:
            return plot.figure.data[0].z.T
        elif ndim == 3:
            trace = plot.figure.data[0]
            return np.array([trace.i, trace.j, trace.k])

    def test_plotting_modes(self, plot, axes):
        trace_type = self._get_trace_class(len(axes))

        plot.update_settings(axes=axes)
        assert isinstance(plot.data[0], trace_type), f"Not displaying grid in {len(axes)}D correctly"

    def test_representation(self, plot, axes, grid_representation):

        kwargs = {"isos": []}

        ndim = len(axes)
        if ndim == 3:
            kwargs["isos"] = [{"frac": 0.5}]

        new_grid, representation = grid_representation

        if new_grid.grid.min() == new_grid.grid.max() and ndim == 3:
            return

        plot.update_settings(axes=axes, represent=representation, **kwargs)
        new_plot = new_grid.plot(axes=axes, represent="real", **kwargs)

        assert np.allclose(
            self._get_plotted_values(plot), self._get_plotted_values(plot=new_plot)
        ), f"'{representation}' representation of the {ndim}D plot is not correct"

    def test_grid(self, plot, test_attrs):
        grid = plot.grid

        assert isinstance(grid, sisl.Grid)
        assert grid.shape == test_attrs["grid_shape"]

    @skip_skimage
    @pytest.mark.skip("We have to reorganize the scan method")
    def test_scan(self, plot):
        # AS_IS SCAN
        # Provide number of steps
        scanned = plot.scan(num=2, mode="as_is")
        assert isinstance(scanned, Animation)
        assert len(scanned.frames) == 2

        # Provide step in Ang
        step = plot.grid.cell[0, 0]/2
        scanned = plot.scan(along=0, step=step, mode="as_is")
        assert len(scanned.frames) == 2

        # Provide breakpoints
        breakpoints = [plot.grid.cell[0, 0]*frac for frac in [1/3, 2/3, 3/3]]
        scanned = plot.scan(along=0, breakpoints=breakpoints, mode="as_is")
        assert len(scanned.frames) == 2

        # Check that it doesn't accept step and breakpoints at the same time
        with pytest.raises(ValueError):
            plot.scan(along=0, step=4.5, breakpoints=breakpoints, mode="as_is")

        # 3D SCAN
        scanned = plot.scan(0, mode="moving_slice", breakpoints=breakpoints)

        assert isinstance(scanned, go.Figure)
        assert len(scanned.frames) == 3 # One cross section for each breakpoint

    @skip_skimage
    def test_supercell(self, plot):

        plot.update_settings(axes=[0, 1], interp=[1, 1, 1], nsc=[1, 1, 1])

        # Check that the initial shapes are right
        prev_shape = (len(plot.data[0].x), len(plot.data[0].y))
        assert prev_shape == (plot.grid.shape[0], plot.grid.shape[1])

        # Check that the supercell is displayed
        plot.update_settings(nsc=[2, 1, 1])
        sc_shape = (len(plot.data[0].x), len(plot.data[0].y))
        assert sc_shape[0] == 2*prev_shape[0]
        assert sc_shape[1] == prev_shape[1]

        plot.update_settings(nsc=[1, 1, 1])

    @pytest.mark.parametrize("reduce_method", ["sum", "average"])
    def test_reduce_method(self, plot, reduce_method, axes, grid_representation):
        new_grid, representation = grid_representation

        # If this is a 3D plot, no dimension is reduced, therefore it makes no sense
        if len(axes) == 3:
            return

        numpy_func = getattr(np, reduce_method)

        plot.update_settings(axes=axes, reduce_method=reduce_method, represent=representation)

        assert np.allclose(
            self._get_plotted_values(plot), numpy_func(new_grid.grid, axis=tuple(ax for ax in [0, 1, 2] if ax not in axes))
        )

    def test_transforms(self, plot, axes, grid_representation):

        if len(axes) == 3:
            return

        new_grid, representation = grid_representation

        plot.update_settings(axes=axes, transforms=["cos"], represent=representation)

        # Check that transforms = ["cos"] applies np.cos
        assert np.allclose(
            self._get_plotted_values(plot), np.cos(new_grid.grid).mean(axis=tuple(ax for ax in [0, 1, 2] if ax not in axes))
        )

