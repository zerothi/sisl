"""

Tests specific functionality of the grid plot.

Different inputs are tested (siesta .RHO and sisl Hamiltonian).

"""
import os.path as osp
import pytest
import numpy as np
import plotly.graph_objs as go

import sisl
from sisl.viz.plotly import GridPlot, Animation
from sisl.viz.plotly.plots.tests.conftest import PlotTester


pytestmark = [pytest.mark.viz, pytest.mark.plotly]
_dir = osp.join('sisl', 'io', 'siesta')

try:
    import skimage
    skip_skimage = pytest.mark.skip(False, reason="scikit-image (skimage) not available")
except ImportError:
    skip_skimage = pytest.mark.skip(True, reason="scikit-image (skimage) not available")


class GridPlotTester(PlotTester):

    _required_attrs = [
        "grid_shape" # Tuple indicating the grid shape
    ]

    @pytest.mark.parametrize("axes,go", [
        ([0], go.Scatter),
        pytest.param([0, 1], go.Heatmap, marks=skip_skimage),
        pytest.param([0, 1, 2], go.Mesh3d, marks=skip_skimage),
    ])
    def test_plotting_modes(self, axes, go):
        plot = self.plot

        plot.update_settings(axes=axes)
        assert isinstance(plot.data[0], go), f"Not displaying grid in {len(axes)}D correctly?"

    @skip_skimage
    def test_complex_representations(self):
        for repr in ["imag", "mod", "rad_phase", "deg_phase", "real"]:
            self.plot.update_settings(represent=repr)

    def test_grid(self):

        assert osp.exists(self.plot.get_setting("grid_file")), "You provided a non-existent grid_file"

        grid = self.plot.grid

        assert isinstance(grid, sisl.Grid)
        assert grid.shape == self.grid_shape

    @skip_skimage
    def test_scan(self):
        # AS_IS SCAN
        # Provide number of steps
        scanned = self.plot.scan(num=2, mode="as_is")
        assert isinstance(scanned, Animation)
        assert len(scanned.frames) == 2

        # Provide step in Ang
        step = self.plot.grid.cell[0, 0]/2
        scanned = self.plot.scan(along=0, step=step, mode="as_is")
        assert len(scanned.frames) == 2

        # Provide breakpoints
        breakpoints = [self.plot.grid.cell[0, 0]*frac for frac in [1/3, 2/3, 3/3]]
        scanned = self.plot.scan(along=0, breakpoints=breakpoints, mode="as_is")
        assert len(scanned.frames) == 2

        # Check that it doesn't accept step and breakpoints at the same time
        with pytest.raises(ValueError):
            self.plot.scan(along=0, step=4.5, breakpoints=breakpoints, mode="as_is")

        # 3D SCAN
        scanned = self.plot.scan(0, mode="moving_slice", breakpoints=breakpoints)

        assert isinstance(scanned, go.Figure)
        assert len(scanned.frames) == 3 # One cross section for each breakpoint

    @skip_skimage
    def test_supercell(self):

        plot = self.plot

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

    def test_transforms(self):

        plot = self.plot

        plot.update_settings(axes=[0], transforms=["cos"])

        # Check that transforms = ["cos"] applies np.cos
        assert np.allclose(plot.data[0].y, np.cos(plot.grid.grid).mean(2).mean(1))


class TestGridPlot(GridPlotTester):

    run_for = {
        # Test with a siesta RHO file
        "siesta_RHO": {
            "plot_file": osp.join(_dir, "SrTiO3.RHO"),
            "grid_shape": (48, 48, 48)
        }
    }
