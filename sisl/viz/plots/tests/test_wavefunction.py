import pytest

from functools import partial

import sisl
from sisl.viz.plots.tests.test_grid import TestGridPlot as _TestGridPlot


class TestWavefunctionPlot(_TestGridPlot):

    @pytest.fixture(scope="class", params=["wfsx file"])
    def init_func_and_attrs(self, request, siesta_test_files):
        name = request.param

        if name == "wfsx file":
            pytest.skip("Basis for bi2se3_3ql.fdf is not available in the test files.")
            fdf = sisl.get_sile(siesta_test_files("bi2se3_3ql.fdf"))
            wfsx = siesta_test_files("bi2se3_3ql.bands.WFSX")
            init_func = partial(
                fdf.plot.wavefunction, wfsx_file=wfsx, k=(0.003, 0.003, 0),
                entry_points_order=["wfsx file"])

            attrs = {"grid_shape": (48, 48, 48)}
        return init_func, attrs
