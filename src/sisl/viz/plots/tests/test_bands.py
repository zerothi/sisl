import pytest

from sisl import Spin
from sisl.viz.data import BandsData
from sisl.viz.plots import bands_plot

pytestmark = [pytest.mark.viz, pytest.mark.plots]


@pytest.fixture(scope="module", params=["plotly", "matplotlib"])
def backend(request):
    pytest.importorskip(request.param)
    return request.param


@pytest.fixture(
    scope="module", params=["unpolarized", "polarized", "noncolinear", "spinorbit"]
)
def spin(request):
    return Spin(request.param)


@pytest.fixture(scope="module")
def gap():
    return 2.5


@pytest.fixture(scope="module")
def bands_data(spin, gap):
    return BandsData.toy_example(spin=spin, gap=gap)


def test_bands_plot(bands_data, backend):
    bands_plot(bands_data, backend=backend)
