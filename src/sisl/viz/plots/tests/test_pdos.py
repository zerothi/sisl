import pytest

from sisl import Spin
from sisl.viz.data import PDOSData
from sisl.viz.plots import pdos_plot

@pytest.fixture(scope="module", params=["plotly", "matplotlib"])
def backend(request):
    return request.param

@pytest.fixture(scope="module", params=["unpolarized", "polarized", "noncolinear", "spinorbit"])
def spin(request):
    return Spin(request.param)

@pytest.fixture(scope="module")
def pdos_data(spin):
    return PDOSData.toy_example(spin=spin)

def test_pdos_plot(pdos_data, backend):
    pdos_plot(pdos_data, backend=backend)