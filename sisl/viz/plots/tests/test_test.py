import pytest

@pytest.fixture(params=[0,1,2])
def name(request):

    
    return [3,4,5][request.param]

@pytest.fixture
def a(request, name):

    if name < 2:
        pytest.skip()

    return [5,6,7][name]

@pytest.fixture
def b(request, name):
    dsdsd
    return [5,6,7][name]

class TestFrame:

    def test_func(self, a, b):
        assert a == b