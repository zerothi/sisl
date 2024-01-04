import pytest

from sisl.viz.processors.logic import swap

pytestmark = [pytest.mark.viz, pytest.mark.processors]


def test_swap():
    assert swap(1, (1, 2)) == 2
    assert swap(2, (1, 2)) == 1

    with pytest.raises(ValueError):
        swap(3, (1, 2))
