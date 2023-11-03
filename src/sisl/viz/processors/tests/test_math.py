import numpy as np
import pytest

from sisl.viz.processors.math import normalize

pytestmark = [pytest.mark.viz, pytest.mark.processors]


def test_normalize():
    data = [0, 1, 2]

    assert np.allclose(normalize(data), [0, 0.5, 1])

    assert np.allclose(normalize(data, vmin=-1, vmax=1), [-1, 0, 1])
