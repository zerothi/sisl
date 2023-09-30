import numpy as np

from sisl.viz.processors.math import normalize

def test_normalize():

    data = [0, 1, 2]

    assert np.allclose(normalize(data), [0, 0.5, 1])

    assert np.allclose(normalize(data, vmin=-1, vmax=1), [-1, 0, 1])

