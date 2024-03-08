import pytest

import sisl
from sisl.viz.plots import atomic_matrix_plot


def test_atomic_matrix_plot():
    pytest.importorskip("skimage", reason="scikit-image not importable")

    graphene = sisl.geom.graphene()
    H = sisl.Hamiltonian(graphene)

    atomic_matrix_plot(H)
