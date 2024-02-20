import sisl
from sisl.viz.plots import atomic_matrix_plot


def test_atomic_matrix_plot():

    graphene = sisl.geom.graphene()
    H = sisl.Hamiltonian(graphene)

    atomic_matrix_plot(H)
