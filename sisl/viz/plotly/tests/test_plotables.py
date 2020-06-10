import plotly.graph_objs as go
import numpy as np

import sisl

r = np.linspace(0, 3.5, 50)
f = np.exp(-r)

orb = sisl.AtomicOrbital('2pzZ', (r, f))
geom = sisl.geom.graphene(orthogonal=True, atom=sisl.Atom(6, orb))
geom = geom.move([0, 0, 5])
H = sisl.Hamiltonian(geom)
H.construct([(0.1, 1.44), (0, -2.7)], )

def test_eigenstate_wf():

    plot = H.eigenstate()[0].plot_wavefunction(geometry=H.geometry)

    assert len(plot.data) > 0
    assert isinstance(plot.data[0], go.Isosurface)

def test_hamiltonian_wf():

    # Check if it works for 3D plots
    plot = H.plot_wavefunction(2)
    assert isinstance(plot.data[0], go.Isosurface)

    # Check that setting plot geom to True adds data traces
    plot.update_settings(plot_geom=False)
    prev_len = len(plot.data)
    plot.update_settings(plot_geom=True)
    assert len(plot.data) > prev_len

    # Now 2D
    plot = H.plot_wavefunction(2, axes=[0,1])
    assert isinstance(plot.data[0], go.Heatmap)

    # Check that setting plot geom to True adds data traces
    plot.update_settings(plot_geom=False)
    prev_len = len(plot.data)
    plot.update_settings(plot_geom=True)
    assert len(plot.data) > prev_len






