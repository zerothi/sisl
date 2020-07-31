import pytest

import numpy as np

import sisl


pytestmark = pytest.mark.plot

mlib = pytest.importorskip('matplotlib')
plt = pytest.importorskip('matplotlib.pyplot')
mlib3d = pytest.importorskip('mpl_toolkits.mplot3d')


def test_supercell_2d():
    g = sisl.geom.graphene()
    sisl.plot(g.sc, axis=[0, 1])
    sisl.plot(g.sc, axis=[0, 2])
    sisl.plot(g.sc, axis=[1, 2])
    plt.close('all')

    ax = plt.subplot(111)
    sisl.plot(g.sc, axis=[1, 2], axes=ax)
    plt.close('all')


def test_supercell_3d():
    g = sisl.geom.graphene()
    sisl.plot(g.sc)
    plt.close('all')


def test_geometry_2d():
    g = sisl.geom.graphene()
    sisl.plot(g, axis=[0, 1])
    sisl.plot(g, axis=[0, 2])
    sisl.plot(g, axis=[1, 2])
    plt.close('all')

    ax = plt.subplot(111)
    sisl.plot(g, axis=[1, 2], axes=ax)
    plt.close('all')


def test_geometry_2d_atom_indices():
    g = sisl.geom.graphene()
    sisl.plot(g, axis=[0, 1])
    sisl.plot(g, axis=[0, 2])
    sisl.plot(g, axis=[1, 2])
    plt.close('all')

    ax = plt.subplot(111)
    sisl.plot(g, axis=[1, 2], axes=ax, atom_indices=True)
    plt.close('all')


def test_geometry_3d():
    g = sisl.geom.graphene()
    sisl.plot(g)
    plt.close('all')


def test_geometry_3d_atom_indices():
    g = sisl.geom.graphene()
    sisl.plot(g, atom_indices=True)
    plt.close('all')


def test_orbital_radial():
    r = np.linspace(0, 10, 1000)
    f = np.exp(- r)
    o = sisl.SphericalOrbital(2, (r, f))
    sisl.plot(o)
    plt.close('all')

    fig = plt.figure()
    sisl.plot(o, axes=fig.gca())
    plt.close('all')


def test_orbital_harmonics():
    r = np.linspace(0, 10, 1000)
    f = np.exp(- r)
    o = sisl.SphericalOrbital(2, (r, f))
    sisl.plot(o, harmonics=True)
    plt.close('all')


def test_not_implemented():
    class Test:
        pass
    t = Test()
    with pytest.raises(NotImplementedError):
        sisl.plot(t)
