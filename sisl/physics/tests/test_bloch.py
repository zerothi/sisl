from __future__ import print_function, division

from itertools import product
import pytest
import numpy as np

from sisl import geom, Hamiltonian, Bloch

pytestmark = pytest.mark.bloch


def get_H(orthogonal=True, *args):
    gr = geom.graphene(orthogonal=orthogonal)
    H = Hamiltonian(gr)
    H.construct([(0.1, 1.44), (0, -2.7)])
    for axis, rep in zip(args[::2], args[1::2]):
        H = H.tile(rep, axis)
    return H


@pytest.mark.parametrize("nx", [1, 2, 4])
@pytest.mark.parametrize("ny", [1, 3, 4])
@pytest.mark.parametrize("nz", [1, 2, 3])
def test_bloch_create(nx, ny, nz):
    b = Bloch([nx, ny, nz])
    assert len(b) == nx * ny * nz
    assert len(b.unfold_points([0] * 3)) == nx * ny * nz


def test_bloch_method():
    b = Bloch([1] * 3)
    assert 'Bloch' in str(b)


def test_bloch_call():
    b = Bloch([2] * 3)
    H = get_H()

    # Manual
    k_unfold = b.unfold_points([0] * 3)
    m = b.unfold([H.Hk(k, format='array') for k in k_unfold], k_unfold)

    assert np.allclose(m, b(H.Hk, [0] * 3, format='array'))


@pytest.mark.parametrize("nx", [1, 3])
@pytest.mark.parametrize("ny", [1, 4])
@pytest.mark.parametrize("nz", [1, 3])
def test_bloch_one_direction(nx, ny, nz):
    H = get_H()
    b = Bloch([nx, ny, nz])

    HB = H.tile(nx, 0).tile(ny, 1).tile(nz, 2)

    KX = [0, 0.1, 0.4358923]
    KY = [0, 0.128359, -0.340925]
    KZ = [0, 0.445]
    for kx, ky, kz in product(KX, KY, KZ):
        K = [kx, ky, kz]
        k_unfold = b.unfold_points(K)

        HK = [H.Hk(k, format='array') for k in k_unfold]
        H_unfold = b.unfold(HK, k_unfold)
        H_big = HB.Hk(K, format='array')

        assert np.allclose(H_unfold, H_big)
