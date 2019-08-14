from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import Geometry, Atom, SuperCell, EnergyDensityMatrix, Spin


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            bond = 1.42
            sq3h = 3.**.5 * 0.5
            self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                          [1.5, -sq3h, 0.],
                                          [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

            C = Atom(Z=6, R=[bond * 1.01] * 3)
            self.g = Geometry(np.array([[0., 0., 0.],
                                        [1., 0., 0.]], np.float64) * bond,
                              atom=C, sc=self.sc)
            self.E = EnergyDensityMatrix(self.g)
            self.ES = EnergyDensityMatrix(self.g, orthogonal=False)

            def func(E, ia, idxs, idxs_xyz):
                idx = E.geom.close(ia, R=(0.1, 1.44), idx=idxs, idx_xyz=idxs_xyz)
                ia = ia * 3

                i0 = idx[0] * 3
                i1 = idx[1] * 3
                # on-site
                p = 1.
                E.E[ia, i0] = p
                E.E[ia+1, i0+1] = p
                E.E[ia+2, i0+2] = p

                # nn
                p = 0.1

                # on-site directions
                E.E[ia, ia+1] = p
                E.E[ia, ia+2] = p
                E.E[ia+1, ia] = p
                E.E[ia+1, ia+2] = p
                E.E[ia+2, ia] = p
                E.E[ia+2, ia+1] = p

                E.E[ia, i1+1] = p
                E.E[ia, i1+2] = p

                E.E[ia+1, i1] = p
                E.E[ia+1, i1+2] = p

                E.E[ia+2, i1] = p
                E.E[ia+2, i1+1] = p

            self.func = func
    return t()


@pytest.mark.density_matrix
@pytest.mark.energydensity_matrix
class TestEnergyDensityMatrix(object):

    def test_objects(self, setup):
        assert len(setup.E.xyz) == 2
        assert setup.g.no == len(setup.E)

    def test_spin(self, setup):
        g = setup.g.copy()
        EnergyDensityMatrix(g)
        EnergyDensityMatrix(g, spin=Spin('P'))
        EnergyDensityMatrix(g, spin=Spin('NC'))
        EnergyDensityMatrix(g, spin=Spin('SO'))

    def test_dtype(self, setup):
        assert setup.E.dtype == np.float64

    def test_ortho(self, setup):
        assert setup.E.orthogonal

    def test_mulliken(self, setup):
        E = setup.E.copy()
        E.construct(setup.func)
        mulliken = E.mulliken('atom')
        assert mulliken.shape == (1, len(E.geometry))
        mulliken = E.mulliken('orbital')
        assert mulliken.shape == (1, len(E))

    def test_mulliken_values_orthogonal(self, setup):
        E = setup.E.copy()
        E[0, 0] = 1.
        E[1, 1] = 2.
        E[1, 2] = 2.
        mulliken = E.mulliken('orbital')
        assert np.allclose(mulliken[0, :2], [1., 2.])
        assert mulliken.sum() == pytest.approx(3)
        mulliken = E.mulliken('atom')
        assert mulliken[0, 0] == pytest.approx(3)
        assert mulliken.sum() == pytest.approx(3)

    def test_mulliken_values_non_orthogonal(self, setup):
        E = setup.ES.copy()
        E[0, 0] = (1., 1.)
        E[1, 1] = (2., 1.)
        E[1, 2] = (2., 0.5)
        mulliken = E.mulliken('orbital')
        assert np.allclose(mulliken[0, :2], [1., 3.])
        assert mulliken.sum() == pytest.approx(4.)
        mulliken = E.mulliken('atom')
        assert mulliken[0, 0] == pytest.approx(4)
        assert mulliken.sum() == pytest.approx(4)

    def test_set1(self, setup):
        E = setup.E.copy()
        E.E[0, 0] = 1.
        assert E[0, 0] == 1.
        assert E[1, 0] == 0.

    def test_pickle(self, setup):
        import pickle as p
        E = setup.E.copy()
        E.construct(setup.func)
        s = p.dumps(E)
        e = p.loads(s)
        assert e.spsame(E)
        assert np.allclose(e.eigh(), E.eigh())
