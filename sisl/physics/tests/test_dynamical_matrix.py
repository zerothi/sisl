from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import Geometry, Atom, SuperCell, DynamicalMatrix


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
            self.D = DynamicalMatrix(self.g)

            def func(D, ia, idxs, idxs_xyz):
                idx = D.geom.close(ia, R=(0.1, 1.44), idx=idxs, idx_xyz=idxs_xyz)
                ia = ia * 3

                i0 = idx[0] * 3
                i1 = idx[1] * 3
                # on-site
                p = 1.
                D.D[ia, i0] = p
                D.D[ia+1, i0+1] = p
                D.D[ia+2, i0+2] = p

                # nn
                p = 0.1

                # on-site directions
                D.D[ia, ia+1] = p
                D.D[ia, ia+2] = p
                D.D[ia+1, ia] = p
                D.D[ia+1, ia+2] = p
                D.D[ia+2, ia] = p
                D.D[ia+2, ia+1] = p

                D.D[ia, i1+1] = p
                D.D[ia, i1+2] = p

                D.D[ia+1, i1] = p
                D.D[ia+1, i1+2] = p

                D.D[ia+2, i1] = p
                D.D[ia+2, i1+1] = p

            self.func = func
    return t()


@pytest.mark.dynamicalmatrix
class TestDynamicalMatrix(object):

    def test_objects(self, setup):
        assert len(setup.D.xyz) == 2
        assert setup.g.no == len(setup.D)

    def test_dtype(self, setup):
        assert setup.D.dtype == np.float64

    def test_ortho(self, setup):
        assert setup.D.orthogonal

    def test_set1(self, setup):
        setup.D.D[0, 0] = 1.
        assert setup.D[0, 0] == 1.
        assert setup.D[1, 0] == 0.
        setup.D.empty()

    def test_apply_newton(self, setup):
        setup.D.construct(setup.func)
        assert setup.D[0, 0] == 1.
        assert setup.D[1, 0] == 0.1
        assert setup.D[0, 1] == 0.1
        setup.D.apply_newton()
        setup.D.empty()

    def test_eig(self, setup):
        D = setup.D.copy()
        D.construct(setup.func)
        ev = D.eigenvalue()
        em = D.eigenmode()
        assert np.allclose(ev.hw, em.hw)
        assert np.allclose(em.norm2(), 1)

    def test_change_gauge(self, setup):
        D = setup.D.copy()
        D.construct(setup.func)
        em = D.eigenmode(k=(0.2, 0.2, 0.2))
        em2 = em.copy()
        em2.change_gauge('r')
        assert not np.allclose(em.mode, em2.mode)
        em2.change_gauge('R')
        assert np.allclose(em.mode, em2.mode)

    def test_dos_pdos_velocity(self, setup):
        D = setup.D.copy()
        D.construct(setup.func)
        E = np.linspace(0, .5, 10)
        em = D.eigenmode()
        assert np.allclose(D.DOS(E), D.PDOS(E).sum(0))
        assert np.allclose(D.DOS(E), em.DOS(E))
        assert np.allclose(D.PDOS(E), em.PDOS(E))
        assert np.allclose(D.displacement(), em.displacement())
        assert np.allclose(D.velocity(), em.velocity())

    def test_pickle(self, setup):
        import pickle as p
        D = setup.D.copy()
        D.construct(setup.func)
        s = p.dumps(D)
        d = p.loads(s)
        assert d.spsame(D)
        assert np.allclose(d.eigh(), D.eigh())
