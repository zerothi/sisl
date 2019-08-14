from __future__ import print_function, division

import pytest
from pytest import approx

import math as m
import numpy as np

import sisl.linalg as lin
from sisl import SuperCell, SuperCellChild
from sisl.geom import graphene


pytestmark = [pytest.mark.supercell, pytest.mark.sc]


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            alat = 1.42
            sq3h = 3.**.5 * 0.5
            self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                          [1.5, -sq3h, 0.],
                                          [0., 0., 10.]], np.float64) * alat, nsc=[3, 3, 1])
    return t()


class TestSuperCell(object):

    def test_str(self, setup):
        str(setup.sc)
        str(setup.sc)
        assert setup.sc != 'Not a SuperCell'

    def test_nsc1(self, setup):
        sc = setup.sc.copy()
        sc.set_nsc([5, 5, 0])
        assert np.allclose([5, 5, 1], sc.nsc)
        assert len(sc.sc_off) == np.prod(sc.nsc)

    def test_nsc2(self, setup):
        sc = setup.sc.copy()
        sc.set_nsc([0, 1, 0])
        assert np.allclose([1, 1, 1], sc.nsc)
        assert len(sc.sc_off) == np.prod(sc.nsc)
        sc.set_nsc(a=3)
        assert np.allclose([3, 1, 1], sc.nsc)
        assert len(sc.sc_off) == np.prod(sc.nsc)
        sc.set_nsc(b=3)
        assert np.allclose([3, 3, 1], sc.nsc)
        assert len(sc.sc_off) == np.prod(sc.nsc)
        sc.set_nsc(c=5)
        assert np.allclose([3, 3, 5], sc.nsc)
        assert len(sc.sc_off) == np.prod(sc.nsc)

    def test_nsc3(self, setup):
        assert setup.sc.sc_index([0, 0, 0]) == 0
        for s in range(setup.sc.n_s):
            assert setup.sc.sc_index(setup.sc.sc_off[s, :]) == s
        arng = np.arange(setup.sc.n_s)
        np.random.shuffle(arng)
        sc_off = setup.sc.sc_off[arng, :]
        assert np.all(setup.sc.sc_index(sc_off) == arng)

    @pytest.mark.xfail(raises=ValueError)
    def test_nsc4(self, setup):
        setup.sc.set_nsc(a=2)

    @pytest.mark.xfail(raises=ValueError)
    def test_nsc5(self, setup):
        setup.sc.set_nsc([1, 2, 3])

    def test_area1(self, setup):
        setup.sc.area(0, 1)

    def test_fill(self, setup):
        sc = setup.sc.swapaxes(1, 2)
        i = sc._fill([1, 1])
        assert i.dtype == np.int32
        i = sc._fill([1., 1.])
        assert i.dtype == np.float64
        for dt in [np.int32, np.int64, np.float32, np.float64, np.complex64]:
            i = sc._fill([1., 1.], dt)
            assert i.dtype == dt
            i = sc._fill(np.ones([2], dt))
            assert i.dtype == dt

    def test_add_vacuum1(self, setup):
        sc = setup.sc.copy()
        for i in range(3):
            s = sc.add_vacuum(10, i)
            ax = setup.sc.cell[i, :]
            ax += ax / np.sum(ax ** 2) ** .5 * 10
            assert np.allclose(ax, s.cell[i, :])

    def test_add1(self, setup):
        sc = setup.sc.copy()
        for R in range(1, 10):
            s = sc + R
            assert np.allclose(s.cell, sc.cell + np.diag([R] * 3))

    def test_rotation1(self, setup):
        rot = setup.sc.rotate(180, [0, 0, 1])
        rot.cell[2, 2] *= -1
        assert np.allclose(-rot.cell, setup.sc.cell)

        rot = setup.sc.rotate(m.pi, [0, 0, 1], rad=True)
        rot.cell[2, 2] *= -1
        assert np.allclose(-rot.cell, setup.sc.cell)

        rot = rot.rotate(180, [0, 0, 1])
        rot.cell[2, 2] *= -1
        assert np.allclose(rot.cell, setup.sc.cell)

    def test_rotation2(self, setup):
        rot = setup.sc.rotate(180, setup.sc.cell[2, :])
        rot.cell[2, 2] *= -1
        assert np.allclose(-rot.cell, setup.sc.cell)

        rot = setup.sc.rotate(m.pi, setup.sc.cell[2, :], rad=True)
        rot.cell[2, 2] *= -1
        assert np.allclose(-rot.cell, setup.sc.cell)

        rot = rot.rotate(180, setup.sc.cell[2, :])
        rot.cell[2, 2] *= -1
        assert np.allclose(rot.cell, setup.sc.cell)

    def test_swapaxes1(self, setup):
        sab = setup.sc.swapaxes(0, 1)
        assert np.allclose(sab.cell[0, :], setup.sc.cell[1, :])
        assert np.allclose(sab.cell[1, :], setup.sc.cell[0, :])

    def test_swapaxes2(self, setup):
        sab = setup.sc.swapaxes(0, 2)
        assert np.allclose(sab.cell[0, :], setup.sc.cell[2, :])
        assert np.allclose(sab.cell[2, :], setup.sc.cell[0, :])

    def test_swapaxes3(self, setup):
        sab = setup.sc.swapaxes(1, 2)
        assert np.allclose(sab.cell[1, :], setup.sc.cell[2, :])
        assert np.allclose(sab.cell[2, :], setup.sc.cell[1, :])

    def test_offset1(self, setup):
        off = setup.sc.offset()
        assert np.allclose(off, [0, 0, 0])
        off = setup.sc.offset([1, 1, 1])
        cell = setup.sc.cell[:, :]
        assert np.allclose(off, cell[0, :] + cell[1, :] + cell[2, :])

    def test_sc_index1(self, setup):
        sc_index = setup.sc.sc_index([0, 0, 0])
        assert sc_index == 0
        sc_index = setup.sc.sc_index([0, 0, None])
        assert len(sc_index) == setup.sc.nsc[2]

    def test_sc_index2(self, setup):
        sc_index = setup.sc.sc_index([[0, 0, 0],
                                      [1, 1, 0]])
        s = str(sc_index)
        assert len(sc_index) == 2

    @pytest.mark.xfail(raises=Exception)
    def test_sc_index3(self, setup):
        setup.sc.sc_index([100, 100, 100])

    def test_cut1(self, setup):
        cut = setup.sc.cut(2, 0)
        assert np.allclose(cut.cell[0, :] * 2, setup.sc.cell[0, :])
        assert np.allclose(cut.cell[1, :], setup.sc.cell[1, :])

    def test_creation1(self, setup):
        # full cell
        tmp1 = SuperCell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # diagonal cell
        tmp2 = SuperCell([1, 1, 1])
        # cell parameters
        tmp3 = SuperCell([1, 1, 1, 90, 90, 90])
        tmp4 = SuperCell([1])
        assert np.allclose(tmp1.cell, tmp2.cell)
        assert np.allclose(tmp1.cell, tmp3.cell)
        assert np.allclose(tmp1.cell, tmp4.cell)

    def test_creation2(self, setup):
        # full cell
        class P(SuperCellChild):

            def copy(self):
                a = P()
                a.set_supercell(setup.sc)
                return a
        tmp1 = P()
        tmp1.set_supercell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # diagonal cell
        tmp2 = P()
        tmp2.set_supercell([1, 1, 1])
        # cell parameters
        tmp3 = P()
        tmp3.set_supercell([1, 1, 1, 90, 90, 90])
        tmp4 = P()
        tmp4.set_supercell([1])
        assert np.allclose(tmp1.cell, tmp2.cell)
        assert np.allclose(tmp1.cell, tmp3.cell)
        assert np.allclose(tmp1.cell, tmp4.cell)
        assert len(tmp1._fill([0, 0, 0])) == 3
        assert len(tmp1._fill_sc([0, 0, 0])) == 3
        assert tmp1.is_orthogonal()
        for i in range(3):
            t2 = tmp2.add_vacuum(10, i)
            assert tmp1.cell[i, i] + 10 == t2.cell[i, i]

    @pytest.mark.xfail(raises=ValueError)
    def test_creation3(self, setup):
        setup.sc.tocell([3, 6])

    @pytest.mark.xfail(raises=ValueError)
    def test_creation4(self, setup):
        setup.sc.tocell([3, 4, 5, 6])

    @pytest.mark.xfail(raises=ValueError)
    def test_creation5(self, setup):
        setup.sc.tocell([3, 4, 5, 6, 7, 6, 7])

    def test_creation_rotate(self, setup):
        # cell parameters
        param = np.array([1, 2, 3, 45, 60, 80], np.float64)
        parama = param.copy()
        parama[3:] *= np.pi / 180
        sc = SuperCell(param)
        assert np.allclose(param, sc.parameters())
        assert np.allclose(parama, sc.parameters(True))
        for ang in range(0, 91, 5):
            s = sc.rotate(ang, sc.cell[0, :]).rotate(ang, sc.cell[1, :]).rotate(ang, sc.cell[2, :])
            assert np.allclose(param, s.parameters())
            assert np.allclose(parama, s.parameters(True))

    def test_rcell(self, setup):
        # LAPACK inverse algorithm implicitly does
        # a transpose.
        rcell = lin.inv(setup.sc.cell) * 2. * np.pi
        assert np.allclose(rcell.T, setup.sc.rcell)
        assert np.allclose(rcell.T / (2 * np.pi), setup.sc.icell)

    def test_icell(self, setup):
        assert np.allclose(setup.sc.rcell, setup.sc.icell * 2 * np.pi)

    def test_translate1(self, setup):
        sc = setup.sc.translate([0, 0, 10])
        assert np.allclose(sc.cell[2, :2], setup.sc.cell[2, :2])
        assert np.allclose(sc.cell[2, 2], setup.sc.cell[2, 2]+10)

    def test_center1(self, setup):
        assert np.allclose(setup.sc.center(), np.sum(setup.sc.cell, axis=0) / 2)
        for i in [0, 1, 2]:
            assert np.allclose(setup.sc.center(i), setup.sc.cell[i, :] / 2)

    def test_pickle(self, setup):
        import pickle as p
        s = p.dumps(setup.sc)
        n = p.loads(s)
        assert setup.sc == n
        s = SuperCell([1, 1, 1])
        assert setup.sc != s

    def test_orthogonal(self, setup):
        assert not setup.sc.is_orthogonal()

    def test_fit1(self, setup):
        g = graphene()
        gbig = g.repeat(40, 0).repeat(40, 1)
        gbig.xyz[:, :] += (np.random.rand(len(gbig), 3) - 0.5) * 0.01
        sc = g.sc.fit(gbig)
        assert np.allclose(sc.cell, gbig.cell)

    def test_fit2(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        gbig.xyz[:, :] += (np.random.rand(len(gbig), 3) - 0.5) * 0.01
        sc = g.sc.fit(gbig)
        assert np.allclose(sc.cell, gbig.cell)

    def test_fit3(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        gbig.xyz[:, :] += (np.random.rand(len(gbig), 3) - 0.5) * 0.01
        sc = g.sc.fit(gbig, axis=0)
        assert np.allclose(sc.cell[0, :], gbig.cell[0, :])
        assert np.allclose(sc.cell[1:, :], g.cell[1:, :])

    def test_fit4(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        gbig.xyz[:, :] += (np.random.rand(len(gbig), 3) - 0.5) * 0.01
        sc = g.sc.fit(gbig, axis=[0, 1])
        assert np.allclose(sc.cell[0:2, :], gbig.cell[0:2, :])
        assert np.allclose(sc.cell[2, :], g.cell[2, :])

    def test_parallel1(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        assert g.sc.parallel(gbig.sc)
        assert gbig.sc.parallel(g.sc)
        g = g.rotate(90, g.cell[0, :])
        assert not g.sc.parallel(gbig.sc)

    def test_tile_multiply_orthogonal(self):
        sc = graphene(orthogonal=True).sc
        assert np.allclose(sc.tile(3, 0).tile(2, 1).tile(4, 2).cell, (sc * (3, 2, 4)).cell)
        assert np.allclose(sc.tile(3, 0).tile(2, 1).cell, (sc * [3, 2]).cell)
        assert np.allclose(sc.tile(3, 0).tile(3, 1).tile(3, 2).cell, (sc * 3).cell)

    def test_tile_multiply_non_orthogonal(self):
        sc = graphene(orthogonal=False).sc
        assert np.allclose(sc.tile(3, 0).tile(2, 1).tile(4, 2).cell, (sc * (3, 2, 4)).cell)
        assert np.allclose(sc.tile(3, 0).tile(2, 1).cell, (sc * [3, 2]).cell)
        assert np.allclose(sc.tile(3, 0).tile(3, 1).tile(3, 2).cell, (sc * 3).cell)

    def test_angle1(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        assert g.sc.angle(0, 1) == 90

    def test_angle2(self, setup):
        sc = SuperCell([1, 1, 1])
        assert sc.angle(0, 1) == 90
        assert sc.angle(0, 2) == 90
        assert sc.angle(1, 2) == 90
        sc = SuperCell([[1, 1, 0],
                        [1, -1, 0],
                        [0, 0, 2]])
        assert sc.angle(0, 1) == 90
        assert sc.angle(0, 2) == 90
        assert sc.angle(1, 2) == 90
        sc = SuperCell([[3, 4, 0],
                        [4, 3, 0],
                        [0, 0, 2]])
        assert sc.angle(0, 1, rad=True) == approx(0.28379, abs=1e-4)
        assert sc.angle(0, 2) == 90
        assert sc.angle(1, 2) == 90

    def test_cell_length(self):
        gr = graphene(orthogonal=True)
        sc = (gr * (40, 40, 1)).rotate(24, gr.cell[2, :]).sc
        assert np.allclose(sc.length, (sc.cell_length(sc.length) ** 2).sum(1) ** 0.5)
        assert np.allclose(1, (sc.cell_length(1) ** 2).sum(0))

    @pytest.mark.xfail(raises=ValueError)
    def test_set_nsc1(self, setup):
        sc = setup.sc.copy()
        sc.sc_off = np.zeros([10000, 3])
        setup.sc.set_nsc(a=2)


def _dot(u, v):
    """ Dot product u . v """
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def test_plane1():
    sc = SuperCell([1] * 3)
    # Check point [0.5, 0.5, 0.5]
    pp = np.array([0.5] * 3)

    n, p = sc.plane(0, 1, True)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = sc.plane(0, 2, True)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = sc.plane(1, 2, True)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = sc.plane(0, 1, False)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = sc.plane(0, 2, False)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = sc.plane(1, 2, False)
    assert -0.5 == approx(_dot(n, pp - p))


def test_plane2():
    sc = SuperCell([1] * 3)
    # Check point [-0.5, -0.5, -0.5]
    pp = np.array([-0.5] * 3)

    n, p = sc.plane(0, 1, True)
    assert 0.5 == approx(_dot(n, pp - p))
    n, p = sc.plane(0, 2, True)
    assert 0.5 == approx(_dot(n, pp - p))
    n, p = sc.plane(1, 2, True)
    assert 0.5 == approx(_dot(n, pp - p))
    n, p = sc.plane(0, 1, False)
    assert -1.5 == approx(_dot(n, pp - p))
    n, p = sc.plane(0, 2, False)
    assert -1.5 == approx(_dot(n, pp - p))
    n, p = sc.plane(1, 2, False)
    assert -1.5 == approx(_dot(n, pp - p))


def test_tocuboid_simple():
    sc = SuperCell([1, 1, 1, 90, 90, 90])
    c1 = sc.toCuboid()
    assert np.allclose(sc.cell, c1._v)
    c2 = sc.toCuboid(True)
    assert np.allclose(c1._v, c2._v)


def test_tocuboid_complex():
    sc = SuperCell([1, 1, 1, 60, 60, 60])
    s = str(sc.cell)
    c1 = sc.toCuboid()
    assert np.allclose(sc.cell, c1._v)
    c2 = sc.toCuboid(True)
    assert not np.allclose(np.diagonal(c1._v), np.diagonal(c2._v))
