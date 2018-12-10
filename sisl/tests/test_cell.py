from __future__ import print_function, division

import pytest
from pytest import approx

import math as m
import numpy as np

import sisl.linalg as lin
from sisl import Cell, CellChild
from sisl.geom import graphene


pytestmark = [pytest.mark.cell, pytest.mark.sc]


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            alat = 1.42
            sq3h = 3.**.5 * 0.5
            self.cell = Cell(np.array([[1.5, sq3h, 0.],
                                          [1.5, -sq3h, 0.],
                                          [0., 0., 10.]], np.float64) * alat, nsc=[3, 3, 1])
    return t()


class TestCell(object):

    def test_str(self, setup):
        str(setup.cell)
        str(setup.cell)
        assert setup.cell != 'Not a Cell'

    def test_nsc1(self, setup):
        cell = setup.cell.copy()
        cell.set_nsc([5, 5, 0])
        assert np.allclose([5, 5, 1], cell.nsc)
        assert len(cell.sc_off) == np.prod(cell.nsc)

    def test_nsc2(self, setup):
        cell = setup.cell.copy()
        cell.set_nsc([0, 1, 0])
        assert np.allclose([1, 1, 1], cell.nsc)
        assert len(cell.sc_off) == np.prod(cell.nsc)
        cell.set_nsc(a=3)
        assert np.allclose([3, 1, 1], cell.nsc)
        assert len(cell.sc_off) == np.prod(cell.nsc)
        cell.set_nsc(b=3)
        assert np.allclose([3, 3, 1], cell.nsc)
        assert len(cell.sc_off) == np.prod(cell.nsc)
        cell.set_nsc(c=5)
        assert np.allclose([3, 3, 5], cell.nsc)
        assert len(cell.sc_off) == np.prod(cell.nsc)

    def test_nsc3(self, setup):
        assert setup.cell.sc_index([0, 0, 0]) == 0
        for s in range(setup.cell.n_s):
            assert setup.cell.sc_index(setup.cell.sc_off[s, :]) == s
        arng = np.arange(setup.cell.n_s)
        np.random.shuffle(arng)
        sc_off = setup.cell.sc_off[arng, :]
        assert np.all(setup.cell.sc_index(sc_off) == arng)

    @pytest.mark.xfail(raises=ValueError)
    def test_nsc4(self, setup):
        setup.cell.set_nsc(a=2)

    @pytest.mark.xfail(raises=ValueError)
    def test_nsc5(self, setup):
        setup.cell.set_nsc([1, 2, 3])

    def test_area1(self, setup):
        setup.cell.area(0, 1)

    def test_fill(self, setup):
        cell = setup.cell.swapaxes(1, 2)
        i = cell._fill([1, 1])
        assert i.dtype == np.int32
        i = cell._fill([1., 1.])
        assert i.dtype == np.float64
        for dt in [np.int32, np.int64, np.float32, np.float64, np.complex64]:
            i = cell._fill([1., 1.], dt)
            assert i.dtype == dt
            i = cell._fill(np.ones([2], dt))
            assert i.dtype == dt

    def test_add_vacuum1(self, setup):
        cell = setup.cell.copy()
        for i in range(3):
            s = cell.add_vacuum(10, i)
            ax = setup.cell.cell[i, :]
            ax += ax / np.sum(ax ** 2) ** .5 * 10
            assert np.allclose(ax, s.cell[i, :])

    def test_add1(self, setup):
        cell = setup.cell.copy()
        for R in range(1, 10):
            s = cell + R
            assert np.allclose(s.cell, cell.cell + np.diag([R] * 3))

    def test_rotation1(self, setup):
        rot = setup.cell.rotate(180, [0, 0, 1])
        rot.cell[2, 2] *= -1
        assert np.allclose(-rot.cell, setup.cell.cell)

        rot = setup.cell.rotate(m.pi, [0, 0, 1], rad=True)
        rot.cell[2, 2] *= -1
        assert np.allclose(-rot.cell, setup.cell.cell)

        rot = rot.rotate(180, [0, 0, 1])
        rot.cell[2, 2] *= -1
        assert np.allclose(rot.cell, setup.cell.cell)

    def test_rotation2(self, setup):
        rot = setup.cell.rotatec(180)
        rot.cell[2, 2] *= -1
        assert np.allclose(-rot.cell, setup.cell.cell)

        rot = setup.cell.rotatec(m.pi, rad=True)
        rot.cell[2, 2] *= -1
        assert np.allclose(-rot.cell, setup.cell.cell)

        rot = rot.rotatec(180)
        rot.cell[2, 2] *= -1
        assert np.allclose(rot.cell, setup.cell.cell)

    def test_rotation3(self, setup):
        rot = setup.cell.rotatea(180)
        assert np.allclose(rot.cell[0, :], setup.cell.cell[0, :])
        assert np.allclose(-rot.cell[2, 2], setup.cell.cell[2, 2])

        rot = setup.cell.rotateb(m.pi, rad=True)
        assert np.allclose(rot.cell[1, :], setup.cell.cell[1, :])
        assert np.allclose(-rot.cell[2, 2], setup.cell.cell[2, 2])

    def test_swapaxes1(self, setup):
        sab = setup.cell.swapaxes(0, 1)
        assert np.allclose(sab.cell[0, :], setup.cell.cell[1, :])
        assert np.allclose(sab.cell[1, :], setup.cell.cell[0, :])

    def test_swapaxes2(self, setup):
        sab = setup.cell.swapaxes(0, 2)
        assert np.allclose(sab.cell[0, :], setup.cell.cell[2, :])
        assert np.allclose(sab.cell[2, :], setup.cell.cell[0, :])

    def test_swapaxes3(self, setup):
        sab = setup.cell.swapaxes(1, 2)
        assert np.allclose(sab.cell[1, :], setup.cell.cell[2, :])
        assert np.allclose(sab.cell[2, :], setup.cell.cell[1, :])

    def test_offset1(self, setup):
        off = setup.cell.offset()
        assert np.allclose(off, [0, 0, 0])
        off = setup.cell.offset([1, 1, 1])
        cell = setup.cell.cell[:, :]
        assert np.allclose(off, cell[0, :] + cell[1, :] + cell[2, :])

    def test_sc_index1(self, setup):
        sc_index = setup.cell.sc_index([0, 0, 0])
        assert sc_index == 0
        sc_index = setup.cell.sc_index([0, 0, None])
        assert len(sc_index) == setup.cell.nsc[2]

    def test_sc_index2(self, setup):
        sc_index = setup.cell.sc_index([[0, 0, 0],
                                      [1, 1, 0]])
        print(sc_index)
        assert len(sc_index) == 2

    @pytest.mark.xfail(raises=Exception)
    def test_sc_index3(self, setup):
        setup.cell.sc_index([100, 100, 100])

    def test_cut1(self, setup):
        cut = setup.cell.cut(2, 0)
        assert np.allclose(cut.cell[0, :] * 2, setup.cell.cell[0, :])
        assert np.allclose(cut.cell[1, :], setup.cell.cell[1, :])

    def test_creation1(self, setup):
        # full cell
        tmp1 = Cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # diagonal cell
        tmp2 = Cell([1, 1, 1])
        # cell parameters
        tmp3 = Cell([1, 1, 1, 90, 90, 90])
        tmp4 = Cell([1])
        assert np.allclose(tmp1.cell, tmp2.cell)
        assert np.allclose(tmp1.cell, tmp3.cell)
        assert np.allclose(tmp1.cell, tmp4.cell)

    def test_creation2(self, setup):
        # full cell
        class P(CellChild):

            def copy(self):
                a = P()
                a.set_cell(setup.cell)
                return a
        tmp1 = P()
        tmp1.set_cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # diagonal cell
        tmp2 = P()
        tmp2.set_cell([1, 1, 1])
        # cell parameters
        tmp3 = P()
        tmp3.set_cell([1, 1, 1, 90, 90, 90])
        tmp4 = P()
        tmp4.set_cell([1])
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
        setup.cell.tocell([3, 6])

    @pytest.mark.xfail(raises=ValueError)
    def test_creation4(self, setup):
        setup.cell.tocell([3, 4, 5, 6])

    @pytest.mark.xfail(raises=ValueError)
    def test_creation5(self, setup):
        setup.cell.tocell([3, 4, 5, 6, 7, 6, 7])

    def test_creation_rotate(self, setup):
        # cell parameters
        param = np.array([1, 2, 3, 45, 60, 80], np.float64)
        parama = param.copy()
        parama[3:] *= np.pi / 180
        cell = Cell(param)
        assert np.allclose(param, cell.parameters())
        assert np.allclose(parama, cell.parameters(True))
        for ang in range(0, 91, 5):
            s = cell.rotatea(ang).rotateb(ang).rotatec(ang)
            assert np.allclose(param, s.parameters())
            assert np.allclose(parama, s.parameters(True))

    def test_rcell(self, setup):
        # LAPACK inverse algorithm implicitly does
        # a transpose.
        rcell = lin.inv(setup.cell.cell) * 2. * np.pi
        assert np.allclose(rcell.T, setup.cell.rcell)
        assert np.allclose(rcell.T / (2 * np.pi), setup.cell.icell)

    def test_icell(self, setup):
        assert np.allclose(setup.cell.rcell, setup.cell.icell * 2 * np.pi)

    def test_translate1(self, setup):
        cell = setup.cell.translate([0, 0, 10])
        assert np.allclose(cell.cell[2, :2], setup.cell.cell[2, :2])
        assert np.allclose(cell.cell[2, 2], setup.cell.cell[2, 2]+10)

    def test_center1(self, setup):
        assert np.allclose(setup.cell.center(), np.sum(setup.cell.cell, axis=0) / 2)
        for i in [0, 1, 2]:
            assert np.allclose(setup.cell.center(i), setup.cell.cell[i, :] / 2)

    def test_pickle(self, setup):
        import pickle as p
        s = p.dumps(setup.cell)
        n = p.loads(s)
        assert setup.cell == n
        s = Cell([1, 1, 1])
        assert setup.cell != s

    def test_orthogonal(self, setup):
        assert not setup.cell.is_orthogonal()

    def test_fit1(self, setup):
        g = graphene()
        gbig = g.repeat(40, 0).repeat(40, 1)
        gbig.xyz[:, :] += (np.random.rand(len(gbig), 3) - 0.5) * 0.01
        cell = g.sc.fit(gbig)
        assert np.allclose(cell.cell, gbig.cell)

    def test_fit2(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        gbig.xyz[:, :] += (np.random.rand(len(gbig), 3) - 0.5) * 0.01
        cell = g.sc.fit(gbig)
        assert np.allclose(cell.cell, gbig.cell)

    def test_fit3(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        gbig.xyz[:, :] += (np.random.rand(len(gbig), 3) - 0.5) * 0.01
        cell = g.sc.fit(gbig, axis=0)
        assert np.allclose(cell.cell[0, :], gbig.cell[0, :])
        assert np.allclose(cell.cell[1:, :], g.cell[1:, :])

    def test_fit4(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        gbig.xyz[:, :] += (np.random.rand(len(gbig), 3) - 0.5) * 0.01
        cell = g.sc.fit(gbig, axis=[0, 1])
        assert np.allclose(cell.cell[0:2, :], gbig.cell[0:2, :])
        assert np.allclose(cell.cell[2, :], g.cell[2, :])

    def test_parallel1(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        assert g.sc.parallel(gbig.sc)
        assert gbig.sc.parallel(g.sc)
        g = g.rotatea(90)
        assert not g.sc.parallel(gbig.sc)

    def test_tile_multiply_orthogonal(self):
        cell = graphene(orthogonal=True).sc
        assert np.allclose(cell.tile(3, 0).tile(2, 1).tile(4, 2).cell, (cell * (3, 2, 4)).cell)
        assert np.allclose(cell.tile(3, 0).tile(2, 1).cell, (cell * [3, 2]).cell)
        assert np.allclose(cell.tile(3, 0).tile(3, 1).tile(3, 2).cell, (cell * 3).cell)

    def test_tile_multiply_non_orthogonal(self):
        cell = graphene(orthogonal=False).sc
        assert np.allclose(cell.tile(3, 0).tile(2, 1).tile(4, 2).cell, (cell * (3, 2, 4)).cell)
        assert np.allclose(cell.tile(3, 0).tile(2, 1).cell, (cell * [3, 2]).cell)
        assert np.allclose(cell.tile(3, 0).tile(3, 1).tile(3, 2).cell, (cell * 3).cell)

    def test_angle1(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        assert g.sc.angle(0, 1) == 90

    def test_angle2(self, setup):
        cell = Cell([1, 1, 1])
        assert cell.angle(0, 1) == 90
        assert cell.angle(0, 2) == 90
        assert cell.angle(1, 2) == 90
        cell = Cell([[1, 1, 0],
                        [1, -1, 0],
                        [0, 0, 2]])
        assert cell.angle(0, 1) == 90
        assert cell.angle(0, 2) == 90
        assert cell.angle(1, 2) == 90
        cell = Cell([[3, 4, 0],
                        [4, 3, 0],
                        [0, 0, 2]])
        assert cell.angle(0, 1, rad=True) == approx(0.28379, abs=1e-4)
        assert cell.angle(0, 2) == 90
        assert cell.angle(1, 2) == 90

    def test_cell_length(self):
        sc = (graphene(orthogonal=True) * (40, 40, 1)).rotatec(24).sc
        assert np.allclose(sc.length, (sc.cell_length(sc.length) ** 2).sum(1) ** 0.5)
        assert np.allclose(1, (sc.cell_length(1) ** 2).sum(0))

    @pytest.mark.xfail(raises=ValueError)
    def test_set_nsc1(self, setup):
        cell = setup.cell.copy()
        cell.sc_off = np.zeros([10000, 3])
        setup.cell.set_nsc(a=2)


def _dot(u, v):
    """ Dot product u . v """
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def test_plane1():
    cell = Cell([1] * 3)
    # Check point [0.5, 0.5, 0.5]
    pp = np.array([0.5] * 3)

    n, p = cell.plane(0, 1, True)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = cell.plane(0, 2, True)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = cell.plane(1, 2, True)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = cell.plane(0, 1, False)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = cell.plane(0, 2, False)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = cell.plane(1, 2, False)
    assert -0.5 == approx(_dot(n, pp - p))


def test_plane2():
    cell = Cell([1] * 3)
    # Check point [-0.5, -0.5, -0.5]
    pp = np.array([-0.5] * 3)

    n, p = cell.plane(0, 1, True)
    assert 0.5 == approx(_dot(n, pp - p))
    n, p = cell.plane(0, 2, True)
    assert 0.5 == approx(_dot(n, pp - p))
    n, p = cell.plane(1, 2, True)
    assert 0.5 == approx(_dot(n, pp - p))
    n, p = cell.plane(0, 1, False)
    assert -1.5 == approx(_dot(n, pp - p))
    n, p = cell.plane(0, 2, False)
    assert -1.5 == approx(_dot(n, pp - p))
    n, p = cell.plane(1, 2, False)
    assert -1.5 == approx(_dot(n, pp - p))


def test_tocuboid_simple():
    cell = Cell([1, 1, 1, 90, 90, 90])
    c1 = cell.toCuboid()
    assert np.allclose(cell.cell, c1._v)
    c2 = cell.toCuboid(True)
    assert np.allclose(c1._v, c2._v)


def test_tocuboid_complex():
    cell = Cell([1, 1, 1, 60, 60, 60])
    print(cell.cell)
    c1 = cell.toCuboid()
    assert np.allclose(cell.cell, c1._v)
    c2 = cell.toCuboid(True)
    assert not np.allclose(np.diagonal(c1._v), np.diagonal(c2._v))
