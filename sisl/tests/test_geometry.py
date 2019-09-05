from __future__ import print_function, division

import pytest

import itertools
import math as m
import numpy as np

import sisl.geom as sisl_geom
from sisl import SislWarning, SislError
from sisl import Cube, Sphere
from sisl import Geometry, Atom, SuperCell


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            bond = 1.42
            sq3h = 3.**.5 * 0.5
            self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                          [1.5, -sq3h, 0.],
                                          [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])
            C = Atom(Z=6, R=[bond * 1.01]*2)
            self.g = Geometry(np.array([[0., 0., 0.],
                                        [1., 0., 0.]], np.float64) * bond,
                              atom=C, sc=self.sc)

            self.mol = Geometry([[i, 0, 0] for i in range(10)], sc=[50])
    return t()


@pytest.mark.geom
@pytest.mark.geometry
class TestGeometry(object):

    def test_objects(self, setup):
        str(setup.g)
        assert len(setup.g) == 2
        assert len(setup.g.xyz) == 2
        assert np.allclose(setup.g[0], np.zeros([3]))
        assert np.allclose(setup.g[None, 0], setup.g.xyz[:, 0])

        i = 0
        for ia in setup.g:
            i += 1
        assert i == len(setup.g)
        assert setup.g.no_s == 2 * len(setup.g) * np.prod(setup.g.sc.nsc)

    def test_properties(self, setup):
        assert 2 == len(setup.g)
        assert 2 == setup.g.na
        assert 3*3 == setup.g.n_s
        assert 2*3*3 == setup.g.na_s
        assert 2*2 == setup.g.no
        assert 2*2*3*3 == setup.g.no_s

    def test_iter1(self, setup):
        i = 0
        for ia in setup.g:
            i += 1
        assert i == 2

    def test_iter2(self, setup):
        for ia in setup.g:
            assert np.allclose(setup.g[ia], setup.g.xyz[ia, :])

    def test_iter3(self, setup):
        i = 0
        for ia, io in setup.g.iter_orbitals(0):
            assert ia == 0
            assert io < 2
            i += 1
        for ia, io in setup.g.iter_orbitals(1):
            assert ia == 1
            assert io < 2
            i += 1
        assert i == 4
        i = 0
        for ia, io in setup.g.iter_orbitals():
            assert ia in [0, 1]
            assert io < 2
            i += 1
        assert i == 4

        i = 0
        for ia, io in setup.g.iter_orbitals(1, local=False):
            assert ia == 1
            assert io >= 2
            i += 1
        assert i == 2

    @pytest.mark.xfail(raises=ValueError)
    def test_tile0(self, setup):
        t = setup.g.tile(0, 0)

    def test_tile1(self, setup):
        cell = np.copy(setup.g.sc.cell)
        cell[0, :] *= 2
        t = setup.g.tile(2, 0)
        assert np.allclose(cell, t.sc.cell)
        cell[1, :] *= 2
        t = t.tile(2, 1)
        assert np.allclose(cell, t.sc.cell)
        cell[2, :] *= 2
        t = t.tile(2, 2)
        assert np.allclose(cell, t.sc.cell)

    def test_tile2(self, setup):
        cell = np.copy(setup.g.sc.cell)
        cell[:, :] *= 2
        t = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        assert np.allclose(cell, t.sc.cell)

    def test_sort(self, setup):
        t = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        ts = t.sort()
        t = setup.g.tile(2, 1).tile(2, 2).tile(2, 0)
        tS = t.sort()
        assert np.allclose(ts.xyz, tS.xyz)

    def test_tile3(self, setup):
        cell = np.copy(setup.g.sc.cell)
        cell[:, :] *= 2
        t1 = setup.g * 2
        cell = np.copy(setup.g.sc.cell)
        cell[0, :] *= 2
        t1 = setup.g * (2, 0)
        assert np.allclose(cell, t1.sc.cell)
        t = setup.g * ((2, 0), 'tile')
        assert np.allclose(cell, t.sc.cell)
        assert np.allclose(t1.xyz, t.xyz)
        cell[1, :] *= 2
        t1 = t * (2, 1)
        assert np.allclose(cell, t1.sc.cell)
        t = t * ((2, 1), 'tile')
        assert np.allclose(cell, t.sc.cell)
        assert np.allclose(t1.xyz, t.xyz)
        cell[2, :] *= 2
        t1 = t * (2, 2)
        assert np.allclose(cell, t1.sc.cell)
        t = t * ((2, 2), 'tile')
        assert np.allclose(cell, t.sc.cell)
        assert np.allclose(t1.xyz, t.xyz)

        # Full
        t = setup.g * [2, 2, 2]
        assert np.allclose(cell, t.sc.cell)
        assert np.allclose(t1.xyz, t.xyz)
        t = setup.g * ([2, 2, 2], 't')
        assert np.allclose(cell, t.sc.cell)
        assert np.allclose(t1.xyz, t.xyz)

    def test_tile4(self, setup):
        t1 = setup.g.tile(2, 0).tile(2, 2)
        t = setup.g * ([2, 0], 't') * [2, 2]
        assert np.allclose(t1.xyz, t.xyz)

    def test_tile5(self, setup):
        t = setup.g.tile(2, 0).tile(2, 2)
        assert np.allclose(t[:len(setup.g), :], setup.g.xyz)

    @pytest.mark.xfail(raises=ValueError)
    def test_repeat0(self, setup):
        t = setup.g.repeat(0, 0)

    def test_repeat1(self, setup):
        cell = np.copy(setup.g.sc.cell)
        cell[0, :] *= 2
        t = setup.g.repeat(2, 0)
        assert np.allclose(cell, t.sc.cell)
        cell[1, :] *= 2
        t = t.repeat(2, 1)
        assert np.allclose(cell, t.sc.cell)
        cell[2, :] *= 2
        t = t.repeat(2, 2)
        assert np.allclose(cell, t.sc.cell)

    def test_repeat2(self, setup):
        cell = np.copy(setup.g.sc.cell)
        cell[:, :] *= 2
        t = setup.g.repeat(2, 0).repeat(2, 1).repeat(2, 2)
        assert np.allclose(cell, t.sc.cell)

    def test_repeat3(self, setup):
        cell = np.copy(setup.g.sc.cell)
        cell[0, :] *= 2
        t1 = setup.g.repeat(2, 0)
        assert np.allclose(cell, t1.sc.cell)
        t = setup.g * ((2, 0), 'repeat')
        assert np.allclose(cell, t.sc.cell)
        assert np.allclose(t1.xyz, t.xyz)
        cell[1, :] *= 2
        t1 = t.repeat(2, 1)
        assert np.allclose(cell, t1.sc.cell)
        t = t * ((2, 1), 'r')
        assert np.allclose(cell, t.sc.cell)
        assert np.allclose(t1.xyz, t.xyz)
        cell[2, :] *= 2
        t1 = t.repeat(2, 2)
        assert np.allclose(cell, t1.sc.cell)
        t = t * ((2, 2), 'repeat')
        assert np.allclose(cell, t.sc.cell)
        assert np.allclose(t1.xyz, t.xyz)

        # Full
        t = setup.g * ([2, 2, 2], 'r')
        assert np.allclose(cell, t.sc.cell)
        assert np.allclose(t1.xyz, t.xyz)

    def test_repeat4(self, setup):
        t1 = setup.g.repeat(2, 0).repeat(2, 2)
        t = setup.g * ([2, 0], 'repeat') * ([2, 2], 'r')
        assert np.allclose(t1.xyz, t.xyz)

    def test_repeat5(self, setup):
        t = setup.g.repeat(2, 0).repeat(2, 2)
        assert np.allclose(t.xyz[::4, :], setup.g.xyz)

    def test_a2o1(self, setup):
        assert 0 == setup.g.a2o(0)
        assert setup.g.atom[0].no == setup.g.a2o(1)
        assert setup.g.no == setup.g.a2o(setup.g.na)

    def test_sub1(self, setup):
        assert len(setup.g.sub([0])) == 1
        assert len(setup.g.sub([0, 1])) == 2
        assert len(setup.g.sub([-1])) == 1

        assert np.allclose(setup.g.sub([0]).xyz, setup.g.sub(np.array([True, False])).xyz)

    def test_sub2(self, setup):
        assert len(setup.g.sub(range(1))) == 1
        assert len(setup.g.sub(range(2))) == 2

    def test_fxyz(self, setup):
        fxyz = setup.g.fxyz
        assert np.allclose(fxyz, [[0, 0, 0], [1./3, 1./3, 0]])
        assert np.allclose(np.dot(fxyz, setup.g.cell), setup.g.xyz)

    def test_axyz(self, setup):
        assert np.allclose(setup.g[:], setup.g.xyz[:])
        assert np.allclose(setup.g[0], setup.g.xyz[0, :])
        assert np.allclose(setup.g[2], setup.g.axyz(2))
        isc = setup.g.a2isc(2)
        off = setup.g.sc.offset(isc)
        assert np.allclose(setup.g.xyz[0] + off, setup.g.axyz(2))

    def test_atranspose_indices(self, setup):
        g = setup.g
        # All supercell indices
        ia2 = np.arange(g.na * g.n_s)
        ja2, ja1 = g.a2transpose(0, ia2)
        assert (ja2 < g.na).sum() == ja2.size
        assert (ja1 % g.na == 0).sum() == ja1.size

        IA1, IA2 = g.a2transpose(ja2, ja1)
        assert np.all(IA1 == 0)
        assert np.all(IA2 == ia2)

    def test_otranspose_indices(self, setup):
        g = setup.g
        # All supercell indices
        io2 = np.arange(g.no * g.n_s)
        jo2, jo1 = g.o2transpose(0, io2)
        assert (jo2 < g.no).sum() == jo2.size
        assert (jo1 % g.no == 0).sum() == jo1.size

        IO1, IO2 = g.o2transpose(jo2, jo1)
        assert np.all(IO1 == 0)
        assert np.all(IO2 == io2)

    def test_auc2sc(self, setup):
        g = setup.g
        # All supercell indices
        asc = g.uc2sc(0)
        assert asc.size == g.n_s
        assert (asc % g.na == 0).sum() == g.n_s

    def test_ouc2sc(self, setup):
        g = setup.g
        # All supercell indices
        asc = g.ouc2sc(0)
        assert asc.size == g.n_s
        assert (asc % g.no == 0).sum() == g.n_s

    def test_rij1(self, setup):
        assert np.allclose(setup.g.rij(0, 1), 1.42)
        assert np.allclose(setup.g.rij(0, [0, 1]), [0., 1.42])

    def test_orij1(self, setup):
        assert np.allclose(setup.g.orij(0, 2), 1.42)
        assert np.allclose(setup.g.orij(0, [0, 2]), [0., 1.42])

    def test_Rij1(self, setup):
        assert np.allclose(setup.g.Rij(0, 1), [1.42, 0, 0])

    def test_oRij1(self, setup):
        assert np.allclose(setup.g.oRij(0, 1), [0., 0, 0])
        assert np.allclose(setup.g.oRij(0, 2), [1.42, 0, 0])
        assert np.allclose(setup.g.oRij(0, [0, 1, 2]), [[0., 0, 0],
                                                        [0., 0, 0],
                                                        [1.42, 0, 0]])
        assert np.allclose(setup.g.oRij(0, 2), [1.42, 0, 0])

    def test_cut(self, setup):
        with pytest.warns(SislWarning) as warns:
            assert len(setup.g.cut(1, 1)) == 2
            assert len(setup.g.cut(2, 1)) == 1
            assert len(setup.g.cut(2, 1, 1)) == 1
        assert len(warns) == 2

    def test_cut2(self, setup):
        with pytest.warns(SislWarning) as warns:
            c1 = setup.g.cut(2, 1)
            c2 = setup.g.cut(2, 1, 1)
        assert len(warns) == 2
        assert np.allclose(c1.xyz[0, :], setup.g.xyz[0, :])
        assert np.allclose(c2.xyz[0, :], setup.g.xyz[1, :])

    def test_cut3(self, setup):
        nr = range(2, 5)
        g = setup.g.copy()
        for x in nr:
            gx = g.tile(x, 0)
            for y in nr:
                gy = gx.tile(y, 1)
                for z in nr:
                    gz = gy.tile(z, 2)
                    G = gz.cut(z, 2)
                    assert np.allclose(G.xyz, gy.xyz)
                    assert np.allclose(G.cell, gy.cell)
                G = gy.cut(y, 1)
                assert np.allclose(G.xyz, gx.xyz)
                assert np.allclose(G.cell, gx.cell)
            G = gx.cut(x, 0)
            assert np.allclose(G.xyz, g.xyz)
            assert np.allclose(G.cell, g.cell)

    def test_remove1(self, setup):
        assert len(setup.g.remove([0])) == 1
        assert len(setup.g.remove([])) == 2
        assert len(setup.g.remove([-1])) == 1
        assert len(setup.g.remove([-0])) == 1

    def test_remove2(self, setup):
        assert len(setup.g.remove(range(1))) == 1
        assert len(setup.g.remove(range(0))) == 2

    def test_copy(self, setup):
        assert setup.g == setup.g.copy()

    def test_nsc1(self, setup):
        sc = setup.g.sc.copy()
        nsc = np.copy(sc.nsc)
        sc.set_nsc([5, 5, 0])
        assert np.allclose([5, 5, 1], sc.nsc)
        assert len(sc.sc_off) == np.prod(sc.nsc)

    def test_nsc2(self, setup):
        sc = setup.g.sc.copy()
        nsc = np.copy(sc.nsc)
        sc.set_nsc([0, 1, 0])
        assert np.allclose([1, 1, 1], sc.nsc)
        assert len(sc.sc_off) == np.prod(sc.nsc)

    def test_rotation1(self, setup):
        rot = setup.g.rotate(180, [0, 0, 1])
        rot.sc.cell[2, 2] *= -1
        assert np.allclose(-rot.sc.cell, setup.g.sc.cell)
        assert np.allclose(-rot.xyz, setup.g.xyz)

        rot = setup.g.rotate(np.pi, [0, 0, 1], rad=True)
        rot.sc.cell[2, 2] *= -1
        assert np.allclose(-rot.sc.cell, setup.g.sc.cell)
        assert np.allclose(-rot.xyz, setup.g.xyz)

        rot = rot.rotate(180, [0, 0, 1])
        rot.sc.cell[2, 2] *= -1
        assert np.allclose(rot.sc.cell, setup.g.sc.cell)
        assert np.allclose(rot.xyz, setup.g.xyz)

    def test_rotation2(self, setup):
        rot = setup.g.rotate(180, [0, 0, 1], only='abc')
        rot.sc.cell[2, 2] *= -1
        assert np.allclose(-rot.sc.cell, setup.g.sc.cell)
        assert np.allclose(rot.xyz, setup.g.xyz)

        rot = setup.g.rotate(np.pi, [0, 0, 1], rad=True, only='abc')
        rot.sc.cell[2, 2] *= -1
        assert np.allclose(-rot.sc.cell, setup.g.sc.cell)
        assert np.allclose(rot.xyz, setup.g.xyz)

        rot = rot.rotate(180, [0, 0, 1], only='abc')
        rot.sc.cell[2, 2] *= -1
        assert np.allclose(rot.sc.cell, setup.g.sc.cell)
        assert np.allclose(rot.xyz, setup.g.xyz)

    def test_rotation3(self, setup):
        rot = setup.g.rotate(180, [0, 0, 1], only='xyz')
        assert np.allclose(rot.sc.cell, setup.g.sc.cell)
        assert np.allclose(-rot.xyz, setup.g.xyz)

        rot = setup.g.rotate(np.pi, [0, 0, 1], rad=True, only='xyz')
        assert np.allclose(rot.sc.cell, setup.g.sc.cell)
        assert np.allclose(-rot.xyz, setup.g.xyz)

        rot = rot.rotate(180, [0, 0, 1], only='xyz')
        assert np.allclose(rot.sc.cell, setup.g.sc.cell)
        assert np.allclose(rot.xyz, setup.g.xyz)

    def test_translate(self, setup):
        t = setup.g.translate([0, 0, 1])
        assert np.allclose(setup.g.xyz[:, 0], t.xyz[:, 0])
        assert np.allclose(setup.g.xyz[:, 1], t.xyz[:, 1])
        assert np.allclose(setup.g.xyz[:, 2] + 1, t.xyz[:, 2])
        t = setup.g.move([0, 0, 1])
        assert np.allclose(setup.g.xyz[:, 0], t.xyz[:, 0])
        assert np.allclose(setup.g.xyz[:, 1], t.xyz[:, 1])
        assert np.allclose(setup.g.xyz[:, 2] + 1, t.xyz[:, 2])

    def test_iter_block1(self, setup):
        for i, iaaspec in enumerate(setup.g.iter_species()):
            ia, a, spec = iaaspec
            assert i == ia
            assert setup.g.atom[ia] == a
        for ia, a, spec in setup.g.iter_species([1]):
            assert 1 == ia
            assert setup.g.atom[ia] == a
        for ia in setup.g:
            assert ia >= 0
        i = 0
        for ias, idx in setup.g.iter_block():
            for ia in ias:
                i += 1
        assert i == len(setup.g)

        i = 0
        for ias, idx in setup.g.iter_block(atom=1):
            for ia in ias:
                i += 1
        assert i == 1

    @pytest.mark.slow
    def test_iter_block2(self, setup):
        g = setup.g.tile(30, 0).tile(30, 1)
        i = 0
        for ias, _ in g.iter_block():
            i += len(ias)
        assert i == len(g)

    def test_iter_shape1(self, setup):
        i = 0
        for ias, _ in setup.g.iter_block(method='sphere'):
            i += len(ias)
        assert i == len(setup.g)
        i = 0
        for ias, _ in setup.g.iter_block(method='cube'):
            i += len(ias)
        assert i == len(setup.g)

    @pytest.mark.slow
    def test_iter_shape2(self, setup):
        g = setup.g.tile(30, 0).tile(30, 1)
        i = 0
        for ias, _ in g.iter_block(method='sphere'):
            i += len(ias)
        assert i == len(g)
        i = 0
        for ias, _ in g.iter_block(method='cube'):
            i += len(ias)
        assert i == len(g)
        i = 0
        for ias, _ in g.iter_block_shape(Cube(g.maxR() * 20)):
            i += len(ias)
        assert i == len(g)

    @pytest.mark.slow
    def test_iter_shape3(self, setup):
        g = setup.g.tile(50, 0).tile(50, 1)
        i = 0
        for ias, _ in g.iter_block(method='sphere'):
            i += len(ias)
        assert i == len(g)
        i = 0
        for ias, _ in g.iter_block(method='cube'):
            i += len(ias)
        assert i == len(g)
        i = 0
        for ias, _ in g.iter_block_shape(Sphere(g.maxR() * 20)):
            i += len(ias)
        assert i == len(g)

    def test_swap(self, setup):
        s = setup.g.swap(0, 1)
        for i in [0, 1, 2]:
            assert np.allclose(setup.g.xyz[::-1, i], s.xyz[:, i])

    def test_append1(self, setup):
        for axis in [0, 1, 2]:
            s = setup.g.append(setup.g, axis)
            assert len(s) == len(setup.g) * 2
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :]* 2)
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :]* 2)
            s = setup.g.prepend(setup.g, axis)
            assert len(s) == len(setup.g) * 2
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :]* 2)
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :]* 2)
            s = setup.g.append(setup.g.sc, axis)
            assert len(s) == len(setup.g)
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :]* 2)
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :]* 2)
            s = setup.g.prepend(setup.g.sc, axis)
            assert len(s) == len(setup.g)
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :]* 2)
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :]* 2)

    @pytest.mark.xfail(raises=ValueError)
    def test_append_xfail(self, setup):
        s = setup.g.append(setup.g, 0, align='not')

    @pytest.mark.xfail(raises=ValueError)
    def test_prepend_xfail(self, setup):
        s = setup.g.prepend(setup.g, 0, align='not')

    def test_append_prepend_align(self, setup):
        for axis in [0, 1, 2]:
            t = setup.g.sc.cell[axis, :].copy()
            t *= 10. / (t ** 2).sum() ** 0.5
            s1 = setup.g.copy()
            s2 = setup.g.translate(t)

            S = s1.append(s2, axis, align='min')
            s = setup.g.append(setup.g, axis)

            assert np.allclose(s.cell[axis, :], S.cell[axis, :])
            assert np.allclose(s.xyz, S.xyz)

            P = s2.prepend(s1, axis, align='min')
            p = setup.g.prepend(setup.g, axis)

            assert np.allclose(p.cell[axis, :], P.cell[axis, :])
            assert np.allclose(p.xyz, P.xyz)

    def test_swapaxes(self, setup):
        s = setup.g.swapaxes(0, 1)
        assert np.allclose(setup.g.xyz[:, 0], s.xyz[:, 1])
        assert np.allclose(setup.g.xyz[:, 1], s.xyz[:, 0])
        assert np.allclose(setup.g.cell[0, :], s.cell[1, :])
        assert np.allclose(setup.g.cell[1, :], s.cell[0, :])

    def test_center(self, setup):
        one = setup.g.center(atom=[0])
        assert np.allclose(setup.g[0], one)
        al = setup.g.center()
        assert np.allclose(np.mean(setup.g.xyz, axis=0), al)
        al = setup.g.center(what='mass')
        al = setup.g.center(what='mm(xyz)')

    @pytest.mark.xfail(raises=ValueError)
    def test_center_raise(self, setup):
        al = setup.g.center(what='unknown')

    def test___add1__(self, setup):
        n = len(setup.g)
        double = setup.g + setup.g + setup.g.sc
        assert len(double) == n * 2
        assert np.allclose(setup.g.cell * 2, double.cell)
        assert np.allclose(setup.g.xyz[:n, :], double.xyz[:n, :])

        double = (setup.g, 1) + setup.g
        d = setup.g.prepend(setup.g, 1)
        assert len(double) == n * 2
        assert np.allclose(setup.g.cell[::2, :], double.cell[::2, :])
        assert np.allclose(double.xyz, d.xyz)

        double = setup.g + (setup.g, 1)
        d = setup.g.append(setup.g, 1)
        assert len(double) == n * 2
        assert np.allclose(setup.g.cell[::2, :], double.cell[::2, :])
        assert np.allclose(double.xyz, d.xyz)

    def test___add2__(self, setup):
        g1 = setup.g.rotate(15, setup.g.cell[2, :])
        g2 = setup.g.rotate(30, setup.g.cell[2, :])

        assert g1 != g2
        assert g1 + g2 == g1.add(g2)
        assert g1 + g2 != g2.add(g1)
        assert g2 + g1 == g2.add(g1)
        assert g2 + g1 != g1.add(g2)
        for i in range(3):
            assert g1 + (g2, i) == g1.append(g2, i)
            assert (g1, i) + g2 == g2.append(g1, i)

    def test___mul__(self, setup):
        g = setup.g.copy()
        assert g * 2 == g.tile(2, 0).tile(2, 1).tile(2, 2)
        assert g * [2, 1] == g.tile(2, 1)
        assert g * (2, 2, 2) == g.tile(2, 0).tile(2, 1).tile(2, 2)
        assert g * [1, 2, 2] == g.tile(1, 0).tile(2, 1).tile(2, 2)
        assert g * [1, 3, 2] == g.tile(1, 0).tile(3, 1).tile(2, 2)
        assert g * ([1, 3, 2], 'r') == g.repeat(1, 0).repeat(3, 1).repeat(2, 2)
        assert g * ([1, 3, 2], 'repeat') == g.repeat(1, 0).repeat(3, 1).repeat(2, 2)
        assert g * ([1, 3, 2], 'tile') == g.tile(1, 0).tile(3, 1).tile(2, 2)
        assert g * ([1, 3, 2], 't') == g.tile(1, 0).tile(3, 1).tile(2, 2)
        assert g * ([3, 2], 't') == g.tile(3, 2)
        assert g * ([3, 2], 'r') == g.repeat(3, 2)

    def test_add(self, setup):
        double = setup.g.add(setup.g)
        assert len(double) == len(setup.g) * 2
        assert np.allclose(setup.g.cell, double.cell)
        double = setup.g.add(setup.g).add(setup.g.sc)
        assert len(double) == len(setup.g) * 2
        assert np.allclose(setup.g.cell * 2, double.cell)

    def test_insert(self, setup):
        double = setup.g.insert(0, setup.g)
        assert len(double) == len(setup.g) * 2
        assert np.allclose(setup.g.cell, double.cell)

    def test_a2o(self, setup):
        # There are 2 orbitals per C atom
        assert setup.g.a2o(1) == setup.g.atom[0].no
        assert np.all(setup.g.a2o(1, True) == [2, 3])
        setup.g.reorder()

    def test_o2a(self, setup):
        # There are 2 orbitals per C atom
        assert setup.g.o2a(2) == 1
        assert setup.g.o2a(3) == 1
        assert setup.g.o2a(4) == 2
        assert np.all(setup.g.o2a([0, 2, 4]) == [0, 1, 2])

    def test_angle(self, setup):
        # There are 2 orbitals per C atom
        g = Geometry([[0] * 3, [1, 0, 0]])
        g.angle([0])
        g.angle([0], ref=1)

    def test_2uc(self, setup):
        # functions for any-thing to UC
        assert setup.g.sc2uc(2) == 0
        assert np.all(setup.g.sc2uc([2, 3]) == [0, 1])
        assert setup.g.asc2uc(2) == 0
        assert np.all(setup.g.asc2uc([2, 3]) == [0, 1])
        assert setup.g.osc2uc(4) == 0
        assert setup.g.osc2uc(5) == 1
        assert np.all(setup.g.osc2uc([4, 5]) == [0, 1])

    def test_2sc(self, setup):
        # functions for any-thing to SC
        c = setup.g.cell

        # check indices
        assert np.all(setup.g.a2isc([1, 2]) == [[0,  0, 0],
                                                [-1, -1, 0]])
        assert np.all(setup.g.a2isc(2) == [-1, -1, 0])
        assert np.allclose(setup.g.a2sc(2), -c[0, :] - c[1, :])
        assert np.all(setup.g.o2isc([1, 5]) == [[0,  0, 0],
                                                [-1, -1, 0]])
        assert np.all(setup.g.o2isc(5) == [-1, -1, 0])
        assert np.allclose(setup.g.o2sc(5), -c[0, :] - c[1, :])

        # Check off-sets
        assert np.allclose(setup.g.a2sc([1, 2]), [[0.,  0., 0.],
                                                      -c[0, :] - c[1, :]])
        assert np.allclose(setup.g.o2sc([1, 5]), [[0.,  0., 0.],
                                                      -c[0, :] - c[1, :]])

    def test_reverse(self, setup):
        rev = setup.g.reverse()
        assert len(rev) == 2
        assert np.allclose(rev.xyz[::-1, :], setup.g.xyz)
        rev = setup.g.reverse(atom=list(range(len(setup.g))))
        assert len(rev) == 2
        assert np.allclose(rev.xyz[::-1, :], setup.g.xyz)

    def test_scale1(self, setup):
        two = setup.g.scale(2)
        assert len(two) == len(setup.g)
        assert np.allclose(two.xyz[:, :] / 2., setup.g.xyz)

    def test_close1(self, setup):
        three = range(3)
        for ia in setup.mol:
            i = setup.mol.close(ia, R=(0.1, 1.1), idx=three)
            if ia < 3:
                assert len(i[0]) == 1
            else:
                assert len(i[0]) == 0
            # Will only return results from [0,1,2]
            # but the fourth atom connects to
            # the third
            if ia in [0, 2, 3]:
                assert len(i[1]) == 1
            elif ia == 1:
                assert len(i[1]) == 2
            else:
                assert len(i[1]) == 0

    def test_close2(self, setup):
        mol = range(3, 5)
        for ia in setup.mol:
            i = setup.mol.close(ia, R=(0.1, 1.1), idx=mol)
            assert len(i) == 2
        i = setup.mol.close([100, 100, 100], R=0.1)
        assert len(i) == 0
        i = setup.mol.close([100, 100, 100], R=0.1, ret_rij=True)
        for el in i:
            assert len(el) == 0
        i = setup.mol.close([100, 100, 100], R=0.1, ret_rij=True, ret_xyz=True)
        for el in i:
            assert len(el) == 0

    @pytest.mark.slow
    def test_close4(self, setup):
        # 2 * 200 ** 2
        g = setup.g * (200, 200, 1)
        i = g.close(0, R=(0.1, 1.43))
        assert len(i) == 2
        assert len(i[0]) == 1
        assert len(i[1]) == 3

    def test_close_within1(self, setup):
        three = range(3)
        for ia in setup.mol:
            shapes = [Sphere(0.1, setup.mol[ia]),
                      Sphere(1.1, setup.mol[ia])]
            i = setup.mol.close(ia, R=(0.1, 1.1), idx=three)
            ii = setup.mol.within(shapes, idx=three)
            assert np.all(i[0] == ii[0])
            assert np.all(i[1] == ii[1])

    def test_close_within2(self, setup):
        g = setup.g.repeat(6, 0).repeat(6, 1)
        for ia in g:
            shapes = [Sphere(0.1, g[ia]),
                      Sphere(1.5, g[ia])]
            i = g.close(ia, R=(0.1, 1.5))
            ii = g.within(shapes)
            assert np.all(i[0] == ii[0])
            assert np.all(i[1] == ii[1])

    def test_close_within3(self, setup):
        g = setup.g.repeat(6, 0).repeat(6, 1)
        args = {'ret_xyz': True, 'ret_rij': True}
        for ia in g:
            shapes = [Sphere(0.1, g[ia]),
                      Sphere(1.5, g[ia])]
            i, xa, d = g.close(ia, R=(0.1, 1.5), **args)
            ii, xai, di = g.within(shapes, **args)
            for j in [0, 1]:
                assert np.all(i[j] == ii[j])
                assert np.allclose(xa[j], xai[j])
                assert np.allclose(d[j], di[j])

    def test_within_inf1(self, setup):
        g = setup.g.translate([0.05] * 3)
        sc_3x3 = g.sc.tile(3, 0).tile(3, 1)
        assert len(g.within_inf(sc_3x3)[0]) == len(g) * 3 ** 2

    def test_within_inf2(self, setup):
        g = setup.mol.translate([0.05] * 3)
        sc = SuperCell(1.5)
        for o in range(10):
            origo = [o - 0.5, -0.5, -0.5]
            sc.origo = origo
            idx = g.within_inf(sc)[0]
            assert len(idx) == 1
            assert idx[0] == o

    def test_within_inf_duplicates(self, setup):
        g = setup.g.copy()
        sc_3x3 = g.sc.tile(3, 0).tile(3, 1)
        assert len(g.within_inf(sc_3x3)[0]) == len(g) * 3 ** 2 + 7 # 3 per vector and 1 in the upper right corner

    def test_close_sizes(self, setup):
        point = 0

        # Return index
        idx = setup.mol.close(point, R=.1)
        assert len(idx) == 1
        # Return index of two things
        idx = setup.mol.close(point, R=(.1, 1.1))
        assert len(idx) == 2
        assert len(idx[0]) == 1
        assert not isinstance(idx[0], list)
        # Longer
        idx = setup.mol.close(point, R=(.1, 1.1, 2.1))
        assert len(idx) == 3
        assert len(idx[0]) == 1

        # Return index
        idx = setup.mol.close(point, R=.1, ret_xyz=True)
        assert len(idx) == 2
        assert len(idx[0]) == 1
        assert len(idx[1]) == 1
        assert idx[1].shape[0] == 1 # equivalent to above
        assert idx[1].shape[1] == 3

        # Return index of two things
        idx = setup.mol.close(point, R=(.1, 1.1), ret_xyz=True)
        # [[idx-1, idx-2], [coord-1, coord-2]]
        assert len(idx) == 2
        assert len(idx[0]) == 2
        assert len(idx[1]) == 2
        # idx-1
        assert len(idx[0][0].shape) == 1
        assert idx[0][0].shape[0] == 1
        # idx-2
        assert idx[0][1].shape[0] == 1
        # coord-1
        assert len(idx[1][0].shape) == 2
        assert idx[1][0].shape[1] == 3
        # coord-2
        assert idx[1][1].shape[1] == 3

        # Return index of two things
        idx = setup.mol.close(point, R=(.1, 1.1), ret_xyz=True, ret_rij=True)
        # [[idx-1, idx-2], [coord-1, coord-2], [dist-1, dist-2]]
        assert len(idx) == 3
        assert len(idx[0]) == 2
        assert len(idx[1]) == 2
        # idx-1
        assert len(idx[0][0].shape) == 1
        assert idx[0][0].shape[0] == 1
        # idx-2
        assert idx[0][1].shape[0] == 1
        # coord-1
        assert len(idx[1][0].shape) == 2
        assert idx[1][0].shape[1] == 3
        # coord-2
        assert idx[1][1].shape[1] == 3
        # dist-1
        assert len(idx[2][0].shape) == 1
        assert idx[2][0].shape[0] == 1
        # dist-2
        assert idx[2][1].shape[0] == 1

        # Return index of two things
        idx = setup.mol.close(point, R=(.1, 1.1), ret_rij=True)
        # [[idx-1, idx-2], [dist-1, dist-2]]
        assert len(idx) == 2
        assert len(idx[0]) == 2
        assert len(idx[1]) == 2
        # idx-1
        assert len(idx[0][0].shape) == 1
        assert idx[0][0].shape[0] == 1
        # idx-2
        assert idx[0][1].shape[0] == 1
        # dist-1
        assert len(idx[1][0].shape) == 1
        assert idx[1][0].shape[0] == 1
        # dist-2
        assert idx[1][1].shape[0] == 1

    def test_close_sizes_none(self, setup):
        point = [100., 100., 100.]

        # Return index
        idx = setup.mol.close(point, R=.1)
        assert len(idx) == 0
        # Return index of two things
        idx = setup.mol.close(point, R=(.1, 1.1))
        assert len(idx) == 2
        assert len(idx[0]) == 0
        assert not isinstance(idx[0], list)
        # Longer
        idx = setup.mol.close(point, R=(.1, 1.1, 2.1))
        assert len(idx) == 3
        assert len(idx[0]) == 0

        # Return index
        idx = setup.mol.close(point, R=.1, ret_xyz=True)
        assert len(idx) == 2
        assert len(idx[0]) == 0
        assert len(idx[1]) == 0
        assert idx[1].shape[0] == 0 # equivalent to above
        assert idx[1].shape[1] == 3

        # Return index of two things
        idx = setup.mol.close(point, R=(.1, 1.1), ret_xyz=True)
        # [[idx-1, idx-2], [coord-1, coord-2]]
        assert len(idx) == 2
        assert len(idx[0]) == 2
        assert len(idx[1]) == 2
        # idx-1
        assert len(idx[0][0].shape) == 1
        assert idx[0][0].shape[0] == 0
        # idx-2
        assert idx[0][1].shape[0] == 0
        # coord-1
        assert len(idx[1][0].shape) == 2
        assert idx[1][0].shape[1] == 3
        # coord-2
        assert idx[1][1].shape[1] == 3

        # Return index of two things
        idx = setup.mol.close(point, R=(.1, 1.1), ret_xyz=True, ret_rij=True)
        # [[idx-1, idx-2], [coord-1, coord-2], [dist-1, dist-2]]
        assert len(idx) == 3
        assert len(idx[0]) == 2
        assert len(idx[1]) == 2
        # idx-1
        assert len(idx[0][0].shape) == 1
        assert idx[0][0].shape[0] == 0
        # idx-2
        assert idx[0][1].shape[0] == 0
        # coord-1
        assert len(idx[1][0].shape) == 2
        assert idx[1][0].shape[0] == 0
        assert idx[1][0].shape[1] == 3
        # coord-2
        assert idx[1][1].shape[0] == 0
        assert idx[1][1].shape[1] == 3
        # dist-1
        assert len(idx[2][0].shape) == 1
        assert idx[2][0].shape[0] == 0
        # dist-2
        assert idx[2][1].shape[0] == 0

        # Return index of two things
        idx = setup.mol.close(point, R=(.1, 1.1), ret_rij=True)
        # [[idx-1, idx-2], [dist-1, dist-2]]
        assert len(idx) == 2
        assert len(idx[0]) == 2
        assert len(idx[1]) == 2
        # idx-1
        assert len(idx[0][0].shape) == 1
        assert idx[0][0].shape[0] == 0
        # idx-2
        assert idx[0][1].shape[0] == 0
        # dist-1
        assert len(idx[1][0].shape) == 1
        assert idx[1][0].shape[0] == 0
        # dist-2
        assert idx[1][1].shape[0] == 0

    def test_sparserij1(self, setup):
        rij = setup.g.sparserij()

    def test_bond_correct(self, setup):
        # Create ribbon
        rib = setup.g.tile(2, 1)
        # Convert the last atom to a H atom
        rib.atom[-1] = Atom[1]
        ia = len(rib) - 1
        # Get bond-length
        idx, d = rib.close(ia, R=(.1, 1000), ret_rij=True)
        i = np.argmin(d[1])
        d = d[1][i]
        rib.bond_correct(ia, idx[1][i])
        idx, d2 = rib.close(ia, R=(.1, 1000), ret_rij=True)
        i = np.argmin(d2[1])
        d2 = d2[1][i]
        assert d != d2
        # Calculate actual radius
        assert d2 == (Atom[1].radius() + Atom[6].radius())

    def test_unit_cell_estimation1(self, setup):
        # Create new geometry with only the coordinates
        # and atoms
        geom = Geometry(setup.g.xyz, Atom[6])
        # Only check the two distances we know have sizes
        for i in range(2):
            # It cannot guess skewed axis
            assert not np.allclose(geom.cell[i, :], setup.g.cell[i, :])

    def test_unit_cell_estimation2(self, setup):
        # Create new geometry with only the coordinates
        # and atoms
        s1 = SuperCell([2, 2, 2])
        g1 = Geometry([[0, 0, 0], [1, 1, 1]], sc=s1)
        g2 = Geometry(np.copy(g1.xyz))
        assert np.allclose(g1.cell, g2.cell)

        # Assert that it correctly calculates the bond-length in the
        # directions of actual distance
        g1 = Geometry([[0, 0, 0], [1, 1, 0]], atom='H', sc=s1)
        g2 = Geometry(np.copy(g1.xyz))
        for i in range(2):
            assert np.allclose(g1.cell[i, :], g2.cell[i, :])
        assert not np.allclose(g1.cell[2, :], g2.cell[2, :])

    @pytest.mark.xfail(raises=ValueError)
    def test_distance1(self, setup):
        geom = Geometry(setup.g.xyz, Atom[6])
        # maxR is undefined
        d = geom.distance()

    @pytest.mark.xfail(raises=ValueError)
    def test_distance2(self, setup):
        geom = Geometry(setup.g.xyz, Atom[6])
        d = geom.distance(R=1.42, method='unknown_numpy_function')

    def test_distance3(self, setup):
        geom = setup.g.copy()
        d = geom.distance()
        assert len(d) == 1
        assert np.allclose(d, [1.42])

    def test_distance4(self, setup):
        geom = setup.g.copy()
        d = geom.distance(method=np.min)
        assert len(d) == 1
        assert np.allclose(d, [1.42])
        d = geom.distance(method=np.max)
        assert len(d) == 1
        assert np.allclose(d, [1.42])
        d = geom.distance(method='max')
        assert len(d) == 1
        assert np.allclose(d, [1.42])

    def test_distance5(self, setup):
        geom = setup.g.copy()
        d = geom.distance(R=np.inf)
        assert len(d) == 6
        d = geom.distance(0, R=1.42)
        assert len(d) == 1
        assert np.allclose(d, [1.42])

    def test_distance6(self, setup):
        # Create a 1D chain
        geom = Geometry([0]*3, Atom(1, R=1.), sc=1)
        geom.set_nsc([77, 1, 1])
        d = geom.distance(0)
        assert len(d) == 1
        assert np.allclose(d, [1.])

        # Do twice
        d = geom.distance(R=2)
        assert len(d) == 2
        assert np.allclose(d, [1., 2.])

        # Do all
        d = geom.distance(R=np.inf)
        assert len(d) == 77 // 2
        # Add one due arange not adding the last item
        assert np.allclose(d, range(1, 78 // 2))

        # Create a 2D grid
        geom.set_nsc([3, 3, 1])
        d = geom.distance(R=2, tol=[.4, .3, .2, .1])
        assert len(d) == 2 # 1, sqrt(2)
        # Add one due arange not adding the last item
        assert np.allclose(d, [1, 2 ** .5])

        # Create a 2D grid
        geom.set_nsc([5, 5, 1])
        d = geom.distance(R=2, tol=[.4, .3, .2, .1])
        assert len(d) == 3 # 1, sqrt(2), 2
        # Add one due arange not adding the last item
        assert np.allclose(d, [1, 2 ** .5, 2])

    def test_distance7(self, setup):
        # Create a 1D chain
        geom = Geometry([0]*3, Atom(1, R=1.), sc=1)
        geom.set_nsc([77, 1, 1])
        # Try with a short R and a long tolerance list
        # We know that the tolerance list prevails, because
        d = geom.distance(R=1, tol=np.ones(10) * .5)
        assert len(d) == 1
        assert np.allclose(d, [1.])

    def test_distance8(self, setup):
        geom = Geometry([0]*3, Atom(1, R=1.), sc=1)
        geom.set_nsc([77, 1, 1])
        d = geom.distance(0, method='min')
        assert len(d) == 1
        d = geom.distance(0, method='median')
        assert len(d) == 1
        d = geom.distance(0, method='mode')
        assert len(d) == 1

    def test_optimize_nsc1(self, setup):
        # Create a 1D chain
        geom = Geometry([0]*3, Atom(1, R=1.), sc=1)
        geom.set_nsc([77, 77, 77])
        assert np.allclose(geom.optimize_nsc(), [3, 3, 3])
        geom.set_nsc([77, 77, 77])
        assert np.allclose(geom.optimize_nsc(1), [77, 3, 77])
        geom.set_nsc([77, 77, 77])
        assert np.allclose(geom.optimize_nsc([0, 2]), [3, 77, 3])
        geom.set_nsc([77, 77, 77])
        assert np.allclose(geom.optimize_nsc([0, 2], R=2.00000001), [5, 77, 5])
        geom.set_nsc([1, 1, 1])
        assert np.allclose(geom.optimize_nsc([0, 2], R=2.0000001), [5, 1, 5])
        geom.set_nsc([5, 1, 5])
        assert np.allclose(geom.optimize_nsc([0, 2], R=0.9999), [1, 1, 1])

    def test_optimize_nsc2(self, setup):
        # 2 ** 0.5 ensures lattice vectors with length 1
        geom = sisl_geom.fcc(2 ** 0.5, Atom(1, R=1.0001))
        geom.set_nsc([77, 77, 77])
        assert np.allclose(geom.optimize_nsc(), [3, 3, 3])
        geom.set_nsc([77, 77, 77])
        assert np.allclose(geom.optimize_nsc(1), [77, 3, 77])
        geom.set_nsc([77, 77, 77])
        assert np.allclose(geom.optimize_nsc([0, 2]), [3, 77, 3])
        geom.set_nsc([77, 77, 77])
        assert np.allclose(geom.optimize_nsc([0, 2], R=2.000001), [5, 77, 5])
        geom.set_nsc([1, 1, 1])
        assert np.allclose(geom.optimize_nsc([0, 2], R=2.0000001), [5, 1, 5])
        geom.set_nsc([5, 1, 5])
        assert np.allclose(geom.optimize_nsc([0, 2], R=0.9999), [1, 1, 1])

    def test_argumentparser1(self, setup):
        setup.g.ArgumentParser()
        setup.g.ArgumentParser(**setup.g._ArgumentParser_args_single())

    def test_argumentparser2(self, setup, **kwargs):
        p, ns = setup.g.ArgumentParser(**kwargs)

        # Try all options
        opts = ['--origin',
                '--center-of', 'mass',
                '--center-of', 'xyz',
                '--center-of', 'position',
                '--center-of', 'cell',
                '--unit-cell', 'translate',
                '--unit-cell', 'mod',
                '--rotate', '90', 'x',
                '--rotate', '90', 'y',
                '--rotate', '90', 'z',
                '--add', '0,0,0', '6',
                '--swap', '0', '1',
                '--repeat', '2', 'x',
                '--repeat', '2', 'y',
                '--repeat', '2', 'z',
                '--tile', '2', 'x',
                '--tile', '2', 'y',
                '--tile', '2', 'z',
                '--cut', '2', 'z',
                '--cut', '2', 'y',
                '--cut', '2', 'x',
        ]
        if kwargs.get('limit_arguments', True):
            opts.extend(['--rotate', '-90', 'x',
                         '--rotate', '-90', 'y',
                         '--rotate', '-90', 'z'])
        else:
            opts.extend(['--rotate-x', ' -90',
                         '--rotate-y', ' -90',
                         '--rotate-z', ' -90',
                         '--repeat-x', '2',
                         '--repeat-y', '2',
                         '--repeat-z', '2'])

        args = p.parse_args(opts, namespace=ns)

        if len(kwargs) == 0:
            self.test_argumentparser2(setup, **setup.g._ArgumentParser_args_single())

    def test_set_sc(self, setup):
        # Create new geometry with only the coordinates
        # and atoms
        s1 = SuperCell([2, 2, 2])
        g1 = Geometry([[0, 0, 0], [1, 1, 1]], sc=[2, 2, 1])
        g1.set_sc(s1)
        assert g1.sc == s1

    def test_attach1(self, setup):
        g = setup.g.attach(0, setup.mol, 0, dist=1.42, axis=2)
        g = setup.g.attach(0, setup.mol, 0, dist='calc', axis=2)
        g = setup.g.attach(0, setup.mol, 0, dist=[0, 0, 1.42])

    def test_mirror1(self, setup):
        for plane in ['xy', 'xz', 'yz']:
            setup.g.mirror(plane)

    def test_pickle(self, setup):
        import pickle as p
        s = p.dumps(setup.g)
        n = p.loads(s)
        assert n == setup.g

    def test_geometry_names(self):
        g = sisl_geom.graphene()

        assert len(g.names) == 0
        g['A'] = 1
        assert len(g.names) == 1
        g['B'] = [1, 2]
        assert len(g.names) == 2
        g.names.delete_name('B')
        assert len(g.names) == 1

        # Add new group
        g['B'] = [0, 2]

        for name in g.names:
            assert name in ['A', 'B']

        str(g)

        assert np.allclose(g['B'], g[[0, 2], :])
        assert np.allclose(g.axyz('B'), g[[0, 2], :])

        del g.names['B']
        assert len(g.names) == 1

    @pytest.mark.xfail(raises=SislError)
    def test_geometry_groups_raise(self):
        g = sisl_geom.graphene()
        g['A'] = 1
        g['A'] = [1, 2]

    @pytest.mark.parametrize("geometry", [sisl_geom.graphene(),
                                          sisl_geom.diamond(),
                                          sisl_geom.sc(1.4, Atom[1]),
                                          sisl_geom.fcc(1.4, Atom[1]),
                                          sisl_geom.bcc(1.4, Atom[1]),
                                          sisl_geom.hcp(1.4, Atom[1])])
    def test_geometry_as_primary(self, geometry):
        prod = itertools.product
        x_reps = [1, 4, 3]
        y_reps = [1, 4, 5]
        z_reps = [1, 4, 6]
        tile_rep = ['r', 't']

        na_primary = len(geometry)
        for x, y, z in prod(x_reps, y_reps, z_reps):
            if x == y == z == 1:
                continue
            for a, b, c in prod(tile_rep, tile_rep, tile_rep):
                G = ((geometry * ([x, 1, 1], a)) * ([1, y, 1], b)) * ([1, 1, z], c)
                p = G.as_primary(na_primary)
                assert np.allclose(p.xyz, geometry.xyz)
                assert np.allclose(p.cell, geometry.cell)

    def test_geometry_as_primary_without_super(self):
        g = sisl_geom.graphene()
        p = g.as_primary(len(g))
        assert g == p

        G = g.tile(2, 0).tile(3, 1)
        p, supercell = G.as_primary(len(g), ret_super=True)
        assert np.allclose(p.xyz, g.xyz)
        assert np.allclose(p.cell, g.cell)
        assert np.all(supercell == [2, 3, 1])

    # Test ASE (but only fail if present)

    def test_geometry_toASE(self):
        ase = pytest.importorskip("ase")
        sisl_geom.graphene().toASE()

    def test_geometry_fromASE(self):
        ase = pytest.importorskip("ase")
        gr = sisl_geom.graphene()
        ase_geom = gr.toASE()
        from_ase = Geometry.fromASE(ase_geom)
        assert np.allclose(gr.xyz, from_ase.xyz)
        assert np.allclose(gr.cell, from_ase.cell)
        assert np.allclose(gr.atoms.Z, from_ase.atoms.Z)

    def test_geometry_overlapping_atoms(self):
        gr22 = sisl_geom.graphene().tile(2, 0).tile(2, 1)
        gr44 = gr22.tile(2, 0).tile(2, 1)
        offset = np.array([0.2, 0.4, 0.4])
        gr22 = gr22.translate(offset)
        idx = np.arange(gr22.na)
        np.random.shuffle(idx)
        gr22 = gr22.sub(idx)
        idx22, idx44 = gr22.overlap(gr44, offset=-offset)
        assert np.allclose(idx22, np.arange(gr22.na))
        assert np.allclose(idx44, idx)
