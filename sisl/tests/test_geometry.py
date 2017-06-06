from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np
import warnings as warn

from sisl import Sphere
from sisl import Geometry, Atom, SuperCell


@attr('geometry')
class TestGeometry(object):

    def setUp(self):
        bond = 1.42
        sq3h = 3.**.5 * 0.5
        self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                      [1.5, -sq3h, 0.],
                                      [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])
        C = Atom(Z=6, R=bond * 1.01, orbs=2)
        self.g = Geometry(np.array([[0., 0., 0.],
                                    [1., 0., 0.]], np.float64) * bond,
                          atom=C, sc=self.sc)

        self.mol = Geometry([[i, 0, 0] for i in range(10)], sc=[50])

    def tearDown(self):
        del self.g
        del self.sc
        del self.mol

    def test_objects(self):
        # just make sure __repr__ works
        print(self.g)
        assert_true(len(self.g) == 2)
        assert_true(len(self.g.xyz) == 2)
        assert_true(np.allclose(self.g[0], np.zeros([3])))

        i = 0
        for ia in self.g:
            i += 1
        assert_true(i == len(self.g))
        assert_true(self.g.no_s == 2 * len(self.g) * np.prod(self.g.sc.nsc))

    def test_properties(self):
        assert_true(2 == len(self.g))
        assert_true(2 == self.g.na)
        assert_true(3*3 == self.g.n_s)
        assert_true(2*3*3 == self.g.na_s)
        assert_true(2*2 == self.g.no)
        assert_true(2*2*3*3 == self.g.no_s)

    def test_iter1(self):
        i = 0
        for ia in self.g:
            i += 1
        assert_true(i == 2)

    def test_iter2(self):
        for ia in self.g:
            assert_true(np.allclose(self.g[ia], self.g.xyz[ia, :]))

    def test_iter3(self):
        i = 0
        for ia, io in self.g.iter_orbitals(0):
            assert_equal(ia, 0)
            assert_true(io < 2)
            i += 1
        for ia, io in self.g.iter_orbitals(1):
            assert_equal(ia, 1)
            assert_true(io < 2)
            i += 1
        assert_true(i == 4)
        i = 0
        for ia, io in self.g.iter_orbitals():
            assert_true(ia in [0, 1])
            assert_true(io < 2)
            i += 1
        assert_true(i == 4)

    def test_tile1(self):
        cell = np.copy(self.g.sc.cell)
        cell[0, :] *= 2
        t = self.g.tile(2, 0)
        assert_true(np.allclose(cell, t.sc.cell))
        cell[1, :] *= 2
        t = t.tile(2, 1)
        assert_true(np.allclose(cell, t.sc.cell))
        cell[2, :] *= 2
        t = t.tile(2, 2)
        assert_true(np.allclose(cell, t.sc.cell))

    def test_tile2(self):
        cell = np.copy(self.g.sc.cell)
        cell[:, :] *= 2
        t = self.g.tile(2, 0).tile(2, 1).tile(2, 2)
        assert_true(np.allclose(cell, t.sc.cell))

    def test_tile3(self):
        cell = np.copy(self.g.sc.cell)
        cell[0, :] *= 2
        t1 = self.g * (2, 0)
        assert_true(np.allclose(cell, t1.sc.cell))
        t = self.g * ((2, 0), 'tile')
        assert_true(np.allclose(cell, t.sc.cell))
        assert_true(np.allclose(t1.xyz, t.xyz))
        cell[1, :] *= 2
        t1 = t * (2, 1)
        assert_true(np.allclose(cell, t1.sc.cell))
        t = t * ((2, 1), 'tile')
        assert_true(np.allclose(cell, t.sc.cell))
        assert_true(np.allclose(t1.xyz, t.xyz))
        cell[2, :] *= 2
        t1 = t * (2, 2)
        assert_true(np.allclose(cell, t1.sc.cell))
        t = t * ((2, 2), 'tile')
        assert_true(np.allclose(cell, t.sc.cell))
        assert_true(np.allclose(t1.xyz, t.xyz))

        # Full
        t = self.g * [2, 2, 2]
        assert_true(np.allclose(cell, t.sc.cell))
        assert_true(np.allclose(t1.xyz, t.xyz))
        t = self.g * ([2, 2, 2], 't')
        assert_true(np.allclose(cell, t.sc.cell))
        assert_true(np.allclose(t1.xyz, t.xyz))

    def test_tile4(self):
        t1 = self.g.tile(2, 0).tile(2, 2)
        t = self.g * ([2, 0], 't') * [2, 2]
        assert_true(np.allclose(t1.xyz, t.xyz))

    def test_repeat1(self):
        cell = np.copy(self.g.sc.cell)
        cell[0, :] *= 2
        t = self.g.repeat(2, 0)
        assert_true(np.allclose(cell, t.sc.cell))
        cell[1, :] *= 2
        t = t.repeat(2, 1)
        assert_true(np.allclose(cell, t.sc.cell))
        cell[2, :] *= 2
        t = t.repeat(2, 2)
        assert_true(np.allclose(cell, t.sc.cell))

    def test_repeat2(self):
        cell = np.copy(self.g.sc.cell)
        cell[:, :] *= 2
        t = self.g.repeat(2, 0).repeat(2, 1).repeat(2, 2)
        assert_true(np.allclose(cell, t.sc.cell))

    def test_repeat3(self):
        cell = np.copy(self.g.sc.cell)
        cell[0, :] *= 2
        t1 = self.g.repeat(2, 0)
        assert_true(np.allclose(cell, t1.sc.cell))
        t = self.g * ((2, 0), 'repeat')
        assert_true(np.allclose(cell, t.sc.cell))
        assert_true(np.allclose(t1.xyz, t.xyz))
        cell[1, :] *= 2
        t1 = t.repeat(2, 1)
        assert_true(np.allclose(cell, t1.sc.cell))
        t = t * ((2, 1), 'r')
        assert_true(np.allclose(cell, t.sc.cell))
        assert_true(np.allclose(t1.xyz, t.xyz))
        cell[2, :] *= 2
        t1 = t.repeat(2, 2)
        assert_true(np.allclose(cell, t1.sc.cell))
        t = t * ((2, 2), 'repeat')
        assert_true(np.allclose(cell, t.sc.cell))
        assert_true(np.allclose(t1.xyz, t.xyz))

        # Full
        t = self.g * ([2, 2, 2], 'r')
        assert_true(np.allclose(cell, t.sc.cell))
        assert_true(np.allclose(t1.xyz, t.xyz))

    def test_repeat4(self):
        t1 = self.g.repeat(2, 0).repeat(2, 2)
        t = self.g * ([2, 0], 'repeat') * ([2, 2], 'r')
        assert_true(np.allclose(t1.xyz, t.xyz))

    def test_a2o1(self):
        assert_true(0 == self.g.a2o(0))
        assert_true(self.g.atom[0].orbs == self.g.a2o(1))
        assert_true(self.g.no == self.g.a2o(self.g.na))

    def test_sub1(self):
        assert_true(len(self.g.sub([0])) == 1)
        assert_true(len(self.g.sub([0, 1])) == 2)
        assert_true(len(self.g.sub([-1])) == 1)

    def test_sub2(self):
        assert_true(len(self.g.sub(range(1))) == 1)
        assert_true(len(self.g.sub(range(2))) == 2)

    def test_fxyz(self):
        assert_true(np.allclose(self.g.fxyz, [[0,    0, 0],
                                              [1./3, 1./3, 0]]))

    def test_axyz(self):
        assert_true(np.allclose(self.g[:], self.g.xyz[:]))
        assert_true(np.allclose(self.g[0], self.g.xyz[0, :]))
        assert_true(np.allclose(self.g[2], self.g.axyz(2)))
        isc = self.g.a2isc(2)
        off = self.g.sc.offset(isc)
        assert_true(np.allclose(self.g.xyz[0] + off, self.g.axyz(2)))

    def test_rij1(self):
        assert_true(np.allclose(self.g.rij(0, 1), 1.42))
        assert_true(np.allclose(self.g.rij(0, [0, 1]), [0., 1.42]))

    def test_orij1(self):
        assert_true(np.allclose(self.g.orij(0, 2), 1.42))
        assert_true(np.allclose(self.g.orij(0, [0, 2]), [0., 1.42]))

    def test_cut(self):
        with warn.catch_warnings():
            warn.simplefilter('ignore', category=UserWarning)
            assert_true(len(self.g.cut(1, 1)) == 2)
            assert_true(len(self.g.cut(2, 1)) == 1)
            assert_true(len(self.g.cut(2, 1, 1)) == 1)

    def test_cut2(self):
        c1 = self.g.cut(2, 1)
        c2 = self.g.cut(2, 1, 1)
        assert_true(np.allclose(c1.xyz[0, :], self.g.xyz[0, :]))
        assert_true(np.allclose(c2.xyz[0, :], self.g.xyz[1, :]))

    def test_remove1(self):
        assert_true(len(self.g.remove([0])) == 1)
        assert_true(len(self.g.remove([])) == 2)
        assert_true(len(self.g.remove([-1])) == 1)
        assert_true(len(self.g.remove([-0])) == 1)

    def test_remove2(self):
        assert_true(len(self.g.remove(range(1))) == 1)
        assert_true(len(self.g.remove(range(0))) == 2)

    def test_copy(self):
        assert_true(self.g == self.g.copy())

    def test_nsc1(self):
        nsc = np.copy(self.g.nsc)
        self.g.sc.set_nsc([5, 5, 0])
        assert_true(np.allclose([5, 5, 1], self.g.nsc))
        assert_true(len(self.g.sc_off) == np.prod(self.g.nsc))

    def test_nsc2(self):
        nsc = np.copy(self.g.nsc)
        self.g.sc.set_nsc([0, 1, 0])
        assert_true(np.allclose([1, 1, 1], self.g.nsc))
        assert_true(len(self.g.sc_off) == np.prod(self.g.nsc))

    def test_rotation1(self):
        rot = self.g.rotate(180, [0, 0, 1])
        rot.sc.cell[2, 2] *= -1
        assert_true(np.allclose(-rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(-rot.xyz, self.g.xyz))

        rot = self.g.rotate(np.pi, [0, 0, 1], radians=True)
        rot.sc.cell[2, 2] *= -1
        assert_true(np.allclose(-rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(-rot.xyz, self.g.xyz))

        rot = rot.rotate(180, [0, 0, 1])
        rot.sc.cell[2, 2] *= -1
        assert_true(np.allclose(rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(rot.xyz, self.g.xyz))

    def test_rotation2(self):
        rot = self.g.rotate(180, [0, 0, 1], only='abc')
        rot.sc.cell[2, 2] *= -1
        assert_true(np.allclose(-rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(rot.xyz, self.g.xyz))

        rot = self.g.rotate(np.pi, [0, 0, 1], radians=True, only='abc')
        rot.sc.cell[2, 2] *= -1
        assert_true(np.allclose(-rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(rot.xyz, self.g.xyz))

        rot = rot.rotate(180, [0, 0, 1], only='abc')
        rot.sc.cell[2, 2] *= -1
        assert_true(np.allclose(rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(rot.xyz, self.g.xyz))

    def test_rotation3(self):
        rot = self.g.rotate(180, [0, 0, 1], only='xyz')
        assert_true(np.allclose(rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(-rot.xyz, self.g.xyz))

        rot = self.g.rotate(np.pi, [0, 0, 1], radians=True, only='xyz')
        assert_true(np.allclose(rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(-rot.xyz, self.g.xyz))

        rot = rot.rotate(180, [0, 0, 1], only='xyz')
        assert_true(np.allclose(rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(rot.xyz, self.g.xyz))

    def test_rotation4(self):
        rot = self.g.rotatea(180, only='xyz')
        rot = self.g.rotateb(180, only='xyz')
        rot = self.g.rotatec(180, only='xyz')

    def test_translate(self):
        t = self.g.translate([0, 0, 1])
        assert_true(np.allclose(self.g.xyz[:, 0], t.xyz[:, 0]))
        assert_true(np.allclose(self.g.xyz[:, 1], t.xyz[:, 1]))
        assert_true(np.allclose(self.g.xyz[:, 2] + 1, t.xyz[:, 2]))
        t = self.g.move([0, 0, 1])
        assert_true(np.allclose(self.g.xyz[:, 0], t.xyz[:, 0]))
        assert_true(np.allclose(self.g.xyz[:, 1], t.xyz[:, 1]))
        assert_true(np.allclose(self.g.xyz[:, 2] + 1, t.xyz[:, 2]))

    def test_iter_block1(self):
        for i, iaaspec in enumerate(self.g.iter_species()):
            ia, a, spec = iaaspec
            assert_true(i == ia)
            assert_true(self.g.atom[ia] == a)
        for ia, a, spec in self.g.iter_species([1]):
            assert_true(1 == ia)
            assert_true(self.g.atom[ia] == a)
        for ia in self.g:
            assert_true(ia >= 0)
        i = 0
        for ias, idx in self.g.iter_block():
            for ia in ias:
                i += 1
        assert_true(i == len(self.g))

    @attr('slow')
    def test_iter_block2(self):
        g = self.g.tile(30, 0).tile(30, 1)
        i = 0
        for ias, _ in g.iter_block():
            i += len(ias)
        assert_true(i == len(g))

    def test_iter_shape1(self):
        i = 0
        for ias, _ in self.g.iter_block(method='sphere'):
            i += len(ias)
        assert_true(i == len(self.g))
        i = 0
        for ias, _ in self.g.iter_block(method='cube'):
            i += len(ias)
        assert_true(i == len(self.g))

    @attr('slow')
    def test_iter_shape2(self):
        g = self.g.tile(30, 0).tile(30, 1)
        i = 0
        for ias, _ in g.iter_block(method='sphere'):
            i += len(ias)
        assert_true(i == len(g))
        i = 0
        for ias, _ in g.iter_block(method='cube'):
            i += len(ias)
        assert_true(i == len(g))

    @attr('slow')
    def test_iter_shape3(self):
        g = self.g.tile(50, 0).tile(50, 1)
        i = 0
        for ias, _ in g.iter_block(method='sphere'):
            i += len(ias)
        assert_true(i == len(g))
        i = 0
        for ias, _ in g.iter_block(method='cube'):
            i += len(ias)
        assert_true(i == len(g))

    def test_swap(self):
        s = self.g.swap(0, 1)
        for i in [0, 1, 2]:
            assert_true(np.allclose(self.g.xyz[::-1, i], s.xyz[:, i]))

    def test_append1(self):
        for axis in [0, 1, 2]:
            s = self.g.append(self.g, axis)
            assert_equal(len(s), len(self.g) * 2)
            assert_true(np.allclose(s.cell[axis, :], self.g.cell[axis, :]* 2))
            assert_true(np.allclose(s.cell[axis, :], self.g.cell[axis, :]* 2))
            s = self.g.prepend(self.g, axis)
            assert_equal(len(s), len(self.g) * 2)
            assert_true(np.allclose(s.cell[axis, :], self.g.cell[axis, :]* 2))
            assert_true(np.allclose(s.cell[axis, :], self.g.cell[axis, :]* 2))
            s = self.g.append(self.g.sc, axis)
            assert_equal(len(s), len(self.g))
            assert_true(np.allclose(s.cell[axis, :], self.g.cell[axis, :]* 2))
            assert_true(np.allclose(s.cell[axis, :], self.g.cell[axis, :]* 2))
            s = self.g.prepend(self.g.sc, axis)
            assert_equal(len(s), len(self.g))
            assert_true(np.allclose(s.cell[axis, :], self.g.cell[axis, :]* 2))
            assert_true(np.allclose(s.cell[axis, :], self.g.cell[axis, :]* 2))

    def test_swapaxes(self):
        s = self.g.swapaxes(0, 1)
        assert_true(np.allclose(self.g.xyz[:, 0], s.xyz[:, 1]))
        assert_true(np.allclose(self.g.xyz[:, 1], s.xyz[:, 0]))
        assert_true(np.allclose(self.g.cell[0, :], s.cell[1, :]))
        assert_true(np.allclose(self.g.cell[1, :], s.cell[0, :]))

    def test_center(self):
        one = self.g.center(atom=[0])
        assert_true(np.allclose(self.g[0], one))
        al = self.g.center()
        assert_true(np.allclose(np.mean(self.g.xyz, axis=0), al))
        al = self.g.center(which='mass')

    @raises(ValueError)
    def test_center_raise(self):
        al = self.g.center(which='unknown')

    def test___add__(self):
        n = len(self.g)
        double = self.g + self.g
        assert_equal(len(double), n * 2)
        assert_true(np.allclose(self.g.cell, double.cell))
        assert_true(np.allclose(self.g.xyz[:n, :], double.xyz[:n, :]))

        double = (self.g, 1) + self.g
        d = self.g.prepend(self.g, 1)
        assert_equal(len(double), n * 2)
        assert_true(np.allclose(self.g.cell[::2, :], double.cell[::2, :]))
        assert_true(np.allclose(double.xyz, d.xyz))

        double = self.g + (self.g, 1)
        d = self.g.append(self.g, 1)
        assert_equal(len(double), n * 2)
        assert_true(np.allclose(self.g.cell[::2, :], double.cell[::2, :]))
        assert_true(np.allclose(double.xyz, d.xyz))

    def test___mul__(self):
        g = self.g.copy()
        assert_equal(g * 2, g.tile(2, 0).tile(2, 1).tile(2, 2))
        assert_equal(g * [2, 1], g.tile(2, 1))
        assert_equal(g * (2, 2, 2), g.tile(2, 0).tile(2, 1).tile(2, 2))
        assert_equal(g * [1, 2, 2], g.tile(1, 0).tile(2, 1).tile(2, 2))
        assert_equal(g * [1, 3, 2], g.tile(1, 0).tile(3, 1).tile(2, 2))
        assert_equal(g * ([1, 3, 2], 'r'), g.repeat(1, 0).repeat(3, 1).repeat(2, 2))
        assert_equal(g * ([1, 3, 2], 'repeat'), g.repeat(1, 0).repeat(3, 1).repeat(2, 2))
        assert_equal(g * ([1, 3, 2], 'tile'), g.tile(1, 0).tile(3, 1).tile(2, 2))
        assert_equal(g * ([1, 3, 2], 't'), g.tile(1, 0).tile(3, 1).tile(2, 2))
        assert_equal(g * ([3, 2], 't'), g.tile(3, 2))
        assert_equal(g * ([3, 2], 'r'), g.repeat(3, 2))

    def test_add(self):
        double = self.g.add(self.g)
        assert_equal(len(double), len(self.g) * 2)
        assert_true(np.allclose(self.g.cell, double.cell))

    def test_insert(self):
        double = self.g.insert(0, self.g)
        assert_equal(len(double), len(self.g) * 2)
        assert_true(np.allclose(self.g.cell, double.cell))

    def test_a2o(self):
        # There are 2 orbitals per C atom
        assert_equal(self.g.a2o(1), self.g.atom[0].orbs)
        assert_true(np.all(self.g.a2o(1, True) == [2, 3]))

    def test_o2a(self):
        # There are 2 orbitals per C atom
        assert_equal(self.g.o2a(2), 1)

    def test_2uc(self):
        # functions for any-thing to UC
        assert_equal(self.g.sc2uc(2), 0)
        assert_true(np.all(self.g.sc2uc([2, 3]) == [0, 1]))
        assert_equal(self.g.asc2uc(2), 0)
        assert_true(np.all(self.g.asc2uc([2, 3]) == [0, 1]))
        assert_equal(self.g.osc2uc(4), 0)
        assert_equal(self.g.osc2uc(5), 1)
        assert_true(np.all(self.g.osc2uc([4, 5]) == [0, 1]))

    def test_2sc(self):
        # functions for any-thing to SC
        c = self.g.cell

        # check indices
        assert_true(np.all(self.g.a2isc([1, 2]) == [[0,  0, 0],
                                                    [-1, -1, 0]]))
        assert_true(np.all(self.g.a2isc(2) == [-1, -1, 0]))
        assert_true(np.allclose(self.g.a2sc(2), -c[0, :] - c[1, :]))
        assert_true(np.all(self.g.o2isc([1, 5]) == [[0,  0, 0],
                                                    [-1, -1, 0]]))
        assert_true(np.all(self.g.o2isc(5) == [-1, -1, 0]))
        assert_true(np.allclose(self.g.o2sc(5), -c[0, :] - c[1, :]))

        # Check off-sets
        assert_true(np.allclose(self.g.a2sc([1, 2]), [[0.,  0., 0.],
                                                      -c[0, :] - c[1, :]]))
        assert_true(np.allclose(self.g.o2sc([1, 5]), [[0.,  0., 0.],
                                                      -c[0, :] - c[1, :]]))

    def test_reverse(self):
        rev = self.g.reverse()
        assert_true(len(rev) == 2)
        assert_true(np.allclose(rev.xyz[::-1, :], self.g.xyz))
        rev = self.g.reverse(atom=list(range(len(self.g))))
        assert_true(len(rev) == 2)
        assert_true(np.allclose(rev.xyz[::-1, :], self.g.xyz))

    def test_scale1(self):
        two = self.g.scale(2)
        assert_true(len(two) == len(self.g))
        assert_true(np.allclose(two.xyz[:, :] / 2., self.g.xyz))

    def test_close1(self):
        three = range(3)
        for ia in self.mol:
            i = self.mol.close(ia, dR=(0.1, 1.1), idx=three)
            if ia < 3:
                assert_equal(len(i[0]), 1)
            else:
                assert_equal(len(i[0]), 0)
            # Will only return results from [0,1,2]
            # but the fourth atom connects to
            # the third
            if ia in [0, 2, 3]:
                assert_equal(len(i[1]), 1)
            elif ia == 1:
                assert_equal(len(i[1]), 2)
            else:
                assert_equal(len(i[1]), 0)

    def test_close2(self):
        mol = range(3, 5)
        for ia in self.mol:
            i = self.mol.close(ia, dR=(0.1, 1.1), idx=mol)
            assert_equal(len(i), 2)
        i = self.mol.close([100, 100, 100], dR=0.1)
        assert_equal(len(i), 0)
        i = self.mol.close([100, 100, 100], dR=0.1, ret_rij=True)
        for el in i:
            assert_equal(len(el), 0)
        i = self.mol.close([100, 100, 100], dR=0.1, ret_rij=True, ret_xyz=True)
        for el in i:
            assert_equal(len(el), 0)

    def test_close_within1(self):
        three = range(3)
        for ia in self.mol:
            shapes = [Sphere(0.1, self.mol[ia]),
                      Sphere(1.1, self.mol[ia])]
            i = self.mol.close(ia, dR=(0.1, 1.1), idx=three)
            ii = self.mol.within(shapes, idx=three)
            assert_true(np.all(i[0] == ii[0]))
            assert_true(np.all(i[1] == ii[1]))

    def test_close_within2(self):
        g = self.g.repeat(6, 0).repeat(6, 1)
        for ia in g:
            shapes = [Sphere(0.1, g[ia]),
                      Sphere(1.5, g[ia])]
            i = g.close(ia, dR=(0.1, 1.5))
            ii = g.within(shapes)
            assert_true(np.all(i[0] == ii[0]))
            assert_true(np.all(i[1] == ii[1]))

    def test_close_within3(self):
        g = self.g.repeat(6, 0).repeat(6, 1)
        args = {'ret_xyz': True, 'ret_rij': True}
        for ia in g:
            shapes = [Sphere(0.1, g[ia]),
                      Sphere(1.5, g[ia])]
            i, xa, d = g.close(ia, dR=(0.1, 1.5), **args)
            ii, xai, di = g.within(shapes, **args)
            for j in [0, 1]:
                assert_true(np.all(i[j] == ii[j]))
                assert_true(np.allclose(xa[j], xai[j]))
                assert_true(np.allclose(d[j], di[j]))

    def test_close_sizes(self):
        point = 0

        # Return index
        idx = self.mol.close(point, dR=.1)
        assert_equal(len(idx), 1)
        # Return index of two things
        idx = self.mol.close(point, dR=(.1, 1.1))
        assert_equal(len(idx), 2)
        assert_equal(len(idx[0]), 1)
        assert_false(isinstance(idx[0], list))
        # Longer
        idx = self.mol.close(point, dR=(.1, 1.1, 2.1))
        assert_equal(len(idx), 3)
        assert_equal(len(idx[0]), 1)

        # Return index
        idx = self.mol.close(point, dR=.1, ret_xyz=True)
        assert_equal(len(idx), 2)
        assert_equal(len(idx[0]), 1)
        assert_equal(len(idx[1]), 1)
        assert_equal(idx[1].shape[0], 1) # equivalent to above
        assert_equal(idx[1].shape[1], 3)

        # Return index of two things
        idx = self.mol.close(point, dR=(.1, 1.1), ret_xyz=True)
        # [[idx-1, idx-2], [coord-1, coord-2]]
        assert_equal(len(idx), 2)
        assert_equal(len(idx[0]), 2)
        assert_equal(len(idx[1]), 2)
        # idx-1
        assert_equal(len(idx[0][0].shape), 1)
        assert_equal(idx[0][0].shape[0], 1)
        # idx-2
        assert_equal(idx[0][1].shape[0], 1)
        # coord-1
        assert_equal(len(idx[1][0].shape), 2)
        assert_equal(idx[1][0].shape[1], 3)
        # coord-2
        assert_equal(idx[1][1].shape[1], 3)

        # Return index of two things
        idx = self.mol.close(point, dR=(.1, 1.1), ret_xyz=True, ret_rij=True)
        # [[idx-1, idx-2], [coord-1, coord-2], [dist-1, dist-2]]
        assert_equal(len(idx), 3)
        assert_equal(len(idx[0]), 2)
        assert_equal(len(idx[1]), 2)
        # idx-1
        assert_equal(len(idx[0][0].shape), 1)
        assert_equal(idx[0][0].shape[0], 1)
        # idx-2
        assert_equal(idx[0][1].shape[0], 1)
        # coord-1
        assert_equal(len(idx[1][0].shape), 2)
        assert_equal(idx[1][0].shape[1], 3)
        # coord-2
        assert_equal(idx[1][1].shape[1], 3)
        # dist-1
        assert_equal(len(idx[2][0].shape), 1)
        assert_equal(idx[2][0].shape[0], 1)
        # dist-2
        assert_equal(idx[2][1].shape[0], 1)

        # Return index of two things
        idx = self.mol.close(point, dR=(.1, 1.1), ret_rij=True)
        # [[idx-1, idx-2], [dist-1, dist-2]]
        assert_equal(len(idx), 2)
        assert_equal(len(idx[0]), 2)
        assert_equal(len(idx[1]), 2)
        # idx-1
        assert_equal(len(idx[0][0].shape), 1)
        assert_equal(idx[0][0].shape[0], 1)
        # idx-2
        assert_equal(idx[0][1].shape[0], 1)
        # dist-1
        assert_equal(len(idx[1][0].shape), 1)
        assert_equal(idx[1][0].shape[0], 1)
        # dist-2
        assert_equal(idx[1][1].shape[0], 1)

    def test_close_sizes_none(self):
        point = [100., 100., 100.]

        # Return index
        idx = self.mol.close(point, dR=.1)
        assert_equal(len(idx), 0)
        # Return index of two things
        idx = self.mol.close(point, dR=(.1, 1.1))
        assert_equal(len(idx), 2)
        assert_equal(len(idx[0]), 0)
        assert_false(isinstance(idx[0], list))
        # Longer
        idx = self.mol.close(point, dR=(.1, 1.1, 2.1))
        assert_equal(len(idx), 3)
        assert_equal(len(idx[0]), 0)

        # Return index
        idx = self.mol.close(point, dR=.1, ret_xyz=True)
        assert_equal(len(idx), 2)
        assert_equal(len(idx[0]), 0)
        assert_equal(len(idx[1]), 0)
        assert_equal(idx[1].shape[0], 0) # equivalent to above
        assert_equal(idx[1].shape[1], 3)

        # Return index of two things
        idx = self.mol.close(point, dR=(.1, 1.1), ret_xyz=True)
        # [[idx-1, idx-2], [coord-1, coord-2]]
        assert_equal(len(idx), 2)
        assert_equal(len(idx[0]), 2)
        assert_equal(len(idx[1]), 2)
        # idx-1
        assert_equal(len(idx[0][0].shape), 1)
        assert_equal(idx[0][0].shape[0], 0)
        # idx-2
        assert_equal(idx[0][1].shape[0], 0)
        # coord-1
        assert_equal(len(idx[1][0].shape), 2)
        assert_equal(idx[1][0].shape[1], 3)
        # coord-2
        assert_equal(idx[1][1].shape[1], 3)

        # Return index of two things
        idx = self.mol.close(point, dR=(.1, 1.1), ret_xyz=True, ret_rij=True)
        # [[idx-1, idx-2], [coord-1, coord-2], [dist-1, dist-2]]
        assert_equal(len(idx), 3)
        assert_equal(len(idx[0]), 2)
        assert_equal(len(idx[1]), 2)
        # idx-1
        assert_equal(len(idx[0][0].shape), 1)
        assert_equal(idx[0][0].shape[0], 0)
        # idx-2
        assert_equal(idx[0][1].shape[0], 0)
        # coord-1
        assert_equal(len(idx[1][0].shape), 2)
        assert_equal(idx[1][0].shape[0], 0)
        assert_equal(idx[1][0].shape[1], 3)
        # coord-2
        assert_equal(idx[1][1].shape[0], 0)
        assert_equal(idx[1][1].shape[1], 3)
        # dist-1
        assert_equal(len(idx[2][0].shape), 1)
        assert_equal(idx[2][0].shape[0], 0)
        # dist-2
        assert_equal(idx[2][1].shape[0], 0)

        # Return index of two things
        idx = self.mol.close(point, dR=(.1, 1.1), ret_rij=True)
        # [[idx-1, idx-2], [dist-1, dist-2]]
        assert_equal(len(idx), 2)
        assert_equal(len(idx[0]), 2)
        assert_equal(len(idx[1]), 2)
        # idx-1
        assert_equal(len(idx[0][0].shape), 1)
        assert_equal(idx[0][0].shape[0], 0)
        # idx-2
        assert_equal(idx[0][1].shape[0], 0)
        # dist-1
        assert_equal(len(idx[1][0].shape), 1)
        assert_equal(idx[1][0].shape[0], 0)
        # dist-2
        assert_equal(idx[1][1].shape[0], 0)

    def test_sparserij1(self):
        rij = self.g.sparserij()

    def test_bond_correct(self):
        # Create ribbon
        rib = self.g.tile(2, 1)
        # Convert the last atom to a H atom
        rib.atom[-1] = Atom[1]
        ia = len(rib) - 1
        # Get bond-length
        idx, d = rib.close(ia, dR=(.1, 1000), ret_rij=True)
        i = np.argmin(d[1])
        d = d[1][i]
        rib.bond_correct(ia, idx[1][i])
        idx, d2 = rib.close(ia, dR=(.1, 1000), ret_rij=True)
        i = np.argmin(d2[1])
        d2 = d2[1][i]
        assert_false(d == d2)
        # Calculate actual radius
        assert_true(d2 == (Atom[1].radius() + Atom[6].radius()))

    def test_unit_cell_estimation1(self):
        # Create new geometry with only the coordinates
        # and atoms
        geom = Geometry(self.g.xyz, Atom[6])
        # Only check the two distances we know have sizes
        for i in range(2):
            # It cannot guess skewed axis
            assert_false(np.allclose(geom.cell[i, :], self.g.cell[i, :]))

    def test_unit_cell_estimation2(self):
        # Create new geometry with only the coordinates
        # and atoms
        s1 = SuperCell([2, 2, 2])
        g1 = Geometry([[0, 0, 0], [1, 1, 1]], sc=s1)
        g2 = Geometry(np.copy(g1.xyz))
        assert_true(np.allclose(g1.cell, g2.cell))

        # Assert that it correctly calculates the bond-length in the
        # directions of actual distance
        g1 = Geometry([[0, 0, 0], [1, 1, 0]], atom='H', sc=s1)
        g2 = Geometry(np.copy(g1.xyz))
        for i in range(2):
            assert_true(np.allclose(g1.cell[i, :], g2.cell[i, :]))
        assert_false(np.allclose(g1.cell[2, :], g2.cell[2, :]))

    def test_argumentparser1(self):
        self.g.ArgumentParser()
        self.g.ArgumentParser(**self.g._ArgumentParser_args_single())

    def test_argumentparser2(self, **kwargs):
        p, ns = self.g.ArgumentParser(**kwargs)

        # Try all options
        opts = ['--origin',
                '--center-of', 'mass',
                '--center-of', 'xyz',
                '--center-of', 'position',
                '--center-of', 'cell',
                '--unit-cell', 'translate',
                '--unit-cell', 'mod',
                '--rotate', 'x', '90',
                '--rotate', 'y', '90',
                '--rotate', 'z', '90',
                '--add', '0,0,0', '6',
                '--swap', '0', '1',
                '--repeat', 'x', '2',
                '--repeat', 'y', '2',
                '--repeat', 'z', '2',
                '--tile', 'x', '2',
                '--tile', 'y', '2',
                '--tile', 'z', '2',
                '--cut', 'z', '2',
                '--cut', 'y', '2',
                '--cut', 'x', '2',
        ]
        if kwargs.get('limit_arguments', True):
            opts.extend(['--rotate', 'x', '-90',
                         '--rotate', 'y', '-90',
                         '--rotate', 'z', '-90'])
        else:
            opts.extend(['--rotate-x', ' -90',
                         '--rotate-y', ' -90',
                         '--rotate-z', ' -90',
                         '--repeat-x', '2',
                         '--repeat-y', '2',
                         '--repeat-z', '2'])

        args = p.parse_args(opts, namespace=ns)

        if len(kwargs) == 0:
            self.test_argumentparser2(**self.g._ArgumentParser_args_single())

    def test_set_sc(self):
        # Create new geometry with only the coordinates
        # and atoms
        s1 = SuperCell([2, 2, 2])
        g1 = Geometry([[0, 0, 0], [1, 1, 1]], sc=[2, 2, 1])
        g1.set_sc(s1)
        assert_true(g1.sc == s1)

    def test_attach1(self):
        g = self.g.attach(0, self.mol, 0, dist=1.42, axis=2)
        g = self.g.attach(0, self.mol, 0, dist='calc', axis=2)
        g = self.g.attach(0, self.mol, 0, dist=[0, 0, 1.42])

    def test_mirror1(self):
        for plane in ['xy', 'xz', 'yz']:
            self.g.mirror(plane)

    def test_pickle(self):
        import pickle as p
        s = p.dumps(self.g)
        n = p.loads(s)
        assert_true(n == self.g)
        assert_false(n != self.g)
