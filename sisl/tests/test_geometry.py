from __future__ import print_function, division

from nose.tools import *

from sisl import Geometry, Atom, SuperCell

import math as m
import numpy as np


class TestGeometry(object):
    # Base test class for MaskedArrays.

    def setUp(self):
        alat = 1.42
        sq3h = 3.**.5 * 0.5
        self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                      [1.5, -sq3h, 0.],
                                      [0., 0., 10.]], np.float64) * alat, nsc=[3, 3, 1])
        C = Atom(Z=6, R=alat * 1.01, orbs=2)
        self.g = Geometry(np.array([[0., 0., 0.],
                                    [1., 0., 0.]], np.float64) * alat,
                          atoms=C, sc=self.sc)

    def tearDown(self):
        del self.g
        del self.sc

    def test_objects(self):
        assert_true(len(self.g) == 2)
        assert_true(len(self.g.xyz) == 2)

        i = 0
        for ia in self.g:
            i += 1
        assert_true(i == len(self.g))
        assert_true(self.g.no_s == 2 * len(self.g) * np.prod(self.g.sc.nsc))

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

    def test_a2o1(self):
        assert_true(0 == self.g.a2o(0))
        assert_true(self.g.atoms[0].orbs == self.g.a2o(1))
        assert_true(self.g.no == self.g.a2o(self.g.na))

    def test_sub(self):
        assert_true(len(self.g.sub([0])) == 1)
        assert_true(len(self.g.sub([0, 1])) == 2)
        assert_true(len(self.g.sub([-1])) == 1)

    def test_cut(self):
        assert_true(len(self.g.cut(1, 1)) == 2)
        assert_true(len(self.g.cut(2, 1)) == 1)
        assert_true(len(self.g.cut(2, 1, 1)) == 1)

    def test_cut2(self):
        c1 = self.g.cut(2, 1)
        c2 = self.g.cut(2, 1, 1)
        assert_true(np.allclose(c1.xyz[0, :], self.g.xyz[0, :]))
        assert_true(np.allclose(c2.xyz[0, :], self.g.xyz[1, :]))

    def test_remove(self):
        assert_true(len(self.g.remove([0])) == 1)
        assert_true(len(self.g.remove([])) == 2)
        assert_true(len(self.g.remove([-1])) == 1)
        assert_true(len(self.g.remove([-0])) == 1)

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

        rot = self.g.rotate(np.pi, [0, 0, 1], degree=False)
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

        rot = self.g.rotate(np.pi, [0, 0, 1], degree=False, only='abc')
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

        rot = self.g.rotate(np.pi, [0, 0, 1], degree=False, only='xyz')
        assert_true(np.allclose(rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(-rot.xyz, self.g.xyz))

        rot = rot.rotate(180, [0, 0, 1], only='xyz')
        assert_true(np.allclose(rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(rot.xyz, self.g.xyz))

    def test_bond_correct(self):
        # Create ribbon
        rib = self.g.tile(2, 1)
        # Convert the last atom to an H atom
        rib.atoms[-1] = Atom[1]
        ia = len(rib) - 1
        # Get bond-length
        idx, d = rib.close(ia, dR=(.1, 1000), ret_dist=True)
        i = np.argmin(d[1])
        d = d[1][i]
        rib.bond_correct(ia, idx[1][i])
        idx, d2 = rib.close(ia, dR=(.1, 1000), ret_dist=True)
        i = np.argmin(d2[1])
        d2 = d2[1][i]
        assert_false(d == d2)
        # Calculate actual radii
        assert_true(d2 == (Atom[1].radii() + Atom[6].radii()))

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
        g1 = Geometry([[0, 0, 0], [1, 1, 0]], sc=s1)
        g2 = Geometry(np.copy(g1.xyz))
        for i in range(2):
            assert_true(np.allclose(g1.cell[i, :], g2.cell[i, :]))
        assert_false(np.allclose(g1.cell[2, :], g2.cell[2, :]))

    def test_pickle(self):
        import pickle as p
        s = p.dumps(self.g)
        n = p.loads(s)
        assert_true(n == self.g)
        assert_false(n != self.g)

    def test_set_sc(self):
        # Create new geometry with only the coordinates
        # and atoms
        s1 = SuperCell([2, 2, 2])
        g1 = Geometry([[0, 0, 0], [1, 1, 1]], sc=[2, 2, 1])
        g1.set_sc(s1)
        assert_true(g1.sc == s1)
