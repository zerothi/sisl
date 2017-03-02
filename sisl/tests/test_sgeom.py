from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np

from sisl import Geometry, Atom, SuperCell
from sisl.geometry import sgeom


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

        def sg_g(**kwargs):
            kwargs['ret_geometry'] = True
            if 'geom' not in kwargs:
                kwargs['geom'] = self.g
            return sgeom(**kwargs)

        self.sg_g = sg_g

        def sg_mol(**kwargs):
            kwargs['ret_geometry'] = True
            if 'geom' not in kwargs:
                kwargs['geom'] = self.mol
            return sgeom(**kwargs)

        self.sg_mol = sg_mol

    def tearDown(self):
        del self.g
        del self.sg_g
        del self.sc
        del self.mol
        del self.sg_mol

    def test_tile1(self):
        cell = np.copy(self.g.sc.cell)
        cell[0, :] *= 2
        for tile in ['tile x 2', 'tile-x 2']:
            tx = self.sg_g(argv=('--' + tile).split())
            assert_true(np.allclose(cell, tx.sc.cell))
        cell[1, :] *= 2
        for tile in ['tile y 2', 'tile-y 2']:
            ty = self.sg_g(geom=tx, argv=('--' + tile).split())
            assert_true(np.allclose(cell, ty.sc.cell))
        cell[2, :] *= 2
        for tile in ['tile z 2', 'tile-z 2']:
            tz = self.sg_g(geom=ty, argv=('--' + tile).split())
            assert_true(np.allclose(cell, tz.sc.cell))

    def test_tile2(self):
        cell = np.copy(self.g.sc.cell)
        cell[:, :] *= 2
        for xt in ['tile x 2', 'tile-x 2']:
            xt = '--' + xt
            for yt in ['tile y 2', 'tile-y 2']:
                yt = '--' + yt
                for zt in ['tile z 2', 'tile-z 2']:
                    zt = '--' + zt
                    argv = ' '.join([xt, yt, zt]).split()
                    t = self.sg_g(argv=argv)
                    assert_true(np.allclose(cell, t.sc.cell))

    def test_repeat1(self):
        cell = np.copy(self.g.sc.cell)
        cell[0, :] *= 2
        for repeat in ['repeat x 2', 'repeat-x 2']:
            tx = self.sg_g(argv=('--' + repeat).split())
            assert_true(np.allclose(cell, tx.sc.cell))
        cell[1, :] *= 2
        for repeat in ['repeat y 2', 'repeat-y 2']:
            ty = self.sg_g(geom=tx, argv=('--' + repeat).split())
            assert_true(np.allclose(cell, ty.sc.cell))
        cell[2, :] *= 2
        for repeat in ['repeat z 2', 'repeat-z 2']:
            tz = self.sg_g(geom=ty, argv=('--' + repeat).split())
            assert_true(np.allclose(cell, tz.sc.cell))

    def test_repeat2(self):
        cell = np.copy(self.g.sc.cell)
        cell[:, :] *= 2
        for xt in ['repeat x 2', 'repeat-x 2']:
            xt = '--' + xt
            for yt in ['repeat y 2', 'repeat-y 2']:
                yt = '--' + yt
                for zt in ['repeat z 2', 'repeat-z 2']:
                    zt = '--' + zt
                    argv = ' '.join([xt, yt, zt]).split()
                    t = self.sg_g(argv=argv)
                    assert_true(np.allclose(cell, t.sc.cell))

    def test_sub1(self):
        for a, l in [('0', 1), ('0,1', 2), ('0-1', 2)]:
            g = self.sg_g(argv=['--sub', a])
            assert_equal(len(g), l)

    def test_rotation1(self):
        rot = self.sg_g(argv='--rotate z 180'.split())
        rot.sc.cell[2, 2] *= -1
        assert_true(np.allclose(-rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(-rot.xyz, self.g.xyz))

        rot = self.sg_g(argv='--rotate-z 180'.split())
        rot.sc.cell[2, 2] *= -1
        assert_true(np.allclose(-rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(-rot.xyz, self.g.xyz))

        rot = self.sg_g(argv='--rotate z rpi'.split())
        rot.sc.cell[2, 2] *= -1
        assert_true(np.allclose(-rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(-rot.xyz, self.g.xyz))

        rot = self.sg_g(argv='--rotate-z rpi'.split())
        rot.sc.cell[2, 2] *= -1
        assert_true(np.allclose(-rot.sc.cell, self.g.sc.cell))
        assert_true(np.allclose(-rot.xyz, self.g.xyz))

    def test_swap(self):
        s = self.sg_g(argv='--swap 0 1'.split())
        for i in [0, 1, 2]:
            assert_true(np.allclose(self.g.xyz[::-1, i], s.xyz[:, i]))
