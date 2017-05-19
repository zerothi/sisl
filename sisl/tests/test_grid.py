from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import math as m
import numpy as np

from sisl import Grid, SuperCell


@attr('grid')
class TestGrid(object):

    def setUp(self):
        alat = 1.42
        sq3h = 3.**.5 * 0.5
        self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                      [1.5, -sq3h, 0.],
                                      [0., 0., 10.]], np.float64) * alat, nsc=[3, 3, 1])
        self.g = Grid([10, 10, 100], sc=self.sc)
        self.g[:, :, :] = 2.
        g = Grid(sc=self.sc)

    def tearDown(self):
        del self.sc
        del self.g

    def test_append(self):
        g = self.g.append(self.g, 0)
        assert_true(np.allclose(g.grid.shape, [20, 10, 100]))
        g = self.g.append(self.g, 1)
        assert_true(np.allclose(g.grid.shape, [10, 20, 100]))
        g = self.g.append(self.g, 2)
        assert_true(np.allclose(g.grid.shape, [10, 10, 200]))

    def test_set(self):
        v = self.g[0, 0, 0]
        self.g[0, 0, 0] = 3
        assert_true(self.g.grid[0, 0, 0] == 3)
        assert_true(self.g[0, 0, 0] == 3)
        self.g[0, 0, 0] = v
        assert_true(self.g[0, 0, 0] == v)

    def test_size(self):
        assert_true(np.allclose(self.g.grid.shape, [10, 10, 100]))

    def test_item(self):
        assert_true(np.allclose(self.g[1:2, 1:2, 2:3], self.g.grid[1:2, 1:2, 2:3]))

    def test_dcell(self):
        assert_true(np.all(self.g.dcell*self.g.cell >= 0))

    def test_dvol(self):
        assert_true(self.g.dvol > 0)

    def test_shape(self):
        assert_true(np.all(self.g.shape == self.g.grid.shape))

    def test_dtype(self):
        assert_true(self.g.dtype == self.g.grid.dtype)

    def test_copy(self):
        assert_true(self.g.copy() == self.g)
        assert_false(self.g.copy() != self.g)

    def test_add(self):
        g = self.g + self.g
        assert_true(np.allclose(g.grid, (self.g * 2).grid))
        g = self.g.copy()
        g *= 2
        assert_true(np.allclose(g.grid, (self.g * 2).grid))
        g = self.g.copy()
        g /= 2
        assert_true(np.allclose(g.grid, (self.g / 2).grid))

    def test_swapaxes(self):
        g = self.g.swapaxes(0, 1)
        assert_true(np.allclose(self.g.cell[0, :], g.cell[1, :]))
        assert_true(np.allclose(self.g.cell[1, :], g.cell[0, :]))

    def test_interp(self):
        shape = np.array(self.g.shape, np.int32)
        g = self.g.interp(shape * 2)
        g1 = g.interp(shape)
        # Sadly the interpolation does not work as it really
        # should...
        # One cannot interp down/up and retrieve the same
        # grid... Perhaps this is ok, but not good... :(
        assert_true(np.allclose(self.g.grid, g1.grid))

    def test_index1(self):
        mid = np.array(self.g.shape, np.int32) // 2
        idx = self.g.index(self.sc.center())
        assert_true(np.all(mid == idx))

    def test_sum(self):
        for i in range(3):
            assert_true(self.g.sum(i).shape[i] == 1)

    def test_mean(self):
        for i in range(3):
            assert_true(self.g.mean(i).shape[i] == 1)

    def test_cross_section(self):
        for i in range(3):
            assert_true(self.g.cross_section(1, i).shape[i] == 1)

    def test_remove_part(self):
        for i in range(3):
            assert_true(self.g.remove_part(1, i, above=True).shape[i] == 1)

    def test_sub_part(self):
        for i in range(3):
            assert_true(self.g.sub_part(1, i, above=False).shape[i] == 1)
            assert_true(self.g.sub_part(1, i, above=True).shape[i] == self.g.shape[i] - 1)

    def test_sub(self):
        for i in range(3):
            assert_true(self.g.sub(1, i).shape[i] == 1)
        for i in range(3):
            assert_true(self.g.sub([1, 2], i).shape[i] == 2)

    def test_remove(self):
        for i in range(3):
            assert_true(self.g.remove(1, i).shape[i] == self.g.shape[i]-1)
        for i in range(3):
            assert_true(self.g.remove([1, 2], i).shape[i] == self.g.shape[i]-2)

    def test_argumentparser(self):
        self.g.ArgumentParser()
