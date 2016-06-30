from __future__ import print_function, division

from nose.tools import *

from sisl import Grid, SuperCell

import math as m
import numpy as np


class TestGrid(object):

    def setUp(self):
        alat = 1.42
        sq3h = 3.**.5 * 0.5
        self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                      [1.5, -sq3h, 0.],
                                      [0., 0., 10.]], np.float64) * alat, nsc=[3, 3, 1])
        self.g = Grid([10, 10, 100], sc=self.sc)

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

    def test_size(self):
        assert_true(np.allclose(self.g.grid.shape, [10, 10, 100]))
        
    def test_item(self):
        assert_true(np.allclose(self.g[1:2, 1:2, 2:3], self.g.grid[1:2, 1:2, 2:3]))

    def test_shape(self):
        assert_true(np.all(self.g.shape == self.g.grid.shape))

    def test_dtype(self):
        assert_true(self.g.dtype == self.g.grid.dtype)

    def test_swapaxes(self):
        g = self.g.swapaxes(0, 1)
        assert_true(np.allclose(self.g.cell[0,:], g.cell[1,:]))
        assert_true(np.allclose(self.g.cell[1,:], g.cell[0,:]))

    def test_sum(self):
        for i in range(3):
            assert_true(self.g.sum(i).shape[i] == 1)

    def test_mean(self):
        for i in range(3):
            assert_true(self.g.mean(i).shape[i] == 1)

    
