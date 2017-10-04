from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import Grid, SuperCell


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            alat = 1.42
            sq3h = 3.**.5 * 0.5
            self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                          [1.5, -sq3h, 0.],
                                          [0., 0., 10.]], np.float64) * alat, nsc=[3, 3, 1])
            self.g = Grid([10, 10, 100], sc=self.sc)
            self.g[:, :, :] = 2.
    return t()


@pytest.mark.grid
class TestGrid(object):

    def test_print(self, setup):
        print(setup.g)

    def test_append(self, setup):
        g = setup.g.append(setup.g, 0)
        assert np.allclose(g.grid.shape, [20, 10, 100])
        g = setup.g.append(setup.g, 1)
        assert np.allclose(g.grid.shape, [10, 20, 100])
        g = setup.g.append(setup.g, 2)
        assert np.allclose(g.grid.shape, [10, 10, 200])

    def test_set(self, setup):
        v = setup.g[0, 0, 0]
        setup.g[0, 0, 0] = 3
        assert setup.g.grid[0, 0, 0] == 3
        assert setup.g[0, 0, 0] == 3
        setup.g[0, 0, 0] = v
        assert setup.g[0, 0, 0] == v

    def test_size(self, setup):
        assert np.allclose(setup.g.grid.shape, [10, 10, 100])

    def test_item(self, setup):
        assert np.allclose(setup.g[1:2, 1:2, 2:3], setup.g.grid[1:2, 1:2, 2:3])

    def test_dcell(self, setup):
        assert np.all(setup.g.dcell*setup.g.cell >= 0)

    def test_dvol(self, setup):
        assert setup.g.dvol > 0

    def test_shape(self, setup):
        assert np.all(setup.g.shape == setup.g.grid.shape)

    def test_dtype(self, setup):
        assert setup.g.dtype == setup.g.grid.dtype

    def test_copy(self, setup):
        assert setup.g.copy() == setup.g
        assert not setup.g.copy() != setup.g

    def test_add1(self, setup):
        g = setup.g + setup.g
        assert np.allclose(g.grid, (setup.g * 2).grid)
        g = setup.g.copy()
        g *= 2
        assert np.allclose(g.grid, (setup.g * 2).grid)
        g = setup.g.copy()
        g /= 2
        assert np.allclose(g.grid, (setup.g / 2).grid)

    def test_add2(self, setup):
        g = setup.g + 2.
        assert np.allclose(g.grid, setup.g.grid + 2)
        g = setup.g.copy()
        g += 2.
        g -= 2.
        assert np.allclose(g.grid, setup.g.grid)
        g = setup.g + setup.g
        assert np.allclose(g.grid, setup.g.grid * 2)
        assert np.allclose((g - setup.g).grid, setup.g.grid)

    @pytest.mark.xfail(raises=ValueError)
    def test_add_fail1(self, setup):
        g = Grid(np.array(setup.g.shape) // 2 + 1, sc=setup.g.sc.copy())
        setup.g + g

    def test_iadd1(self):
        g = Grid([10, 10, 10])
        g[:] = 1
        old = g.copy()
        g += g
        g -= old
        assert np.allclose(g.grid, 1)
        g -= g
        assert np.allclose(g.grid, 0)

    def test_op1(self, setup):
        g = setup.g * setup.g
        assert np.allclose(g.grid, setup.g.grid * setup.g.grid)
        g = setup.g.copy()
        g *= setup.g
        assert np.allclose(g.grid, setup.g.grid * setup.g.grid)
        g = setup.g * setup.g
        g /= setup.g
        assert np.allclose(g.grid, setup.g.grid)

    def test_swapaxes(self, setup):
        g = setup.g.swapaxes(0, 1)
        assert np.allclose(setup.g.cell[0, :], g.cell[1, :])
        assert np.allclose(setup.g.cell[1, :], g.cell[0, :])

    def test_interp(self, setup):
        shape = np.array(setup.g.shape, np.int32)
        g = setup.g.interp(shape * 2)
        g1 = g.interp(shape)
        # Sadly the interpolation does not work as it really
        # should...
        # One cannot interp down/up and retrieve the same
        # grid... Perhaps this is ok, but not good... :(
        assert np.allclose(setup.g.grid, g1.grid)

    def test_index1(self, setup):
        mid = np.array(setup.g.shape, np.int32) // 2
        idx = setup.g.index(setup.sc.center())
        assert np.all(mid == idx)

    def test_sum(self, setup):
        for i in range(3):
            assert setup.g.sum(i).shape[i] == 1

    def test_mean(self, setup):
        for i in range(3):
            assert setup.g.mean(i).shape[i] == 1

    def test_cross_section(self, setup):
        for i in range(3):
            assert setup.g.cross_section(1, i).shape[i] == 1

    @pytest.mark.xfail(raises=ValueError)
    def test_cross_section_fail(self, setup):
        setup.g.cross_section(1, -1)

    def test_remove_part(self, setup):
        for i in range(3):
            assert setup.g.remove_part(1, i, above=True).shape[i] == 1

    def test_sub_part(self, setup):
        for i in range(3):
            assert setup.g.sub_part(1, i, above=False).shape[i] == 1
            assert setup.g.sub_part(1, i, above=True).shape[i] == setup.g.shape[i] - 1

    def test_sub(self, setup):
        for i in range(3):
            assert setup.g.sub(1, i).shape[i] == 1
        for i in range(3):
            assert setup.g.sub([1, 2], i).shape[i] == 2

    def test_remove(self, setup):
        for i in range(3):
            assert setup.g.remove(1, i).shape[i] == setup.g.shape[i]-1
        for i in range(3):
            assert setup.g.remove([1, 2], i).shape[i] == setup.g.shape[i]-2

    def test_bc1(self, setup):
        assert np.all(setup.g.bc == setup.g.PERIODIC)
        setup.g.set_bc(a=setup.g.NEUMANN)
        setup.g.set_bc(b=setup.g.NEUMANN, c=setup.g.NEUMANN)
        assert np.all(setup.g.bc == setup.g.NEUMANN)
        setup.g.set_bc(setup.g.PERIODIC)
        assert np.all(setup.g.bc == setup.g.PERIODIC)

    def test_argumentparser(self, setup):
        setup.g.ArgumentParser()
