import pytest

import math as m
import numpy as np
from scipy.sparse import csr_matrix

from sisl import SuperCell, SphericalOrbital, Atom, Geometry
from sisl import Grid
from sisl import Ellipsoid, Cuboid


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
            self.g.fill(2.)
    return t()


@pytest.mark.grid
class TestGrid:

    def test_print(self, setup):
        str(setup.g)

    def test_init(self, setup):
        Grid(0.1, sc=setup.sc)

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

    def test_dvolume(self, setup):
        assert setup.g.dvolume > 0

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

    def test_add_fail1(self, setup):
        g = Grid(np.array(setup.g.shape) // 2 + 1, sc=setup.g.sc.copy())
        with pytest.raises(ValueError):
            setup.g + g

    def test_iadd1(self):
        g = Grid([10, 10, 10])
        g.fill(1)
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

    def test_average(self, setup):
        g = setup.g.copy()
        g.grid.fill(1)
        shape = g.shape
        for i in range(3):
            assert g.average(i).grid.sum() == shape[0] * shape[1] * shape[2] / shape[i]
        assert g.average(0).shape == (1, shape[1], shape[2])
        assert g.average(1).shape == (shape[0], 1, shape[2])
        assert g.average(2).shape == (shape[0], shape[1], 1)

    def test_average_weight(self, setup):
        g = setup.g.copy()
        g.grid.fill(1)
        shape = g.shape
        for i in range(3):
            w = np.zeros(shape[i]) + 0.5
            assert g.average(i, weights=w).grid.sum() == shape[0] * shape[1] * shape[2] / shape[i]

    def test_interp(self, setup):
        shape = np.array(setup.g.shape, np.int32)
        g = setup.g.interp(shape * 2)
        g1 = g.interp(shape)
        # Sadly the interpolation does not work as it really
        # should...
        # One cannot interp down/up and retrieve the same
        # grid... Perhaps this is ok, but not good... :(
        assert np.allclose(setup.g.grid, g1.grid)

    def test_interp_2d(self, setup):
        shape = np.array(setup.g.shape, np.int32)
        g = setup.g.sum(2)
        g = g.interp([g.shape[0] * 2, g.shape[1] * 2, 1])
        g1 = g.interp([shape[0], shape[1], 1])
        # Sadly the interpolation does not work as it really
        # should...
        # One cannot interp down/up and retrieve the same
        # grid... Perhaps this is ok, but not good... :(
        assert np.allclose(setup.g.sum(2).grid, g1.grid)

    def test_interp_1d(self, setup):
        shape = np.array(setup.g.shape, np.int32)
        g = setup.g.sum(2).sum(0)
        g = g.interp([1, g.shape[1] * 2, 1])
        g1 = g.interp([1, shape[1], 1])
        # Sadly the interpolation does not work as it really
        # should...
        # One cannot interp down/up and retrieve the same
        # grid... Perhaps this is ok, but not good... :(
        assert np.allclose(setup.g.sum(2).sum(0).grid, g1.grid)

    def test_interp_extrap(self, setup):
        shape = np.array(setup.g.shape, np.int32)
        g = setup.g.sum(2)
        g = g.interp([g.shape[0] * 2, g.shape[1] * 2, 10])
        g1 = g.interp([shape[0], shape[1], 1])
        # Sadly the interpolation does not work as it really
        # should...
        # One cannot interp down/up and retrieve the same
        # grid... Perhaps this is ok, but not good... :(
        assert np.allclose(setup.g.sum(2).grid, g1.grid)

    def test_isosurface_orthogonal(self, setup):

        pytest.importorskip("skimage", reason="scikit-image not available")

        # Build an empty grid
        grid = Grid(0.1, sc=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # Fill it with some values that have a clear isosurface
        grid.grid = np.tile([1, 2, 3, 4, 5, 4, 3, 2, 1, 0], 100).reshape(10, 10, 10)

        # Calculate the isosurface for the value of 2.5
        verts, *returns = grid.isosurface(2.5)

        # The third dimension should contain only two coordinates
        # [1, 2, (HERE) 3, 4, 5, 4, 3, (HERE) 2, 1, 0]
        assert np.unique(verts[:, 1]).shape == (10,)
        assert np.unique(verts[:, 2]).shape == (2,)

    def test_isosurface_non_orthogonal(self, setup):

        pytest.importorskip("skimage", reason="scikit-image not available")

        # If the grid is non-orthogonal, there should be 20 unique values
        # (10 for each (HERE), see test_isosurface_orthogonal)
        grid = Grid(0.1, sc=[[1, 0, 0], [0, 1, 0], [0, 2, 1]])

        grid.grid = np.tile([1, 2, 3, 4, 5, 4, 3, 2, 1, 0], 100).reshape(10, 10, 10)

        verts, *returns = grid.isosurface(2.5)

        # we have twice as many since the 3rd lattice vector has components in y
        assert np.unique(verts[:, 1]).shape == (20,)
        assert np.unique(verts[:, 2]).shape == (2,)

    def test_smooth_gaussian(self, setup):
        g = Grid(0.1, sc=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        g[10, 10, 10] = 1

        # With a sigma of 0.3 Ang, that single value should be propagated
        # to the whole grid
        smoothed = g.smooth(r=0.7, truncate=4)
        assert not np.any(smoothed.grid == 0)

        # With a sigma of 0.2 Ang, borders should be still 0
        smoothed = g.smooth(r=0.1, truncate=4)
        assert np.any(smoothed.grid == 0)

    def test_smooth_uniform(self, setup):
        g = Grid(0.1, sc=[[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        g[10, 10, 10] = 1

        # With a radius of 0.7 Ang, that single value should be propagated
        # to the whole grid
        smoothed = g.smooth(r=1., method='uniform')
        assert not np.any(smoothed.grid == 0)

        # With a sigma of 0.1 Ang, borders should still be 0
        smoothed = g.smooth(r=0.9, method='uniform')
        assert np.any(smoothed.grid == 0)

    def test_index_ndim1(self, setup):
        mid = np.array(setup.g.shape, np.int32) // 2 - 1
        v = [0.001, 0., 0.001]
        idx = setup.g.index(setup.sc.center() - v)
        assert np.all(mid == idx)
        for i in range(3):
            idx = setup.g.index(setup.sc.center() - v, axis=i)
            assert idx == mid[i]

    def test_index_fail(self, setup):
        with pytest.raises(ValueError):
            setup.g.index([0.1, 0.2])

    def test_index_ndim2(self, setup):
        mid = np.array(setup.g.shape, np.int32) // 2 - 1
        v = [0.001, 0., 0.001]
        idx = setup.g.index([[0]*3, setup.sc.center() - v])
        assert np.allclose([[0] * 3, mid], idx)

        for i in range(3):
            idx = setup.g.index([[0]*3, setup.sc.center() - v], axis=i)
            assert np.allclose([[0, 0, 0][i], mid[i]], idx)

    def test_index_shape1(self, setup):
        g = setup.g.copy()
        n = 0
        for r in [0.5, 1., 1.5]:
            s = Ellipsoid(r)
            idx = g.index(s)
            assert len(idx) > n
            n = len(idx)

        # Check that if we place the sphere an integer
        # amount of cells away we retain an equal amount of indices
        # Also we can check whether they are the same if we add the
        # offset
        v = g.dcell.sum(0)
        vd = v * 0.001
        s = Ellipsoid(1.)
        idx0 = g.index(s)
        idx0.sort(0)
        for d in [10, 15, 20, 60, 100, 340]:
            idx = g.index(v * d + vd)
            s = Ellipsoid(1., center=v * d + vd)
            idx1 = g.index(s)
            idx1.sort(0)
            assert len(idx1) == len(idx0)
            assert np.all(idx0 == idx1 - idx.reshape(1, 3))

    def test_index_shape2(self, setup):
        g = setup.g.copy()
        n = 0
        for r in [0.5, 1., 1.5]:
            s = Cuboid(r)
            idx = g.index(s)
            assert len(idx) > n
            n = len(idx)

        # Check that if we place the sphere an integer
        # amount of cells away we retain an equal amount of indices
        # Also we can check whether they are the same if we add the
        # offset
        v = g.dcell.sum(0)
        vd = v * 0.001
        s = Cuboid(1.)
        idx0 = g.index(s)
        idx0.sort(0)
        for d in [10, 15, 20, 60, 100, 340]:
            idx = g.index(v * d + vd)
            s = Cuboid(1., center=v * d + vd)
            idx1 = g.index(s)
            idx1.sort(0)
            assert len(idx1) == len(idx0)
            assert np.all(idx0 == idx1 - idx.reshape(1, 3))

    def test_sum(self, setup):
        for i in range(3):
            assert setup.g.sum(i).shape[i] == 1

    def test_mean(self, setup):
        for i in range(3):
            assert setup.g.mean(i).shape[i] == 1

    def test_cross_section(self, setup):
        for i in range(3):
            assert setup.g.cross_section(1, i).shape[i] == 1

    def test_cross_section_fail(self, setup):
        with pytest.raises(ValueError):
            setup.g.cross_section(1, -1)

    def test_remove_part(self, setup):
        for i in range(3):
            assert setup.g.remove_part(1, i, above=True).shape[i] == 1

    def test_sub_fail(self, setup):
        g = Grid(np.array(setup.g.shape) // 2 + 1, sc=setup.g.sc.copy())
        with pytest.raises(ValueError):
            g.sub([], 0)

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

    def test_set_grid1(self, setup):
        g = setup.g.copy()
        g.set_grid([2, 2, 2])
        assert np.all(np.array(g.shape) == 2)
        g.set_grid([2, 2, 3])
        assert np.all(np.array(g.shape) == [2, 2, 3])

    def test_set_grid2(self, setup):
        g = setup.g.copy()
        with pytest.raises(ValueError):
            g.set_grid([2, 2, 2, 4])

    def test_bc1(self, setup):
        assert np.all(setup.g.bc == setup.g.PERIODIC)
        setup.g.set_bc(a=setup.g.NEUMANN)
        setup.g.set_bc(b=setup.g.NEUMANN, c=setup.g.NEUMANN)
        assert np.all(setup.g.bc == setup.g.NEUMANN)
        setup.g.set_bc(setup.g.PERIODIC)
        assert np.all(setup.g.bc == setup.g.PERIODIC)

    def test_bc2(self, setup):
        g = setup.g.copy()
        P = g.PERIODIC
        D = g.DIRICHLET
        N = g.NEUMANN
        assert np.all(g.bc == P)
        g.set_bc(N)
        assert np.all(g.bc == setup.g.NEUMANN)
        bc = [[P, P], [N, D], [D, N]]
        g.set_bc(bc)
        assert np.all(g.bc == bc)

    def test_argumentparser(self, setup):
        setup.g.ArgumentParser()

    def test_pyamg1(self, setup):
        g = setup.g.copy()
        g.set_bc(g.PERIODIC) # periodic boundary conditions
        n = np.prod(g.shape)
        A = csr_matrix((n, n))
        b = np.zeros(A.shape[0])

        lb = g.mgrid(slice(0, 1), slice(0, g.shape[1]), slice(0, g.shape[2]))
        lb2 = g.mgrid([slice(0, 1), slice(0, g.shape[1]), slice(0, g.shape[2])])
        assert np.allclose(lb, lb2)
        del lb2

        # Retrieve linear indices
        index = g.pyamg_index(lb)
        g.pyamg_source(b, index, 1)
        assert int(b.sum()) == len(index)
        b[:] = 0

        g.pyamg_fix(A, b, index, 1)
        assert int(b.sum()) == len(index)
        assert A.getnnz() == len(index)

    def test_pyamg2(self, setup):
        # Currently this simply runs the stuff.
        # Nothing is actually tested other than succesfull run,
        # the correctness of the values are not.
        g = setup.g.copy()
        bc = [[g.PERIODIC] * 2,
              [g.NEUMANN, g.DIRICHLET],
              [g.DIRICHLET, g.NEUMANN]]
        g.set_bc(bc)
        n = np.prod(g.shape)
        A = csr_matrix((n, n))
        b = np.zeros(A.shape[0])
        g.pyamg_boundary_condition(A, b)


def test_grid_fold():
    grid = Grid([4, 5, 6])
    # Assert shapes
    assert grid.index_fold([-1] * 3).shape == (3,)
    assert grid.index_fold([[-1] * 3] * 2, False).shape == (2, 3)
    assert grid.index_fold([[-1] * 3] * 2, True).shape == (1, 3)

    assert np.all(grid.index_fold([-1, -1, -1]) == [3, 4, 5])
    assert np.all(grid.index_fold([[-1, -1, -1]] * 2) == [3, 4, 5])
    assert np.all(grid.index_fold([[-1, -1, -1]] * 2, False) == [[3, 4, 5]] * 2)

    idx = [[-1, 0, 0],
           [3, 0, 0]]
    assert np.all(grid.index_fold(idx) == [3, 0, 0])
    assert np.all(grid.index_fold(idx, False) == [[3, 0, 0]] * 2)

    idx = [[3, 0, 0],
           [2, 0, 0]]
    assert np.all(grid.index_fold(idx, False) == idx)
    assert not np.all(grid.index_fold(idx) == idx) # sorted from unique
    assert np.all(grid.index_fold(idx) == np.sort(idx, axis=0))


def test_grid_tile_sc():
    grid = Grid([4, 5, 6])
    grid2 = grid.tile(2, 2)
    assert grid.shape[:2] == grid2.shape[:2]
    assert grid.shape[2] == grid2.shape[2] // 2
    assert grid.volume * 2 == pytest.approx(grid2.volume)

    grid4 = grid2.tile(2, 1)
    assert grid.shape[0] == grid4.shape[0]
    assert grid.shape[1] == grid4.shape[1] // 2
    assert grid.shape[2] == grid4.shape[2] // 2
    assert grid.volume * 4 == pytest.approx(grid4.volume)


def test_grid_tile_geom():
    grid = Grid([4, 5, 6], geometry=Geometry([0] * 3, Atom[4], sc=4.))
    grid2 = grid.tile(2, 2)
    assert grid.shape[:2] == grid2.shape[:2]
    assert grid.shape[2] == grid2.shape[2] // 2
    assert grid.volume * 2 == pytest.approx(grid2.volume)
    assert grid.geometry.na * 2 == grid2.geometry.na

    grid4 = grid2.tile(2, 1)
    assert grid.shape[0] == grid4.shape[0]
    assert grid.shape[1] == grid4.shape[1] // 2
    assert grid.shape[2] == grid4.shape[2] // 2
    assert grid.volume * 4 == pytest.approx(grid4.volume)
    assert grid.geometry.na * 4 == grid4.geometry.na
