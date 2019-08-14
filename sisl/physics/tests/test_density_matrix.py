from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import Geometry, Atom, SphericalOrbital, AtomicOrbital, SuperCell
from sisl import Grid, Spin
from sisl import DensityMatrix


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            bond = 1.42
            sq3h = 3.**.5 * 0.5
            self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                          [1.5, -sq3h, 0.],
                                          [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

            n = 60
            rf = np.linspace(0, bond * 1.01, n)
            rf = (rf, rf)
            orb = SphericalOrbital(1, rf, 2.)
            C = Atom(6, orb.toAtomicOrbital())
            self.g = Geometry(np.array([[0., 0., 0.],
                                        [1., 0., 0.]], np.float64) * bond,
                              atom=C, sc=self.sc)
            self.D = DensityMatrix(self.g)
            self.DS = DensityMatrix(self.g, orthogonal=False)

            def func(D, ia, idxs, idxs_xyz):
                idx = D.geom.close(ia, R=(0.1, 1.44), idx=idxs, idx_xyz=idxs_xyz)
                ia = ia * 3

                i0 = idx[0] * 3
                i1 = idx[1] * 3
                # on-site
                p = 1.
                D.D[ia, i0] = p
                D.D[ia+1, i0+1] = p
                D.D[ia+2, i0+2] = p

                # nn
                p = 0.1

                # on-site directions
                D.D[ia, ia+1] = p
                D.D[ia, ia+2] = p
                D.D[ia+1, ia] = p
                D.D[ia+1, ia+2] = p
                D.D[ia+2, ia] = p
                D.D[ia+2, ia+1] = p

                D.D[ia, i1+1] = p
                D.D[ia, i1+2] = p

                D.D[ia+1, i1] = p
                D.D[ia+1, i1+2] = p

                D.D[ia+2, i1] = p
                D.D[ia+2, i1+1] = p

            self.func = func
    return t()


@pytest.mark.density_matrix
class TestDensityMatrix(object):

    def test_objects(self, setup):
        assert len(setup.D.xyz) == 2
        assert setup.g.no == len(setup.D)

    def test_dtype(self, setup):
        assert setup.D.dtype == np.float64

    def test_ortho(self, setup):
        assert setup.D.orthogonal

    def test_set1(self, setup):
        D = setup.D.copy()
        D.D[0, 0] = 1.
        assert D[0, 0] == 1.
        assert D[1, 0] == 0.

    def test_mulliken(self, setup):
        D = setup.D.copy()
        D.construct(setup.func)
        mulliken = D.mulliken('atom')
        assert mulliken.shape == (1, len(D.geometry))
        mulliken = D.mulliken('orbital')
        assert mulliken.shape == (1, len(D))

    def test_mulliken_values_orthogonal(self, setup):
        D = setup.D.copy()
        D.D[0, 0] = 1.
        D.D[1, 1] = 2.
        D.D[1, 2] = 2.
        mulliken = D.mulliken('orbital')
        assert np.allclose(mulliken[0, :2], [1., 2.])
        assert mulliken.sum() == pytest.approx(3)
        mulliken = D.mulliken('atom')
        assert mulliken[0, 0] == pytest.approx(3)
        assert mulliken.sum() == pytest.approx(3)

    def test_mulliken_values_non_orthogonal(self, setup):
        D = setup.DS.copy()
        D[0, 0] = (1., 1.)
        D[1, 1] = (2., 1.)
        D[1, 2] = (2., 0.5)
        mulliken = D.mulliken('orbital')
        assert np.allclose(mulliken[0, :2], [1., 3.])
        assert mulliken.sum() == pytest.approx(4.)
        mulliken = D.mulliken('atom')
        assert mulliken[0, 0] == pytest.approx(4)
        assert mulliken.sum() == pytest.approx(4)

    def test_rho1(self, setup):
        D = setup.D.copy()
        D.construct(setup.func)
        grid = Grid(0.2, geometry=setup.D.geom)
        D.density(grid)

    def test_rho2(self, setup):
        bond = 1.42
        sq3h = 3.**.5 * 0.5
        sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                      [1.5, -sq3h, 0.],
                                      [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

        n = 60
        rf = np.linspace(0, bond * 1.01, n)
        rf = (rf, rf)
        orb = SphericalOrbital(1, rf, 2.)
        C = Atom(6, orb)
        g = Geometry(np.array([[0., 0., 0.],
                                    [1., 0., 0.]], np.float64) * bond,
                        atom=C, sc=sc)
        D = DensityMatrix(g)
        D.construct([[0.1, bond + 0.01], [1., 0.1]])
        grid = Grid(0.2, geometry=D.geom)
        D.density(grid)

        D = DensityMatrix(g, spin=Spin('P'))
        D.construct([[0.1, bond + 0.01], [(1., 0.5), (0.1, 0.1)]])
        grid = Grid(0.2, geometry=D.geom)
        D.density(grid)
        D.density(grid, [1., -1])
        D.density(grid, 0)
        D.density(grid, 1)

        D = DensityMatrix(g, spin=Spin('NC'))
        D.construct([[0.1, bond + 0.01], [(1., 0.5, 0.01, 0.01), (0.1, 0.1, 0.1, 0.1)]])
        grid = Grid(0.2, geometry=D.geom)
        D.density(grid)
        D.density(grid, [[1., 0.], [0., -1]])

        D = DensityMatrix(g, spin=Spin('SO'))
        D.construct([[0.1, bond + 0.01], [(1., 0.5, 0.01, 0.01, 0.01, 0.01, 0., 0.), (0.1, 0.1, 0.1, 0.1, 0., 0., 0., 0.)]])
        grid = Grid(0.2, geometry=D.geom)
        D.density(grid)
        D.density(grid, [[1., 0.], [0., -1]])
        D.density(grid, Spin.X)
        D.density(grid, Spin.Y)
        D.density(grid, Spin.Z)

    def test_orbital_momentum(self, setup):
        bond = 1.42
        sq3h = 3.**.5 * 0.5
        sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                      [1.5, -sq3h, 0.],
                                      [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

        orb = AtomicOrbital('px', R=bond * 1.001)
        C = Atom(6, orb)
        g = Geometry(np.array([[0., 0., 0.],
                                    [1., 0., 0.]], np.float64) * bond,
                        atom=C, sc=sc)
        D = DensityMatrix(g, spin=Spin('SO'))
        D.construct([[0.1, bond + 0.01], [(1., 0.5, 0.01, 0.01, 0.01, 0.01, 0., 0.), (0.1, 0.1, 0.1, 0.1, 0., 0., 0., 0.)]])
        D.orbital_momentum("atom")
        D.orbital_momentum("orbital")

    def test_rho_eta(self, setup):
        D = setup.D.copy()
        D.construct(setup.func)
        grid = Grid(0.2, geometry=setup.D.geom)
        D.density(grid, eta=True)

    def test_rho_smaller_grid1(self, setup):
        D = setup.D.copy()
        D.construct(setup.func)
        sc = setup.D.geom.cell.copy() / 2
        grid = Grid(0.2, geometry=setup.D.geom.copy(), sc=sc)
        D.density(grid)

    @pytest.mark.xfail(raises=ValueError)
    def test_rho_fail_p(self, setup):
        bond = 1.42
        sq3h = 3.**.5 * 0.5
        sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                      [1.5, -sq3h, 0.],
                                      [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

        n = 60
        rf = np.linspace(0, bond * 1.01, n)
        rf = (rf, rf)
        orb = SphericalOrbital(1, rf, 2.)
        C = Atom(6, orb)
        g = Geometry(np.array([[0., 0., 0.],
                                    [1., 0., 0.]], np.float64) * bond,
                        atom=C, sc=sc)

        D = DensityMatrix(g, spin=Spin('P'))
        D.construct([[0.1, bond + 0.01], [(1., 0.5), (0.1, 0.1)]])
        grid = Grid(0.2, geometry=D.geom)
        D.density(grid, [1., -1, 0.])

    @pytest.mark.xfail(raises=ValueError)
    def test_rho_fail_nc(self, setup):
        bond = 1.42
        sq3h = 3.**.5 * 0.5
        sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                      [1.5, -sq3h, 0.],
                                      [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

        n = 60
        rf = np.linspace(0, bond * 1.01, n)
        rf = (rf, rf)
        orb = SphericalOrbital(1, rf, 2.)
        C = Atom(6, orb)
        g = Geometry(np.array([[0., 0., 0.],
                                    [1., 0., 0.]], np.float64) * bond,
                        atom=C, sc=sc)

        D = DensityMatrix(g, spin=Spin('NC'))
        D.construct([[0.1, bond + 0.01], [(1., 0.5, 0.01, 0.01), (0.1, 0.1, 0.1, 0.1)]])
        grid = Grid(0.2, geometry=D.geom)
        D.density(grid, [1., 0.])

    def test_pickle(self, setup):
        import pickle as p
        D = setup.D.copy()
        D.construct(setup.func)
        s = p.dumps(D)
        d = p.loads(s)
        assert d.spsame(D)
        assert np.allclose(d.eigh(), D.eigh())
