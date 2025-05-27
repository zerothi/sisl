# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

from sisl import (
    Atom,
    AtomicOrbital,
    DensityMatrix,
    Geometry,
    Grid,
    Lattice,
    SparseAtom,
    SparseOrbital,
    SphericalOrbital,
    Spin,
)


@pytest.fixture
def setup():
    class t:
        def __init__(self):
            self.bond = bond = 1.42
            sq3h = 3.0**0.5 * 0.5
            self.lattice = Lattice(
                np.array(
                    [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
                )
                * bond,
                nsc=[3, 3, 1],
            )

            n = 60
            rf = np.linspace(0, bond * 1.01, n)
            rf = (rf, rf)
            orb = SphericalOrbital(1, rf, 2.0)
            C = Atom(6, orb.toAtomicOrbital())
            self.g = Geometry(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
                atoms=C,
                lattice=self.lattice,
            )
            self.D = DensityMatrix(self.g)
            self.DS = DensityMatrix(self.g, orthogonal=False)

            def func(D, ia, atoms, atoms_xyz):
                idx = D.geometry.close(
                    ia, R=(0.1, 1.44), atoms=atoms, atoms_xyz=atoms_xyz
                )
                ia = ia * 3

                i0 = idx[0] * 3
                i1 = idx[1] * 3
                # on-site
                p = 1.0
                D.D[ia, i0] = p
                D.D[ia + 1, i0 + 1] = p
                D.D[ia + 2, i0 + 2] = p

                # nn
                p = 0.1

                # on-site directions
                D.D[ia, ia + 1] = p
                D.D[ia, ia + 2] = p
                D.D[ia + 1, ia] = p
                D.D[ia + 1, ia + 2] = p
                D.D[ia + 2, ia] = p
                D.D[ia + 2, ia + 1] = p

                D.D[ia, i1 + 1] = p
                D.D[ia, i1 + 2] = p

                D.D[ia + 1, i1] = p
                D.D[ia + 1, i1 + 2] = p

                D.D[ia + 2, i1] = p
                D.D[ia + 2, i1 + 1] = p

            self.func = func

    return t()


@pytest.fixture(
    scope="module",
    params=["direct", "pre-compute"],
)
def density_method(request):
    return request.param


@pytest.mark.physics
@pytest.mark.densitymatrix
class TestDensityMatrix:
    def test_objects(self, setup):
        assert len(setup.D.xyz) == 2
        assert setup.g.no == len(setup.D)

    def test_dtype(self, setup):
        assert setup.D.dtype == np.float64

    def test_ortho(self, setup):
        assert setup.D.orthogonal

    def test_set1(self, setup):
        D = setup.D.copy()
        D.D[0, 0] = 1.0
        assert D[0, 0] == 1.0
        assert D[1, 0] == 0.0

    def test_mulliken(self, setup):
        D = setup.D.copy()
        D.construct(setup.func)
        mulliken = D.mulliken("atom")
        assert mulliken.shape == (len(D.geometry),)
        mulliken = D.mulliken("orbital")
        assert mulliken.shape == (len(D),)

    def test_mulliken_values_orthogonal(self, setup):
        D = setup.D.copy()
        D.D[0, 0] = 1.0
        D.D[1, 1] = 2.0
        D.D[1, 2] = 2.0
        mulliken = D.mulliken("orbital")
        assert np.allclose(mulliken[:2], [1.0, 2.0])
        assert mulliken.sum() == pytest.approx(3)
        mulliken = D.mulliken("atom")
        assert mulliken[0] == pytest.approx(3)
        assert mulliken.sum() == pytest.approx(3)

    def test_mulliken_values_non_orthogonal(self, setup):
        D = setup.DS.copy()
        D[0, 0] = (1.0, 1.0)
        D[1, 1] = (2.0, 1.0)
        D[1, 2] = (2.0, 0.5)
        mulliken = D.mulliken("orbital")
        assert np.allclose(mulliken[:2], [1.0, 3.0])
        assert mulliken.sum() == pytest.approx(4)
        mulliken = D.mulliken("atom")
        assert mulliken[0] == pytest.approx(4)
        assert mulliken.sum() == pytest.approx(4)

    def test_mulliken_polarized(self):
        bond = 1.42
        sq3h = 3.0**0.5 * 0.5
        lattice = Lattice(
            np.array(
                [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
            )
            * bond,
            nsc=[3, 3, 1],
        )

        orb = AtomicOrbital("px", R=bond * 1.001)
        C = Atom(6, orb)
        g = Geometry(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
            atoms=C,
            lattice=lattice,
        )
        D = DensityMatrix(g, spin=Spin("P"))
        # 1 charge onsite for each spin-up
        # 0.5 charge onsite for each spin-down
        D.construct([[0.1, bond + 0.01], [(1.0, 0.5), (0.1, 0.1)]])

        m = D.mulliken("orbital")
        assert m[0].sum() == pytest.approx(3)
        assert m[1].sum() == pytest.approx(1)
        m = D.mulliken("atom")
        assert m[0].sum() == pytest.approx(3)
        assert m[1].sum() == pytest.approx(1)

    @pytest.mark.parametrize("method", ["wiberg", "mayer"])
    @pytest.mark.parametrize("option", ["", ":spin"])
    @pytest.mark.parametrize("projection", ["atom", "orbitals"])
    def test_bond_order(self, setup, method, option, projection):
        D = setup.D.copy()
        D.construct(setup.func)
        BO = D.bond_order(method + option, projection)
        if projection == "atom":
            assert isinstance(BO, SparseAtom)
            assert BO.shape[:2] == (D.geometry.na, D.geometry.na_s)
        elif projection == "orbitals":
            assert isinstance(BO, SparseOrbital)
            assert BO.shape[:2] == (D.geometry.no, D.geometry.no_s)

    def test_rho1(self, setup, density_method):
        D = setup.D.copy()
        D.construct(setup.func)
        grid = Grid(0.2, geometry=setup.D.geometry)
        D.density(grid, method=density_method)

    @pytest.mark.filterwarnings("ignore", message="*non-Hermitian on-site")
    def test_rho2(self, density_method):
        bond = 1.42
        sq3h = 3.0**0.5 * 0.5
        lattice = Lattice(
            np.array(
                [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
            )
            * bond,
            nsc=[3, 3, 1],
        )

        n = 60
        rf = np.linspace(0, bond * 1.01, n)
        rf = (rf, rf)
        orb = SphericalOrbital(1, rf, 2.0)
        C = Atom(6, orb)
        g = Geometry(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
            atoms=C,
            lattice=lattice,
        )
        D = DensityMatrix(g)
        D.construct([[0.1, bond + 0.01], [1.0, 0.1]])
        grid = Grid(0.2, geometry=D.geometry)
        D.density(grid, method=density_method)

        D = DensityMatrix(g, spin=Spin("P"))
        D.construct([[0.1, bond + 0.01], [(1.0, 0.5), (0.1, 0.1)]])
        grid = Grid(0.2, geometry=D.geometry)
        D.density(grid, method=density_method)
        D.density(grid, [1.0, -1], method=density_method)
        D.density(grid, 0, method=density_method)
        D.density(grid, 1, method=density_method)

        D = DensityMatrix(g, spin=Spin("NC"))
        D.construct(
            [[0.1, bond + 0.01], [(1.0, 0.5, 0.01, 0.01), (0.1, 0.1, 0.1, 0.1)]]
        )
        grid = Grid(0.2, geometry=D.geometry)
        D.density(grid, method=density_method)
        D.density(grid, [[1.0, 0.0], [0.0, -1]], method=density_method)

        D = DensityMatrix(g, spin=Spin("SO"))
        D.construct(
            [
                [0.1, bond + 0.01],
                [
                    (1.0, 0.5, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0),
                    (0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0),
                ],
            ]
        )
        grid = Grid(0.2, geometry=D.geometry)
        D.density(grid, method=density_method)
        D.density(grid, [[1.0, 0.0], [0.0, -1]], method=density_method)
        D.density(grid, Spin.X, method=density_method)
        D.density(grid, Spin.Y, method=density_method)
        D.density(grid, Spin.Z, method=density_method)

    @pytest.mark.filterwarnings("ignore", message="*non-Hermitian on-site")
    def test_orbital_momentum(self):
        bond = 1.42
        sq3h = 3.0**0.5 * 0.5
        lattice = Lattice(
            np.array(
                [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
            )
            * bond,
            nsc=[3, 3, 1],
        )

        orb = AtomicOrbital("px", R=bond * 1.001)
        C = Atom(6, orb)
        g = Geometry(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
            atoms=C,
            lattice=lattice,
        )
        D = DensityMatrix(g, spin=Spin("SO"))
        D.construct(
            [
                [0.1, bond + 0.01],
                [
                    (1.0, 0.5, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0),
                    (0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0),
                ],
            ]
        )
        D.orbital_momentum("atom")
        D.orbital_momentum("orbital")

    def test_spin_align_pol(self):
        bond = 1.42
        sq3h = 3.0**0.5 * 0.5
        lattice = Lattice(
            np.array(
                [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
            )
            * bond,
            nsc=[3, 3, 1],
        )

        orb = AtomicOrbital("px", R=bond * 1.001)
        C = Atom(6, orb)
        g = Geometry(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
            atoms=C,
            lattice=lattice,
        )
        D = DensityMatrix(g, spin=Spin("p"))
        D.construct([[0.1, bond + 0.01], [(1.0, 0.5), (0.1, 0.2)]])
        D_mull = D.mulliken()
        assert D_mull.shape == (2, len(D))

        v = np.array([1, 2, 3])
        d = D.spin_align(v)
        d_mull = d.mulliken()
        assert np.allclose(d_mull, d.astype(np.complex128).mulliken())
        assert d_mull.shape == (4, len(D))

        assert not np.allclose(D_mull[1], d_mull[3])
        assert np.allclose(D_mull[0], d_mull[0])

    def test_spin_align_nc(self):
        bond = 1.42
        sq3h = 3.0**0.5 * 0.5
        lattice = Lattice(
            np.array(
                [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
            )
            * bond,
            nsc=[3, 3, 1],
        )

        orb = AtomicOrbital("px", R=bond * 1.001)
        C = Atom(6, orb)
        g = Geometry(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
            atoms=C,
            lattice=lattice,
        )
        D = DensityMatrix(g, spin=Spin("nc"))
        D.construct(
            [[0.1, bond + 0.01], [(1.0, 0.5, 0.01, 0.01), (0.1, 0.2, 0.1, 0.1)]]
        )
        D_mull = D.mulliken()
        v = np.array([1, 2, 3])
        d = D.spin_align(v)
        d_mull = d.mulliken()
        assert np.allclose(d_mull, d.astype(np.complex128).mulliken())
        assert not np.allclose(D_mull, d_mull)
        assert np.allclose(D_mull[0], d_mull[0])

    @pytest.mark.filterwarnings("ignore", message="*non-Hermitian on-site")
    def test_spin_align_so(self):
        bond = 1.42
        sq3h = 3.0**0.5 * 0.5
        lattice = Lattice(
            np.array(
                [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
            )
            * bond,
            nsc=[3, 3, 1],
        )

        orb = AtomicOrbital("px", R=bond * 1.001)
        C = Atom(6, orb)
        g = Geometry(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
            atoms=C,
            lattice=lattice,
        )
        D = DensityMatrix(g, spin=Spin("SO"))
        D.construct(
            [
                [0.1, bond + 0.01],
                [
                    (1.0, 0.5, 0.01, 0.01, 0.01, 0.01, 0.2, 0.2),
                    (0.1, 0.2, 0.1, 0.1, 0.0, 0.1, 0.2, 0.3),
                ],
            ]
        )
        D_mull = D.mulliken()
        v = np.array([1, 2, 3])
        d = D.spin_align(v, atoms=0)
        d_mull = d.mulliken()
        assert np.allclose(d_mull, d.astype(np.complex128).mulliken())
        assert not np.allclose(D_mull, d_mull)
        assert np.allclose(D_mull[0], d_mull[0])

    def test_spin_rotate_pol(self):
        bond = 1.42
        sq3h = 3.0**0.5 * 0.5
        lattice = Lattice(
            np.array(
                [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
            )
            * bond,
            nsc=[3, 3, 1],
        )

        orb = AtomicOrbital("px", R=bond * 1.001)
        C = Atom(6, orb)
        g = Geometry(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
            atoms=C,
            lattice=lattice,
        )
        D = DensityMatrix(g, spin=Spin("p"))
        D.construct([[0.1, bond + 0.01], [(1.0, 0.5), (0.1, 0.2)]])

        D_mull = D.mulliken()
        assert np.allclose(D_mull, D.astype(np.complex128).mulliken())
        assert D_mull.shape == (2, len(D))

        d = D.spin_rotate([45, 60, 90], rad=False)
        d_mull = d.mulliken()
        assert d_mull.shape == (4, len(D))

        assert not np.allclose(D_mull[1], d_mull[3])
        assert np.allclose(D_mull[0], d_mull[0])

    def test_spin_rotate_pol_full(self):
        bond = 1.42
        sq3h = 3.0**0.5 * 0.5
        lattice = Lattice(
            np.array(
                [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
            )
            * bond,
            nsc=[3, 3, 1],
        )

        orb = AtomicOrbital("px", R=bond * 1.001)
        C = Atom(6, orb)
        g = Geometry(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
            atoms=C,
            lattice=lattice,
        )
        D = DensityMatrix(g, spin=Spin("p"))
        D.construct([[0.1, bond + 0.01], [(1.0, 0), (0.1, 0.0)]])

        D_mull = D.mulliken()
        assert np.allclose(D_mull, D.astype(np.complex128).mulliken())
        assert D_mull.shape == (2, len(D))

        # Euler (noop)
        d = D.spin_rotate([0, 0, 64], rad=False)
        assert d.spin.is_polarized
        assert np.allclose(d.mulliken()[1], D.mulliken()[1])
        d = D.spin_rotate([180, 180, 64], rad=False)
        assert d.spin.is_polarized
        assert np.allclose(d.mulliken()[1], D.mulliken()[1])

        # Euler (full)
        d = D.spin_rotate([0, 180, 64], rad=False)
        assert d.spin.is_polarized
        assert np.allclose(d.mulliken()[1], -D.mulliken()[1])
        d = D.spin_rotate([180, 0, 64], rad=False)
        assert d.spin.is_polarized
        assert np.allclose(d.mulliken()[1], -D.mulliken()[1])

    def test_spin_rotate_nc(self):
        bond = 1.42
        sq3h = 3.0**0.5 * 0.5
        lattice = Lattice(
            np.array(
                [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
            )
            * bond,
            nsc=[3, 3, 1],
        )

        orb = AtomicOrbital("px", R=bond * 1.001)
        C = Atom(6, orb)
        g = Geometry(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
            atoms=C,
            lattice=lattice,
        )
        D = DensityMatrix(g, spin=Spin("nc"))
        D.construct(
            [[0.1, bond + 0.01], [(1.0, 0.5, 0.01, 0.01), (0.1, 0.2, 0.1, 0.1)]]
        )

        D_mull = D.mulliken()
        assert np.allclose(D_mull, D.astype(np.complex128).mulliken())
        d = D.spin_rotate([45, 60, 90], rad=False)

        d_mull = d.mulliken()

        assert not np.allclose(D_mull, d_mull)
        assert np.allclose(D_mull[0], d_mull[0])

    @pytest.mark.filterwarnings("ignore", message="*non-Hermitian on-site")
    def test_spin_rotate_so(self):
        bond = 1.42
        sq3h = 3.0**0.5 * 0.5
        lattice = Lattice(
            np.array(
                [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
            )
            * bond,
            nsc=[3, 3, 1],
        )

        orb = AtomicOrbital("px", R=bond * 1.001)
        C = Atom(6, orb)
        g = Geometry(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
            atoms=C,
            lattice=lattice,
        )
        D = DensityMatrix(g, spin=Spin("SO"))
        D.construct(
            [
                [0.1, bond + 0.01],
                [
                    (1.0, 0.5, 0.01, 0.01, 0.01, 0.01, 0.2, 0.2),
                    (0.1, 0.2, 0.1, 0.1, 0.0, 0.1, 0.2, 0.3),
                ],
            ]
        )
        D_mull = D.mulliken()
        assert np.allclose(D_mull, D.astype(np.complex128).mulliken())
        d = D.spin_rotate([45, 60, 90], rad=False)
        d_mull = d.mulliken()
        assert not np.allclose(D_mull, d_mull)
        assert np.allclose(D_mull[0], d_mull[0])

        d1 = (
            d.spin_rotate([0, 0, -90], rad=False)
            .spin_rotate([0, -60, 0], rad=False)
            .spin_rotate([-45, 0, 0], rad=False)
        )
        d1_mull = d1.mulliken()
        assert not np.allclose(d_mull, d1_mull)
        assert np.allclose(D_mull, d1_mull)

    def test_rho_eta(self, setup, density_method):
        D = setup.D.copy()
        D.construct(setup.func)
        grid = Grid(0.2, geometry=setup.D.geometry)
        D.density(grid, eta=True, method=density_method)

    def test_rho_smaller_grid1(self, setup, density_method):
        D = setup.D.copy()
        D.construct(setup.func)
        lattice = setup.D.geometry.cell.copy() / 2
        grid = Grid(0.2, geometry=setup.D.geometry.copy(), lattice=lattice)
        D.density(grid, method=density_method)

    def test_rho_fail_p(self, density_method):
        bond = 1.42
        sq3h = 3.0**0.5 * 0.5
        lattice = Lattice(
            np.array(
                [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
            )
            * bond,
            nsc=[3, 3, 1],
        )

        n = 60
        rf = np.linspace(0, bond * 1.01, n)
        rf = (rf, rf)
        orb = SphericalOrbital(1, rf, 2.0)
        C = Atom(6, orb)
        g = Geometry(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
            atoms=C,
            lattice=lattice,
        )

        D = DensityMatrix(g, spin=Spin("P"))
        D.construct([[0.1, bond + 0.01], [(1.0, 0.5), (0.1, 0.1)]])
        grid = Grid(0.2, geometry=D.geometry)
        with pytest.raises(ValueError):
            D.density(grid, [1.0, -1, 0.0], method=density_method)

    def test_rho_fail_nc(self, density_method):
        bond = 1.42
        sq3h = 3.0**0.5 * 0.5
        lattice = Lattice(
            np.array(
                [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
            )
            * bond,
            nsc=[3, 3, 1],
        )

        n = 60
        rf = np.linspace(0, bond * 1.01, n)
        rf = (rf, rf)
        orb = SphericalOrbital(1, rf, 2.0)
        C = Atom(6, orb)
        g = Geometry(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
            atoms=C,
            lattice=lattice,
        )

        D = DensityMatrix(g, spin=Spin("NC"))
        D.construct(
            [[0.1, bond + 0.01], [(1.0, 0.5, 0.01, 0.01), (0.1, 0.1, 0.1, 0.1)]]
        )
        grid = Grid(0.2, geometry=D.geometry)
        with pytest.raises(ValueError):
            D.density(grid, [1.0, 0.0], method=density_method)

    def test_pickle(self, setup):
        import pickle as p

        D = setup.D.copy()
        D.construct(setup.func)
        s = p.dumps(D)
        d = p.loads(s)
        assert d.spsame(D)
        assert np.allclose(d.eigh(), D.eigh())

    def test_transform(self, setup):
        D = DensityMatrix(setup.g, spin="so")
        a = np.arange(8)
        for ia in setup.g:
            D[ia, ia] = a
        Dcsr = [D.tocsr(i) for i in range(D.shape[2])]

        Dt = D.transform(spin="unpolarized").astype(np.float32)
        assert np.abs(0.5 * Dcsr[0] + 0.5 * Dcsr[1] - Dt.tocsr(0)).sum() == 0

        Dt = D.transform(spin="polarized", orthogonal=False)
        assert np.abs(Dcsr[0] - Dt.tocsr(0)).sum() == 0
        assert np.abs(Dcsr[1] - Dt.tocsr(1)).sum() == 0
        assert np.abs(Dt.tocsr(2)).sum() != 0

        Dt = D.transform(spin="non-colinear", orthogonal=False)
        assert np.abs(Dcsr[0] - Dt.tocsr(0)).sum() == 0
        assert np.abs(Dcsr[1] - Dt.tocsr(1)).sum() == 0
        assert np.abs(Dcsr[2] - Dt.tocsr(2)).sum() == 0
        assert np.abs(Dcsr[3] - Dt.tocsr(3)).sum() == 0
        assert np.abs(Dt.tocsr(-1)).sum() != 0

    def test_transform_nonortho(self, setup):
        D = DensityMatrix(setup.g, spin="polarized", orthogonal=False)
        a = np.arange(3)
        a[-1] = 1.0
        for ia in setup.g:
            D[ia, ia] = a

        Dt = D.transform(spin="unpolarized").astype(np.float32)
        assert np.abs(0.5 * D.tocsr(0) + 0.5 * D.tocsr(1) - Dt.tocsr(0)).sum() == 0
        assert np.abs(D.tocsr(-1) - Dt.tocsr(-1)).sum() == 0
        Dt = D.transform(spin="polarized")
        assert np.abs(D.tocsr(0) - Dt.tocsr(0)).sum() == 0
        assert np.abs(D.tocsr(1) - Dt.tocsr(1)).sum() == 0
        Dt = D.transform(spin="polarized", orthogonal=True)
        assert np.abs(D.tocsr(0) - Dt.tocsr(0)).sum() == 0
        assert np.abs(D.tocsr(1) - Dt.tocsr(1)).sum() == 0
        Dt = D.transform(spin="non-colinear", orthogonal=False)
        assert np.abs(D.tocsr(0) - Dt.tocsr(0)).sum() == 0
        assert np.abs(D.tocsr(1) - Dt.tocsr(1)).sum() == 0
        assert np.abs(Dt.tocsr(2)).sum() == 0
        assert np.abs(Dt.tocsr(3)).sum() == 0
        assert np.abs(D.tocsr(-1) - Dt.tocsr(-1)).sum() == 0
        Dt = D.transform(spin="so", orthogonal=True)
        assert np.abs(D.tocsr(0) - Dt.tocsr(0)).sum() == 0
        assert np.abs(D.tocsr(1) - Dt.tocsr(1)).sum() == 0
        assert np.abs(Dt.tocsr(-1)).sum() == 0
