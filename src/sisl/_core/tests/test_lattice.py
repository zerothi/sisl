# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import math as m

import numpy as np
import pytest
from pytest import approx

import sisl
import sisl.linalg as lin
from sisl import BoundaryCondition, Lattice, LatticeChild, SuperCell
from sisl.geom import graphene

pytestmark = [pytest.mark.supercell, pytest.mark.sc, pytest.mark.lattice]


@pytest.fixture
def setup():
    class t:
        def __init__(self):
            alat = 1.42
            sq3h = 3.0**0.5 * 0.5
            self.lattice = Lattice(
                np.array(
                    [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
                )
                * alat,
                nsc=[3, 3, 1],
            )

    return t()


class TestLattice:
    def test_str(self, setup):
        str(setup.lattice)
        str(setup.lattice)
        assert setup.lattice != "Not a Lattice"

    def test_nsc1(self, setup):
        lattice = setup.lattice.copy()
        lattice.set_nsc([5, 5, 0])
        assert np.allclose([5, 5, 1], lattice.nsc)
        assert len(lattice.sc_off) == np.prod(lattice.nsc)

    def test_nsc2(self, setup):
        lattice = setup.lattice.copy()
        lattice.set_nsc([0, 1, 0])
        assert np.allclose([1, 1, 1], lattice.nsc)
        assert len(lattice.sc_off) == np.prod(lattice.nsc)
        lattice.set_nsc(a=3)
        assert np.allclose([3, 1, 1], lattice.nsc)
        assert len(lattice.sc_off) == np.prod(lattice.nsc)
        lattice.set_nsc(b=3)
        assert np.allclose([3, 3, 1], lattice.nsc)
        assert len(lattice.sc_off) == np.prod(lattice.nsc)
        lattice.set_nsc(c=5)
        assert np.allclose([3, 3, 5], lattice.nsc)
        assert len(lattice.sc_off) == np.prod(lattice.nsc)

    def test_nsc3(self, setup):
        assert setup.lattice.sc_index([0, 0, 0]) == 0
        for s in range(setup.lattice.n_s):
            assert setup.lattice.sc_index(setup.lattice.sc_off[s, :]) == s
        arng = np.arange(setup.lattice.n_s)
        np.random.seed(42)
        np.random.shuffle(arng)
        sc_off = setup.lattice.sc_off[arng, :]
        assert np.all(setup.lattice.sc_index(sc_off) == arng)

    def test_nsc_fail_even(self, setup):
        with pytest.raises(ValueError):
            setup.lattice.set_nsc(a=2)

    def test_nsc_fail_even_and_odd(self, setup):
        with pytest.raises(ValueError):
            setup.lattice.set_nsc([1, 2, 3])

    def test_area1(self, setup):
        setup.lattice.area(0, 1)

    def test_fill(self, setup):
        sc = setup.lattice.swapaxes(1, 2)
        i = sc._fill([1, 1])
        assert i.dtype == np.int32
        i = sc._fill([1.0, 1.0])
        assert i.dtype == np.float64
        for dt in [np.int32, np.float32, np.float64, np.complex64]:
            i = sc._fill([1.0, 1.0], dt)
            assert i.dtype == dt
            i = sc._fill(np.ones([2], dt))
            assert i.dtype == dt

    def test_add_vacuum_direct(self, setup):
        sc = setup.lattice.copy()
        for i in range(3):
            s = sc.add_vacuum(10, i)
            ax = setup.lattice.cell[i, :]
            ax += ax / np.sum(ax**2) ** 0.5 * 10
            assert np.allclose(ax, s.cell[i, :])

    def test_add_vacuum_orthogonal(self, setup):
        sc = setup.lattice.copy()
        s = sc.add_vacuum(10, 2, orthogonal_to_plane=True)
        assert np.allclose(s.cell[2], [0, 0, sc.cell[2, 2] + 10])

        # now check for the skewed ones
        s = sc.add_vacuum(sc.length[0], 0, orthogonal_to_plane=True)
        assert (s.length[0] / sc.length[0] - 1) * m.cos(m.radians(30)) == pytest.approx(
            1.0
        )

        # now check for the skewed ones
        s = sc.add_vacuum(sc.length[1], 1, orthogonal_to_plane=True)
        assert (s.length[1] / sc.length[1] - 1) * m.cos(m.radians(30)) == pytest.approx(
            1.0
        )

    def test_add1(self, setup):
        sc = setup.lattice.copy()
        for R in range(1, 10):
            s = sc + R
            assert np.allclose(s.cell, sc.cell + np.diag([R] * 3))

    def test_rotation1(self, setup):
        rot = setup.lattice.rotate(180, [0, 0, 1])
        rot.cell[2, 2] *= -1
        assert np.allclose(-rot.cell, setup.lattice.cell)

        rot = setup.lattice.rotate(m.pi, [0, 0, 1], rad=True)
        rot.cell[2, 2] *= -1
        assert np.allclose(-rot.cell, setup.lattice.cell)

        rot = rot.rotate(180, [0, 0, 1])
        rot.cell[2, 2] *= -1
        assert np.allclose(rot.cell, setup.lattice.cell)

    def test_rotation2(self, setup):
        rot = setup.lattice.rotate(180, setup.lattice.cell[2, :])
        rot.cell[2, 2] *= -1
        assert np.allclose(-rot.cell, setup.lattice.cell)

        rot = setup.lattice.rotate(m.pi, setup.lattice.cell[2, :], rad=True)
        rot.cell[2, 2] *= -1
        assert np.allclose(-rot.cell, setup.lattice.cell)

        rot = rot.rotate(180, setup.lattice.cell[2, :])
        rot.cell[2, 2] *= -1
        assert np.allclose(rot.cell, setup.lattice.cell)

    @pytest.mark.parametrize("a,b", [[0, 1], [0, 2], [1, 2]])
    def test_swapaxes_lattice_vector(self, setup, a, b):
        lattice = setup.lattice
        sab = lattice.swapaxes(a, b)
        assert np.allclose(lattice.origin, sab.origin)
        assert np.allclose(lattice.nsc[[b, a]], sab.nsc[[a, b]])
        assert np.allclose(lattice.cell[[b, a]], sab.cell[[a, b]])

        # and string input
        sab = setup.lattice.swapaxes("abc"[a], "abc"[b])
        assert np.allclose(sab.cell[[b, a]], setup.lattice.cell[[a, b]])
        assert np.allclose(sab.origin, setup.lattice.origin)
        assert np.allclose(sab.nsc[[b, a]], setup.lattice.nsc[[a, b]])

    @pytest.mark.parametrize("a,b", [[0, 1], [0, 2], [1, 2]])
    def test_swapaxes_xyz(self, setup, a, b):
        lattice = setup.lattice
        sab = lattice.swapaxes(1, 2, "xyz")
        assert np.allclose(lattice.nsc, sab.nsc)
        assert np.allclose(lattice.origin[[b, a]], sab.origin[[a, b]])
        assert np.allclose(lattice.origin[[b, a]], sab.origin[[a, b]])

        # and string input
        sab = setup.lattice.swapaxes("xyz"[a], "xyz"[b])
        assert np.allclose(lattice.nsc, sab.nsc)
        assert np.allclose(lattice.origin[[b, a]], sab.origin[[a, b]])
        assert np.allclose(lattice.origin[[b, a]], sab.origin[[a, b]])

    def test_swapaxes_complicated(self, setup):
        # swap a couple of lattice vectors and cartesian coordinates
        a = "azby"
        b = "bxcz"
        # this will result in
        # 0. abc, xyz
        # 1. bac, xyz
        # 2. bac, zyx
        # 3. bca, zyx
        # 4. bca, zxy
        sab = setup.lattice.swapaxes(a, b)
        idx_abc = [1, 2, 0]
        idx_xyz = [2, 0, 1]
        assert np.allclose(sab.cell, setup.lattice.cell[idx_abc][:, idx_xyz])
        assert np.allclose(sab.origin, setup.lattice.origin[idx_xyz])
        assert np.allclose(sab.nsc, setup.lattice.nsc[idx_abc])

    def test_offset1(self, setup):
        off = setup.lattice.offset()
        assert np.allclose(off, [0, 0, 0])
        off = setup.lattice.offset([1, 1, 1])
        cell = setup.lattice.cell[:, :]
        assert np.allclose(off, cell[0, :] + cell[1, :] + cell[2, :])

    def test_sc_index1(self, setup):
        sc_index = setup.lattice.sc_index([0, 0, 0])
        assert sc_index == 0
        sc_index = setup.lattice.sc_index([0, 0, None])
        assert len(sc_index) == setup.lattice.nsc[2]

    def test_sc_index2(self, setup):
        sc_index = setup.lattice.sc_index([[0, 0, 0], [1, 1, 0]])
        s = str(sc_index)
        assert len(sc_index) == 2

    def test_sc_index3(self, setup):
        with pytest.raises(Exception):
            setup.lattice.sc_index([100, 100, 100])

    def test_vertices(self, setup):
        verts = setup.lattice.vertices()

        assert verts.shape == (2, 2, 2, 3)
        assert np.allclose(verts[0, 0, 0], [0, 0, 0])
        assert np.allclose(verts[1, 0, 0], setup.lattice.cell[0])
        assert np.allclose(verts[1, 1, 1], setup.lattice.cell.sum(axis=0))

    def test_untile1(self, setup):
        cut = setup.lattice.untile(2, 0)
        assert np.allclose(cut.cell[0, :] * 2, setup.lattice.cell[0, :])
        assert np.allclose(cut.cell[1, :], setup.lattice.cell[1, :])

    def test_creation1(self, setup):
        # full cell
        tmp1 = Lattice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # diagonal cell
        tmp2 = Lattice([1, 1, 1])
        # cell parameters
        tmp3 = Lattice([1, 1, 1, 90, 90, 90])
        tmp4 = Lattice([1])
        assert np.allclose(tmp1.cell, tmp2.cell)
        assert np.allclose(tmp1.cell, tmp3.cell)
        assert np.allclose(tmp1.cell, tmp4.cell)

    def test_creation_latticechild_dispatch(self, setup):
        # full cell
        class P(LatticeChild):
            def copy(self):
                a = P()
                a.set_lattice(setup.lattice)
                return a

        tmp1 = P()
        tmp1.set_lattice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # test dispatch method
        tmp2 = Lattice.new(tmp1)
        assert tmp1 == tmp2

    def test_creation_latticechild(self, setup):
        # full cell
        class P(LatticeChild):
            def copy(self):
                a = P()
                a.set_lattice(setup.lattice)
                return a

        tmp1 = P()
        tmp1.set_lattice([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # diagonal cell
        tmp2 = P()
        tmp2.set_lattice([1, 1, 1])
        # cell parameters
        tmp3 = P()
        tmp3.set_lattice([1, 1, 1, 90, 90, 90])
        tmp4 = P()
        tmp4.set_lattice([1])
        tmp5 = P()
        tmp5.set_lattice([[1, 1, 1], [90, 90, 90]])
        assert np.allclose(tmp1.cell, tmp2.cell)
        assert np.allclose(tmp1.cell, tmp3.cell)
        assert np.allclose(tmp1.cell, tmp4.cell)
        assert np.allclose(tmp1.cell, tmp5.cell)
        assert len(tmp1.lattice._fill([0, 0, 0])) == 3
        assert len(tmp1.lattice._fill_sc([0, 0, 0])) == 3
        assert tmp1.lattice.is_orthogonal()
        for i in range(3):
            t2 = tmp2.lattice.add_vacuum(10, i)
            assert tmp1.cell[i, i] + 10 == t2.cell[i, i]

    def test_creation3(self, setup):
        with pytest.raises(ValueError):
            setup.lattice.tocell([3, 6])

    def test_creation4(self, setup):
        with pytest.raises(ValueError):
            setup.lattice.tocell([3, 4, 5, 6])

    def test_creation5(self, setup):
        with pytest.raises(ValueError):
            setup.lattice.tocell([3, 4, 5, 6, 7, 6, 7])

    def test_creation_rotate(self, setup):
        # cell parameters
        param = np.array([1, 2, 3, 45, 60, 80], np.float64)
        parama = param.copy()
        parama[3:] *= np.pi / 180

        lattice = Lattice(param)

        param.shape = (2, 3)
        parama.shape = (2, 3)

        assert np.allclose(param, np.array(lattice.parameters()))
        assert np.allclose(parama, np.array(lattice.parameters(True)))
        for ang in range(0, 91, 5):
            s = (
                lattice.rotate(ang, lattice.cell[0, :])
                .rotate(ang, lattice.cell[1, :])
                .rotate(ang, lattice.cell[2, :])
            )
            assert np.allclose(param, np.array(s.parameters()))
            assert np.allclose(parama, np.array(s.parameters(True)))

    def test_rcell(self, setup):
        # LAPACK inverse algorithm implicitly does
        # a transpose.
        rcell = lin.inv(setup.lattice.cell) * 2.0 * np.pi
        assert np.allclose(rcell.T, setup.lattice.rcell)
        assert np.allclose(rcell.T / (2 * np.pi), setup.lattice.icell)

    def test_icell(self, setup):
        assert np.allclose(setup.lattice.rcell, setup.lattice.icell * 2 * np.pi)

    def test_center1(self, setup):
        assert np.allclose(
            setup.lattice.center(), np.sum(setup.lattice.cell, axis=0) / 2
        )
        for i in [0, 1, 2]:
            assert np.allclose(setup.lattice.center(i), setup.lattice.cell[i, :] / 2)

    def test_pickle(self, setup):
        import pickle as p

        s = p.dumps(setup.lattice)
        n = p.loads(s)
        assert setup.lattice == n
        s = Lattice([1, 1, 1])
        assert setup.lattice != s

    def test_orthogonal(self, setup):
        assert not setup.lattice.is_orthogonal()

    def test_fit1(self, setup):
        g = graphene()
        gbig = g.repeat(40, 0).repeat(40, 1)
        gbig.xyz[:, :] += (np.random.rand(len(gbig), 3) - 0.5) * 0.01
        lattice = g.lattice.fit(gbig)
        assert np.allclose(lattice.cell, gbig.cell)

    def test_fit2(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        gbig.xyz[:, :] += (np.random.rand(len(gbig), 3) - 0.5) * 0.01
        lattice = g.lattice.fit(gbig)
        assert np.allclose(lattice.cell, gbig.cell)

    def test_fit3(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        gbig.xyz[:, :] += (np.random.rand(len(gbig), 3) - 0.5) * 0.01
        lattice = g.lattice.fit(gbig, axes=0)
        assert np.allclose(lattice.cell[0, :], gbig.cell[0, :])
        assert np.allclose(lattice.cell[1:, :], g.cell[1:, :])

    def test_fit4(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        gbig.xyz[:, :] += (np.random.rand(len(gbig), 3) - 0.5) * 0.01
        lattice = g.lattice.fit(gbig, axes=[0, 1])
        assert np.allclose(lattice.cell[0:2, :], gbig.cell[0:2, :])
        assert np.allclose(lattice.cell[2, :], g.cell[2, :])

    def test_parallel1(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        assert g.lattice.parallel(gbig.lattice)
        assert gbig.lattice.parallel(g.lattice)
        g = g.rotate(90, g.cell[0, :], what="abc")
        assert not g.lattice.parallel(gbig.lattice)

    def test_tile_multiply_orthogonal(self):
        lattice = graphene(orthogonal=True).lattice
        assert np.allclose(
            lattice.tile(3, 0).tile(2, 1).tile(4, 2).cell, (lattice * (3, 2, 4)).cell
        )
        assert np.allclose(lattice.tile(3, 0).tile(2, 1).cell, (lattice * [3, 2]).cell)
        assert np.allclose(
            lattice.tile(3, 0).tile(3, 1).tile(3, 2).cell, (lattice * 3).cell
        )

    def test_tile_multiply_non_orthogonal(self):
        lattice = graphene(orthogonal=False).lattice
        assert np.allclose(
            lattice.tile(3, 0).tile(2, 1).tile(4, 2).cell, (lattice * (3, 2, 4)).cell
        )
        assert np.allclose(lattice.tile(3, 0).tile(2, 1).cell, (lattice * [3, 2]).cell)
        assert np.allclose(
            lattice.tile(3, 0).tile(3, 1).tile(3, 2).cell, (lattice * 3).cell
        )

    def test_angle1(self, setup):
        g = graphene(orthogonal=True)
        gbig = g.repeat(40, 0).repeat(40, 1)
        assert g.lattice.angle(0, 1) == 90

    def test_angle2(self, setup):
        lattice = Lattice([1, 1, 1])
        assert lattice.angle(0, 1) == 90
        assert lattice.angle(0, 2) == 90
        assert lattice.angle(1, 2) == 90
        lattice = Lattice([[1, 1, 0], [1, -1, 0], [0, 0, 2]])
        assert lattice.angle(0, 1) == 90
        assert lattice.angle(0, 2) == 90
        assert lattice.angle(1, 2) == 90
        lattice = Lattice([[3, 4, 0], [4, 3, 0], [0, 0, 2]])
        assert lattice.angle(0, 1, rad=True) == approx(0.28379, abs=1e-4)
        assert lattice.angle(0, 2) == 90
        assert lattice.angle(1, 2) == 90

    def test_cell2length(self):
        gr = graphene(orthogonal=True)
        lattice = (gr * (40, 40, 1)).rotate(24, gr.cell[2, :]).lattice
        assert np.allclose(
            lattice.length, (lattice.cell2length(lattice.length) ** 2).sum(1) ** 0.5
        )
        assert np.allclose(1, (lattice.cell2length(1) ** 2).sum(0))

    def test_set_lattice_off_wrong_size(self, setup):
        lattice = setup.lattice.copy()
        with pytest.raises(ValueError):
            lattice.sc_off = np.zeros([10000, 3])


def test_zero_length():
    with pytest.warns(sisl.SislWarning):
        Lattice([1, 0, 0])


def _dot(u, v):
    """Dot product u . v"""
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def test_plane1():
    lattice = Lattice([1] * 3)
    # Check point [0.5, 0.5, 0.5]
    pp = np.array([0.5] * 3)

    n, p = lattice.plane(0, 1, True)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = lattice.plane(0, 2, True)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = lattice.plane(1, 2, True)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = lattice.plane(0, 1, False)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = lattice.plane(0, 2, False)
    assert -0.5 == approx(_dot(n, pp - p))
    n, p = lattice.plane(1, 2, False)
    assert -0.5 == approx(_dot(n, pp - p))


def test_plane2():
    lattice = Lattice([1] * 3)
    # Check point [-0.5, -0.5, -0.5]
    pp = np.array([-0.5] * 3)

    n, p = lattice.plane(0, 1, True)
    assert 0.5 == approx(_dot(n, pp - p))
    n, p = lattice.plane(0, 2, True)
    assert 0.5 == approx(_dot(n, pp - p))
    n, p = lattice.plane(1, 2, True)
    assert 0.5 == approx(_dot(n, pp - p))
    n, p = lattice.plane(0, 1, False)
    assert -1.5 == approx(_dot(n, pp - p))
    n, p = lattice.plane(0, 2, False)
    assert -1.5 == approx(_dot(n, pp - p))
    n, p = lattice.plane(1, 2, False)
    assert -1.5 == approx(_dot(n, pp - p))


def test_tocuboid_simple():
    lattice = Lattice([1, 1, 1, 90, 90, 90])
    with pytest.warns(sisl.SislDeprecation):
        c1 = lattice.toCuboid()
    assert np.allclose(lattice.cell, c1._v)
    c1 = lattice.to.Cuboid()
    assert np.allclose(lattice.cell, c1._v)

    with pytest.warns(sisl.SislDeprecation):
        c2 = lattice.toCuboid(orthogonal=True)
    assert np.allclose(c1._v, c2._v)
    c2 = lattice.to.Cuboid(orthogonal=True)
    assert np.allclose(c1._v, c2._v)


def test_tocuboid_complex():
    lattice = Lattice([1, 1, 1, 60, 60, 60])
    s = str(lattice.cell)
    with pytest.warns(sisl.SislDeprecation):
        c1 = lattice.toCuboid()
    assert np.allclose(lattice.cell, c1._v)
    c1 = lattice.to.Cuboid()
    assert np.allclose(lattice.cell, c1._v)

    c1 = lattice.to.Cuboid(orthogonal=True)
    assert not np.allclose(np.diagonal(lattice.cell), np.diagonal(c1._v))
    with pytest.warns(sisl.SislDeprecation):
        c2 = lattice.toCuboid(orthogonal=True)
    assert np.allclose(np.diagonal(c1._v), np.diagonal(c2._v))


def test_fortran_contiguous():
    # for #748
    abc = np.zeros([3, 3], order="F")
    latt = Lattice(abc)
    assert latt.cell.flags.c_contiguous


def test_lattice_indices():
    lattice = Lattice([1] * 3, nsc=[3, 5, 7])
    for i in range(lattice.n_s):
        assert i == lattice.sc_index(lattice.sc_off[i])


def test_supercell_warn():
    with pytest.warns(sisl.SislDeprecation):
        lattice = SuperCell([1] * 3, nsc=[3, 5, 7])


#####
# boundary-condition tests
#####


@pytest.mark.parametrize("bc", list(BoundaryCondition))
def test_lattice_bc_init(bc):
    Lattice(1, boundary_condition=bc)
    Lattice(1, boundary_condition=[bc, bc, bc])
    Lattice(1, boundary_condition=[[bc, bc], [bc, bc], [bc, bc]])


def test_lattice_bc_set():
    lat = Lattice(1, boundary_condition=Lattice.BC.PERIODIC)
    assert lat.pbc.all()
    lat.boundary_condition = Lattice.BC.UNKNOWN
    assert not lat.pbc.any()
    assert (lat.boundary_condition == Lattice.BC.UNKNOWN).all()

    lat.boundary_condition = ["per", "unkn", [3, 4]]

    for n in "abc":
        lat.set_boundary_condition(**{n: "per"})
        lat.set_boundary_condition(**{n: [3, Lattice.BC.UNKNOWN]})
        lat.set_boundary_condition(**{n: [True, Lattice.BC.PERIODIC]})

    bc = ["per", ["Dirichlet", Lattice.BC.NEUMANN], ["un", "neu"]]
    lat.set_boundary_condition(bc)
    assert np.all(
        lat.boundary_condition[1] == [Lattice.BC.DIRICHLET, Lattice.BC.NEUMANN]
    )
    assert np.all(lat.boundary_condition[2] == [Lattice.BC.UNKNOWN, Lattice.BC.NEUMANN])
    lat.set_boundary_condition(["per", None, ["dirichlet", "unkno"]])
    assert np.all(
        lat.boundary_condition[1] == [Lattice.BC.DIRICHLET, Lattice.BC.NEUMANN]
    )
    assert np.all(
        lat.boundary_condition[2] == [Lattice.BC.DIRICHLET, Lattice.BC.UNKNOWN]
    )
    lat.set_boundary_condition("ppu")
    assert np.all(lat.pbc == [True, True, False])


def test_lattice_bc_fail():
    lat = Lattice(1)
    with pytest.raises(ValueError):
        lat.boundary_condition = False
    with pytest.raises(ValueError):
        lat.set_boundary_condition(b=False)
    with pytest.raises(KeyError):
        lat.set_boundary_condition(b="eusoatuhesoau")


def test_new_inherit_class():
    l = Lattice(1, boundary_condition=Lattice.BC.NEUMANN)

    class MyClass(Lattice):
        pass

    l2 = MyClass.new(l)
    assert l2.__class__ == MyClass
    assert l2 == l
    l2 = Lattice.new(l)
    assert l2.__class__ == Lattice
    assert l2 == l


def test_lattice_info():
    lat = Lattice(1, nsc=[3, 3, 3])
    with pytest.warns(sisl.SislWarning) as record:
        lat.set_boundary_condition(b=Lattice.BC.DIRICHLET)
        lat.set_boundary_condition(c=Lattice.BC.PERIODIC)
        assert len(record) == 1


def test_lattice_pbc_setter():
    lat = Lattice(1, nsc=[3, 3, 3])
    assert np.all(lat.pbc)
    pbc = [True, False, True]
    lat.pbc = pbc
    assert np.all(lat.pbc == pbc)
    # check that None won't change anything
    pbc = [False, None, None]
    lat.pbc = pbc
    assert np.all(lat.pbc == [False, False, True])
