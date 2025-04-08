# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import itertools

import numpy as np
import pytest

import sisl as si
import sisl.geom as sisl_geom
from sisl import (
    Atom,
    Cube,
    Geometry,
    Lattice,
    SislDeprecation,
    SislError,
    SislWarning,
    Sphere,
)

pytestmark = [pytest.mark.geom, pytest.mark.geometry]


@pytest.fixture
def setup():
    class t:
        def __init__(self):
            bond = 1.42
            sq3h = 3.0**0.5 * 0.5
            self.lattice = Lattice(
                np.array(
                    [[1.5, sq3h, 0.0], [1.5, -sq3h, 0.0], [0.0, 0.0, 10.0]], np.float64
                )
                * bond,
                nsc=[3, 3, 1],
            )
            C = Atom(Z=6, R=[bond * 1.01] * 2)
            self.g = Geometry(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
                atoms=C,
                lattice=self.lattice,
            )

            self.mol = Geometry([[i, 0, 0] for i in range(10)], lattice=[50])

    return t()


class TestGeometry:
    def test_objects(self, setup):
        str(setup.g)
        assert len(setup.g) == 2
        assert len(setup.g.xyz) == 2
        assert np.allclose(setup.g[0], np.zeros([3]))
        assert np.allclose(setup.g[None, 0], setup.g.xyz[:, 0])

        i = 0
        for ia in setup.g:
            i += 1
        assert i == len(setup.g)
        assert setup.g.no_s == 2 * len(setup.g) * np.prod(setup.g.lattice.nsc)

    def test_properties(self, setup):
        assert 2 == len(setup.g)
        assert 2 == setup.g.na
        assert 3 * 3 == setup.g.n_s
        assert 2 * 3 * 3 == setup.g.na_s
        assert 2 * 2 == setup.g.no
        assert 2 * 2 * 3 * 3 == setup.g.no_s

    def test_iter1(self, setup):
        i = 0
        for ia in setup.g:
            i += 1
        assert i == 2

    def test_iter2(self, setup):
        for ia in setup.g:
            assert np.allclose(setup.g[ia], setup.g.xyz[ia, :])

    def test_iter3(self, setup):
        i = 0
        for ia, io in setup.g.iter_orbitals(0):
            assert ia == 0
            assert io < 2
            i += 1
        for ia, io in setup.g.iter_orbitals(1):
            assert ia == 1
            assert io < 2
            i += 1
        assert i == 4
        i = 0
        for ia, io in setup.g.iter_orbitals():
            assert ia in [0, 1]
            assert io < 2
            i += 1
        assert i == 4

        i = 0
        for ia, io in setup.g.iter_orbitals(1, local=False):
            assert ia == 1
            assert io >= 2
            i += 1
        assert i == 2

    def test_tile0(self, setup):
        with pytest.raises(ValueError):
            t = setup.g.tile(0, 0)

    def test_tile1(self, setup):
        cell = np.copy(setup.g.lattice.cell)
        cell[0, :] *= 2
        t = setup.g.tile(2, 0)
        assert np.allclose(cell, t.lattice.cell)
        cell[1, :] *= 2
        t = t.tile(2, 1)
        assert np.allclose(cell, t.lattice.cell)
        cell[2, :] *= 2
        t = t.tile(2, 2)
        assert np.allclose(cell, t.lattice.cell)

    def test_tile2(self, setup):
        cell = np.copy(setup.g.lattice.cell)
        cell[:, :] *= 2
        t = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        assert np.allclose(cell, t.lattice.cell)

    def test_sort(self, setup):
        t = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        ts = t.sort()
        t = setup.g.tile(2, 1).tile(2, 2).tile(2, 0)
        tS = t.sort()
        assert np.allclose(ts.xyz, tS.xyz)

    def test_tile3(self, setup):
        cell = np.copy(setup.g.lattice.cell)
        cell[:, :] *= 2
        t1 = setup.g * 2
        cell = np.copy(setup.g.lattice.cell)
        cell[0, :] *= 2
        t1 = setup.g * (2, 0)
        assert np.allclose(cell, t1.lattice.cell)
        t = setup.g * ((2, 0), "tile")
        assert np.allclose(cell, t.lattice.cell)
        assert np.allclose(t1.xyz, t.xyz)
        cell[1, :] *= 2
        t1 = t * (2, 1)
        assert np.allclose(cell, t1.lattice.cell)
        t = t * ((2, 1), "tile")
        assert np.allclose(cell, t.lattice.cell)
        assert np.allclose(t1.xyz, t.xyz)
        cell[2, :] *= 2
        t1 = t * (2, 2)
        assert np.allclose(cell, t1.lattice.cell)
        t = t * ((2, 2), "tile")
        assert np.allclose(cell, t.lattice.cell)
        assert np.allclose(t1.xyz, t.xyz)

        # Full
        t = setup.g * [2, 2, 2]
        assert np.allclose(cell, t.lattice.cell)
        assert np.allclose(t1.xyz, t.xyz)
        t = setup.g * ([2, 2, 2], "t")
        assert np.allclose(cell, t.lattice.cell)
        assert np.allclose(t1.xyz, t.xyz)

    def test_tile4(self, setup):
        t1 = setup.g.tile(2, 0).tile(2, 2)
        t = setup.g * ([2, 0], "t") * [2, 2]
        assert np.allclose(t1.xyz, t.xyz)

    def test_tile5(self, setup):
        t = setup.g.tile(2, 0).tile(2, 2)
        assert np.allclose(t[: len(setup.g), :], setup.g.xyz)

    def test_repeat0(self, setup):
        with pytest.raises(ValueError):
            t = setup.g.repeat(0, 0)

    def test_repeat1(self, setup):
        cell = np.copy(setup.g.lattice.cell)
        cell[0, :] *= 2
        t = setup.g.repeat(2, 0)
        assert np.allclose(cell, t.lattice.cell)
        cell[1, :] *= 2
        t = t.repeat(2, 1)
        assert np.allclose(cell, t.lattice.cell)
        cell[2, :] *= 2
        t = t.repeat(2, 2)
        assert np.allclose(cell, t.lattice.cell)

    def test_repeat2(self, setup):
        cell = np.copy(setup.g.lattice.cell)
        cell[:, :] *= 2
        t = setup.g.repeat(2, 0).repeat(2, 1).repeat(2, 2)
        assert np.allclose(cell, t.lattice.cell)

    def test_repeat3(self, setup):
        cell = np.copy(setup.g.lattice.cell)
        cell[0, :] *= 2
        t1 = setup.g.repeat(2, 0)
        assert np.allclose(cell, t1.lattice.cell)
        t = setup.g * ((2, 0), "repeat")
        assert np.allclose(cell, t.lattice.cell)
        assert np.allclose(t1.xyz, t.xyz)
        cell[1, :] *= 2
        t1 = t.repeat(2, 1)
        assert np.allclose(cell, t1.lattice.cell)
        t = t * ((2, 1), "r")
        assert np.allclose(cell, t.lattice.cell)
        assert np.allclose(t1.xyz, t.xyz)
        cell[2, :] *= 2
        t1 = t.repeat(2, 2)
        assert np.allclose(cell, t1.lattice.cell)
        t = t * ((2, 2), "repeat")
        assert np.allclose(cell, t.lattice.cell)
        assert np.allclose(t1.xyz, t.xyz)

        # Full
        t = setup.g * ([2, 2, 2], "r")
        assert np.allclose(cell, t.lattice.cell)
        assert np.allclose(t1.xyz, t.xyz)

    def test_repeat4(self, setup):
        t1 = setup.g.repeat(2, 0).repeat(2, 2)
        t = setup.g * ([2, 0], "repeat") * ([2, 2], "r")
        assert np.allclose(t1.xyz, t.xyz)

    def test_repeat5(self, setup):
        t = setup.g.repeat(2, 0).repeat(2, 2)
        assert np.allclose(t.xyz[::4, :], setup.g.xyz)

    def test_a2o1(self, setup):
        assert 0 == setup.g.a2o(0)
        assert setup.g.atoms[0].no == setup.g.a2o(1)
        assert setup.g.no == setup.g.a2o(setup.g.na)

    def test_sub1(self, setup):
        assert len(setup.g.sub([0])) == 1
        assert len(setup.g.sub([0, 1])) == 2
        assert len(setup.g.sub([-1])) == 1

        assert np.allclose(
            setup.g.sub([0]).xyz, setup.g.sub(np.array([True, False])).xyz
        )

    def test_sub2(self, setup):
        assert len(setup.g.sub(range(1))) == 1
        assert len(setup.g.sub(range(2))) == 2

    def test_fxyz(self, setup):
        fxyz = setup.g.fxyz
        assert np.allclose(fxyz, [[0, 0, 0], [1.0 / 3, 1.0 / 3, 0]])
        assert np.allclose(np.dot(fxyz, setup.g.cell), setup.g.xyz)

    def test_axyz(self, setup):
        g = setup.g
        assert np.allclose(g[:], g.xyz[:])
        assert np.allclose(g[0], g.xyz[0, :])
        assert np.allclose(g[2], g.axyz(2))
        isc = g.a2isc(2)
        off = g.lattice.offset(isc)
        assert np.allclose(g.xyz[0] + off, g.axyz(2))
        assert np.allclose(g.xyz[0] + off, g.axyz(0, isc))
        assert np.allclose(g.xyz[0] + off, g.axyz(isc=isc)[0])

        # Also check multidimensional things
        assert g.axyz([[0], [1]]).shape == (2, 1, 3)

    def test_atranspose_indices(self, setup):
        g = setup.g
        # All supercell indices
        ia2 = np.arange(g.na * g.n_s)
        ja2, ja1 = g.a2transpose(0, ia2)
        assert (ja2 < g.na).sum() == ja2.size
        assert (ja1 % g.na == 0).sum() == ja1.size

        IA1, IA2 = g.a2transpose(ja2, ja1)
        assert np.all(IA1 == 0)
        assert np.all(IA2 == ia2)

    def test_otranspose_indices(self, setup):
        g = setup.g
        # All supercell indices
        io2 = np.arange(g.no * g.n_s)
        jo2, jo1 = g.o2transpose(0, io2)
        assert (jo2 < g.no).sum() == jo2.size
        assert (jo1 % g.no == 0).sum() == jo1.size

        IO1, IO2 = g.o2transpose(jo2, jo1)
        assert np.all(IO1 == 0)
        assert np.all(IO2 == io2)

    def test_auc2sc(self, setup):
        g = setup.g
        # All supercell indices
        asc = g.auc2sc(0)
        assert asc.size == g.n_s
        assert (asc % g.na == 0).sum() == g.n_s

    def test_ouc2sc(self, setup):
        g = setup.g
        # All supercell indices
        asc = g.ouc2sc(0)
        assert asc.size == g.n_s
        assert (asc % g.no == 0).sum() == g.n_s

    def test_rij1(self, setup):
        assert np.allclose(setup.g.rij(0, 1), 1.42)
        assert np.allclose(setup.g.rij(0, [0, 1]), [0.0, 1.42])

    def test_orij1(self, setup):
        assert np.allclose(setup.g.orij(0, 2), 1.42)
        assert np.allclose(setup.g.orij(0, [0, 2]), [0.0, 1.42])

    def test_Rij1(self, setup):
        assert np.allclose(setup.g.Rij(0, 1), [1.42, 0, 0])

    def test_oRij1(self, setup):
        assert np.allclose(setup.g.oRij(0, 1), [0.0, 0, 0])
        assert np.allclose(setup.g.oRij(0, 2), [1.42, 0, 0])
        assert np.allclose(
            setup.g.oRij(0, [0, 1, 2]), [[0.0, 0, 0], [0.0, 0, 0], [1.42, 0, 0]]
        )
        assert np.allclose(setup.g.oRij(0, 2), [1.42, 0, 0])

    def test_untile_warns(self, setup):
        with pytest.warns(SislWarning) as warns:
            assert len(setup.g.untile(1, 1)) == 2
            assert len(setup.g.untile(2, 1)) == 1
            assert len(setup.g.untile(2, 1, 1)) == 1
        assert len(warns) == 2

    def test_untile_check_same(self, setup):
        with pytest.warns(SislWarning) as warns:
            c1 = setup.g.untile(2, 1)
            c2 = setup.g.untile(2, 1, 1)
        assert len(warns) == 2
        assert np.allclose(c1.xyz[0, :], setup.g.xyz[0, :])
        assert np.allclose(c2.xyz[0, :], setup.g.xyz[1, :])

    def test_untile_algo(self, setup):
        nr = range(2, 5)
        g = setup.g.copy()
        for x in nr:
            gx = g.tile(x, 0)
            for y in nr:
                gy = gx.tile(y, 1)
                for z in nr:
                    gz = gy.tile(z, 2)
                    G = gz.untile(z, 2)
                    assert np.allclose(G.xyz, gy.xyz)
                    assert np.allclose(G.cell, gy.cell)
                G = gy.untile(y, 1)
                assert np.allclose(G.xyz, gx.xyz)
                assert np.allclose(G.cell, gx.cell)
            G = gx.untile(x, 0)
            assert np.allclose(G.xyz, g.xyz)
            assert np.allclose(G.cell, g.cell)

    def test_unrepeat_algo(self, setup):
        nr = range(2, 5)
        g = setup.g.copy()
        for x in nr:
            gx = g.repeat(x, 0)
            for y in nr:
                gy = gx.repeat(y, 1)
                for z in nr:
                    gz = gy.repeat(z, 2)
                    G = gz.unrepeat(z, 2)
                    assert np.allclose(G.xyz, gy.xyz)
                    assert np.allclose(G.cell, gy.cell)
                G = gy.unrepeat(y, 1)
                assert np.allclose(G.xyz, gx.xyz)
                assert np.allclose(G.cell, gx.cell)
            G = gx.unrepeat(x, 0)
            assert np.allclose(G.xyz, g.xyz)
            assert np.allclose(G.cell, g.cell)

    def test_remove1(self, setup):
        assert len(setup.g.remove([0])) == 1
        assert len(setup.g.remove([])) == 2
        assert len(setup.g.remove([-1])) == 1
        assert len(setup.g.remove([-0])) == 1

    def test_remove2(self, setup):
        assert len(setup.g.remove(range(1))) == 1
        assert len(setup.g.remove(range(0))) == 2

    def test_copy(self, setup):
        assert setup.g == setup.g.copy()

    def test_nsc1(self, setup):
        lattice = setup.g.lattice.copy()
        nsc = np.copy(lattice.nsc)
        lattice.set_nsc([5, 5, 0])
        assert np.allclose([5, 5, 1], lattice.nsc)
        assert len(lattice.sc_off) == np.prod(lattice.nsc)

    def test_nsc2(self, setup):
        lattice = setup.g.lattice.copy()
        nsc = np.copy(lattice.nsc)
        lattice.set_nsc([0, 1, 0])
        assert np.allclose([1, 1, 1], lattice.nsc)
        assert len(lattice.sc_off) == np.prod(lattice.nsc)

    def test_rotation1(self, setup):
        rot = setup.g.rotate(180, [0, 0, 1], what="xyz+abc")
        rot.lattice.cell[2, 2] *= -1
        assert np.allclose(-rot.lattice.cell, setup.g.lattice.cell)
        assert np.allclose(-rot.xyz, setup.g.xyz)

        rot = setup.g.rotate(np.pi, [0, 0, 1], rad=True, what="xyz+abc")
        rot.lattice.cell[2, 2] *= -1
        assert np.allclose(-rot.lattice.cell, setup.g.lattice.cell)
        assert np.allclose(-rot.xyz, setup.g.xyz)

        rot = rot.rotate(180, "z", what="xyz+abc")
        rot.lattice.cell[2, 2] *= -1
        assert np.allclose(rot.lattice.cell, setup.g.lattice.cell)
        assert np.allclose(rot.xyz, setup.g.xyz)

    def test_rotation2(self, setup):
        rot = setup.g.rotate(180, "z", what="abc")
        rot.lattice.cell[2, 2] *= -1
        assert np.allclose(-rot.lattice.cell, setup.g.lattice.cell)
        assert np.allclose(rot.xyz, setup.g.xyz)

        rot = setup.g.rotate(np.pi, [0, 0, 1], rad=True, what="abc")
        rot.lattice.cell[2, 2] *= -1
        assert np.allclose(-rot.lattice.cell, setup.g.lattice.cell)
        assert np.allclose(rot.xyz, setup.g.xyz)

        rot = rot.rotate(180, [0, 0, 1], what="abc")
        rot.lattice.cell[2, 2] *= -1
        assert np.allclose(rot.lattice.cell, setup.g.lattice.cell)
        assert np.allclose(rot.xyz, setup.g.xyz)

    def test_rotation3(self, setup):
        rot = setup.g.rotate(180, [0, 0, 1], what="xyz")
        assert np.allclose(rot.lattice.cell, setup.g.lattice.cell)
        assert np.allclose(-rot.xyz, setup.g.xyz)

        rot = setup.g.rotate(np.pi, [0, 0, 1], rad=True, what="xyz")
        assert np.allclose(rot.lattice.cell, setup.g.lattice.cell)
        assert np.allclose(-rot.xyz, setup.g.xyz)

        rot = rot.rotate(180, "z", what="xyz")
        assert np.allclose(rot.lattice.cell, setup.g.lattice.cell)
        assert np.allclose(rot.xyz, setup.g.xyz)

    def test_rotation4(self, setup):
        ref = setup.g.tile(2, 0).tile(2, 1)

        rot = ref.rotate(10, "z", atoms=1)
        assert not np.allclose(ref.xyz[1], rot.xyz[1])

        rot = ref.rotate(10, "z", atoms=[1, 2])
        assert not np.allclose(ref.xyz[1], rot.xyz[1])
        assert not np.allclose(ref.xyz[2], rot.xyz[2])
        assert np.allclose(ref.xyz[3], rot.xyz[3])

        rot = ref.rotate(10, "z", atoms=[1, 2], what="y")
        assert ref.xyz[1, 0] == rot.xyz[1, 0]
        assert ref.xyz[1, 1] != rot.xyz[1, 1]
        assert ref.xyz[1, 2] == rot.xyz[1, 2]

        rot = ref.rotate(10, "z", atoms=[1, 2], what="xy", origin=ref.xyz[2])
        assert ref.xyz[1, 0] != rot.xyz[1, 0]
        assert ref.xyz[1, 1] != rot.xyz[1, 1]
        assert ref.xyz[1, 2] == rot.xyz[1, 2]
        assert np.allclose(ref.xyz[2], rot.xyz[2])

    def test_translate(self, setup):
        t = setup.g.translate([0, 0, 1])
        assert np.allclose(setup.g.xyz[:, 0], t.xyz[:, 0])
        assert np.allclose(setup.g.xyz[:, 1], t.xyz[:, 1])
        assert np.allclose(setup.g.xyz[:, 2] + 1, t.xyz[:, 2])
        t = setup.g.move([0, 0, 1])
        assert np.allclose(setup.g.xyz[:, 0], t.xyz[:, 0])
        assert np.allclose(setup.g.xyz[:, 1], t.xyz[:, 1])
        assert np.allclose(setup.g.xyz[:, 2] + 1, t.xyz[:, 2])

    def test_iter_block1(self, setup):
        for i, iaaspec in enumerate(setup.g.iter_species()):
            ia, a, spec = iaaspec
            assert i == ia
            assert setup.g.atoms[ia] == a
        for ia, a, spec in setup.g.iter_species([1]):
            assert 1 == ia
            assert setup.g.atoms[ia] == a
        for ia in setup.g:
            assert ia >= 0
        i = 0
        for ias, idx in setup.g.iter_block():
            for ia in ias:
                i += 1
        assert i == len(setup.g)

        i = 0
        for ias, idx in setup.g.iter_block(atoms=1):
            for ia in ias:
                i += 1
        assert i == 1

    @pytest.mark.slow
    def test_iter_block2(self, setup):
        g = setup.g.tile(30, 0).tile(30, 1)
        i = 0
        for ias, _ in g.iter_block():
            i += len(ias)
        assert i == len(g)

    def test_iter_shape1(self, setup):
        i = 0
        for ias, _ in setup.g.iter_block(method="sphere"):
            i += len(ias)
        assert i == len(setup.g)
        i = 0
        for ias, _ in setup.g.iter_block(method="cube"):
            i += len(ias)
        assert i == len(setup.g)

    @pytest.mark.slow
    def test_iter_shape2(self, setup):
        g = setup.g.tile(30, 0).tile(30, 1)
        i = 0
        for ias, _ in g.iter_block(method="sphere"):
            i += len(ias)
        assert i == len(g)
        i = 0
        for ias, _ in g.iter_block(method="cube"):
            i += len(ias)
        assert i == len(g)
        i = 0
        for ias, _ in g.iter_block_shape(Cube(g.maxR() * 20)):
            i += len(ias)
        assert i == len(g)

    @pytest.mark.slow
    def test_iter_shape3(self, setup):
        g = setup.g.tile(50, 0).tile(50, 1)
        i = 0
        for ias, _ in g.iter_block(method="sphere"):
            i += len(ias)
        assert i == len(g)
        i = 0
        for ias, _ in g.iter_block(method="cube"):
            i += len(ias)
        assert i == len(g)
        i = 0
        for ias, _ in g.iter_block_shape(Sphere(g.maxR() * 20)):
            i += len(ias)
        assert i == len(g)

    def test_swap(self, setup):
        s = setup.g.swap(0, 1)
        for i in [0, 1, 2]:
            assert np.allclose(setup.g.xyz[::-1, i], s.xyz[:, i])

    def test_append1(self, setup):
        for axis in [0, 1, 2]:
            s = setup.g.append(setup.g, axis)
            assert len(s) == len(setup.g) * 2
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :] * 2)
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :] * 2)
            s = setup.g.prepend(setup.g, axis)
            assert len(s) == len(setup.g) * 2
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :] * 2)
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :] * 2)
            s = setup.g.append(setup.g.lattice, axis)
            assert len(s) == len(setup.g)
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :] * 2)
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :] * 2)
            s = setup.g.prepend(setup.g.lattice, axis)
            assert len(s) == len(setup.g)
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :] * 2)
            assert np.allclose(s.cell[axis, :], setup.g.cell[axis, :] * 2)

    def test_append_raise_valueerror(self, setup):
        with pytest.raises(ValueError):
            s = setup.g.append(setup.g, 0, offset="not")

    def test_prepend_raise_valueerror(self, setup):
        with pytest.raises(ValueError):
            s = setup.g.prepend(setup.g, 0, offset="not")

    def test_append_prepend_offset(self, setup):
        for axis in [0, 1, 2]:
            t = setup.g.lattice.cell[axis, :].copy()
            t *= 10.0 / (t**2).sum() ** 0.5
            s1 = setup.g.copy()
            s2 = setup.g.translate(t)

            S = s1.append(s2, axis, offset="min")
            s = setup.g.append(setup.g, axis)

            assert np.allclose(s.cell[axis, :], S.cell[axis, :])
            assert np.allclose(s.xyz, S.xyz)

            P = s2.prepend(s1, axis, offset="min")
            p = setup.g.prepend(setup.g, axis)

            assert np.allclose(p.cell[axis, :], P.cell[axis, :])
            assert np.allclose(p.xyz, P.xyz)

    @pytest.mark.parametrize("a,b", [[0, 1], [0, 2], [1, 2]])
    def test_swapaxes_lattice_vectors(self, setup, a, b):
        s = setup.g.swapaxes(a, b)
        assert np.allclose(setup.g.xyz, s.xyz)

    @pytest.mark.parametrize("a,b", [[0, 1], [0, 2], [1, 2]])
    def test_swapaxes_lattice_xyz(self, setup, a, b):
        g = setup.g
        s = g.swapaxes(a, b, what="xyz")
        idx = [0, 1, 2]
        idx[a], idx[b] = idx[b], idx[a]
        assert np.allclose(g.cell[:, idx], s.cell)
        assert np.allclose(g.xyz[:, idx], s.xyz)

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
        sab = setup.g.swapaxes(a, b)
        idx_abc = [1, 2, 0]
        idx_xyz = [2, 0, 1]
        assert np.allclose(setup.g.xyz[:, idx_xyz], sab.xyz)
        assert np.allclose(setup.g.cell[idx_abc][:, idx_xyz], sab.cell)

    def test_center(self, setup):
        g = setup.g.copy()
        assert np.allclose(g[1], g.center(atoms=[1]))
        assert np.allclose(np.mean(g.xyz, axis=0), g.center())
        # in this case the pbc COM is equivalent to the simple one
        assert np.allclose(g.center(what="mass"), g.center(what="mass:pbc"))
        assert np.allclose(g.center(what="mm:xyz"), g.center(what="mm(xyz)"))

    def test_center_raise(self, setup):
        with pytest.raises(ValueError):
            al = setup.g.center(what="unknown")

    def test___add1__(self, setup):
        n = len(setup.g)
        double = setup.g + setup.g + setup.g.lattice
        assert len(double) == n * 2
        assert np.allclose(setup.g.cell * 2, double.cell)
        assert np.allclose(setup.g.xyz[:n, :], double.xyz[:n, :])

        double = (setup.g, 1) + setup.g
        d = setup.g.prepend(setup.g, 1)
        assert len(double) == n * 2
        assert np.allclose(setup.g.cell[::2, :], double.cell[::2, :])
        assert np.allclose(double.xyz, d.xyz)

        double = setup.g + (setup.g, 1)
        d = setup.g.append(setup.g, 1)
        assert len(double) == n * 2
        assert np.allclose(setup.g.cell[::2, :], double.cell[::2, :])
        assert np.allclose(double.xyz, d.xyz)

    def test___add2__(self, setup):
        g1 = setup.g.rotate(15, setup.g.cell[2, :])
        g2 = setup.g.rotate(30, setup.g.cell[2, :])

        assert g1 != g2
        assert g1 + g2 == g1.add(g2)
        assert g1 + g2 != g2.add(g1)
        assert g2 + g1 == g2.add(g1)
        assert g2 + g1 != g1.add(g2)
        for i in range(3):
            assert g1 + (g2, i) == g1.append(g2, i)
            assert (g1, i) + g2 == g2.append(g1, i)

    def test___mul__(self, setup):
        g = setup.g.copy()
        assert g * 2 == g.tile(2, 0).tile(2, 1).tile(2, 2)
        assert g * [2, 1] == g.tile(2, 1)
        assert [2, 1] * g == g.repeat(2, 1)
        assert g * (2, 2, 2) == g.tile(2, 0).tile(2, 1).tile(2, 2)
        assert g * [1, 2, 2] == g.tile(1, 0).tile(2, 1).tile(2, 2)
        assert g * [1, 3, 2] == g.tile(1, 0).tile(3, 1).tile(2, 2)
        assert g * ([1, 3, 2], "r") == g.repeat(1, 0).repeat(3, 1).repeat(2, 2)
        assert g * ([1, 3, 2], "repeat") == g.repeat(1, 0).repeat(3, 1).repeat(2, 2)
        assert g * ([1, 3, 2], "tile") == g.tile(1, 0).tile(3, 1).tile(2, 2)
        assert g * ([1, 3, 2], "t") == g.tile(1, 0).tile(3, 1).tile(2, 2)
        assert g * ([3, 2], "t") == g.tile(3, 2)
        assert g * ([3, 2], "r") == g.repeat(3, 2)

    def test_add(self, setup):
        double = setup.g.add(setup.g)
        assert len(double) == len(setup.g) * 2
        assert np.allclose(setup.g.cell, double.cell)
        double = setup.g.add(setup.g).add(setup.g.lattice)
        assert len(double) == len(setup.g) * 2
        assert np.allclose(setup.g.cell * 2, double.cell)

    def test_add_vacuum(self, setup):
        g = setup.g
        added = setup.g.add_vacuum(g.cell[2, 2], 2)
        assert len(added) == len(g)
        assert np.allclose(
            added.lattice.length, g.lattice.length + [0, 0, g.cell[2, 2]]
        )
        assert np.allclose(added.xyz, g.xyz)
        added = g.add_vacuum(g.cell[2, 2], 2, offset=(1, 2, 3))
        assert len(added) == len(g)
        assert np.allclose(
            added.lattice.length, g.lattice.length + [0, 0, g.cell[2, 2]]
        )
        assert np.allclose(added.xyz, g.xyz + [1, 2, 3])

    def test_insert(self, setup):
        double = setup.g.insert(0, setup.g)
        assert len(double) == len(setup.g) * 2
        assert np.allclose(setup.g.cell, double.cell)

    def test_a2o(self, setup):
        # There are 2 orbitals per C atom
        assert setup.g.a2o(1) == setup.g.atoms[0].no
        assert np.all(setup.g.a2o(1, True) == [2, 3])
        setup.g.reorder()

    def test_o2a(self, setup):
        g = setup.g
        # There are 2 orbitals per C atom
        assert g.o2a(2) == 1
        assert g.o2a(3) == 1
        assert g.o2a(4) == 2
        assert np.all(g.o2a([0, 2, 4]) == [0, 1, 2])
        assert np.all(g.o2a([[0], [2], [4]]) == [[0], [1], [2]])
        assert np.all(g.o2a([[0], [2], [4]], unique=True) == [0, 1, 2])

    def test_angle(self, setup):
        # There are 2 orbitals per C atom
        g = Geometry([[0] * 3, [1, 0, 0]])
        cell = g.cell.copy()
        g.angle([0])
        g.angle([0], ref=1)
        g.angle([0], dir=1)
        assert np.allclose(g.cell[1], cell[1])

    def test_dihedral(self):
        g = sisl_geom.graphene() * (2, 2, 1)
        g.xyz[-1, 2] = 1
        assert g.dihedral(range(4)) == 180
        assert g.dihedral(range(4), rad=True) == np.pi
        assert g.dihedral(range(4, 8)) == pytest.approx(-140.88304639377796)
        assert np.allclose(
            g.dihedral([range(4), range(1, 5), range(4, 8)]),
            [180, 0, -140.88304639377796],
        )

    def test_2uc(self, setup):
        # functions for any-thing to UC
        g = setup.g
        assert g.asc2uc(2) == 0
        assert np.all(g.asc2uc([2, 3]) == [0, 1])
        assert g.asc2uc(2) == 0
        assert np.all(g.asc2uc([2, 3]) == [0, 1])
        assert g.osc2uc(4) == 0
        assert g.osc2uc(5) == 1
        assert np.all(g.osc2uc([4, 5]) == [0, 1])

    def test_2uc_many_axes(self, setup):
        # 2 orbitals per atom
        g = setup.g
        # functions for any-thing to SC
        idx = [[1], [2]]
        assert np.all(g.asc2uc(idx) == [[1], [0]])
        idx = [[2], [4]]
        assert np.all(g.osc2uc(idx) == [[2], [0]])

    def test_2sc(self, setup):
        # functions for any-thing to SC
        g = setup.g
        c = g.cell

        # check indices
        assert np.all(g.a2isc([1, 2]) == [[0, 0, 0], [-1, -1, 0]])
        assert np.all(g.a2isc(2) == [-1, -1, 0])
        assert np.allclose(g.a2sc(2), -c[0, :] - c[1, :])
        assert np.all(g.o2isc([1, 5]) == [[0, 0, 0], [-1, -1, 0]])
        assert np.all(g.o2isc(5) == [-1, -1, 0])
        assert np.allclose(g.o2sc(5), -c[0, :] - c[1, :])

        # Check off-sets
        assert np.allclose(setup.g.a2sc([1, 2]), [[0.0, 0.0, 0.0], -c[0, :] - c[1, :]])
        assert np.allclose(setup.g.o2sc([1, 5]), [[0.0, 0.0, 0.0], -c[0, :] - c[1, :]])

    def test_2sc_many_axes(self, setup):
        # 2 orbitals per atom
        g = setup.g
        # functions for any-thing to SC
        idx = [[1], [2]]
        assert np.all(g.a2isc(idx) == [[[0, 0, 0]], [[-1, -1, 0]]])
        assert g.auc2sc(idx).shape == (2, g.n_s)
        idx = [[2], [4]]
        assert np.all(g.o2isc(idx) == [[[0, 0, 0]], [[-1, -1, 0]]])
        assert g.ouc2sc(idx).shape == (2, g.n_s)

    def test_reverse(self, setup):
        rev = setup.g.reverse()
        assert len(rev) == 2
        assert np.allclose(rev.xyz[::-1, :], setup.g.xyz)
        rev = setup.g.reverse(atoms=list(range(len(setup.g))))
        assert len(rev) == 2
        assert np.allclose(rev.xyz[::-1, :], setup.g.xyz)

    def test_scale1(self, setup):
        two = setup.g.scale(2)
        assert len(two) == len(setup.g)
        assert np.allclose(two.xyz[:, :] / 2.0, setup.g.xyz)

    def test_scale_vector_abc(self, setup):
        two = setup.g.scale([2, 1, 1], what="abc")
        assert len(two) == len(setup.g)
        # Check that cell has been scaled accordingly
        assert np.allclose(two.cell[0] / 2.0, setup.g.cell[0])
        assert np.allclose(two.cell[1:], setup.g.cell[1:])
        # Now check that fractional coordinates are still the same
        assert np.allclose(two.fxyz, setup.g.fxyz)

    def test_scale_vector_xyz(self, setup):
        two = setup.g.scale([2, 1, 1], what="xyz")
        assert len(two) == len(setup.g)
        # Check that cell has been scaled accordingly
        assert np.allclose(two.cell[:, 0] / 2.0, setup.g.cell[:, 0])
        assert np.allclose(two.cell[:, 1:], setup.g.cell[:, 1:])
        # Now check that fractional coordinates are still the same
        assert np.allclose(two.fxyz, setup.g.fxyz)

    def test_close1(self, setup):
        three = range(3)
        for ia in setup.mol:
            i = setup.mol.close(ia, R=(0.1, 1.1), atoms=three)
            if ia < 3:
                assert len(i[0]) == 1
            else:
                assert len(i[0]) == 0
            # Will only return results from [0,1,2]
            # but the fourth atom connects to
            # the third
            if ia in [0, 2, 3]:
                assert len(i[1]) == 1
            elif ia == 1:
                assert len(i[1]) == 2
            else:
                assert len(i[1]) == 0

    def test_close2(self, setup):
        mol = range(3, 5)
        for ia in setup.mol:
            i = setup.mol.close(ia, R=(0.1, 1.1), atoms=mol)
            assert len(i) == 2
        i = setup.mol.close([100, 100, 100], R=0.1)
        assert len(i) == 0
        i = setup.mol.close([100, 100, 100], R=0.1, ret_rij=True)
        for el in i:
            assert len(el) == 0
        i = setup.mol.close([100, 100, 100], R=0.1, ret_rij=True, ret_xyz=True)
        for el in i:
            assert len(el) == 0

    def test_close_arguments_2R(self, setup):
        g = setup.mol.copy()
        for args in ("isc", "xyz", "rij"):
            kwargs = {f"ret_{args}": True}
            ia, other = g.close(0, R=(0.1, 1.1), **kwargs)
            assert len(ia) == 2
            assert len(other) == 2
        ia, a, b, c = g.close(0, R=(0.1, 1.1), ret_isc=True, ret_rij=True, ret_xyz=True)
        assert len(ia) == 2
        assert len(a) == 2
        assert len(b) == 2
        assert len(c) == 2

    def test_close_arguments_1R(self, setup):
        g = setup.mol.copy()
        for args in ("isc", "xyz", "rij"):
            kwargs = {f"ret_{args}": True}
            ia, other = g.close(0, R=1.1, **kwargs)
            assert len(ia) == len(other)
        ia, a, b, c = g.close(0, R=1.1, ret_isc=True, ret_rij=True, ret_xyz=True)
        assert len(ia) == len(a)
        assert len(a) == len(b)
        assert len(b) == len(c)

    @pytest.mark.slow
    def test_close4(self, setup):
        # 2 * 200 ** 2
        g = setup.g * (200, 200, 1)
        i = g.close(0, R=(0.1, 1.43))
        assert len(i) == 2
        assert len(i[0]) == 1
        assert len(i[1]) == 3

    def test_close_within1(self, setup):
        three = range(3)
        for ia in setup.mol:
            shapes = [Sphere(0.1, setup.mol[ia]), Sphere(1.1, setup.mol[ia])]
            i = setup.mol.close(ia, R=(0.1, 1.1), atoms=three)
            ii = setup.mol.within(shapes, atoms=three)
            assert np.all(i[0] == ii[0])
            assert np.all(i[1] == ii[1])

    def test_close_within2(self, setup):
        g = setup.g.repeat(6, 0).repeat(6, 1)
        for ia in g:
            shapes = [Sphere(0.1, g[ia]), Sphere(1.5, g[ia])]
            i = g.close(ia, R=(0.1, 1.5))
            ii = g.within(shapes)
            assert np.all(i[0] == ii[0])
            assert np.all(i[1] == ii[1])

    def test_close_within3(self, setup):
        g = setup.g.repeat(6, 0).repeat(6, 1)
        args = {"ret_xyz": True, "ret_rij": True, "ret_isc": True}
        for ia in g:
            shapes = [Sphere(0.1, g[ia]), Sphere(1.5, g[ia])]
            i, xa, d, isc = g.close(ia, R=(0.1, 1.5), **args)
            ii, xai, di, isci = g.within(shapes, **args)
            for j in [0, 1]:
                assert np.all(i[j] == ii[j])
                assert np.allclose(xa[j], xai[j])
                assert np.allclose(d[j], di[j])
                assert np.allclose(isc[j], isci[j])

    def test_within_inf_small_translated(self, setup):
        g = setup.g.translate([0.05] * 3)
        lattice_3x3 = g.lattice.tile(3, 0).tile(3, 1)
        assert len(g.within_inf(lattice_3x3)[0]) == len(g) * 3**2

    def test_within_inf_nonperiodic(self, setup):
        g = setup.g.copy()

        # Even if the geometry has nsc > 1, if we set periodic=False
        # we should get only the atoms in the unit cell.
        g.set_nsc([3, 3, 1])

        ia, xyz, isc = g.within_inf(g.lattice, periodic=[False, False, False])

        assert len(ia) == g.na
        assert np.all(isc == 0)

        # Make sure that it also works with mixed periodic/non periodic directions
        ia, xyz, isc = g.within_inf(g.lattice, periodic=[True, False, False])

        assert len(ia) > g.na
        assert np.any(isc[:, 0] != 0)
        assert np.all(isc[:, 1] == 0)

    def test_within_inf_molecule(self, setup):
        g = setup.mol.translate([0.05] * 3)
        lattice = Lattice(1.5)
        for o in range(10):
            origin = [o - 0.5, -0.5, -0.5]
            lattice.origin = origin
            idx = g.within_inf(lattice)[0]
            assert len(idx) == 1
            assert idx[0] == o

    def test_within_inf_duplicates(self, setup):
        g = setup.g.copy()
        g.lattice.pbc = [True, True, False]
        lattice_3x3 = g.lattice.tile(3, 0).tile(3, 1)
        assert len(g.within_inf(lattice_3x3)[0]) == 25

    def test_within_inf_gh649(self):
        # see https://github.com/zerothi/sisl/issues/649

        # Create a geometry with an atom outside of the unit cell
        geometry = Geometry([-0.5, 0, 0], lattice=np.diag([2, 10, 10]))

        search = Lattice(np.diag([3, 10, 10]))
        ia, xyz, isc = geometry.within_inf(search, periodic=True)
        assert np.allclose(ia, 0)
        assert np.allclose(isc, [1, 0, 0])

        search = Lattice(np.diag([2, 10, 10]))
        ia, xyz, isc = geometry.within_inf(search, periodic=True)
        assert np.allclose(ia, 0)
        assert np.allclose(isc, [1, 0, 0])

    def test_close_sizes(self, setup):
        point = 0

        # Return index
        idx = setup.mol.close(point, R=0.1)
        assert len(idx) == 1
        # Return index of two things
        idx = setup.mol.close(point, R=(0.1, 1.1))
        assert len(idx) == 2
        assert len(idx[0]) == 1
        assert not isinstance(idx[0], list)
        # Longer
        idx = setup.mol.close(point, R=(0.1, 1.1, 2.1))
        assert len(idx) == 3
        assert len(idx[0]) == 1

        # Return index
        idx = setup.mol.close(point, R=0.1, ret_xyz=True)
        assert len(idx) == 2
        assert len(idx[0]) == 1
        assert len(idx[1]) == 1
        assert idx[1].shape[0] == 1  # equivalent to above
        assert idx[1].shape[1] == 3

        # Return index of two things
        idx = setup.mol.close(point, R=(0.1, 1.1), ret_xyz=True)
        # [[idx-1, idx-2], [coord-1, coord-2]]
        assert len(idx) == 2
        assert len(idx[0]) == 2
        assert len(idx[1]) == 2
        # idx-1
        assert len(idx[0][0].shape) == 1
        assert idx[0][0].shape[0] == 1
        # idx-2
        assert idx[0][1].shape[0] == 1
        # coord-1
        assert len(idx[1][0].shape) == 2
        assert idx[1][0].shape[1] == 3
        # coord-2
        assert idx[1][1].shape[1] == 3

        # Return index of two things
        idx = setup.mol.close(
            point, R=(0.1, 1.1), ret_xyz=True, ret_rij=True, ret_isc=True
        )
        # [[idx-1, idx-2], [coord-1, coord-2], [dist-1, dist-2], [isc-1, isc-2]]
        assert len(idx) == 4
        assert len(idx[0]) == 2
        assert len(idx[1]) == 2
        assert len(idx[2]) == 2
        assert len(idx[3]) == 2
        assert all(len(idx[0][0]) == len(t[0]) for t in idx)
        assert all(len(idx[0][1]) == len(t[1]) for t in idx)
        # idx-1
        assert len(idx[0][0].shape) == 1
        assert idx[0][0].shape[0] == 1
        # idx-2
        assert idx[0][1].shape[0] == 1
        # coord-1
        assert len(idx[1][0].shape) == 2
        assert idx[1][0].shape[1] == 3
        # coord-2
        assert idx[1][1].shape[1] == 3
        # dist-1
        assert len(idx[2][0].shape) == 1
        assert idx[2][0].shape[0] == 1
        # dist-2
        assert idx[2][1].shape[0] == 1
        # isc-1
        assert len(idx[3][0].shape) == 2
        assert idx[3][0].shape[1] == 3
        # isc-2
        assert idx[3][1].shape[1] == 3

        # Return index of two things
        idx = setup.mol.close(point, R=(0.1, 1.1), ret_rij=True)
        # [[idx-1, idx-2], [dist-1, dist-2]]
        assert len(idx) == 2
        assert len(idx[0]) == 2
        assert len(idx[1]) == 2
        # idx-1
        assert len(idx[0][0].shape) == 1
        assert idx[0][0].shape[0] == 1
        # idx-2
        assert idx[0][1].shape[0] == 1
        # dist-1
        assert len(idx[1][0].shape) == 1
        assert idx[1][0].shape[0] == 1
        # dist-2
        assert idx[1][1].shape[0] == 1

    def test_close_sizes_none(self, setup):
        point = [100.0, 100.0, 100.0]

        # Return index
        idx = setup.mol.close(point, R=0.1)
        assert len(idx) == 0
        # Return index of two things
        idx = setup.mol.close(point, R=(0.1, 1.1))
        assert len(idx) == 2
        assert len(idx[0]) == 0
        assert not isinstance(idx[0], list)
        # Longer
        idx = setup.mol.close(point, R=(0.1, 1.1, 2.1))
        assert len(idx) == 3
        assert len(idx[0]) == 0

        # Return index
        idx = setup.mol.close(point, R=0.1, ret_xyz=True)
        assert len(idx) == 2
        assert len(idx[0]) == 0
        assert len(idx[1]) == 0
        assert idx[1].shape[0] == 0  # equivalent to above
        assert idx[1].shape[1] == 3

        # Return index of two things
        idx = setup.mol.close(point, R=(0.1, 1.1), ret_xyz=True)
        # [[idx-1, idx-2], [coord-1, coord-2]]
        assert len(idx) == 2
        assert len(idx[0]) == 2
        assert len(idx[1]) == 2
        # idx-1
        assert len(idx[0][0].shape) == 1
        assert idx[0][0].shape[0] == 0
        # idx-2
        assert idx[0][1].shape[0] == 0
        # coord-1
        assert len(idx[1][0].shape) == 2
        assert idx[1][0].shape[1] == 3
        # coord-2
        assert idx[1][1].shape[1] == 3

        # Return index of two things
        idx = setup.mol.close(point, R=(0.1, 1.1), ret_xyz=True, ret_rij=True)
        # [[idx-1, idx-2], [coord-1, coord-2], [dist-1, dist-2]]
        assert len(idx) == 3
        assert len(idx[0]) == 2
        assert len(idx[1]) == 2
        # idx-1
        assert len(idx[0][0].shape) == 1
        assert idx[0][0].shape[0] == 0
        # idx-2
        assert idx[0][1].shape[0] == 0
        # coord-1
        assert len(idx[1][0].shape) == 2
        assert idx[1][0].shape[0] == 0
        assert idx[1][0].shape[1] == 3
        # coord-2
        assert idx[1][1].shape[0] == 0
        assert idx[1][1].shape[1] == 3
        # dist-1
        assert len(idx[2][0].shape) == 1
        assert idx[2][0].shape[0] == 0
        # dist-2
        assert idx[2][1].shape[0] == 0

        # Return index of two things
        idx = setup.mol.close(point, R=(0.1, 1.1), ret_rij=True)
        # [[idx-1, idx-2], [dist-1, dist-2]]
        assert len(idx) == 2
        assert len(idx[0]) == 2
        assert len(idx[1]) == 2
        # idx-1
        assert len(idx[0][0].shape) == 1
        assert idx[0][0].shape[0] == 0
        # idx-2
        assert idx[0][1].shape[0] == 0
        # dist-1
        assert len(idx[1][0].shape) == 1
        assert idx[1][0].shape[0] == 0
        # dist-2
        assert idx[1][1].shape[0] == 0

    def test_sparserij1(self, setup):
        rij = setup.g.sparserij()

    def test_bond_correct(self, setup):
        # Create ribbon
        rib = setup.g.tile(2, 1)
        # Convert the last atom to a H atom
        rib.atoms[-1] = Atom[1]
        ia = len(rib) - 1
        # Get bond-length
        idx, d = rib.close(ia, R=(0.1, 1000), ret_rij=True)
        i = np.argmin(d[1])
        d = d[1][i]
        rib.bond_correct(ia, idx[1][i])
        idx, d2 = rib.close(ia, R=(0.1, 1000), ret_rij=True)
        i = np.argmin(d2[1])
        d2 = d2[1][i]
        assert d != d2
        # Calculate actual radius
        assert d2 == (Atom[1].radius() + Atom[6].radius())

    def test_unit_cell_estimation1(self, setup):
        # Create new geometry with only the coordinates
        # and atoms
        geom = Geometry(setup.g.xyz, Atom[6])
        # Only check the two distances we know have sizes
        for i in range(2):
            # It cannot guess skewed axis
            assert not np.allclose(geom.cell[i, :], setup.g.cell[i, :])

    def test_unit_cell_estimation2(self, setup):
        # Create new geometry with only the coordinates
        # and atoms
        s1 = Lattice([2, 2, 2])
        g1 = Geometry([[0, 0, 0], [1, 1, 1]], lattice=s1)
        g2 = Geometry(np.copy(g1.xyz))
        assert np.allclose(g1.cell, g2.cell)

        # Assert that it correctly calculates the bond-length in the
        # directions of actual distance
        g1 = Geometry([[0, 0, 0], [1, 1, 0]], atoms="H", lattice=s1)
        g2 = Geometry(np.copy(g1.xyz))
        for i in range(2):
            assert np.allclose(g1.cell[i, :], g2.cell[i, :])
        assert not np.allclose(g1.cell[2, :], g2.cell[2, :])

    def test_distance1(self, setup):
        geom = Geometry(setup.g.xyz, Atom[6])
        # maxR is undefined
        with pytest.raises(ValueError):
            d = geom.distance()

    def test_distance2(self, setup):
        geom = Geometry(setup.g.xyz, Atom[6])
        with pytest.raises(ValueError):
            d = geom.distance(R=1.42, method="unknown_numpy_function")

    def test_distance3(self, setup):
        geom = setup.g.copy()
        d = geom.distance()
        assert len(d) == 1
        assert np.allclose(d, [1.42])

    def test_distance4(self, setup):
        geom = setup.g.copy()
        d = geom.distance(method=np.min)
        assert len(d) == 1
        assert np.allclose(d, [1.42])
        d = geom.distance(method=np.max)
        assert len(d) == 1
        assert np.allclose(d, [1.42])
        d = geom.distance(method="max")
        assert len(d) == 1
        assert np.allclose(d, [1.42])

    def test_distance5(self, setup):
        geom = setup.g.copy()
        d = geom.distance(R=np.inf)
        assert len(d) == 6
        d = geom.distance(0, R=1.42)
        assert len(d) == 1
        assert np.allclose(d, [1.42])

    def test_distance6(self, setup):
        # Create a 1D chain
        geom = Geometry([0] * 3, Atom(1, R=1.0), lattice=1)
        geom.set_nsc([77, 1, 1])
        d = geom.distance(0)
        assert len(d) == 1
        assert np.allclose(d, [1.0])

        # Do twice
        d = geom.distance(R=2)
        assert len(d) == 2
        assert np.allclose(d, [1.0, 2.0])

        # Do all
        d = geom.distance(R=np.inf)
        assert len(d) == 77 // 2
        # Add one due arange not adding the last item
        assert np.allclose(d, range(1, 78 // 2))

        # Create a 2D grid
        geom.set_nsc([3, 3, 1])
        d = geom.distance(R=2, atol=[0.4, 0.3, 0.2, 0.1])
        assert len(d) == 2  # 1, sqrt(2)
        # Add one due arange not adding the last item
        assert np.allclose(d, [1, 2**0.5])

        # Create a 2D grid
        geom.set_nsc([5, 5, 1])
        d = geom.distance(R=2, atol=[0.4, 0.3, 0.2, 0.1])
        assert len(d) == 3  # 1, sqrt(2), 2
        # Add one due arange not adding the last item
        assert np.allclose(d, [1, 2**0.5, 2])

    def test_distance7(self, setup):
        # Create a 1D chain
        geom = Geometry([0] * 3, Atom(1, R=1.0), lattice=1)
        geom.set_nsc([77, 1, 1])
        # Try with a short R and a long tolerance list
        # We know that the tolerance list prevails, because
        d = geom.distance(R=1, atol=np.ones(10) * 0.5)
        assert len(d) == 1
        assert np.allclose(d, [1.0])

    def test_distance8(self, setup):
        geom = Geometry([0] * 3, Atom(1, R=1.0), lattice=1)
        geom.set_nsc([77, 1, 1])
        d = geom.distance(0, method="min")
        assert len(d) == 1
        d = geom.distance(0, method="median")
        assert len(d) == 1
        d = geom.distance(0, method="mode")
        assert len(d) == 1

    def test_find_nsc1(self, setup):
        # Create a 1D chain
        geom = Geometry([0] * 3, Atom(1, R=1.0), lattice=1)
        geom.set_nsc([77, 77, 77])

        assert np.allclose(geom.find_nsc(), [3, 3, 3])
        assert np.allclose(geom.find_nsc(1), [77, 3, 77])
        assert np.allclose(geom.find_nsc([0, 2]), [3, 77, 3])
        assert np.allclose(geom.find_nsc([0, 2], R=2.00000001), [5, 77, 5])

        geom.set_nsc([1, 1, 1])
        assert np.allclose(geom.find_nsc([0, 2], R=2.00000001), [5, 1, 5])

        geom.set_nsc([5, 1, 5])
        assert np.allclose(geom.find_nsc([0, 2], R=0.9999), [1, 1, 1])

    def test_find_nsc2(self, setup):
        # 2 ** 0.5 ensures lattice vectors with length 1
        geom = sisl_geom.fcc(2**0.5, Atom(1, R=1.0001))
        geom.set_nsc([77, 77, 77])

        assert np.allclose(geom.find_nsc(), [3, 3, 3])
        assert np.allclose(geom.find_nsc(1), [77, 3, 77])
        assert np.allclose(geom.find_nsc([0, 2]), [3, 77, 3])
        assert np.allclose(geom.find_nsc([0, 2], R=2.00000001), [5, 77, 5])

        geom.set_nsc([1, 1, 1])
        assert np.allclose(geom.find_nsc([0, 2], R=2.00000001), [5, 1, 5])

        geom.set_nsc([5, 1, 5])
        assert np.allclose(geom.find_nsc([0, 2], R=0.9999), [1, 1, 1])

    def test_argumentparser1(self, setup):
        setup.g.ArgumentParser()
        setup.g.ArgumentParser(**setup.g._ArgumentParser_args_single())

    def test_argumentparser2(self, setup, **kwargs):
        p, ns = setup.g.ArgumentParser(**kwargs)

        # Try all options
        opts = [
            "--origin",
            "--center-of",
            "mass",
            "--center-of",
            "xyz",
            "--center-of",
            "position",
            "--center-of",
            "cell",
            "--unit-cell",
            "translate",
            "--unit-cell",
            "mod",
            "--rotate",
            "90",
            "x",
            "--rotate",
            "90",
            "y",
            "--rotate",
            "90",
            "z",
            "--add",
            "0,0,0",
            "6",
            "--swap",
            "0",
            "1",
            "--repeat",
            "2",
            "x",
            "--repeat",
            "2",
            "y",
            "--repeat",
            "2",
            "z",
            "--tile",
            "2",
            "x",
            "--tile",
            "2",
            "y",
            "--tile",
            "2",
            "z",
            "--untile",
            "2",
            "z",
            "--untile",
            "2",
            "y",
            "--untile",
            "2",
            "x",
        ]
        if kwargs.get("limit_arguments", True):
            opts.extend(
                ["--rotate", "-90", "x", "--rotate", "-90", "y", "--rotate", "-90", "z"]
            )
        else:
            opts.extend(
                [
                    "--rotate-x",
                    " -90",
                    "--rotate-y",
                    " -90",
                    "--rotate-z",
                    " -90",
                    "--repeat-x",
                    "2",
                    "--repeat-y",
                    "2",
                    "--repeat-z",
                    "2",
                ]
            )

        args = p.parse_args(opts, namespace=ns)

        if len(kwargs) == 0:
            self.test_argumentparser2(setup, **setup.g._ArgumentParser_args_single())

    def test_set_supercell(self, setup):
        # check for deprecation
        s1 = Lattice([2, 2, 2])
        g1 = Geometry([[0, 0, 0], [1, 1, 1]], lattice=[2, 2, 1])
        with pytest.warns(SislDeprecation) as deps:
            g1.set_supercell(s1)
        assert g1.lattice == s1
        assert len(deps) == 1

    def test_attach1(self, setup):
        g = setup.g.attach(0, setup.mol, 0, dist=1.42, axis=2)
        g = setup.g.attach(0, setup.mol, 0, dist="calc", axis=2)
        g = setup.g.attach(0, setup.mol, 0, dist=[0, 0, 1.42])

    def test_mirror_function(self, setup):
        g = setup.g
        for plane in ["xy", "xz", "yz", "ab", "bc", "ac"]:
            g.mirror(plane)

        assert g.mirror("xy") == g.mirror("z")
        assert g.mirror("xy") == g.mirror([0, 0, 1])

        assert g.mirror("xy", [0]) == g.mirror([0, 0, 1], [0])

    def test_mirror_point(self):
        g = Geometry([[0, 0, 0], [0, 0, 1]])
        out = g.mirror("z")
        assert np.allclose(out.xyz[:, 2], [0, -1])
        assert np.allclose(out.xyz[:, :2], 0)
        out = g.mirror("z", point=(0, 0, 0.5))
        assert np.allclose(out.xyz[:, 2], [1, 0])
        assert np.allclose(out.xyz[:, :2], 0)
        out = g.mirror("z", point=(0, 0, 1))
        assert np.allclose(out.xyz[:, 2], [2, 1])
        assert np.allclose(out.xyz[:, :2], 0)

    def test_pickle(self, setup):
        import pickle as p

        s = p.dumps(setup.g)
        n = p.loads(s)
        assert n == setup.g

    def test_geometry_names(self):
        g = sisl_geom.graphene()

        assert len(g.names) == 0
        g["A"] = 1
        assert len(g.names) == 1
        g[[1, 2]] = "B"
        assert len(g.names) == 2
        g.names.delete_name("B")
        assert len(g.names) == 1

        # Add new group
        g["B"] = [0, 2]

        for name in g.names:
            assert name in ["A", "B"]

        str(g)

        assert np.allclose(g["B"], g[[0, 2], :])
        assert np.allclose(g.axyz("B"), g[[0, 2], :])

        del g.names["B"]
        assert len(g.names) == 1

    def test_geometry_groups_raise(self):
        g = sisl_geom.graphene()
        g["A"] = 1
        with pytest.raises(SislError):
            g["A"] = [1, 2]

    def test_geometry_as_primary_raise_nondivisable(self):
        g = sisl_geom.graphene()
        with pytest.raises(ValueError):
            g.as_primary(3)

    def test_geometry_untile_raise_nondivisable(self):
        g = sisl_geom.graphene()
        with pytest.raises(ValueError):
            g.untile(3, 0)

    def test_geometry_iR_negative_R(self):
        g = sisl_geom.graphene()
        with pytest.raises(ValueError):
            g.iR(R=-1.0)

    @pytest.mark.parametrize(
        "geometry",
        [
            sisl_geom.graphene(),
            sisl_geom.diamond(),
            sisl_geom.sc(1.4, Atom[1]),
            sisl_geom.fcc(1.4, Atom[1]),
            sisl_geom.bcc(1.4, Atom[1]),
            sisl_geom.hcp(1.4, Atom[1]),
        ],
    )
    def test_geometry_as_primary(self, geometry):
        prod = itertools.product
        x_reps = [1, 4, 3]
        y_reps = [1, 4, 5]
        z_reps = [1, 4, 6]
        tile_rep = ["r", "t"]

        na_primary = len(geometry)
        for x, y, z in prod(x_reps, y_reps, z_reps):
            if x == y == z == 1:
                continue
            for a, b, c in prod(tile_rep, tile_rep, tile_rep):
                G = ((geometry * ([x, 1, 1], a)) * ([1, y, 1], b)) * ([1, 1, z], c)
                p = G.as_primary(na_primary)
                assert np.allclose(p.xyz, geometry.xyz)
                assert np.allclose(p.cell, geometry.cell)

    def test_geometry_as_primary_without_super(self):
        g = sisl_geom.graphene()
        p = g.as_primary(len(g))
        assert g == p

        G = g.tile(2, 0).tile(3, 1)
        p, supercell = G.as_primary(len(g), ret_super=True)
        assert np.allclose(p.xyz, g.xyz)
        assert np.allclose(p.cell, g.cell)
        assert np.all(supercell == [2, 3, 1])

    # Test ASE (but only fail if present)

    def test_geometry_dispatch(self):
        pytest.importorskip("ase", reason="ase not available")
        gr = sisl_geom.graphene()
        to_ase = gr.to.ase()

        ase_rotate = si.rotate(to_ase, 30, [0, 0, 1])
        assert isinstance(ase_rotate, type(to_ase))
        ase_sisl_rotate = si.rotate(to_ase, 30, [0, 0, 1], ret_sisl=True)
        assert isinstance(ase_sisl_rotate, Geometry)
        geom_rotate = si.rotate(gr, 30, [0, 0, 1])

        assert geom_rotate.equal(ase_sisl_rotate, R=False)

    def test_geometry_ase_new_to(self):
        pytest.importorskip("ase", reason="ase not available")
        gr = sisl_geom.graphene()
        to_ase = gr.to.ase()
        from_ase = gr.new(to_ase)
        assert gr.equal(from_ase, R=False)

    def test_geometry_ase_run_center(self):
        pytest.importorskip("ase", reason="ase not available")
        gr = sisl_geom.graphene()
        ase_atoms = gr.to.ase()
        from_ase = si.center(ase_atoms)
        assert np.allclose(gr.center(), from_ase)

    @pytest.mark.xfail(
        reason="pymatgen backconversion sets nsc=[3, 3, 3], we need to figure this out"
    )
    def test_geometry_pymatgen_to(self):
        pytest.importorskip("pymatgen", reason="pymatgen not available")
        gr = sisl_geom.graphene()
        to_pymatgen = gr.to.pymatgen()
        from_pymatgen = gr.new(to_pymatgen)
        assert gr.equal(from_pymatgen, R=False)
        # TODO add test for Molecule as well (cell will then be different)

    def test_geometry_pandas_to(self):
        pytest.importorskip("pandas", reason="pandas not available")
        gr = sisl_geom.graphene()
        df = gr.to.dataframe()
        assert np.allclose(df["x"], gr.xyz[:, 0])
        assert np.allclose(df["y"], gr.xyz[:, 1])
        assert np.allclose(df["z"], gr.xyz[:, 2])

    def test_geometry_overlapping_atoms(self):
        gr22 = sisl_geom.graphene().tile(2, 0).tile(2, 1)
        gr44 = gr22.tile(2, 0).tile(2, 1)
        offset = np.array([0.2, 0.4, 0.4])
        gr22 = gr22.translate(offset)
        idx = np.arange(gr22.na)
        np.random.shuffle(idx)
        gr22 = gr22.sub(idx)
        idx22, idx44 = gr22.overlap(gr44, offset=-offset)
        assert np.allclose(idx22, np.arange(gr22.na))
        assert np.allclose(idx44, idx)


def test_geometry_sort_simple():
    bi = sisl_geom.bilayer().tile(2, 0).repeat(3, 1)

    # the default tolerance is 1e-9, and since we are sorting
    # in each group, we may in the end find another ordering
    atol = 1e-9

    for i in [0, 1, 2]:
        s = bi.sort(axes=i)
        assert np.all(np.diff(s.xyz[:, i]) >= -atol)
        s = bi.sort(lattice=i)
        assert np.all(np.diff(s.fxyz[:, i] * bi.lattice.length[i]) >= -atol)

    s, idx = bi.sort(axes=0, lattice=1, ret_atoms=True)
    assert np.all(np.diff(s.xyz[:, 0]) >= -atol)
    for ix in idx:
        assert np.all(np.diff(bi.fxyz[ix, 1]) >= -atol)

    s, idx = bi.sort(
        axes=0, ascending=False, lattice=1, vector=[0, 0, 1], ret_atoms=True
    )
    assert np.all(np.diff(s.xyz[:, 0]) >= -atol)
    for ix in idx:
        # idx is according to bi
        assert np.all(np.diff(bi.fxyz[ix, 1] * bi.lattice.length[i]) <= atol)


def test_geometry_sort_int():
    bi = sisl_geom.bilayer().tile(2, 0).repeat(3, 1)

    # the default tolerance is 1e-9, and since we are sorting
    # in each group, we may in the end find another ordering
    atol = 1e-9

    for i in [0, 1, 2]:
        s = bi.sort(axes0=i)
        assert np.all(np.diff(s.xyz[:, i]) >= -atol)
        s = bi.sort(lattice3=i)
        assert np.all(np.diff(s.fxyz[:, i] * bi.lattice.length[i]) >= -atol)

    s, idx = bi.sort(axes12314=0, lattice0=1, ret_atoms=True)
    assert np.all(np.diff(s.xyz[:, 0]) >= -atol)
    for ix in idx:
        assert np.all(np.diff(bi.fxyz[ix, 1]) >= -atol)

    s, idx = bi.sort(
        ascending1=True, axes15=0, ascending0=False, lattice235=1, ret_atoms=True
    )
    assert np.all(np.diff(s.xyz[:, 0]) >= -atol)
    for ix in idx:
        # idx is according to bi
        assert np.all(np.diff(bi.fxyz[ix, 1] * bi.lattice.length[i]) <= atol)


def test_geometry_ellipsis():
    gr = sisl_geom.graphene()
    assert np.allclose(gr.axyz(...), gr.axyz(None))


def test_geometry_sort_atom():
    bi = sisl_geom.bilayer().tile(2, 0).repeat(2, 1)

    atom = [[2, 0], [3, 1]]
    out, atom = bi.sort(atoms=atom, ret_atoms=True)

    atom = np.concatenate(atom)
    all_atoms = np.arange(len(bi))
    all_atoms[np.sort(atom)] = atom[:]

    assert np.allclose(out.xyz, bi.sub(all_atoms).xyz)


def test_geometry_sort_func():
    bi = sisl_geom.bilayer().tile(2, 0).repeat(2, 1)

    def reverse(geometry, atoms, **kwargs):
        return atoms[::-1]

    atoms = [[2, 0], [3, 1]]
    out = bi.sort(func=reverse, atoms=atoms)

    all_atoms = np.arange(len(bi))
    all_atoms[1] = 2
    all_atoms[2] = 1

    assert np.allclose(out.xyz, bi.sub(all_atoms).xyz)

    bi_again = bi.sort(func=(reverse, reverse), atoms=atoms)

    # Ensure that they are swapped
    atoms = [2, 0]
    out = bi.sort(func=reverse, atoms=atoms)

    assert np.allclose(out.xyz, bi.xyz)

    out = bi.sort(func=reverse)
    all_atoms = np.arange(len(bi))[::-1]
    assert np.allclose(out.xyz, bi.sub(all_atoms).xyz)


def test_geometry_sort_func_sort():
    bi = sisl_geom.bilayer().tile(2, 0).repeat(2, 1)

    # Sort according to another cell fractional coordinates
    fcc = sisl_geom.fcc(2.4, Atom(6))

    def fcc_fracs(axis):
        def _(geometry):
            return np.dot(geometry.xyz, fcc.icell.T)[:, axis]

        return _

    out = bi.sort(func_sort=(fcc_fracs(0), fcc_fracs(2)))


def test_geometry_sort_group():
    bi = (
        sisl_geom.bilayer(bottom_atoms=Atom[6], top_atoms=(Atom[5], Atom[7]))
        .tile(2, 0)
        .repeat(2, 1)
    )

    out = bi.sort(group="Z")

    assert np.allclose(out.atoms.Z[:4], 5)
    assert np.allclose(out.atoms.Z[4:12], 6)
    assert np.allclose(out.atoms.Z[12:16], 7)

    out = bi.sort(group=("symbol", "C", None))

    assert np.allclose(out.atoms.Z[:8], 6)

    C = bi.sort(group=("symbol", "C", None))
    BN = bi.sort(group=("symbol", None, "C"))
    BN2 = bi.sort(group=("symbol", ["B", "N"], "C"))
    # For these simple geometries symbol and tag are the same
    BN3 = bi.sort(group=("tag", ["B", "N"], "C"))

    # none of these atoms should be the same
    assert not np.any(np.isclose(C.atoms.Z, BN.atoms.Z))
    # All these sorting algorithms are the same
    assert np.allclose(BN.atoms.Z, BN2.atoms.Z)
    assert np.allclose(BN.atoms.Z, BN3.atoms.Z)

    mass = bi.sort(group="mass")
    Z = bi.sort(group="Z")
    assert np.allclose(mass.atoms.Z, Z.atoms.Z)


def test_geometry_sort_fail_keyword():
    with pytest.raises(ValueError):
        sisl_geom.bilayer().sort(not_found_keyword=True)


@pytest.mark.category
@pytest.mark.geom_category
def test_geometry_sanitize_atom_category():
    bi = (
        sisl_geom.bilayer(bottom_atoms=Atom[6], top_atoms=(Atom[5], Atom[7]))
        .tile(2, 0)
        .repeat(2, 1)
    )
    C_idx = (bi.atoms.Z == 6).nonzero()[0]
    check_C = bi.axyz(C_idx)
    only_C = bi.axyz(Atom[6])
    assert np.allclose(only_C, check_C)
    only_C = bi.axyz(bi.atoms.Z == 6)
    assert np.allclose(only_C, check_C)
    # with dict redirect
    only_C = bi.axyz({"Z": 6})
    assert np.allclose(only_C, check_C)
    # Using a dict that has multiple keys. This basically checks
    # that it accepts generic categories such as the AndCategory
    bi2 = bi.copy()
    bi2.atoms["C"] = Atom("C", R=1.9)
    only_C = bi2.axyz({"Z": 6, "neighbors": 3})
    assert np.allclose(only_C, check_C)

    tup_01 = (0, 2)
    list_01 = [0, 2]
    ndarray_01 = np.array(list_01)
    assert np.allclose(bi._sanitize_atoms(tup_01), bi._sanitize_atoms(list_01))
    assert np.allclose(bi._sanitize_atoms(ndarray_01), bi._sanitize_atoms(list_01))


def test_geometry_sanitize_atom_shape():
    bi = (
        sisl_geom.bilayer(bottom_atoms=Atom[6], top_atoms=(Atom[5], Atom[7]))
        .tile(2, 0)
        .repeat(2, 1)
    )
    cube = Cube(10)
    assert len(bi.axyz(cube)) != 0


def test_geometry_sanitize_atom_0_length():
    gr = sisl_geom.graphene()
    assert len(gr.axyz([])) == 0


@pytest.mark.parametrize("atoms", [[True, False], (True, False), [0], (0,)])
def test_geometry_sanitize_atom_other_bool(atoms):
    gr = sisl_geom.graphene()
    assert len(gr.axyz(atoms)) == 1


def test_geometry_sanitize_atom_0_length_float_fail():
    gr = sisl_geom.graphene()
    with pytest.raises(IndexError):
        # it raises an error because np.float64 is used
        gr.axyz(np.array([1], dtype=np.float64))


def test_geometry_sanitize_atom_bool():
    gr = sisl_geom.graphene()
    assert gr.axyz(True).shape == (gr.na, 3)
    assert gr.axyz(False).shape == (0, 3)


def test_geometry_sanitize_orbs():
    bot = Atom(6, [1, 2, 3])
    top = [Atom(5, [1, 2]), Atom(7, [1, 2])]

    bi = sisl_geom.bilayer(bottom_atoms=bot, top_atoms=top).tile(2, 0).repeat(2, 1)

    C_idx = (bi.atoms.Z == 6).nonzero()[0]
    assert np.allclose(bi._sanitize_orbs({bot: [0]}), bi.firsto[C_idx])
    assert np.allclose(bi._sanitize_orbs({bot: 1}), bi.firsto[C_idx] + 1)
    assert np.allclose(
        bi._sanitize_orbs({bot: [1, 2]}), np.add.outer(bi.firsto[C_idx], [1, 2]).ravel()
    )


def test_geometry_sub_orbitals():
    atom = Atom(6, [1, 2, 3])
    gr = sisl_geom.graphene(atoms=atom)
    assert gr.no == 6

    # try and do sub
    gr2 = gr.sub_orbital(atom, [atom.orbitals[0], atom.orbitals[2]])
    assert gr2.atoms[0][0] == atom.orbitals[0]
    assert gr2.atoms[0][1] == atom.orbitals[2]

    gr2 = gr.sub_orbital(atom, atom.orbitals[1])
    assert gr2.atoms[0][0] == atom.orbitals[1]


def test_geometry_new_xyz(sisl_tmp):
    # test that Geometry.new works
    out = sisl_tmp("out.xyz")
    C = Atom[6]
    gr = sisl_geom.graphene(atoms=C)
    # writing doesn't save orbital information, so we force
    # an empty atom
    gr.write(out)

    gr2 = Geometry.new(out)
    assert np.allclose(gr.xyz, gr2.xyz)
    assert gr == gr2

    gr2 = gr.new(out)
    assert np.allclose(gr.xyz, gr2.xyz)
    assert gr == gr2


def test_translate2uc():
    gr = sisl_geom.graphene() * (2, 3, 1)
    gr = gr.move([5, 5, 5])
    gr2 = gr.translate2uc()
    assert not np.allclose(gr.xyz, gr2.xyz)


def test_translate2uc_axes():
    gr = sisl_geom.graphene() * (2, 3, 1)
    gr = gr.move([5, 5, 5])
    gr_once = gr.translate2uc()
    gr_individual = gr.translate2uc(axes=0)
    assert not np.allclose(gr_once.xyz, gr_individual.xyz)
    gr_individual = gr_individual.translate2uc(axes=1)
    assert np.allclose(gr_once.xyz, gr_individual.xyz)
    gr_individual = gr_individual.translate2uc(axes=2)
    assert np.allclose(gr_once.xyz, gr_individual.xyz)


def test_fortran_contiguous():
    # for #748
    xyz = np.zeros([10, 3], order="F")
    geom = Geometry(xyz)
    assert geom.xyz.flags.c_contiguous


def test_as_supercell_graphene():
    gr = sisl_geom.graphene()
    grsc = gr.as_supercell()
    assert np.allclose(grsc.xyz[: len(gr)], gr.xyz)
    assert np.allclose(grsc.axyz(np.arange(gr.na_s)), gr.axyz(np.arange(gr.na_s)))


def test_as_supercell_fcc():
    g = sisl_geom.fcc(2**0.5, Atom(1, R=1.0001))
    gsc = g.as_supercell()
    assert np.allclose(gsc.xyz[: len(g)], g.xyz)
    assert np.allclose(gsc.axyz(np.arange(g.na_s)), g.axyz(np.arange(g.na_s)))


def test_new_inherit_class():
    g = sisl_geom.fcc(2**0.5, Atom(1, R=1.0001))

    class MyClass(Geometry):
        pass

    g2 = MyClass.new(g)
    assert g2.__class__ == MyClass
    assert g2 == g
    g2 = Geometry.new(g)
    assert g2.__class__ == Geometry
    assert g2 == g


def test_sc_warn():
    with pytest.warns(SislDeprecation):
        lattice = sisl_geom.graphene().sc


def test_geometry_apply():
    from functools import partial

    g = sisl_geom.fcc(2**0.5, Atom(1, R=(1.0001, 2)))
    data = np.random.rand(3, g.no, 4)
    data_dup = g.apply(data, "sum", lambda x: x, segments="orbitals", axis=1)
    assert data_dup.shape == data.shape
    assert np.allclose(data, data_dup)

    data_atom = g.apply(data, "sum", partial(g.a2o, all=True), axis=1)
    assert data_atom.shape == (3, g.na, 4)
    assert np.allclose(data_atom[:, 0], data[:, :2].sum(1))
