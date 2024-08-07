# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
import scipy as sc

from sisl import Atom, Geometry, Lattice, SislWarning
from sisl._core.sparse_geometry import *
from sisl.geom import fcc, graphene

pytestmark = [pytest.mark.sparse, pytest.mark.sparse_geometry]


@pytest.fixture
def setup():
    class t:
        def __init__(self):
            self.g = fcc(1.0, Atom(1, R=1.495)) * 2
            self.s1 = SparseAtom(self.g)
            self.s2 = SparseAtom(self.g, 2)

    return t()


class TestSparseAtom:
    def test_fail_align1(self, setup):
        s = SparseAtom(setup.g * 2)
        str(s)
        with pytest.raises(ValueError):
            setup.s1.spalign(s)

    def test_align1(self, setup):
        s = SparseAtom(setup.g)
        setup.s1.spalign(s)
        setup.s1.spalign(s._csr)

    def test_create1(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        assert setup.s1.nnz == 3
        assert setup.s1[0, 1] == 1
        setup.s1[2, [1, 2, 3]] = 1
        assert setup.s1.nnz == 6
        setup.s1.empty(keep_nnz=True)
        assert setup.s1.nnz == 6
        setup.s1.empty()
        assert setup.s1.nnz == 0
        # nan produces zeros
        setup.s1[0, 0] = np.nan
        assert setup.s1.nnz == 1
        setup.s1.empty()

    def test_create2(self, setup):
        setup.s1[0, [1, 2, 3], (1, 1, 1)] = 1
        assert setup.s1[0, 2, (1, 1, 1)] == 1
        setup.s1.empty()

    def test_create3(self, setup):
        setup.s2[0, [1, 2, 3], (1, 1, 1)] = [1, 2]
        assert np.allclose(setup.s2[0, 2, (1, 1, 1)], [1, 2])
        assert setup.s2[0, 2, 0, (1, 1, 1)] == 1
        assert setup.s2[0, 2, 1, (1, 1, 1)] == 2
        setup.s2.empty()

    def test_in1(self, setup):
        setup.s1[0, [1, 2, 3]] = 1
        assert setup.s1[0, 2] == 1
        assert [0, 4] not in setup.s1
        assert [0, 2] in setup.s1
        setup.s1.empty()

    def test_reset1(self, setup):
        setup.s1[0, [1, 2, 3], (1, 1, 1)] = 1
        assert setup.s1[0, 2, (1, 1, 1)] == 1
        setup.s1.reset()
        assert setup.s1.nnz == 0
        setup.s1.empty()

    def test_eliminate_zeros1(self, setup):
        setup.s1[0, [1, 2, 3]] = 0
        assert setup.s1.nnz == 3
        setup.s1.eliminate_zeros()
        assert setup.s1.nnz == 0
        setup.s1.empty()

    def test_nonzero1(self, setup):
        s2 = setup.s2.copy()
        s2[0, [0, 2, 3]] = [1, 2]
        s2[3, [1, 2, 3]] = [1, 2]

        r, c = s2.nonzero()
        assert np.allclose(r, [0, 0, 0, 3, 3, 3])
        assert np.allclose(c, [0, 2, 3, 1, 2, 3])
        c = s2.nonzero(only_cols=True)
        assert np.allclose(c, [0, 2, 3, 1, 2, 3])
        r, c = s2.nonzero(atoms=1)
        assert len(r) == 0
        assert len(c) == 0
        r, c = s2.nonzero(atoms=0)
        assert np.allclose(r, [0, 0, 0])
        assert np.allclose(c, [0, 2, 3])
        c = s2.nonzero(atoms=0, only_cols=True)
        assert np.allclose(c, [0, 2, 3])

    def test_create_construct_different_length(self, setup):
        s1 = SparseAtom(setup.g)
        with pytest.raises(ValueError):
            s1.construct([[0.1, 1.5], [1]])

    def test_untile1(self, setup):
        s1 = SparseAtom(setup.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        s2 = SparseAtom(setup.g * 2)
        s2.construct([[0.1, 1.5], [1, 2]])
        s2 = s2.untile(2, 2).untile(2, 1).untile(2, 0)
        assert s1.spsame(s2)

        s1 = SparseAtom(setup.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        s2 = SparseAtom(setup.g * [2, 1, 1])
        s2.construct([[0.1, 1.5], [1, 2]])
        s2 = s2.untile(2, 0)
        assert s1.spsame(s2)

        s1 = SparseAtom(setup.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        s2 = SparseAtom(setup.g * [1, 2, 1])
        s2.construct([[0.1, 1.5], [1, 2]])
        s2 = s2.untile(2, 1)
        assert s1.spsame(s2)

    def test_untile_wrong_usage(self):
        # one should not untile
        geometry = Geometry([0] * 3, Atom(1, R=1.001), Lattice(1, nsc=[1] * 3))
        geometry = geometry.tile(4, 0)
        s = SparseAtom(geometry)
        s.construct([[0.1, 1.01], [1, 2]])
        s[0, 3] = 2
        s[3, 0] = 2

        # check that untiling twice is not the same as untiling 4 times and coupling it
        with pytest.warns(
            SislWarning, match=r"may have connections crossing the entire"
        ):
            s2 = s.untile(2, 0)
        s4 = s.untile(4, 0).tile(2, 0)
        ds = s2 - s4
        ds.finalize()
        assert np.absolute(ds)._csr._D.sum() > 0

    def test_untile_segment_single(self):
        # one should not untile
        geometry = Geometry([0] * 3, Atom(1, R=1.001), Lattice(1, nsc=[1] * 3))
        geometry = geometry.tile(4, 0)
        s = SparseAtom(geometry)
        s.construct([[0.1, 1.01], [1, 2]])
        s[0, 3] = 2
        s[3, 0] = 2

        # check that untiling twice is not the same as untiling 4 times and coupling it
        s4 = s.untile(4, 0).tile(2, 0)
        for seg in range(1, 4):
            sx = s.untile(4, 0, segment=seg).tile(2, 0)
            ds = s4 - sx
            ds.finalize()
            assert np.absolute(ds)._csr._D.sum() == pytest.approx(0.0)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_untile_segment_three(self, axis):
        # one should not untile
        nsc = [3] * 3
        nsc[axis] = 1
        geometry = Geometry([0] * 3, Atom(1, R=1.001), Lattice(1, nsc=nsc))
        geometry = geometry.tile(4, axis)
        s = SparseAtom(geometry)
        s.construct([[0.1, 1.01], [1, 2]])
        s[0, 3] = 2
        s[3, 0] = 2

        # check that untiling twice is not the same as untiling 4 times and coupling it
        s4 = s.untile(4, axis).tile(2, axis)
        for seg in range(1, 4):
            sx = s.untile(4, axis, segment=seg).tile(2, axis)
            ds = s4 - sx
            ds.finalize()
            assert np.absolute(ds)._csr._D.sum() == pytest.approx(0.0)

    def test_unrepeat_setup(self, setup):
        s1 = SparseAtom(setup.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        s2 = SparseAtom(setup.g * ((2, 2, 2), "r"))
        s2.construct([[0.1, 1.5], [1, 2]])
        s2 = s2.unrepeat(2, 2).unrepeat(2, 1).unrepeat(2, 0)
        assert s1.spsame(s2)

        s1 = SparseAtom(setup.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        s2 = SparseAtom(setup.g * ([2, 1, 1], "r"))
        s2.construct([[0.1, 1.5], [1, 2]])
        s2 = s2.unrepeat(2, 0)
        assert s1.spsame(s2)

        s1 = SparseAtom(setup.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        s2 = SparseAtom(setup.g * ([1, 2, 1], "r"))
        s2.construct([[0.1, 1.5], [1, 2]])
        s2 = s2.unrepeat(2, 1)
        assert s1.spsame(s2)

    def test_iter(self, setup):
        s1 = SparseAtom(setup.g)
        s1.construct([[0.1, 1.5], [1, 2]])
        i = 0
        for r, c in s1:
            i += 1
        assert i == s1.nnz

    def test_rij_fail1(self, setup):
        s = SparseOrbital(setup.g.copy())
        s.construct([[0.1, 1.5], [1, 2]])
        with pytest.raises(ValueError):
            s.rij(what="none")

    def test_rij_atom(self, setup):
        s = SparseAtom(setup.g.copy())
        s.construct([[0.1, 1.5], [1, 2]])
        atom = s.rij()
        assert atom.spsame(s)

    def test_rij_atom_orbital_compare(self, setup):
        sa = SparseAtom(setup.g.copy())
        so = SparseOrbital(setup.g.copy())
        sa.construct([[0.1, 1.5], [1, 2]])
        so.construct([[0.1, 1.5], [1, 2]])
        atom = sa.rij()
        orbital = so.rij()
        assert atom.spsame(orbital)
        atom = sa.rij()
        orbital = so.rij("atom")
        assert atom.spsame(orbital)
        # This only works because there is 1 orbital per atom
        orbital = so.rij()
        assert so.spsame(orbital)
        so.finalize()
        orbital = so.rij()
        assert so.spsame(orbital)

    def test_sp_orb_remove(self, setup):
        so = SparseOrbital(setup.g.copy())
        so2 = so.remove(0)
        assert so.geometry.na - 1 == so2.geometry.na
        so2 = so.remove([0])
        assert so.geometry.na - 1 == so2.geometry.na
        so2 = so.remove([1])
        assert so.geometry.na - 1 == so2.geometry.na

    def test_sp_orb_remove_atom(self):
        so = SparseOrbital(Geometry([[0] * 3, [1] * 3], [Atom[1], Atom[2]], 2))
        so2 = so.remove(Atom[1])
        assert so.geometry.na - 1 == so2.geometry.na
        assert so.geometry.no - 1 == so2.geometry.no

    def test_remove1(self, setup):
        for i in range(len(setup.g)):
            setup.s1.construct([[0.1, 1.5], [1, 2]])
            s1 = setup.s1.remove(i)
            setup.s1.empty()
            s2 = SparseAtom(setup.g.remove(i))
            s2.construct([[0.1, 1.5], [1, 2]])
            assert s1.spsame(s2)

    def test_sub1(self, setup):
        all = range(len(setup.g))
        for i in range(len(setup.g)):
            setup.s1.construct([[0.1, 1.5], [1, 2]])
            # my new sub
            sub = [j for j in all if i != j]
            s1 = setup.s1.sub(sub)
            setup.s1.empty()
            s2 = SparseAtom(setup.g.sub(sub))
            s2.construct([[0.1, 1.5], [1, 2]])
            assert s1.spsame(s2)
            assert len(s2) == len(sub)

    def test_construct_eta(self, setup):
        s = setup.s1.copy()
        s.construct([[0.1, 1.5], [1, 2]], eta=True)

    def test_tile1(self, setup):
        setup.s1.construct([[0.1, 1.5], [1, 2]])
        setup.s1.finalize()
        s1 = setup.s1.tile(2, 0).tile(2, 1)
        s2 = SparseAtom(setup.g * [2, 2, 1])
        s2.construct([[0.1, 1.5], [1, 2]])
        assert s1.spsame(s2)
        s1.finalize()
        s2.finalize()
        assert np.allclose(s1._csr._D, s2._csr._D)
        s2 = s2.untile(2, 1).untile(2, 0)
        assert setup.s1.spsame(s2)
        s2.finalize()
        assert np.allclose(setup.s1._csr._D, s2._csr._D)
        s1 = s1.untile(2, 1).untile(2, 0)
        assert setup.s1.spsame(s1)
        s1.finalize()
        assert np.allclose(s1._csr._D, setup.s1._csr._D)
        setup.s1.empty()

    def test_tile2(self, setup):
        setup.s1.construct([[0.1, 1.5], [1, 2]])
        setup.s1.finalize()
        s1 = setup.s1.tile(2, 0).tile(2, 1)
        s2 = SparseAtom(setup.g * [2, 2, 1])
        s2.construct([[0.1, 1.5], [1, 2]])
        assert s1.spsame(s2)
        s1.finalize()
        s2.finalize()
        assert np.allclose(s1._csr._D, s2._csr._D)
        s2 = s2.untile(2, 1).untile(2, 0)
        assert setup.s1.spsame(s2)
        s2.finalize()
        assert np.allclose(setup.s1._csr._D, s2._csr._D)
        s1 = s1.untile(2, 1).untile(2, 0)
        assert setup.s1.spsame(s1)
        s1.finalize()
        assert np.allclose(s1._csr._D, setup.s1._csr._D)
        setup.s1.empty()

    def test_repeat1(self, setup):
        setup.s1.construct([[0.1, 1.5], [1, 2]])
        s1 = setup.s1.repeat(2, 0).repeat(2, 1)
        setup.s1.empty()
        s2 = SparseAtom(setup.g * ([2, 2, 1], "r"))
        s2.construct([[0.1, 1.5], [1, 2]])
        assert s1.spsame(s2)
        s1.finalize()
        s2.finalize()
        assert np.allclose(s1._csr._D, s2._csr._D)

    def test_repeat2(self, setup):
        setup.s1.construct([[0.1, 1.5], [1, 2]])
        setup.s1.finalize()
        s1 = setup.s1.repeat(2, 0).repeat(2, 1)
        setup.s1.empty()
        s2 = SparseAtom(setup.g * ([2, 2, 1], "r"))
        s2.construct([[0.1, 1.5], [1, 2]])
        assert s1.spsame(s2)
        s1.finalize()
        s2.finalize()
        assert np.allclose(s1._csr._D, s2._csr._D)

    def test_supercell_poisition1(self, setup):
        g1 = setup.g.copy()
        g2 = setup.g.translate([100, 100, 100])
        # Just check that the atomic coordinates are not equivalent
        # up to 1 angstrom
        assert not np.allclose(g1.xyz, g2.xyz, atol=1)
        s1 = SparseAtom(g1)
        s2 = SparseAtom(g2)
        s1.construct([[0.1, 1.5], [1, 2]])
        s1.finalize()
        s2.construct([[0.1, 1.5], [1, 2]])
        s2.finalize()
        assert s1.spsame(s2)

    def test_set_nsc1(self, setup):
        g = fcc(1.0, Atom(1, R=3.5))
        s = SparseAtom(g)
        s.construct([[0.1, 1.5, 3.5], [1, 2, 3]])
        s.finalize()
        assert s.nnz > 1
        s.set_nsc([1, 1, 1])
        assert s.nnz == 1
        assert s[0, 0] == 1

    def test_set_nsc2(self, setup):
        g = graphene(atoms=Atom(6, R=1.43))
        s = SparseAtom(g)
        s.construct([[0.1, 1.43], [1, 2]])
        s.finalize()
        assert s.nnz == 8
        s.set_nsc(a=1)
        assert s.nnz == 6
        s.set_nsc([None, 1, 1])
        assert s.nnz == 4
        assert s[0, 0] == 1

    def test_set_nsc3(self, setup):
        g = graphene(atoms=Atom(6, R=1.43))
        s = SparseAtom(g)

        s.set_nsc((3, 3, 1))
        s.construct([[0.1, 1.43], [1, 2]])

        s55 = s.copy()
        s55.set_nsc((5, 5, 1))
        s77 = s55.copy()
        s77.set_nsc((7, 7, 1))
        s59 = s77.copy()
        s59.set_nsc((5, 9, 1))
        assert s.nnz == s55.nnz
        assert s.nnz == s77.nnz
        assert s.nnz == s59.nnz

        s55.set_nsc((3, 3, 1))
        s77.set_nsc((3, 3, 1))
        s59.set_nsc((3, 3, 1))
        assert s.nnz == s55.nnz
        assert s.nnz == s77.nnz
        assert s.nnz == s59.nnz

    def test_edges1(self, setup):
        g = graphene(atoms=Atom(6, R=1.43))
        s = SparseAtom(g)
        s.construct([[0.1, 1.43], [1, 2]])
        assert len(s.edges(0)) == 4
        assert len(s.edges(0, exclude=[0])) == 3

    def test_op_numpy_iscalar(self, setup):
        g = graphene(atoms=Atom(6, R=1.43))
        S = SparseAtom(g, dtype=np.complex128)
        I = np.float32(1)
        # Create initial stuff
        for i in range(10):
            j = range(i, i * 2)
            S[0, j] = i
        S.finalize()

        Ssum = S._csr._D.sum()

        S += I
        assert isinstance(S, SparseAtom)
        assert S._csr._D.sum() == Ssum + S.nnz

        S -= I
        assert isinstance(S, SparseAtom)
        assert S._csr._D.sum() == Ssum

        S *= I
        assert isinstance(S, SparseAtom)
        assert S._csr._D.sum() == Ssum

        S /= I
        assert isinstance(S, SparseAtom)
        assert S._csr._D.sum() == Ssum

        S **= I
        assert isinstance(S, SparseAtom)
        assert S._csr._D.sum() == Ssum

    def test_op_numpy_scalar(self, setup):
        g = graphene(atoms=Atom(6, R=1.43))
        S = SparseAtom(g)
        I = np.ones(1, dtype=np.complex128)[0]
        # Create initial stuff
        for i in range(10):
            j = range(i, i * 2)
            S[0, j] = i
        S.finalize()

        Ssum = S._csr._D.sum()

        s = S + I
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum + S.nnz

        s = S - I
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum - S.nnz

        s = I + S
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum + S.nnz

        s = S * I
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum

        s = I * S
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum

        s = S / I
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum

        s = S**I
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128
        assert s._csr._D.sum() == Ssum

        s = I**S
        assert isinstance(s, SparseAtom)
        assert s.dtype == np.complex128

    def test_numpy_reduction(self, setup):
        g = graphene(atoms=Atom(6, R=1.43))
        S = SparseAtom(g)
        I = np.ones(1, dtype=np.complex128)[0]
        # Create initial stuff
        for i in range(2):
            j = range(i, i + 2)
            S[i, j] = 1
        S.finalize()
        assert np.sum(S, axis=(0, 1)) == pytest.approx(1 * 2 * 2)

    def test_sanitize_atoms_assign(self, setup):
        g = graphene(atoms=Atom(6, R=1.43))
        S = SparseAtom(g)
        for i in range(2):
            S[i, 1:4] = 1

    def test_fromsp1(self, setup):
        g = setup.g.repeat(2, 0).tile(2, 1)
        lil = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil[0, [1, 2, 3]] = 1
        lil[1, [2, 4, 1]] = 2
        s1 = SparseAtom.fromsp(g, [lil], unknown_key="hello")
        assert s1.nnz == 6
        assert np.allclose(s1.shape, [g.na, g.na_s, 1])

        assert np.allclose(s1[0, [1, 2, 3]], np.ones([3], np.int32))
        assert np.allclose(s1[1, [1, 2, 4]], np.ones([3], np.int32) * 2)

        # Different instantiating
        s2 = SparseAtom.fromsp(g, lil)
        assert s1.spsame(s2)

    def test_fromsp2(self, setup):
        g = setup.g.repeat(2, 0).tile(2, 1)
        lil1 = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil2 = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil1[0, [1, 2, 3]] = 1
        lil2[1, [2, 4, 1]] = 2
        s1 = SparseAtom.fromsp(g, [lil1, lil2])
        assert s1.nnz == 6
        assert np.allclose(s1.shape, [g.na, g.na_s, 2])

        assert np.allclose(s1[0, [1, 2, 3], 0], np.ones([3], np.int32))
        assert np.allclose(s1[0, [1, 2, 3], 1], np.zeros([3], np.int32))
        assert np.allclose(s1[1, [1, 2, 4], 0], np.zeros([3], np.int32))
        assert np.allclose(s1[1, [1, 2, 4], 1], np.ones([3], np.int32) * 2)

    def test_fromsp4(self, setup):
        g = setup.g.repeat(2, 0).tile(2, 1)
        lil1 = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil2 = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil1[0, [1, 2, 3]] = 1
        lil2[1, [2, 4, 1]] = 2

        # Ensure that one does not mix everything.
        with pytest.raises(ValueError):
            SparseAtom.fromsp(setup.g.copy(), [lil1, lil2])

    def test_pickle(self, setup):
        import pickle as p

        g = setup.g.repeat(2, 0).tile(2, 1)
        lil1 = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil2 = sc.sparse.lil_matrix((g.na, g.na_s), dtype=np.int32)
        lil1[0, [1, 2, 3]] = 1
        lil2[1, [2, 4, 1]] = 2
        S = SparseAtom.fromsp(g, [lil1, lil2])
        n = p.dumps(S)
        s = p.loads(n)
        assert s.spsame(S)


@pytest.mark.parametrize("n0", [1, 2])
@pytest.mark.parametrize("n1", [1, 3])
@pytest.mark.parametrize("n2", [1, 4])
def test_sparse_atom_symmetric(n0, n1, n2):
    g = fcc(1.0, Atom(1, R=1.5)) * 2
    s = SparseAtom(g)
    s.construct([[0.1, 1.51], [1, 2]])
    s = s.tile(n0, 0).tile(n1, 1).tile(n2, 2)
    na = s.geometry.na

    nnz = 0
    for ia in range(na):
        # orbitals connecting to ia
        edges = s.edges(ia)
        # Figure out the transposed supercell indices of the edges
        isc = -s.geometry.a2isc(edges)
        # Convert to supercell
        IA = s.geometry.lattice.sc_index(isc) * na + ia
        # Figure out if 'ia' is also in the back-edges
        for ja, edge in zip(IA, edges % na):
            assert ja in s.edges(edge)
            nnz += 1

    # Check that we have counted all nnz
    assert s.nnz == nnz


@pytest.mark.parametrize("i", [0, 1, 2])
def test_sparse_atom_transpose_single(i):
    """This is problematic when the sparsity pattern is not *filled*"""
    g = fcc(1.0, Atom(1, R=1.5)) * 3
    s = SparseAtom(g)
    s[i, 2] = 1.0
    s[i, 0] = 2.0
    t = s.transpose()

    assert t.nnz == s.nnz
    assert t[2, i] == pytest.approx(1.0)
    assert t[0, i] == pytest.approx(2.0)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_sparse_atom_transpose_more(i):
    """This is problematic when the sparsity pattern is not *filled*"""
    g = fcc(1.0, Atom(1, R=1.5)) * 3
    s = SparseAtom(g)
    s[i, 2] = 1.0
    s[i, 0] = 2.0
    s[i + 2, 3] = 1.0
    s[i + 2, 5] = 2.0
    t = s.transpose()

    assert t.nnz == s.nnz
    assert t[2, i] == pytest.approx(1.0)
    assert t[0, i] == pytest.approx(2.0)
    assert t[3, i + 2] == pytest.approx(1.0)
    assert t[5, i + 2] == pytest.approx(2.0)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_sparse_orbital_transpose_single(i):
    g = fcc(1.0, Atom(1, R=(1.5, 2.1))) * 3
    s = SparseOrbital(g)
    s[i, 2] = 1.0
    s[i, 0] = 2.0
    t = s.transpose()

    assert t.nnz == s.nnz
    assert t[2, i] == pytest.approx(1.0)
    assert t[0, i] == pytest.approx(2.0)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_sparse_orbital_transpose_more(i):
    g = fcc(1.0, Atom(1, R=(1.5, 2.1))) * 3
    s = SparseOrbital(g)
    s[i, 2] = 1.0
    s[i, 0] = 2.0
    s[i + 3, 4] = 1.0
    s[i + 2, 4] = 2.0
    t = s.transpose()

    assert t.nnz == s.nnz
    assert t[2, i] == pytest.approx(1.0)
    assert t[0, i] == pytest.approx(2.0)
    assert t[4, i + 3] == pytest.approx(1.0)
    assert t[4, i + 2] == pytest.approx(2.0)


def test_sparse_orbital_add_axis(setup):
    g = setup.g.copy()
    s = SparseOrbital(g)
    s.construct([[0.1, 1.5], [1, 2]])
    s1 = s.add(s, axis=2)
    with pytest.warns(SislWarning, match=r"with 0 length"):
        s2 = SparseOrbital(g.append(Lattice([0, 0, 10]), 2).add(g, offset=[0, 0, 5]))
    s2.construct([[0.1, 1.5], [1, 2]])
    assert s1.spsame(s2)


def test_sparse_orbital_add_no_axis():
    from sisl.geom import sc

    with pytest.warns(SislWarning, match=r"with 0 length"):
        g = (sc(1.0, Atom(1, R=1.5)) * 2).add(Lattice([0, 0, 5]))
    s = SparseOrbital(g)
    s.construct([[0.1, 1.5], [1, 2]])
    s1 = s.add(s, offset=[0, 0, 3])
    s2 = SparseOrbital(g.add(g, offset=[0, 0, 3]))
    s2.construct([[0.1, 1.5], [1, 2]])
    assert s1.spsame(s2)


def test_sparse_orbital_sub_orbital():
    atom = Atom(1, (1, 2, 3))
    g = fcc(1.0, atom) * 2
    s = SparseOrbital(g)

    # take out some orbitals
    s1 = s.sub_orbital(atom, 1)
    assert s1.geometry.no == s1.geometry.na

    s2 = s.sub_orbital(atom, atom.orbitals[1])
    assert s1 == s2

    s2 = s.sub_orbital(atom, [atom.orbitals[1]])
    assert s1 == s2

    s2 = s.sub_orbital(atom, [atom.orbitals[1], atom.orbitals[0]])
    assert s2.geometry.atoms[0].orbitals[0] == atom.orbitals[0]
    assert s2.geometry.atoms[0].orbitals[1] == atom.orbitals[1]


def test_translate_sparse_atoms():
    # Build the geometry
    H = Atom(1, (1, 2, 3))
    graph = graphene(atoms=H)
    assert np.allclose(graph.nsc, [3, 3, 1])

    # Build a dummy matrix with onsite terms and just one coupling term
    matrix = SparseAtom(graph)
    matrix[0, 0] = 1
    matrix[1, 1] = 2
    matrix[0, 1] = 3

    # Translate the second atom
    transl = matrix._translate_atoms_sc([[0, 0, 0], [1, 0, 0]])

    # Check that the new auxiliary cell is correct.
    assert np.allclose(transl.nsc, [3, 1, 1])
    assert np.allclose(transl.shape, [2, 6, 1])

    # Check coordinates
    assert np.allclose(transl.geometry[0], matrix.geometry[0])
    assert np.allclose(transl.geometry[1], matrix.geometry[1] + matrix.geometry.cell[0])

    # Assert that the matrix elements have been translated
    assert transl[0, 0] == 1
    assert transl[1, 1] == 2
    assert transl[0, 1] == 0
    assert transl[0, 3] == 3

    # Translate back to unit cell
    uc_matrix = transl.translate2uc()

    # Check the auxiliary cell, coordinates and the matrix elements
    assert np.allclose(uc_matrix.nsc, [1, 1, 1])
    assert np.allclose(uc_matrix.shape, [2, 2, 1])

    assert np.allclose(uc_matrix.geometry.xyz, matrix.geometry.xyz)

    assert uc_matrix[0, 0] == 1
    assert uc_matrix[1, 1] == 2
    assert uc_matrix[0, 1] == 3
    assert uc_matrix[0, 3] == 0

    # Instead, test atoms and axes arguments to avoid any translation.
    for kwargs in [{"atoms": [0]}, {"axes": [1, 2]}]:
        not_uc_matrix = transl.translate2uc(**kwargs)

        # Check the auxiliary cell, coordinates and the matrix elements
        assert np.allclose(not_uc_matrix.nsc, transl.nsc)
        assert np.allclose(not_uc_matrix.shape, transl.shape)

        assert np.allclose(not_uc_matrix.geometry.xyz, transl.geometry.xyz)

        assert np.allclose(not_uc_matrix._csr.todense(), transl._csr.todense())

    # Now, translate both atoms
    transl_both = uc_matrix._translate_atoms_sc([[-1, 0, 0], [1, 0, 0]])

    # Check the auxiliary cell, coordinates and the matrix elements
    assert np.allclose(transl_both.nsc, [5, 1, 1])
    assert np.allclose(transl_both.shape, [2, 10, 1])

    assert np.allclose(
        transl_both.geometry[0], uc_matrix.geometry[0] - uc_matrix.geometry.cell[0]
    )
    assert np.allclose(
        transl_both.geometry[1], uc_matrix.geometry[1] + uc_matrix.geometry.cell[0]
    )

    assert transl_both[0, 0] == 1
    assert transl_both[1, 1] == 2
    assert transl_both[0, 1] == 0
    assert transl_both[0, 3] == 3


def test_sanitize_orbs_assign():
    g = graphene(atoms=Atom(6, R=[1.43, 1.66]))
    S = SparseOrbital(g)
    for i in range(2):
        S[i, 1:4] = 1
