from __future__ import print_function, division

from nose.tools import *
from nose.plugins.attrib import attr

import warnings
import math as m
import numpy as np

from sisl import Geometry, Atom, SuperCell, Hamiltonian, Spin, PathBZ


@attr('hamiltonian')
class TestHamiltonian(object):

    def setUp(self):
        bond = 1.42
        sq3h = 3.**.5 * 0.5
        self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                      [1.5, -sq3h, 0.],
                                      [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

        C = Atom(Z=6, R=bond * 1.01, orbs=1)
        self.g = Geometry(np.array([[0., 0., 0.],
                                    [1., 0., 0.]], np.float64) * bond,
                          atom=C, sc=self.sc)
        self.H = Hamiltonian(self.g)
        self.HS = Hamiltonian(self.g, orthogonal=False)

        C = Atom(Z=6, R=bond * 1.01, orbs=2)
        self.g2 = Geometry(np.array([[0., 0., 0.],
                                    [1., 0., 0.]], np.float64) * bond,
                          atom=C, sc=self.sc)
        self.H2 = Hamiltonian(self.g2)
        self.HS2 = Hamiltonian(self.g2, orthogonal=False)

    def tearDown(self):
        del self.sc
        del self.g
        del self.H
        del self.HS
        del self.g2
        del self.H2
        del self.HS2

    def test_objects(self):
        assert_true(len(self.H.xyz) == 2)
        assert_true(self.g.no == len(self.H))
        assert_true(len(self.HS.xyz) == 2)
        assert_true(self.g.no == len(self.HS))

        assert_true(len(self.H2.xyz) == 2)
        assert_true(self.g2.no == len(self.H2))
        assert_true(len(self.HS2.xyz) == 2)
        assert_true(self.g2.no == len(self.HS2))

    def test_dtype(self):
        assert_true(self.H.dtype == np.float64)
        assert_true(self.HS.dtype == np.float64)
        assert_true(self.H2.dtype == np.float64)
        assert_true(self.HS2.dtype == np.float64)

    def test_ortho(self):
        assert_true(self.H.orthogonal)
        assert_false(self.HS.orthogonal)

    def test_set1(self):
        self.H.H[0, 0] = 1.
        assert_true(self.H[0, 0] == 1.)
        assert_true(self.H[1, 0] == 0.)
        self.H.empty()

        self.HS.H[0, 0] = 1.
        assert_true(self.HS.H[0, 0] == 1.)
        assert_true(self.HS.H[1, 0] == 0.)
        assert_true(self.HS.S[0, 0] == 0.)
        assert_true(self.HS.S[1, 0] == 0.)
        self.HS.S[0, 0] = 1.
        assert_true(self.HS.H[0, 0] == 1.)
        assert_true(self.HS.H[1, 0] == 0.)
        assert_true(self.HS.S[0, 0] == 1.)
        assert_true(self.HS.S[1, 0] == 0.)

        # delete before creating the same content
        self.HS.empty()
        # THIS IS A CHECK FOR BACK_WARD COMPATIBILITY!
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.HS[0, 0] = 1., 1.
        assert_true(self.HS.H[0, 0] == 1.)
        assert_true(self.HS.S[0, 0] == 1.)
        self.HS.empty()

    def test_set2(self):
        self.H.construct([(0.1, 1.5), (1., 0.1)])
        assert_true(self.H[0, 0] == 1.)
        assert_true(self.H[1, 0] == 0.1)
        assert_true(self.H[0, 1] == 0.1)
        self.H.empty()

    def test_set3(self):
        self.HS.construct([(0.1, 1.5), ((1., 2.), (0.1, 0.2))])
        assert_true(self.HS.H[0, 0] == 1.)
        assert_true(self.HS.S[0, 0] == 2.)
        assert_true(self.HS.H[1, 1] == 1.)
        assert_true(self.HS.S[1, 1] == 2.)
        assert_true(self.HS.H[1, 0] == 0.1)
        assert_true(self.HS.H[0, 1] == 0.1)
        assert_true(self.HS.S[1, 0] == 0.2)
        assert_true(self.HS.S[0, 1] == 0.2)
        assert_true(self.HS.nnz == len(self.HS) * 4)
        self.HS.empty()

    def test_set4(self):
        for ia, io in self.H:
            # Find atoms close to 'ia'
            idx = self.H.geom.close(ia, R=(0.1, 1.5))
            self.H[io, idx[0]] = 1.
            self.H[io, idx[1]] = 0.1
        assert_true(self.H.H[0, 0] == 1.)
        assert_true(self.H.H[1, 1] == 1.)
        assert_true(self.H.H[1, 0] == 0.1)
        assert_true(self.H.H[0, 1] == 0.1)
        assert_true(self.H.nnz == len(self.H) * 4)
        self.H.empty()

    @attr('slow')
    def test_set5(self):
        # Test of HUGE construct
        g = self.g.tile(10, 0).tile(10, 1).tile(10, 2)
        H = Hamiltonian(g)
        H.construct([(0.1, 1.5), (1., 0.1)])
        assert_true(H.H[0, 0] == 1.)
        assert_true(H.H[1, 1] == 1.)
        assert_true(H.H[1, 0] == 0.1)
        assert_true(H.H[0, 1] == 0.1)
        # This is graphene
        # on-site == len(H)
        # nn == 3 * len(H)
        assert_true(H.nnz == len(H) * 4)
        del H

    def test_iter1(self):
        self.HS.construct([(0.1, 1.5), ((1., 2.), (0.1, 0.2))])
        nnz = 0
        for io, jo in self.HS.iter_nnz():
            nnz = nnz + 1
        assert_equal(nnz, self.HS.nnz)
        nnz = 0
        for io, jo in self.HS.iter_nnz(0):
            nnz = nnz + 1
        # 3 nn and 1 onsite
        assert_equal(nnz, 4)
        self.HS.empty()

    def test_iter2(self):
        self.HS.H[0, 0] = 1.
        nnz = 0
        for io, jo in self.HS.iter_nnz():
            nnz = nnz + 1
        assert_equal(nnz, self.HS.nnz)
        assert_equal(nnz, 1)
        self.HS.empty()

    @raises(ValueError)
    def test_construct_raise(self):
        # Test that construct fails with more than one
        # orbital
        self.H2.construct([(0.1, 1.5), (1., 0.1)])

    def test_getitem1(self):
        H = self.H
        # graphene Hamiltonian
        H.construct([(0.1, 1.5), (0.1, 0.2)])
        # Assert all connections
        assert_equal(H[0, 0], 0.1)
        assert_equal(H[0, 1], 0.2)
        assert_equal(H[0, 1, (-1, 0)], 0.2)
        assert_equal(H[0, 1, (0, -1)], 0.2)
        assert_equal(H[1, 0], 0.2)
        assert_equal(H[1, 1], 0.1)
        assert_equal(H[1, 0, (1, 0)], 0.2)
        assert_equal(H[1, 0, (0, 1)], 0.2)
        H[0, 0, (0, 1)] = 0.3
        assert_equal(H[0, 0, (0, 1)], 0.3)
        H[0, 1, (0, 1)] = -0.2
        assert_equal(H[0, 1, (0, 1)], -0.2)
        H.empty()

    def test_delitem1(self):
        H = self.H
        H.construct([(0.1, 1.5), (0.1, 0.2)])
        assert_equal(H[0, 1], 0.2)
        del H[0, 1]
        assert_equal(H[0, 1], 0.0)
        H.empty()

    def test_sp2HS(self):
        csr = self.H.tocsr(0)
        H = Hamiltonian.fromsp(self.H.geom, csr)
        assert_true(H.spsame(self.H))

    def test_op1(self):
        g = Geometry([[i, 0, 0] for i in range(100)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.int32)
        for i in range(10):
            j = range(i*4, i*4+3)
            H[0, j] = i

            # i+
            H += 1
            for jj in j:
                assert_equal(H[0, jj], i+1)
                assert_equal(H[1, jj], 0)

            # i-
            H -= 1
            for jj in j:
                assert_equal(H[0, jj], i)
                assert_equal(H[1, jj], 0)

            # i*
            H *= 2
            for jj in j:
                assert_equal(H[0, jj], i*2)
                assert_equal(H[1, jj], 0)

            # //
            H //= 2
            for jj in j:
                assert_equal(H[0, jj], i)
                assert_equal(H[1, jj], 0)

            # i**
            H **= 2
            for jj in j:
                assert_equal(H[0, jj], i**2)
                assert_equal(H[1, jj], 0)

    def test_op2(self):
        g = Geometry([[i, 0, 0] for i in range(100)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.int32)
        for i in range(10):
            j = range(i*4, i*4+3)
            H[0, j] = i

            # +
            s = H + 1
            for jj in j:
                assert_equal(s[0, jj], i+1)
                assert_equal(H[0, jj], i)
                assert_equal(s[1, jj], 0)

            # -
            s = H - 1
            for jj in j:
                assert_equal(s[0, jj], i-1)
                assert_equal(H[0, jj], i)
                assert_equal(s[1, jj], 0)

            # -
            s = 1 - H
            for jj in j:
                assert_equal(s[0, jj], 1-i)
                assert_equal(H[0, jj], i)
                assert_equal(s[1, jj], 0)

            # *
            s = H * 2
            for jj in j:
                assert_equal(s[0, jj], i*2)
                assert_equal(H[0, jj], i)
                assert_equal(s[1, jj], 0)

            # //
            s = s // 2
            for jj in j:
                assert_equal(s[0, jj], i)
                assert_equal(H[0, jj], i)
                assert_equal(s[1, jj], 0)

            # **
            s = H ** 2
            for jj in j:
                assert_equal(s[0, jj], i**2)
                assert_equal(H[0, jj], i)
                assert_equal(s[1, jj], 0)

            # ** (r)
            s = 2 ** H
            for jj in j:
                assert_equal(s[0, jj], 2 ** H[0, jj])
                assert_equal(H[0, jj], i)
                assert_equal(s[1, jj], 0)

    def test_op3(self):
        g = Geometry([[i, 0, 0] for i in range(100)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.int32)
        Hc = H.copy()
        del Hc

        # Create initial stuff
        for i in range(10):
            j = range(i*4, i*4+3)
            H[0, j] = i

        for op in ['add', 'sub', 'mul', 'pow']:
            func = getattr(H, '__{}__'.format(op))
            h = func(1)
            assert_equal(h.dtype, np.int32)
            h = func(1.)
            assert_equal(h.dtype, np.float64)
            if op != 'pow':
                h = func(1.j)
                assert_equal(h.dtype, np.complex128)

        H = H.copy(dtype=np.float64)
        for op in ['add', 'sub', 'mul', 'pow']:
            func = getattr(H, '__{}__'.format(op))
            h = func(1)
            assert_equal(h.dtype, np.float64)
            h = func(1.)
            assert_equal(h.dtype, np.float64)
            if op != 'pow':
                h = func(1.j)
                assert_equal(h.dtype, np.complex128)

        H = H.copy(dtype=np.complex128)
        for op in ['add', 'sub', 'mul', 'pow']:
            func = getattr(H, '__{}__'.format(op))
            h = func(1)
            assert_equal(h.dtype, np.complex128)
            h = func(1.)
            assert_equal(h.dtype, np.complex128)
            if op != 'pow':
                h = func(1.j)
                assert_equal(h.dtype, np.complex128)

    def test_op4(self):
        g = Geometry([[i, 0, 0] for i in range(100)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.int32)
        # Create initial stuff
        for i in range(10):
            j = range(i*4, i*4+3)
            H[0, j] = i

        h = 1 + H
        assert_equal(h.dtype, np.int32)
        h = 1. + H
        assert_equal(h.dtype, np.float64)
        h = 1.j + H
        assert_equal(h.dtype, np.complex128)

        h = 1 - H
        assert_equal(h.dtype, np.int32)
        h = 1. - H
        assert_equal(h.dtype, np.float64)
        h = 1.j - H
        assert_equal(h.dtype, np.complex128)

        h = 1 * H
        assert_equal(h.dtype, np.int32)
        h = 1. * H
        assert_equal(h.dtype, np.float64)
        h = 1.j * H
        assert_equal(h.dtype, np.complex128)

        h = 1 ** H
        assert_equal(h.dtype, np.int32)
        h = 1. ** H
        assert_equal(h.dtype, np.float64)
        h = 1.j ** H
        assert_equal(h.dtype, np.complex128)

    def test_cut1(self):
        # Test of eigenvalues using a cut
        # Hamiltonian
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        Hg = Hamiltonian(self.g)
        Hg.construct([R, param])
        g = self.g.tile(2, 0).tile(2, 1)
        H = Hamiltonian(g)
        H.construct([R, param])
        # Create cut Hamiltonian
        Hc = H.cut(2, 1).cut(2, 0)
        eigc = Hc.eigh()
        eigg = Hg.eigh()
        assert_true(np.allclose(eigc, eigg))
        assert_true(np.allclose(Hg.eigh(), Hc.eigh()))
        del Hc, H

    def test_cut2(self):
        # Test of eigenvalues using a cut
        # Hamiltonian
        R, param = [0.1, 1.5], [(1., 1.), (0.1, 0.1)]

        # Create reference
        Hg = Hamiltonian(self.g, orthogonal=False)
        Hg.construct([R, param])

        g = self.g.tile(2, 0).tile(2, 1)
        H = Hamiltonian(g, orthogonal=False)
        H.construct([R, param])
        # Create cut Hamiltonian
        Hc = H.cut(2, 1).cut(2, 0)
        eigc = Hc.eigh()
        eigg = Hg.eigh()
        assert_true(np.allclose(Hg.eigh(), Hc.eigh()))
        del Hc, H

    def test_eig1(self):
        # Test of eigenvalues
        R, param = [0.1, 1.5], [1., 0.1]
        g = self.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param), eta=True)
        eig1 = H.eigh()
        for i in range(2):
            assert_true(np.allclose(eig1, H.eigh()))
        H.eigsh(n=4)
        H.empty()
        del H

    def test_eig2(self):
        # Test of eigenvalues
        HS = self.HS.copy()
        HS.construct([(0.1, 1.5), ((1., 1.), (0.1, 0.1))])
        eig1 = HS.eigh()
        for i in range(2):
            assert_true(np.allclose(eig1, HS.eigh()))
        self.HS.empty()

    def test_eig3(self):
        self.HS.construct([(0.1, 1.5), ((1., 1.), (0.1, 0.1))])
        BS = PathBZ(self.HS.geom, [[0, 0, 0], [0.5, 0.5, 0]], 10)
        eig = self.HS.eigh(BS)
        assert_equal(len(BS), eig.shape[0])
        assert_equal(len(self.HS), eig.shape[1])
        self.HS.empty()

    def test_spin1(self):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.int32, spin=2)
        for i in range(10):
            j = range(i*4, i*4+3)
            H[0, j] = (i, i*2)

        H2 = Hamiltonian(g, 2, dtype=np.int32)
        for i in range(10):
            j = range(i*4, i*4+3)
            H2[0, j] = (i, i*2)
        assert_true(H.spsame(H2))

    def test_spin2(self):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.int32, spin=2)
        for i in range(10):
            j = range(i*4, i*4+3)
            H[0, j] = (i, i*2)

        H2 = Hamiltonian(g, 2, dtype=np.int32)
        for i in range(10):
            j = range(i*4, i*4+3)
            H2[0, j] = (i, i*2)
        assert_true(H.spsame(H2))

        H2 = Hamiltonian(g, Spin(2), dtype=np.int32)
        for i in range(10):
            j = range(i*4, i*4+3)
            H2[0, j] = (i, i*2)
        assert_true(H.spsame(H2))

        H2 = Hamiltonian(g, Spin('polarized'), dtype=np.int32)
        for i in range(10):
            j = range(i*4, i*4+3)
            H2[0, j] = (i, i*2)
        assert_true(H.spsame(H2))

    def test_non_colinear1(self):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.float64, spin=4)
        for i in range(10):
            j = range(i*4, i*4+3)
            H[i, i, 0] = 0.
            H[i, i, 1] = 0.
            H[i, i, 2] = 0.1
            H[i, i, 3] = 0.1
            if i > 0:
                H[i, i-1, 0] = 1.
                H[i, i-1, 1] = 1.
            if i < 9:
                H[i, i+1, 0] = 1.
                H[i, i+1, 1] = 1.
        eig1 = H.eigh()
        # Check TimeSelector
        for i in range(2):
            assert_true(np.allclose(H.eigh(), eig1))
        assert_true(len(eig1) == len(H))

        H1 = Hamiltonian(g, dtype=np.float64, spin=Spin('non-colinear'))
        for i in range(10):
            j = range(i*4, i*4+3)
            H1[i, i, 0] = 0.
            H1[i, i, 1] = 0.
            H1[i, i, 2] = 0.1
            H1[i, i, 3] = 0.1
            if i > 0:
                H1[i, i-1, 0] = 1.
                H1[i, i-1, 1] = 1.
            if i < 9:
                H1[i, i+1, 0] = 1.
                H1[i, i+1, 1] = 1.
        assert_true(H1.spsame(H))
        eig1 = H1.eigh()
        # Check TimeSelector
        for i in range(2):
            assert_true(np.allclose(H1.eigh(), eig1))
        assert_true(np.allclose(H.eigh(), H1.eigh()))

    def test_so1(self):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.float64, spin=8)
        for i in range(10):
            j = range(i*4, i*4+3)
            H[i, i, 0] = 0.
            H[i, i, 1] = 0.
            H[i, i, 2] = 0.1
            H[i, i, 3] = 0.1
            H[i, i, 4] = 0.1
            H[i, i, 5] = 0.1
            H[i, i, 6] = 0.1
            H[i, i, 7] = 0.1
            if i > 0:
                H[i, i-1, 0] = 1.
                H[i, i-1, 1] = 1.
            if i < 9:
                H[i, i+1, 0] = 1.
                H[i, i+1, 1] = 1.
        eig1 = H.eigh()
        # Check TimeSelector
        for i in range(2):
            assert_true(np.allclose(H.eigh(), eig1))
        assert_true(len(H.eigh()) == len(H))

        H1 = Hamiltonian(g, dtype=np.float64, spin=Spin('spin-orbit'))
        for i in range(10):
            j = range(i*4, i*4+3)
            H1[i, i, 0] = 0.
            H1[i, i, 1] = 0.
            H1[i, i, 2] = 0.1
            H1[i, i, 3] = 0.1
            if i > 0:
                H1[i, i-1, 0] = 1.
                H1[i, i-1, 1] = 1.
            if i < 9:
                H1[i, i+1, 0] = 1.
                H1[i, i+1, 1] = 1.
        assert_true(H1.spsame(H))
        eig1 = H1.eigh()
        # Check TimeSelector
        for i in range(2):
            assert_true(np.allclose(H1.eigh(), eig1))
        assert_true(np.allclose(H.eigh(), H1.eigh()))

    def test_finalized(self):
        assert_false(self.H.finalized)
        self.H.H[0, 0] = 1.
        self.H.finalize()
        assert_true(self.H.finalized)
        assert_true(self.H.nnz == 1)
        self.H.empty()
        assert_false(self.HS.finalized)
        self.HS.H[0, 0] = 1.
        self.HS.S[0, 0] = 1.
        self.HS.finalize()
        assert_true(self.HS.finalized)
        assert_true(self.HS.nnz == 1)
        self.HS.empty()

    @attr('slow')
    def test_tile1(self):
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        Hg = Hamiltonian(self.g.tile(2, 0).tile(2, 1).tile(2, 2))
        Hg.construct([R, param])
        Hg.finalize()
        H = Hamiltonian(self.g)
        H.construct([R, param])
        H = H.tile(2, 0).tile(2, 1).tile(2, 2)
        assert_true(Hg.spsame(H))

    @attr('slow')
    def test_tile2(self):
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        Hg = Hamiltonian(self.g.tile(2, 0))
        Hg.construct([R, param])
        Hg.finalize()
        H = Hamiltonian(self.g)
        H.construct([R, param])
        H = H.tile(2, 0)
        assert_true(Hg.spsame(H))

    @attr('slow')
    def test_tile3(self):
        R, param = [0.1, 1.1, 2.1, 3.1], [1., 2., 3., 4.]

        # Create reference
        g = Geometry([[0] * 3], Atom('H', R=[4.]), sc=[1.] * 3)
        g.set_nsc([7] * 3)

        # Now create bigger geometry
        G = g.tile(2, 0).tile(2, 1).tile(2, 2)

        HG = Hamiltonian(G.tile(2, 0).tile(2, 1).tile(2, 2))
        HG.construct([R, param])
        HG.finalize()
        H = Hamiltonian(G)
        H.construct([R, param])
        H.finalize()
        H = H.tile(2, 0).tile(2, 1).tile(2, 2)
        assert_true(HG.spsame(H))

    @attr('slow')
    def test_repeat1(self):
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        Hg = Hamiltonian(self.g.repeat(2, 0))
        Hg.construct([R, param])
        Hg.finalize()
        H = Hamiltonian(self.g)
        H.construct([R, param])
        H = H.repeat(2, 0)
        assert_true(Hg.spsame(H))

    @attr('slow')
    def test_repeat2(self):
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        Hg = Hamiltonian(self.g.repeat(2, 0).repeat(2, 1).repeat(2, 2))
        Hg.construct([R, param])
        Hg.finalize()
        H = Hamiltonian(self.g)
        H.construct([R, param])
        H = H.repeat(2, 0).repeat(2, 1). repeat(2, 2)
        assert_true(Hg.spsame(H))

    @attr('slow')
    def test_repeat3(self):
        R, param = [0.1, 1.1, 2.1, 3.1], [1., 2., 3., 4.]

        # Create reference
        g = Geometry([[0] * 3], Atom('H', R=[4.]), sc=[1.] * 3)
        g.set_nsc([7] * 3)

        # Now create bigger geometry
        G = g.repeat(2, 0).repeat(2, 1).repeat(2, 2)

        HG = Hamiltonian(G.repeat(2, 0).repeat(2, 1).repeat(2, 2))
        HG.construct([R, param])
        HG.finalize()
        H = Hamiltonian(G)
        H.construct([R, param])
        H.finalize()
        H = H.repeat(2, 0).repeat(2, 1).repeat(2, 2)
        assert_true(HG.spsame(H))

    def test_sub1(self):
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        H = Hamiltonian(self.g)
        H.construct([R, param])
        H.finalize()
        # Tiling in this direction will not introduce
        # any new connections.
        # So tiling and removing is a no-op (but
        # increases vacuum in 3rd lattice vector)
        Hg = Hamiltonian(self.g.tile(2, 2))
        Hg.construct([R, param])
        Hg = Hg.sub(range(len(self.g)))
        Hg.finalize()
        assert_true(Hg.spsame(H))
