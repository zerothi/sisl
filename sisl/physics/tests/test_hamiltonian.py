from __future__ import print_function, division

import pytest

import warnings
import math as m
import numpy as np

from sisl import Geometry, Atom, SuperCell, Hamiltonian, Spin, PathBZ


@pytest.fixture
def setup():
    class t():
        def __init__(self):
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
    return t()


@pytest.mark.hamiltonian
class TestHamiltonian(object):

    def test_objects(self, setup):
        assert len(setup.H.xyz) == 2
        assert setup.g.no == len(setup.H)
        assert len(setup.HS.xyz) == 2
        assert setup.g.no == len(setup.HS)

        assert len(setup.H2.xyz) == 2
        assert setup.g2.no == len(setup.H2)
        assert len(setup.HS2.xyz) == 2
        assert setup.g2.no == len(setup.HS2)

    def test_dtype(self, setup):
        assert setup.H.dtype == np.float64
        assert setup.HS.dtype == np.float64
        assert setup.H2.dtype == np.float64
        assert setup.HS2.dtype == np.float64

    def test_ortho(self, setup):
        assert setup.H.orthogonal
        assert not setup.HS.orthogonal
        assert not setup.H.non_orthogonal
        assert setup.HS.non_orthogonal

    def test_set1(self, setup):
        setup.H.H[0, 0] = 1.
        assert setup.H[0, 0] == 1.
        assert setup.H[1, 0] == 0.
        setup.H.empty()

        setup.HS.H[0, 0] = 1.
        assert setup.HS.H[0, 0] == 1.
        assert setup.HS.H[1, 0] == 0.
        assert setup.HS.S[0, 0] == 0.
        assert setup.HS.S[1, 0] == 0.
        setup.HS.S[0, 0] = 1.
        assert setup.HS.H[0, 0] == 1.
        assert setup.HS.H[1, 0] == 0.
        assert setup.HS.S[0, 0] == 1.
        assert setup.HS.S[1, 0] == 0.

        # delete before creating the same content
        setup.HS.empty()
        # THIS IS A CHECK FOR BACK_WARD COMPATIBILITY!
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            setup.HS[0, 0] = 1., 1.
        assert setup.HS.H[0, 0] == 1.
        assert setup.HS.S[0, 0] == 1.
        setup.HS.empty()

    def test_set2(self, setup):
        setup.H.construct([(0.1, 1.5), (1., 0.1)])
        assert setup.H[0, 0] == 1.
        assert setup.H[1, 0] == 0.1
        assert setup.H[0, 1] == 0.1
        setup.H.empty()

    def test_set3(self, setup):
        setup.HS.construct([(0.1, 1.5), ((1., 2.), (0.1, 0.2))])
        assert setup.HS.H[0, 0] == 1.
        assert setup.HS.S[0, 0] == 2.
        assert setup.HS.H[1, 1] == 1.
        assert setup.HS.S[1, 1] == 2.
        assert setup.HS.H[1, 0] == 0.1
        assert setup.HS.H[0, 1] == 0.1
        assert setup.HS.S[1, 0] == 0.2
        assert setup.HS.S[0, 1] == 0.2
        assert setup.HS.nnz == len(setup.HS) * 4
        setup.HS.empty()

    def test_set4(self, setup):
        for ia, io in setup.H:
            # Find atoms close to 'ia'
            idx = setup.H.geom.close(ia, R=(0.1, 1.5))
            setup.H[io, idx[0]] = 1.
            setup.H[io, idx[1]] = 0.1
        assert setup.H.H[0, 0] == 1.
        assert setup.H.H[1, 1] == 1.
        assert setup.H.H[1, 0] == 0.1
        assert setup.H.H[0, 1] == 0.1
        assert setup.H.nnz == len(setup.H) * 4
        setup.H.empty()

    @pytest.mark.slow
    def test_set5(self, setup):
        # Test of HUGE construct
        g = setup.g.tile(10, 0).tile(10, 1).tile(10, 2)
        H = Hamiltonian(g)
        H.construct([(0.1, 1.5), (1., 0.1)])
        assert H.H[0, 0] == 1.
        assert H.H[1, 1] == 1.
        assert H.H[1, 0] == 0.1
        assert H.H[0, 1] == 0.1
        # This is graphene
        # on-site == len(H)
        # nn == 3 * len(H)
        assert H.nnz == len(H) * 4
        del H

    def test_iter1(self, setup):
        setup.HS.construct([(0.1, 1.5), ((1., 2.), (0.1, 0.2))])
        nnz = 0
        for io, jo in setup.HS.iter_nnz():
            nnz = nnz + 1
        assert nnz == setup.HS.nnz
        nnz = 0
        for io, jo in setup.HS.iter_nnz(0):
            nnz = nnz + 1
        # 3 nn and 1 onsite
        assert nnz == 4
        setup.HS.empty()

    def test_iter2(self, setup):
        setup.HS.H[0, 0] = 1.
        nnz = 0
        for io, jo in setup.HS.iter_nnz():
            nnz = nnz + 1
        assert nnz == setup.HS.nnz
        assert nnz == 1
        setup.HS.empty()

    @pytest.mark.filterwarnings('ignore', category=np.ComplexWarning)
    def test_Hk1(self, setup):
        H = setup.HS.copy()
        H.construct([(0.1, 1.5), ((1., 2.), (0.1, 0.2))])
        # The loops ensures that we loop over all selector
        # items
        h = H.copy()
        for i in range(4):
            assert h.Hk().dtype == np.complex128
            assert h.Sk().dtype == np.complex128
        h = H.copy()
        for i in range(4):
            Hk = h.Hk(dtype=np.complex64)
            assert Hk.dtype == np.complex64
        h = H.copy()
        for i in range(4):
            Hk = h.Hk(dtype=np.float64)
            assert Hk.dtype == np.float64
        h = H.copy()
        for i in range(4):
            Hk = h.Hk(dtype=np.float32)
            assert Hk.dtype == np.float32

    def test_Hk2(self, setup):
        H = setup.HS.copy()
        H.construct([(0.1, 1.5), ((1., 2.), (0.1, 0.2))])
        # The loops ensures that we loop over all selector
        # items
        h = H.copy()
        for i in range(4):
            Hk = h.Hk(k=[0.15, 0.15, 0.15])
            assert Hk.dtype == np.complex128
        h = H.copy()
        for i in range(4):
            Hk = h.Hk(k=[0.15, 0.15, 0.15], dtype=np.complex64)
            assert Hk.dtype == np.complex64

    @pytest.mark.xfail(raises=ValueError)
    def test_Hk3(self, setup):
        H = setup.HS.copy()
        H.construct([(0.1, 1.5), ((1., 2.), (0.1, 0.2))])
        # The loops ensures that we loop over all selector
        # items
        grabbed = True
        h = H.copy()
        for i in range(4):
            try:
                Hk = h.Hk(k=[0.15, 0.15, 0.15], dtype=np.float64)
                grabbed = False
            except ValueError:
                grabbed = grabbed and True
        if grabbed:
            raise ValueError('all grabbed')

    @pytest.mark.xfail(raises=ValueError)
    def test_Hk4(self, setup):
        H = setup.HS.copy()
        H.construct([(0.1, 1.5), ((1., 2.), (0.1, 0.2))])
        # The loops ensures that we loop over all selector
        # items
        grabbed = True
        h = H.copy()
        for i in range(4):
            try:
                Hk = h.Hk(k=[0.15, 0.15, 0.15], dtype=np.float32)
                grabbed = False
            except ValueError:
                grabbed = grabbed and True
        if grabbed:
            raise ValueError('all grabbed')

    def test_Hk5(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), (1., 0.1)])
        # The loops ensures that we loop over all selector
        # items
        h = H.copy()
        for i in range(4):
            Hk = h.Hk(k=[0.15, 0.15, 0.15])
            assert Hk.dtype == np.complex128
            Sk = h.Sk(k=[0.15, 0.15, 0.15])
            assert Sk.dtype == np.float64

    @pytest.mark.xfail(raises=ValueError)
    def test_construct_raise(self, setup):
        # Test that construct fails with more than one
        # orbital
        setup.H2.construct([(0.1, 1.5), (1., 0.1)])

    def test_getitem1(self, setup):
        H = setup.H
        # graphene Hamiltonian
        H.construct([(0.1, 1.5), (0.1, 0.2)])
        # Assert all connections
        assert H[0, 0] == 0.1
        assert H[0, 1] == 0.2
        assert H[0, 1, (-1, 0)] == 0.2
        assert H[0, 1, (0, -1)] == 0.2
        assert H[1, 0] == 0.2
        assert H[1, 1] == 0.1
        assert H[1, 0, (1, 0)] == 0.2
        assert H[1, 0, (0, 1)] == 0.2
        H[0, 0, (0, 1)] = 0.3
        assert H[0, 0, (0, 1)] == 0.3
        H[0, 1, (0, 1)] = -0.2
        assert H[0, 1, (0, 1)] == -0.2
        H.empty()

    def test_delitem1(self, setup):
        H = setup.H
        H.construct([(0.1, 1.5), (0.1, 0.2)])
        assert H[0, 1] == 0.2
        del H[0, 1]
        assert H[0, 1] == 0.0
        H.empty()

    def test_fromsp1(self, setup):
        setup.H.construct([(0.1, 1.5), (1., 0.1)])
        csr = setup.H.tocsr(0)
        H = Hamiltonian.fromsp(setup.H.geom, csr)
        assert H.spsame(setup.H)
        setup.H.empty()

    @pytest.mark.xfail(raises=ValueError)
    def test_fromsp2(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), (1., 0.1)])
        csr = H.tocsr(0)
        Hamiltonian.fromsp(setup.H.geom.tile(2, 0), csr)

    def test_fromsp3(self, setup):
        H = setup.HS.copy()
        H.construct([(0.1, 1.5), ([1., 1.], [0.1, 0])])
        h = Hamiltonian.fromsp(H.geom.copy(), H.tocsr(0), H.tocsr(1))
        assert H.spsame(h)

    def test_op1(self, setup):
        g = Geometry([[i, 0, 0] for i in range(100)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.int32)
        for i in range(10):
            j = range(i*4, i*4+3)
            H[0, j] = i

            # i+
            H += 1
            for jj in j:
                assert H[0, jj] == i+1
                assert H[1, jj] == 0

            # i-
            H -= 1
            for jj in j:
                assert H[0, jj] == i
                assert H[1, jj] == 0

            # i*
            H *= 2
            for jj in j:
                assert H[0, jj] == i*2
                assert H[1, jj] == 0

            # //
            H //= 2
            for jj in j:
                assert H[0, jj] == i
                assert H[1, jj] == 0

            # i**
            H **= 2
            for jj in j:
                assert H[0, jj] == i**2
                assert H[1, jj] == 0

    def test_op2(self, setup):
        g = Geometry([[i, 0, 0] for i in range(100)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.int32)
        for i in range(10):
            j = range(i*4, i*4+3)
            H[0, j] = i

            # +
            s = H + 1
            for jj in j:
                assert s[0, jj] == i+1
                assert H[0, jj] == i
                assert s[1, jj] == 0

            # -
            s = H - 1
            for jj in j:
                assert s[0, jj] == i-1
                assert H[0, jj] == i
                assert s[1, jj] == 0

            # -
            s = 1 - H
            for jj in j:
                assert s[0, jj] == 1-i
                assert H[0, jj] == i
                assert s[1, jj] == 0

            # *
            s = H * 2
            for jj in j:
                assert s[0, jj] == i*2
                assert H[0, jj] == i
                assert s[1, jj] == 0

            # //
            s = s // 2
            for jj in j:
                assert s[0, jj] == i
                assert H[0, jj] == i
                assert s[1, jj] == 0

            # **
            s = H ** 2
            for jj in j:
                assert s[0, jj] == i**2
                assert H[0, jj] == i
                assert s[1, jj] == 0

            # ** (r)
            s = 2 ** H
            for jj in j:
                assert s[0, jj], 2 ** H[0 == jj]
                assert H[0, jj] == i
                assert s[1, jj] == 0

    def test_op3(self, setup):
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
            assert h.dtype == np.int32
            h = func(1.)
            assert h.dtype == np.float64
            if op != 'pow':
                h = func(1.j)
                assert h.dtype == np.complex128

        H = H.copy(dtype=np.float64)
        for op in ['add', 'sub', 'mul', 'pow']:
            func = getattr(H, '__{}__'.format(op))
            h = func(1)
            assert h.dtype == np.float64
            h = func(1.)
            assert h.dtype == np.float64
            if op != 'pow':
                h = func(1.j)
                assert h.dtype == np.complex128

        H = H.copy(dtype=np.complex128)
        for op in ['add', 'sub', 'mul', 'pow']:
            func = getattr(H, '__{}__'.format(op))
            h = func(1)
            assert h.dtype == np.complex128
            h = func(1.)
            assert h.dtype == np.complex128
            if op != 'pow':
                h = func(1.j)
                assert h.dtype == np.complex128

    def test_op4(self, setup):
        g = Geometry([[i, 0, 0] for i in range(100)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.int32)
        # Create initial stuff
        for i in range(10):
            j = range(i*4, i*4+3)
            H[0, j] = i

        h = 1 + H
        assert h.dtype == np.int32
        h = 1. + H
        assert h.dtype == np.float64
        h = 1.j + H
        assert h.dtype == np.complex128

        h = 1 - H
        assert h.dtype == np.int32
        h = 1. - H
        assert h.dtype == np.float64
        h = 1.j - H
        assert h.dtype == np.complex128

        h = 1 * H
        assert h.dtype == np.int32
        h = 1. * H
        assert h.dtype == np.float64
        h = 1.j * H
        assert h.dtype == np.complex128

        h = 1 ** H
        assert h.dtype == np.int32
        h = 1. ** H
        assert h.dtype == np.float64
        h = 1.j ** H
        assert h.dtype == np.complex128

    def test_cut1(self, setup):
        # Test of eigenvalues using a cut
        # Hamiltonian
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        Hg = Hamiltonian(setup.g)
        Hg.construct([R, param])
        g = setup.g.tile(2, 0).tile(2, 1)
        H = Hamiltonian(g)
        H.construct([R, param])
        # Create cut Hamiltonian
        Hc = H.cut(2, 1).cut(2, 0)
        eigc = Hc.eigh()
        eigg = Hg.eigh()
        assert np.allclose(eigc, eigg)
        assert np.allclose(Hg.eigh(), Hc.eigh())
        del Hc, H

    def test_cut2(self, setup):
        # Test of eigenvalues using a cut
        # Hamiltonian
        R, param = [0.1, 1.5], [(1., 1.), (0.1, 0.1)]

        # Create reference
        Hg = Hamiltonian(setup.g, orthogonal=False)
        Hg.construct([R, param])

        g = setup.g.tile(2, 0).tile(2, 1)
        H = Hamiltonian(g, orthogonal=False)
        H.construct([R, param])
        # Create cut Hamiltonian
        Hc = H.cut(2, 1).cut(2, 0)
        eigc = Hc.eigh()
        eigg = Hg.eigh()
        assert np.allclose(Hg.eigh(), Hc.eigh())
        del Hc, H

    def test_eig1(self, setup):
        # Test of eigenvalues
        R, param = [0.1, 1.5], [1., 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param), eta=True)
        eig1 = H.eigh()
        for i in range(2):
            assert np.allclose(eig1, H.eigh())
        H.eigsh(n=4)
        H.empty()
        del H

    def test_eig2(self, setup):
        # Test of eigenvalues
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((1., 1.), (0.1, 0.1))])
        eig1 = HS.eigh()
        for i in range(2):
            assert np.allclose(eig1, HS.eigh())
        setup.HS.empty()

    def test_eig3(self, setup):
        setup.HS.construct([(0.1, 1.5), ((1., 1.), (0.1, 0.1))])
        BS = PathBZ(setup.HS, [[0, 0, 0], [0.5, 0.5, 0]], 10)
        eigs = BS.array().eigh()
        assert len(BS) == eigs.shape[0]
        assert len(setup.HS) == eigs.shape[1]
        eig2 = np.array([eig for eig in BS.yields().eigh()])
        assert np.allclose(eigs, eig2)
        setup.HS.empty()

    def test_spin1(self, setup):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.int32, spin=Spin.POLARIZED)
        for i in range(10):
            j = range(i*4, i*4+3)
            H[0, j] = (i, i*2)

        H2 = Hamiltonian(g, 2, dtype=np.int32)
        for i in range(10):
            j = range(i*4, i*4+3)
            H2[0, j] = (i, i*2)
        assert H.spsame(H2)

    def test_spin2(self, setup):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.int32, spin=Spin.POLARIZED)
        for i in range(10):
            j = range(i*4, i*4+3)
            H[0, j] = (i, i*2)

        H2 = Hamiltonian(g, 2, dtype=np.int32)
        for i in range(10):
            j = range(i*4, i*4+3)
            H2[0, j] = (i, i*2)
        assert H.spsame(H2)

        H2 = Hamiltonian(g, Spin(Spin.POLARIZED), dtype=np.int32)
        for i in range(10):
            j = range(i*4, i*4+3)
            H2[0, j] = (i, i*2)
        assert H.spsame(H2)

        H2 = Hamiltonian(g, Spin('polarized'), dtype=np.int32)
        for i in range(10):
            j = range(i*4, i*4+3)
            H2[0, j] = (i, i*2)
        assert H.spsame(H2)

    def test_non_colinear1(self, setup):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.float64, spin=Spin.NONCOLINEAR)
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
            assert np.allclose(H.eigh(), eig1)
        assert len(eig1) == len(H)

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
        assert H1.spsame(H)
        eig1 = H1.eigh()
        # Check TimeSelector
        for i in range(2):
            assert np.allclose(H1.eigh(), eig1)
        assert np.allclose(H.eigh(), H1.eigh())

    def test_so1(self, setup):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=[100])
        H = Hamiltonian(g, dtype=np.float64, spin=Spin.SPINORBIT)
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
            assert np.allclose(H.eigh(), eig1)
        assert len(H.eigh()) == len(H)

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
        assert H1.spsame(H)
        eig1 = H1.eigh()
        # Check TimeSelector
        for i in range(2):
            assert np.allclose(H1.eigh(), eig1)
        assert np.allclose(H.eigh(), H1.eigh())

    def test_finalized(self, setup):
        assert not setup.H.finalized
        setup.H.H[0, 0] = 1.
        setup.H.finalize()
        assert setup.H.finalized
        assert setup.H.nnz == 1
        setup.H.empty()
        assert not setup.HS.finalized
        setup.HS.H[0, 0] = 1.
        setup.HS.S[0, 0] = 1.
        setup.HS.finalize()
        assert setup.HS.finalized
        assert setup.HS.nnz == 1
        setup.HS.empty()

    @pytest.mark.slow
    def test_tile1(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        Hg = Hamiltonian(setup.g.tile(2, 0).tile(2, 1).tile(2, 2))
        Hg.construct([R, param])
        Hg.finalize()
        H = Hamiltonian(setup.g)
        H.construct([R, param])
        H = H.tile(2, 0).tile(2, 1).tile(2, 2)
        assert Hg.spsame(H)
        H.finalize()
        Hg.finalize()
        assert np.allclose(H._csr._D, Hg._csr._D)

    @pytest.mark.slow
    def test_tile2(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        Hg = Hamiltonian(setup.g.tile(2, 0))
        Hg.construct([R, param])
        Hg.finalize()
        H = Hamiltonian(setup.g)
        H.construct([R, param])
        H = H.tile(2, 0)
        assert Hg.spsame(H)
        H.finalize()
        Hg.finalize()
        assert np.allclose(H._csr._D, Hg._csr._D)

    @pytest.mark.slow
    def test_tile3(self, setup):
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
        assert HG.spsame(H)
        H.finalize()
        HG.finalize()
        assert np.allclose(H._csr._D, HG._csr._D)

    def test_tile4(self, setup):
        def func(self, ia, idxs, idxs_xyz=None):
            idx = self.geom.close(ia, R=[0.1, 1.43], idx=idxs)
            io = self.geom.a2o(ia)
            # Set on-site on first and second orbital
            odx = self.geom.a2o(idx[0])
            self[io, odx] = -1.
            self[io+1, odx+1] = 1.

            # Set connecting
            odx = self.geom.a2o(idx[1])
            self[io, odx] = 0.2
            self[io, odx+1] = 0.01
            self[io+1, odx] = 0.01
            self[io+1, odx+1] = 0.3

        setup.H2.construct(func)
        Hbig = setup.H2.tile(3, 0).tile(3, 1)

        gbig = setup.H2.geom.tile(3, 0).tile(3, 1)
        H = Hamiltonian(gbig)
        H.construct(func)
        assert H.spsame(Hbig)
        H.finalize()
        Hbig.finalize()
        assert np.allclose(H._csr._D, Hbig._csr._D)
        setup.H2.empty()

    @pytest.mark.slow
    def test_repeat1(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        Hg = Hamiltonian(setup.g.repeat(2, 0))
        Hg.construct([R, param])
        Hg.finalize()
        H = Hamiltonian(setup.g)
        H.construct([R, param])
        H = H.repeat(2, 0)
        assert Hg.spsame(H)
        H.finalize()
        Hg.finalize()
        assert np.allclose(H._csr._D, Hg._csr._D)

    @pytest.mark.slow
    def test_repeat2(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        Hg = Hamiltonian(setup.g.repeat(2, 0).repeat(2, 1).repeat(2, 2))
        Hg.construct([R, param])
        Hg.finalize()
        H = Hamiltonian(setup.g)
        H.construct([R, param])
        H = H.repeat(2, 0).repeat(2, 1). repeat(2, 2)
        assert Hg.spsame(H)
        H.finalize()
        Hg.finalize()
        assert np.allclose(H._csr._D, Hg._csr._D)

    @pytest.mark.slow
    def test_repeat3(self, setup):
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
        assert HG.spsame(H)
        H.finalize()
        HG.finalize()
        assert np.allclose(H._csr._D, HG._csr._D)

    @pytest.mark.slow
    def test_repeat4(self, setup):
        def func(self, ia, idxs, idxs_xyz=None):
            idx = self.geom.close(ia, R=[0.1, 1.43], idx=idxs)
            io = self.geom.a2o(ia)
            # Set on-site on first and second orbital
            odx = self.geom.a2o(idx[0])
            self[io, odx] = -1.
            self[io+1, odx+1] = 1.

            # Set connecting
            odx = self.geom.a2o(idx[1])
            self[io, odx] = 0.2
            self[io, odx+1] = 0.01
            self[io+1, odx] = 0.01
            self[io+1, odx+1] = 0.3

        setup.H2.construct(func)
        Hbig = setup.H2.repeat(3, 0).repeat(3, 1)

        gbig = setup.H2.geom.repeat(3, 0).repeat(3, 1)
        H = Hamiltonian(gbig)
        H.construct(func)

        assert H.spsame(Hbig)
        H.finalize()
        Hbig.finalize()
        assert np.allclose(H._csr._D, Hbig._csr._D)
        setup.H2.empty()

    def test_sub1(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        H = Hamiltonian(setup.g)
        H.construct([R, param])
        H.finalize()
        # Tiling in this direction will not introduce
        # any new connections.
        # So tiling and removing is a no-op (but
        # increases vacuum in 3rd lattice vector)
        Hg = Hamiltonian(setup.g.tile(2, 2))
        Hg.construct([R, param])
        Hg = Hg.sub(range(len(setup.g)))
        Hg.finalize()
        assert Hg.spsame(H)
        assert len(Hg) == len(setup.g)

    def test_set_nsc1(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        H = Hamiltonian(setup.g.copy())
        H.construct([R, param])
        h = H.copy()
        H.set_nsc(nsc=[None, 1, 1])
        assert H.nnz == 6
        H.set_nsc(nsc=[1, None, 1])
        assert H.nnz == 4
        h.set_nsc(nsc=[1, None, 1])
        assert h.nnz == 6

        g = setup.g.copy()
        g.set_nsc([1] * 3)
        Hg = Hamiltonian(g)
        Hg.construct([R, param])
        assert Hg.nnz == 4
        assert Hg.spsame(H)

    def test_shift1(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]
        H = Hamiltonian(setup.g.copy())
        H.construct([R, param])
        eig0 = H.eigh()[0]
        H.shift(0.2)
        assert H.eigh()[0] == pytest.approx(eig0 + 0.2)

    def test_shift2(self, setup):
        R, param = [0.1, 1.5], [(1., 1.), (0.1, 0.1)]
        H = Hamiltonian(setup.g.copy(), orthogonal=False)
        H.construct([R, param])
        eig0 = H.eigh()[0]
        H.shift(0.2)
        assert H.eigh()[0] == pytest.approx(eig0 + 0.2)

    def test_edges1(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]
        H = Hamiltonian(setup.g)
        H.construct([R, param])
        assert len(H.edges(0)) == 3

    @pytest.mark.xfail(raises=ValueError)
    def test_edges2(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]
        H = Hamiltonian(setup.g)
        H.construct([R, param])
        H.edges()

    def test_edges3(self, setup):
        def func(self, ia, idxs, idxs_xyz=None):
            idx = self.geom.close(ia, R=[0.1, 1.43], idx=idxs)
            io = self.geom.a2o(ia)
            # Set on-site on first and second orbital
            odx = self.geom.a2o(idx[0])
            self[io, odx] = -1.
            self[io+1, odx+1] = 1.

            # Set connecting
            odx = self.geom.a2o(idx[1])
            self[io, odx] = 0.2
            self[io, odx+1] = 0.01
            self[io+1, odx] = 0.01
            self[io+1, odx+1] = 0.3

        H2 = setup.H2.copy()
        H2.construct(func)
        # first atom
        assert len(H2.edges(0)) == 3
        # orbitals of first atom
        edge = H2.edges(orbital=[0, 1])
        assert len(edge) == 6
        assert len(H2.geom.o2a(edge, uniq=True)) == 3

        # first orbital on first two atoms
        edge = H2.edges(orbital=[0, 2])
        # The 1, 3 are still on the first two atoms, but aren't
        # excluded. Hence they are both there
        assert len(edge) == 10
        assert len(H2.geom.o2a(edge, uniq=True)) == 6

        # first orbital on first two atoms
        edge = H2.edges(orbital=[0, 2], exclude=[0, 1, 2, 3])
        assert len(edge) == 8
        assert len(H2.geom.o2a(edge, uniq=True)) == 4
