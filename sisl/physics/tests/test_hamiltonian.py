import pytest

from functools import partial
import warnings
import numpy as np
from scipy.linalg import block_diag

from sisl import Geometry, Atom, SuperCell, Hamiltonian, Spin, BandStructure, MonkhorstPack, BrillouinZone
from sisl import get_distribution
from sisl import oplist
from sisl import Grid, SphericalOrbital, SislError
from sisl.physics.electron import berry_phase, spin_squared, conductivity


pytestmark = pytest.mark.hamiltonian


@pytest.fixture
def setup():
    class t():
        def __init__(self):
            bond = 1.42
            sq3h = 3.**.5 * 0.5
            self.sc = SuperCell(np.array([[1.5, sq3h, 0.],
                                          [1.5, -sq3h, 0.],
                                          [0., 0., 10.]], np.float64) * bond, nsc=[3, 3, 1])

            C = Atom(Z=6, R=[bond * 1.01])
            self.g = Geometry(np.array([[0., 0., 0.],
                                        [1., 0., 0.]], np.float64) * bond,
                              atoms=C, sc=self.sc)
            self.H = Hamiltonian(self.g)
            self.HS = Hamiltonian(self.g, orthogonal=False)

            C = Atom(Z=6, R=[bond * 1.01] * 2)
            self.g2 = Geometry(np.array([[0., 0., 0.],
                                         [1., 0., 0.]], np.float64) * bond,
                               atoms=C, sc=self.sc)
            self.H2 = Hamiltonian(self.g2)
            self.HS2 = Hamiltonian(self.g2, orthogonal=False)
    return t()


def _to_voight(m):
    idx1 = [0, 1, 2, 1, 0, 0]
    idx2 = [0, 1, 2, 2, 2, 1]
    return m[:, idx1, idx2]


@pytest.mark.hamiltonian
class TestHamiltonian:

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
        for ia in setup.H.geometry:
            # Find atoms close to 'ia'
            idx = setup.H.geometry.close(ia, R=(0.1, 1.5))
            setup.H[ia, idx[0]] = 1.
            setup.H[ia, idx[1]] = 0.1
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
        for io, jo in setup.HS:
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
        for io, jo in setup.HS:
            nnz = nnz + 1
        assert nnz == setup.HS.nnz
        assert nnz == 1
        setup.HS.empty()

    @pytest.mark.filterwarnings('ignore', category=np.ComplexWarning)
    def test_Hk1(self, setup):
        H = setup.HS.copy()
        H.construct([(0.1, 1.5), ((1., 2.), (0.1, 0.2))])
        h = H.copy()
        assert h.Hk().dtype == np.float64
        assert h.Sk().dtype == np.float64
        h = H.copy()
        assert h.Hk(dtype=np.complex64).dtype == np.complex64
        h = H.copy()
        assert h.Hk(dtype=np.complex128).dtype == np.complex128
        h = H.copy()
        assert h.Hk(dtype=np.float64).dtype == np.float64
        h = H.copy()
        assert h.Hk(dtype=np.float32).dtype == np.float32

    def test_Hk2(self, setup):
        H = setup.HS.copy()
        H.construct([(0.1, 1.5), ((1., 2.), (0.1, 0.2))])
        h = H.copy()
        Hk = h.Hk(k=[0.15, 0.15, 0.15])
        assert Hk.dtype == np.complex128
        h = H.copy()
        Hk = h.Hk(k=[0.15, 0.15, 0.15], dtype=np.complex64)
        assert Hk.dtype == np.complex64

    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_Hk5(self, setup, dtype):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), (1., 0.1)])
        Hk = H.Hk(k=[0.15, 0.15, 0.15], dtype=dtype)
        assert Hk.dtype == dtype
        Sk = H.Sk(k=[0.15, 0.15, 0.15])
        # orthogonal basis sets always returns a diagonal in float64
        assert Sk.dtype == np.float64
        Sk = H.Sk(k=[0.15, 0.15, 0.15], dtype=dtype)
        # orthogonal basis sets always returns a diagonal in float64
        assert Sk.dtype == dtype

    @pytest.mark.parametrize("k", [[0, 0, 0], [0.15, 0.15, 0.15]])
    def test_Hk_format(self, setup, k):
        H = setup.HS.copy()
        H.construct([(0.1, 1.5), ((1., 2.), (0.1, 0.2))])
        csr = H.Hk(k, format='csr').toarray()
        mat = H.Hk(k, format='matrix')
        arr = H.Hk(k, format='array')
        coo = H.Hk(k, format='coo').toarray()
        assert np.allclose(csr, mat)
        assert np.allclose(csr, arr)
        assert np.allclose(csr, coo)

    @pytest.mark.parametrize("orthogonal", [True, False])
    @pytest.mark.parametrize("gauge", ["R", "r"])
    @pytest.mark.parametrize("spin", ["unpolarized", "polarized", "non-collinear", "spin-orbit"])
    @pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
    def test_format_sc(self, orthogonal, gauge, spin, dtype):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=SuperCell([10, 1, 5.], nsc=[3, 3, 1]))
        H = Hamiltonian(g, dtype=np.float64, orthogonal=orthogonal, spin=Spin(spin))
        nd = H._csr._D.shape[-1]
        # this will correctly account for the double size for NC/SOC
        no = len(H)
        no_s = H.geometry.no_s
        for ia in g:
            idx = g.close(ia, R=(0.1, 1.01))[1]
            H[ia, ia] = 1.
            H[ia, idx] = np.random.rand(nd)
        if dtype == np.complex64:
            atol = 1e-6
            rtol = 1e-12
        else:
            atol = 1e-9
            rtol = 1e-15
        allclose = partial(np.allclose, atol=atol, rtol=rtol)

        H = (H + H.transpose(hermitian=True)) / 2
        n_s = H.geometry.sc.n_s

        for k in [[0, 0, 0], [0.15, 0.1, 0.05]]:
            for attr, kwargs in [("Hk", {"gauge": gauge}), ("Sk", {})]:
                Mk = getattr(H, attr)
                csr = Mk(k, format='csr', **kwargs, dtype=dtype)
                sc_csr1 = Mk(k, format='sc:csr', **kwargs, dtype=dtype)
                sc_csr2 = Mk(k, format='sc', **kwargs, dtype=dtype)
                sc_mat = Mk(k, format='sc:array', **kwargs, dtype=dtype)
                mat = sc_mat.reshape(no, n_s, no).sum(1)

                assert sc_mat.shape == sc_csr1.shape
                assert allclose(csr.toarray(), mat)
                assert allclose(sc_csr1.toarray(), sc_csr2.toarray())
                for isc in range(n_s):
                    csr -= sc_csr1[:, isc * no: (isc + 1) * no]
                assert allclose(csr.toarray(), 0.)

    def test_construct_raise_default(self, setup):
        # Test that construct fails with more than one
        # orbital
        with pytest.raises(ValueError):
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
        H = Hamiltonian.fromsp(setup.H.geometry, csr)
        assert H.spsame(setup.H)
        setup.H.empty()

    def test_fromsp2(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), (1., 0.1)])
        csr = H.tocsr(0)
        with pytest.raises(ValueError):
            Hamiltonian.fromsp(setup.H.geometry.tile(2, 0), csr)

    def test_fromsp3(self, setup):
        H = setup.HS.copy()
        H.construct([(0.1, 1.5), ([1., 1.], [0.1, 0])])
        h = Hamiltonian.fromsp(H.geometry.copy(), H.tocsr(0), H.tocsr(1))
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
            func = getattr(H, f'__{op}__')
            h = func(1)
            assert h.dtype == np.int32
            h = func(1.)
            assert h.dtype == np.float64
            if op != 'pow':
                h = func(1.j)
                assert h.dtype == np.complex128

        H = H.copy(dtype=np.float64)
        for op in ['add', 'sub', 'mul', 'pow']:
            func = getattr(H, f'__{op}__')
            h = func(1)
            assert h.dtype == np.float64
            h = func(1.)
            assert h.dtype == np.float64
            if op != 'pow':
                h = func(1.j)
                assert h.dtype == np.complex128

        H = H.copy(dtype=np.complex128)
        for op in ['add', 'sub', 'mul', 'pow']:
            func = getattr(H, f'__{op}__')
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

    def test_eigh_vs_eig(self, setup):
        # Test of eigenvalues
        R, param = [0.1, 1.5], [1., 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param), eta=True)
        eig1 = H.eigh(dtype=np.complex64)
        eig2 = np.sort(H.eig(dtype=np.complex64).real)
        eig3 = np.sort(H.eig(eigvals_only=False, dtype=np.complex64)[0].real)
        assert np.allclose(eig1, eig2, atol=1e-5)
        assert np.allclose(eig1, eig3, atol=1e-5)

        eig1 = H.eigh([0.01] * 3, dtype=np.complex64)
        eig2 = np.sort(H.eig([0.01] * 3, dtype=np.complex64).real)
        eig3 = np.sort(H.eig([0.01] * 3, eigvals_only=False, dtype=np.complex64)[0].real)
        assert np.allclose(eig1, eig2, atol=1e-5)
        assert np.allclose(eig1, eig3, atol=1e-5)

    def test_eig1(self, setup):
        # Test of eigenvalues
        R, param = [0.1, 1.5], [1., 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param), eta=True)
        eig1 = H.eigh(dtype=np.complex64)
        assert np.allclose(eig1, H.eigh(dtype=np.complex128))
        H.eigsh(n=4)
        H.empty()
        del H

    def test_eig2(self, setup):
        # Test of eigenvalues
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((1., 1.), (0.1, 0.1))])
        eig1 = HS.eigh(dtype=np.complex64)
        assert np.allclose(eig1, HS.eigh(dtype=np.complex128))
        setup.HS.empty()

    def test_eig3(self, setup):
        setup.HS.construct([(0.1, 1.5), ((1., 1.), (0.1, 0.1))])
        BS = BandStructure(setup.HS, [[0, 0, 0], [0.5, 0.5, 0]], 10)
        eigs = BS.apply.array.eigh()
        assert len(BS) == eigs.shape[0]
        assert len(setup.HS) == eigs.shape[1]
        eig2 = np.array([eig for eig in BS.apply.iter.eigh()])
        assert np.allclose(eigs, eig2)
        setup.HS.empty()

    def test_eig4(self, setup):
        # Test of eigenvalues vs eigenstate class
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((1., 1.), (0.1, 0.1))])

        for k in ([0] *3, [0.2] * 3):
            e, v = HS.eigh(k, eigvals_only=False)
            es = HS.eigenstate(k)
            assert np.allclose(e, es.eig)
            assert np.allclose(v, es.state.T)
            assert np.allclose(es.norm2(), 1)
            assert np.allclose(es.inner(diagonal=False) - np.eye(len(es)), 0)

            assert es.inner(es.sub(0)).shape == (1, )
            assert es.inner(es.sub(0), diagonal=False).shape == (len(es), 1)

            eig1 = HS.eigh(k)
            eig2 = np.sort(HS.eig(k).real)
            eig3 = np.sort(HS.eig(k, eigvals_only=False)[0].real)
            assert np.allclose(eig1, eig2, atol=1e-5)
            assert np.allclose(eig1, eig3, atol=1e-5)

    def test_gauge_eig(self, setup):
        # Test of eigenvalues
        R, param = [0.1, 1.5], [1., 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))

        k = [0.1] * 3
        es1 = H.eigenstate(k, gauge='R')
        es2 = H.eigenstate(k, gauge='r')
        assert np.allclose(es1.eig, es2.eig)
        assert not np.allclose(es1.state, es2.state)

        es1 = H.eigenstate(k, gauge='R', dtype=np.complex64)
        es2 = H.eigenstate(k, gauge='r', dtype=np.complex64)
        assert np.allclose(es1.eig, es2.eig)
        assert not np.allclose(es1.state, es2.state)

    def test_gauge_velocity(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))

        k = [0.1] * 3
        es1 = H.eigenstate(k, gauge='R')
        es2 = H.eigenstate(k, gauge='r')
        assert np.allclose(es1.velocity(), es2.velocity())

        es2.change_gauge('R')
        assert np.allclose(es1.velocity(), es2.velocity())

        es2.change_gauge('r')
        es1.change_gauge('r')
        v1 = es1.velocity()
        v2 = es2.velocity()
        assert np.allclose(v1, v2)

        # Projected velocity
        pv1 = es1.velocity(project=True)
        pv2 = es2.velocity(project=True)
        assert np.allclose(pv1.sum(1), v2)
        assert np.allclose(pv2.sum(1), v1)
        # since degenerate states *could* swap states
        # we can't for sure compare states
        # This test is one of those cases
        # hence the below is disabled
        #assert np.allclose(pv1, pv2)

    def test_berry_phase(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))
        bz = BandStructure.param_circle(H, 20, 0.01, [0, 0, 1], [1/3] * 3)
        berry_phase(bz)
        berry_phase(bz, sub=0)
        berry_phase(bz, eigvals=True, sub=0)

    def test_berry_phase_fail_sc(self, setup):
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        bz = BandStructure.param_circle(H.geometry.sc, 20, 0.01, [0, 0, 1], [1/3] * 3)
        with pytest.raises(SislError):
            berry_phase(bz)

    def test_berry_phase_loop(self, setup):
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        bz1 = BandStructure.param_circle(H, 20, 0.01, [0, 0, 1], [1/3] * 3)
        bz2 = BandStructure.param_circle(H, 20, 0.01, [0, 0, 1], [1/3] * 3, loop=True)
        assert np.allclose(berry_phase(bz1), berry_phase(bz2))

    def test_berry_phase_zak(self):
        # SSH model, topological cell
        g = Geometry([[-.6, 0, 0], [0.6, 0, 0]], Atom(1, 1.001), sc=[2, 10, 10])
        g.set_nsc([3, 1, 1])
        H = Hamiltonian(g)
        H.construct([(0.1, 1.0, 1.5), (0, 1., 0.5)])
        # Contour
        k = np.linspace(0.0, 1.0, 101)
        K = np.zeros([k.size, 3])
        K[:, 0] = k
        bz = BrillouinZone(H, K)
        assert np.allclose(np.abs(berry_phase(bz, sub=0, method='zak')), np.pi)
        # Just to do the other branch
        berry_phase(bz, method='zak')

    def test_berry_phase_method_fail(self):
        # wrong method keyword
        g = Geometry([[-.6, 0, 0], [0.6, 0, 0]], Atom(1, 1.001), sc=[2, 10, 10])
        g.set_nsc([3, 1, 1])
        H = Hamiltonian(g)
        H.construct([(0.1, 1.0, 1.5), (0, 1., 0.5)])
        # Contour
        k = np.linspace(0.0, 1.0, 101)
        K = np.zeros([k.size, 3])
        K[:, 0] = k
        bz = BrillouinZone(H, K)
        with pytest.raises(ValueError):
            berry_phase(bz, method='unknown')

    def test_berry_curvature(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))

        k = [0.1] * 3
        ie1 = H.eigenstate(k, gauge='R').berry_curvature()
        ie2 = H.eigenstate(k, gauge='r').berry_curvature()
        assert np.allclose(ie1, ie2)

    def test_conductivity(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))

        mp = MonkhorstPack(H, [11, 11, 1])
        cond = conductivity(mp)

    def test_gauge_inv_eff(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))

        k = [0.1] * 3
        ie1 = H.eigenstate(k, gauge='R').inv_eff_mass_tensor()
        ie2 = H.eigenstate(k, gauge='r').inv_eff_mass_tensor()
        str(ie1)
        str(ie2)
        assert np.allclose(ie1, ie2)

    def test_change_gauge(self, setup):
        # Test of eigenvalues vs eigenstate class
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((1., 1.), (0.1, 0.1))])
        es = HS.eigenstate()
        es2 = es.copy()
        es2.change_gauge('r')
        assert np.allclose(es2.state, es.state)

        es = HS.eigenstate(k=(0.2, 0.2, 0.2))
        es2 = es.copy()
        es2.change_gauge('r')
        assert not np.allclose(es2.state, es.state)
        es2.change_gauge('R')
        assert np.allclose(es2.state, es.state)

    def test_expectation_value(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), ((1., 1.))])
        D = np.ones(len(H))
        I = np.identity(len(H))
        for k in ([0] *3, [0.2] * 3):
            es = H.eigenstate(k)

            d = es.expectation(D)
            assert np.allclose(d, D)
            d = es.expectation(D, diag=False)
            assert np.allclose(d, I)

            d = es.expectation(I)
            assert np.allclose(d, D)
            d = es.expectation(I, diag=False)
            assert np.allclose(d, I)

    def test_velocity_orthogonal(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), ((1., 1.))])
        E = np.linspace(-4, 4, 1000)
        for k in ([0] *3, [0.2] * 3):
            es = H.eigenstate(k)
            v = es.velocity()
            vsub = es.sub([0]).velocity()
            assert np.allclose(v[0, :], vsub)

    def test_velocity_nonorthogonal(self, setup):
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((1., 1.), (0.1, 0.1))])
        E = np.linspace(-4, 4, 1000)
        for k in ([0] *3, [0.2] * 3):
            es = HS.eigenstate(k)
            v = es.velocity()
            vsub = es.sub([0]).velocity()
            assert np.allclose(v[0, :], vsub)

    def test_velocity_matrix_orthogonal(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), ((1., 1.))])
        E = np.linspace(-4, 4, 1000)
        for k in ([0] *3, [0.2] * 3):
            es = H.eigenstate(k)
            v = es.velocity_matrix()
            vsub = es.sub([0, 1]).velocity_matrix()
            assert np.allclose(v[:2, :2, :], vsub)

    def test_velocity_matrix_nonorthogonal(self, setup):
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((1., 1.), (0.1, 0.1))])
        E = np.linspace(-4, 4, 1000)
        for k in ([0] *3, [0.2] * 3):
            es = HS.eigenstate(k)
            v = es.velocity_matrix()
            vsub = es.sub([0, 1]).velocity_matrix()
            assert np.allclose(v[:2, :2, :], vsub)

    def test_inv_eff_mass_tensor_orthogonal(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), ((1., 1.))])
        E = np.linspace(-4, 4, 1000)
        for k in ([0] *3, [0.2] * 3):
            es = H.eigenstate(k)
            v = es.inv_eff_mass_tensor()
            vsub = es.sub([0]).inv_eff_mass_tensor()
            assert np.allclose(v[0, :], vsub)
            vsub = es.sub([0]).inv_eff_mass_tensor(True)
            assert np.allclose(v[0, :], _to_voight(vsub))

    def test_inv_eff_mass_tensor_nonorthogonal(self, setup):
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((1., 1.), (0.1, 0.1))])
        E = np.linspace(-4, 4, 1000)
        for k in ([0] *3, [0.2] * 3):
            es = HS.eigenstate(k)
            v = es.inv_eff_mass_tensor()
            vsub = es.sub([0]).inv_eff_mass_tensor()
            assert np.allclose(v[0, :], vsub)
            vsub = es.sub([0]).inv_eff_mass_tensor(True)
            assert np.allclose(v[0, :], _to_voight(vsub))

    def test_dos1(self, setup):
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((1., 1.), (0.1, 0.1))])
        E = np.linspace(-4, 4, 1000)
        for k in ([0] *3, [0.2] * 3):
            es = HS.eigenstate(k)
            DOS = es.DOS(E)
            assert DOS.dtype.kind == 'f'
            assert np.allclose(DOS, HS.DOS(E, k))
            assert np.allclose(es.norm2(), 1)
            str(es)

    def test_pdos1(self, setup):
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((0., 1.), (1., 0.1))])
        E = np.linspace(-4, 4, 1000)
        for k in ([0] *3, [0.2] * 3):
            es = HS.eigenstate(k)
            DOS = es.DOS(E, 'lorentzian')
            PDOS = es.PDOS(E, 'lorentzian')
            assert PDOS.dtype.kind == 'f'
            assert PDOS.shape[0] == len(HS)
            assert PDOS.shape[1] == len(E)
            assert np.allclose(PDOS.sum(0), DOS)
            assert np.allclose(PDOS, HS.PDOS(E, k, 'lorentzian'))

    def test_pdos2(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), (0., 0.1)])
        E = np.linspace(-4, 4, 1000)
        for k in ([0] *3, [0.2] * 3):
            es = H.eigenstate(k)
            DOS = es.DOS(E)
            PDOS = es.PDOS(E)
            assert PDOS.dtype.kind == 'f'
            assert np.allclose(PDOS.sum(0), DOS)
            assert np.allclose(PDOS, H.PDOS(E, k))

    def test_pdos3(self, setup):
        # check whether the default S(Gamma) works
        # In this case we will assume an orthogonal
        # basis, however, the basis is not orthogonal.
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((0., 1.), (1., 0.1))])
        E = np.linspace(-4, 4, 1000)
        es = HS.eigenstate()
        es.parent = None
        DOS = es.DOS(E)
        PDOS = es.PDOS(E)
        assert not np.allclose(PDOS.sum(0), DOS)

    def test_pdos4(self, setup):
        # check whether the default S(Gamma) works
        # In this case we will assume an orthogonal
        # basis. If the basis *is* orthogonal, then
        # regardless of k, the PDOS will be correct.
        H = setup.H.copy()
        H.construct([(0.1, 1.5), (0., 0.1)])
        E = np.linspace(-4, 4, 1000)
        es = H.eigenstate()
        es.parent = None
        DOS = es.DOS(E)
        PDOS = es.PDOS(E)
        assert PDOS.dtype.kind == 'f'
        assert np.allclose(PDOS.sum(0), DOS)
        es = H.eigenstate([0.25] * 3)
        DOS = es.DOS(E)
        es.parent = None
        PDOS = es.PDOS(E)
        assert PDOS.dtype.kind == 'f'
        assert np.allclose(PDOS.sum(0), DOS)

    def test_spin1(self, setup):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=SuperCell(100, nsc=[3, 3, 1]))
        H = Hamiltonian(g, dtype=np.int32, spin=Spin.POLARIZED)
        for i in range(10):
            j = range(i*2, i*2+3)
            H[0, j] = (i, i*2)

        H2 = Hamiltonian(g, 2, dtype=np.int32)
        for i in range(10):
            j = range(i*2, i*2+3)
            H2[0, j] = (i, i*2)
        assert H.spsame(H2)

    def test_spin2(self, setup):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=SuperCell(100, nsc=[3, 3, 1]))
        H = Hamiltonian(g, dtype=np.int32, spin=Spin.POLARIZED)
        for i in range(10):
            j = range(i*2, i*2+3)
            H[0, j] = (i, i*2)

        H2 = Hamiltonian(g, 2, dtype=np.int32)
        for i in range(10):
            j = range(i*2, i*2+3)
            H2[0, j] = (i, i*2)
        assert H.spsame(H2)

        H2 = Hamiltonian(g, Spin(Spin.POLARIZED), dtype=np.int32)
        for i in range(10):
            j = range(i*2, i*2+3)
            H2[0, j] = (i, i*2)
        assert H.spsame(H2)

        H2 = Hamiltonian(g, Spin('polarized'), dtype=np.int32)
        for i in range(10):
            j = range(i*2, i*2+3)
            H2[0, j] = (i, i*2)
        assert H.spsame(H2)

    @pytest.mark.parametrize("k", [[0, 0, 0], [0.1, 0, 0]])
    def test_spin_squared(self, setup, k):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=SuperCell(1, nsc=[3, 1, 1]))
        H = Hamiltonian(g, spin=Spin.POLARIZED)
        H.construct(([0.1, 1.1], [[0, 0.1], [1, 1.1]]))
        H[0, 0] = (0.1, 0.)
        H[0, 1] = (0.5, 0.4)
        es_alpha = H.eigenstate(k, spin=0)
        es_beta = H.eigenstate(k, spin=1)

        sup, sdn = spin_squared(es_alpha.state, es_beta.state)
        sup1, sdn1 = H.spin_squared(k)
        assert sup.sum() == pytest.approx(sdn.sum())
        assert np.all(sup1 == sup)
        assert np.all(sdn1 == sdn)
        assert len(sup) == es_alpha.shape[0]
        assert len(sdn) == es_beta.shape[0]

        sup, sdn = spin_squared(es_alpha.sub(range(2)).state, es_beta.state)
        assert sup.sum() == pytest.approx(sdn.sum())
        assert len(sup) == 2
        assert len(sdn) == es_beta.shape[0]

        sup, sdn = spin_squared(es_alpha.sub(range(3)).state, es_beta.sub(range(2)).state)
        sup1, sdn1 = H.spin_squared(k, 3, 2)
        assert sup.sum() == pytest.approx(sdn.sum())
        assert np.all(sup1 == sup)
        assert np.all(sdn1 == sdn)
        assert len(sup) == 3
        assert len(sdn) == 2

        sup, sdn = spin_squared(es_alpha.sub(0).state.ravel(), es_beta.sub(range(2)).state)
        assert sup.sum() == pytest.approx(sdn.sum())
        assert sup.ndim == 1
        assert len(sup) == 1
        assert len(sdn) == 2

        sup, sdn = spin_squared(es_alpha.sub(0).state.ravel(), es_beta.sub(0).state.ravel())
        assert sup.sum() == pytest.approx(sdn.sum())
        assert sup.ndim == 0
        assert sdn.ndim == 0

        sup, sdn = spin_squared(es_alpha.sub(range(2)).state, es_beta.sub(0).state.ravel())
        assert sup.sum() == pytest.approx(sdn.sum())
        assert len(sup) == 2
        assert len(sdn) == 1

    def test_non_colinear_orthogonal(self, setup):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=SuperCell(100, nsc=[3, 3, 1]))
        H = Hamiltonian(g, dtype=np.float64, spin=Spin.NONCOLINEAR)
        for i in range(10):
            j = range(i*2, i*2+3)
            H[i, i, 0] = 0.05
            H[i, i, 1] = 0.1
            H[i, i, 2] = 0.1
            H[i, i, 3] = 0.1
            if i > 0:
                H[i, i-1, 0] = 1.
                H[i, i-1, 1] = 1.
            if i < 9:
                H[i, i+1, 0] = 1.
                H[i, i+1, 1] = 1.
        eig1 = H.eigh(dtype=np.complex64)
        assert np.allclose(H.eigh(dtype=np.complex128), eig1)
        assert np.allclose(H.eigh(gauge='r', dtype=np.complex128), eig1)
        assert len(eig1) == len(H)

        H1 = Hamiltonian(g, dtype=np.float64, spin=Spin('non-collinear'))
        for i in range(10):
            j = range(i*2, i*2+3)
            H1[i, i, 0] = 0.05
            H1[i, i, 1] = 0.1
            H1[i, i, 2] = 0.1
            H1[i, i, 3] = 0.1
            if i > 0:
                H1[i, i-1, 0] = 1.
                H1[i, i-1, 1] = 1.
            if i < 9:
                H1[i, i+1, 0] = 1.
                H1[i, i+1, 1] = 1.
        assert H1.spsame(H)
        eig1 = H1.eigh(dtype=np.complex64)
        assert np.allclose(H1.eigh(dtype=np.complex128), eig1)
        assert np.allclose(H.eigh(), H1.eigh())

        # Create the block matrix for expectation
        SZ = block_diag(*([H1.spin.Z] * H1.no))

        for dtype in [np.complex64, np.complex128]:
            es = H1.eigenstate(dtype=dtype)
            assert np.allclose(es.eig, eig1)
            assert np.allclose(es.inner(), 1)

            # Perform spin-moment calculation
            sm = es.spin_moment()
            sm2 = es.expectation(SZ).real
            sm3 = np.diag(np.dot(np.conj(es.state), SZ).dot(es.state.T)).real
            assert np.allclose(sm[:, 2], sm2)
            assert np.allclose(sm[:, 2], sm3)

            om = es.spin_moment(project=True)
            assert np.allclose(sm, om.sum(1))

            PDOS = es.PDOS(np.linspace(-1, 1, 100))
            DOS = es.DOS(np.linspace(-1, 1, 100))
            assert np.allclose(PDOS.sum(1)[0, :], DOS)
            es.velocity_matrix()
            es.inv_eff_mass_tensor()

        # Check the velocities
        # But only compare for np.float64, we need the precision
        v = es.velocity()
        pv = es.velocity(project=True)
        assert np.allclose(pv.sum(1), v)

        # Ensure we can change gauge for NC stuff
        es.change_gauge('R')
        es.change_gauge('r')

    def test_non_colinear_non_orthogonal(self):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=SuperCell(100, nsc=[3, 3, 1]))
        H = Hamiltonian(g, dtype=np.float64, orthogonal=False, spin=Spin.NONCOLINEAR)
        for i in range(10):
            j = range(i*2, i*2+3)
            H[i, i, 0] = 0.1
            H[i, i, 1] = 0.05
            H[i, i, 2] = 0.1
            H[i, i, 3] = 0.1
            if i > 0:
                H[i, i-1, 0] = 1.
                H[i, i-1, 1] = 1.
            if i < 9:
                H[i, i+1, 0] = 1.
                H[i, i+1, 1] = 1.
            H.S[i, i] = 1.
        eig1 = H.eigh(dtype=np.complex64)
        assert np.allclose(H.eigh(dtype=np.complex128), eig1)
        assert len(eig1) == len(H)

        H1 = Hamiltonian(g, dtype=np.float64, orthogonal=False, spin=Spin('non-collinear'))
        for i in range(10):
            j = range(i*2, i*2+3)
            H1[i, i, 0] = 0.1
            H1[i, i, 1] = 0.05
            H1[i, i, 2] = 0.1
            H1[i, i, 3] = 0.1
            if i > 0:
                H1[i, i-1, 0] = 1.
                H1[i, i-1, 1] = 1.
            if i < 9:
                H1[i, i+1, 0] = 1.
                H1[i, i+1, 1] = 1.
            H1.S[i, i] = 1.
        assert H1.spsame(H)
        eig1 = H1.eigh(dtype=np.complex64)
        assert np.allclose(H1.eigh(dtype=np.complex128), eig1)
        assert np.allclose(H.eigh(dtype=np.complex64), H1.eigh(dtype=np.complex128))

        for dtype in [np.complex64, np.complex128]:
            es = H1.eigenstate(dtype=dtype)
            assert np.allclose(es.eig, eig1)

            sm = es.spin_moment()

            om = es.spin_moment(project=True)
            assert np.allclose(sm, om.sum(1))

            PDOS = es.PDOS(np.linspace(-1, 1, 100))
            DOS = es.DOS(np.linspace(-1, 1, 100))
            assert np.allclose(PDOS.sum(1)[0, :], DOS)
            es.velocity_matrix()
            es.inv_eff_mass_tensor()

        # Check the velocities
        # But only compare for np.float64, we need the precision
        v = es.velocity()
        pv = es.velocity(project=True)
        assert np.allclose(pv.sum(1), v)

        # Ensure we can change gauge for NC stuff
        es.change_gauge('R')
        es.change_gauge('r')

    def test_spin_orbit_orthogonal(self):
        g = Geometry([[i, 0, 0] for i in range(10)], Atom(6, R=1.01), sc=SuperCell(100, nsc=[3, 3, 1]))
        H = Hamiltonian(g, dtype=np.float64, spin=Spin.SPINORBIT)
        for i in range(10):
            j = range(i*2, i*2+3)
            H[i, i, 0] = 0.1
            H[i, i, 1] = 0.05
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
        eig1 = H.eigh(dtype=np.complex64)
        assert np.allclose(H.eigh(dtype=np.complex128), eig1)
        assert len(H.eigh()) == len(H)

        H1 = Hamiltonian(g, dtype=np.float64, spin=Spin('spin-orbit'))
        for i in range(10):
            j = range(i*2, i*2+3)
            H1[i, i, 0] = 0.1
            H1[i, i, 1] = 0.05
            H1[i, i, 2] = 0.1
            H1[i, i, 3] = 0.1
            H1[i, i, 4] = 0.1
            H1[i, i, 5] = 0.1
            H1[i, i, 6] = 0.1
            H1[i, i, 7] = 0.1
            if i > 0:
                H1[i, i-1, 0] = 1.
                H1[i, i-1, 1] = 1.
            if i < 9:
                H1[i, i+1, 0] = 1.
                H1[i, i+1, 1] = 1.
        assert H1.spsame(H)
        eig1 = H1.eigh(dtype=np.complex64)
        assert np.allclose(H1.eigh(dtype=np.complex128), eig1, atol=1e-5)
        assert np.allclose(H.eigh(dtype=np.complex64), H1.eigh(dtype=np.complex128), atol=1e-5)

        # Create the block matrix for expectation
        SZ = block_diag(*([H1.spin.Z] * H1.no))

        for dtype in [np.complex64, np.complex128]:
            es = H.eigenstate(dtype=dtype)
            assert np.allclose(es.eig, eig1)

            sm = es.spin_moment()
            sm2 = es.expectation(SZ).real
            sm3 = np.diag(np.dot(np.conj(es.state), SZ).dot(es.state.T)).real
            assert np.allclose(sm[:, 2], sm2)
            assert np.allclose(sm[:, 2], sm3)

            om = es.spin_moment(project=True)
            assert np.allclose(sm, om.sum(1))

            PDOS = es.PDOS(np.linspace(-1, 1, 100))
            DOS = es.DOS(np.linspace(-1, 1, 100))
            assert np.allclose(PDOS.sum(1)[0, :], DOS)
            es.velocity_matrix()
            es.inv_eff_mass_tensor()

        # Check the velocities
        # But only compare for np.float64, we need the precision
        v = es.velocity()
        pv = es.velocity(project=True)
        assert np.allclose(pv.sum(1), v)

        # Ensure we can change gauge for SO stuff
        es.change_gauge('R')
        es.change_gauge('r')

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
    @pytest.mark.parametrize("nx", [1, 4])
    @pytest.mark.parametrize("ny", [1, 5])
    @pytest.mark.parametrize("nz", [1, 6])
    def test_tile_same(self, setup, nx, ny, nz):
        R, param = [0.1, 1.5], [1., 0.1]

        # Create reference
        Hg = Hamiltonian(setup.g.tile(nx, 0).tile(ny, 1).tile(nz, 2))
        Hg.construct([R, param])
        Hg.finalize()
        H = Hamiltonian(setup.g)
        H.construct([R, param])
        H = H.tile(nx, 0).tile(ny, 1).tile(nz, 2)
        assert Hg.spsame(H)
        H.finalize()
        Hg.finalize()
        assert np.allclose(H._csr._D, Hg._csr._D)
        assert np.allclose(Hg.Hk([0.1, 0.2, 0.3], format='array'),
                           H.Hk([0.1, 0.2, 0.3], format='array'))

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
        def func(self, ia, atoms, atoms_xyz=None):
            idx = self.geometry.close(ia, R=[0.1, 1.43], atoms=atoms)
            io = self.geometry.a2o(ia)
            # Set on-site on first and second orbital
            odx = self.geometry.a2o(idx[0])
            self[io, odx] = -1.
            self[io+1, odx+1] = 1.

            # Set connecting
            odx = self.geometry.a2o(idx[1])
            self[io, odx] = 0.2
            self[io, odx+1] = 0.01
            self[io+1, odx] = 0.01
            self[io+1, odx+1] = 0.3

        setup.H2.construct(func)
        Hbig = setup.H2.tile(3, 0).tile(3, 1)

        gbig = setup.H2.geometry.tile(3, 0).tile(3, 1)
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
        def func(self, ia, atoms, atoms_xyz=None):
            idx = self.geometry.close(ia, R=[0.1, 1.43], atoms=atoms)
            io = self.geometry.a2o(ia)
            # Set on-site on first and second orbital
            odx = self.geometry.a2o(idx[0])
            self[io, odx] = -1.
            self[io+1, odx+1] = 1.

            # Set connecting
            odx = self.geometry.a2o(idx[1])
            self[io, odx] = 0.2
            self[io, odx+1] = 0.01
            self[io+1, odx] = 0.01
            self[io+1, odx+1] = 0.3

        setup.H2.construct(func)
        Hbig = setup.H2.repeat(3, 0).repeat(3, 1)

        gbig = setup.H2.geometry.repeat(3, 0).repeat(3, 1)
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

    def test_shift3(self, setup):
        R, param = [0.1, 1.5], [(1., -1., 1.), (0.1, 0.1, 0.1)]
        H = Hamiltonian(setup.g.copy(), spin=Spin('P'), orthogonal=False)
        H.construct([R, param])
        eig0_0 = H.eigh(spin=0)[0]
        eig1_0 = H.eigh(spin=1)[0]
        H.shift(0.2)
        assert H.eigh(spin=0)[0] == pytest.approx(eig0_0 + 0.2)
        assert H.eigh(spin=1)[0] == pytest.approx(eig1_0 + 0.2)
        H.shift([0, -0.2])
        assert H.eigh(spin=0)[0] == pytest.approx(eig0_0 + 0.2)
        assert H.eigh(spin=1)[0] == pytest.approx(eig1_0)

    def test_fermi_level(self, setup):
        R, param = [0.1, 1.5], [(1., 1.), (2.1, 0.1)]
        H = Hamiltonian(setup.g.copy(), orthogonal=False)
        H.construct([R, param])
        bz = MonkhorstPack(H, [10, 10, 1])
        q = 0.9
        Ef = H.fermi_level(bz, q=q)
        H.shift(-Ef)
        assert H.fermi_level(bz, q=q) == pytest.approx(0., abs=1e-6)

    def test_fermi_level_spin(self, setup):
        R, param = [0.1, 1.5], [(1., 1.), (2.1, 0.1)]
        H = Hamiltonian(setup.g.copy(), spin=Spin('P'))
        H.construct([R, param])
        bz = MonkhorstPack(H, [10, 10, 1])
        q = 1.1
        Ef = H.fermi_level(bz, q=q)
        assert np.asarray(Ef).ndim == 0
        H.shift(-Ef)
        assert H.fermi_level(bz, q=q) == pytest.approx(0., abs=1e-6)

    def test_fermi_level_spin_separate(self, setup):
        R, param = [0.1, 1.5], [(1., 1.), (2.1, 0.1)]
        H = Hamiltonian(setup.g.copy(), spin=Spin('P'))
        H.construct([R, param])
        bz = MonkhorstPack(H, [10, 10, 1])
        q = [0.5, 0.3]
        Ef = H.fermi_level(bz, q=q)
        assert len(Ef) == 2
        H.shift(-Ef)
        assert np.allclose(H.fermi_level(bz, q=q), 0.)

    def test_wrap_oplist(self, setup):
        R, param = [0.1, 1.5], [1, 2.1]
        H = Hamiltonian(setup.g.copy())
        H.construct([R, param])
        bz = MonkhorstPack(H, [10, 10, 1])
        E = np.linspace(-4, 4, 500)
        dist = get_distribution('gaussian', smearing=0.05)
        def wrap(es, parent, k, weight):
            DOS = es.DOS(E, distribution=dist)
            PDOS = es.PDOS(E, distribution=dist)
            vel = es.velocity() * es.occupation().reshape(-1, 1)
            return oplist([DOS, PDOS, vel])
        bz_avg = bz.apply.average
        results = bz_avg.eigenstate(wrap=wrap)
        assert np.allclose(bz_avg.DOS(E, distribution=dist), results[0])
        assert np.allclose(bz_avg.PDOS(E, distribution=dist), results[1])

    def test_edges1(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]
        H = Hamiltonian(setup.g)
        H.construct([R, param])
        assert len(H.edges(0)) == 4

    def test_edges2(self, setup):
        R, param = [0.1, 1.5], [1., 0.1]
        H = Hamiltonian(setup.g)
        H.construct([R, param])
        with pytest.raises(ValueError):
            H.edges()

    def test_edges3(self, setup):
        def func(self, ia, atoms, atoms_xyz=None):
            idx = self.geometry.close(ia, R=[0.1, 1.43], atoms=atoms)
            io = self.geometry.a2o(ia)
            # Set on-site on first and second orbital
            odx = self.geometry.a2o(idx[0])
            self[io, odx] = -1.
            self[io+1, odx+1] = 1.

            # Set connecting
            odx = self.geometry.a2o(idx[1])
            self[io, odx] = 0.2
            self[io, odx+1] = 0.01
            self[io+1, odx] = 0.01
            self[io+1, odx+1] = 0.3

        H2 = setup.H2.copy()
        H2.construct(func)
        # first atom
        assert len(H2.edges(0)) == 4
        # orbitals of first atom
        edge = H2.edges(orbitals=[0, 1])
        assert len(edge) == 8
        assert len(H2.geometry.o2a(edge, unique=True)) == 4

        # first orbital on first two atoms
        edge = H2.edges(orbitals=[0, 2])
        # The 1, 3 are still on the first two atoms, but aren't
        # excluded. Hence they are both there
        assert len(edge) == 12
        assert len(H2.geometry.o2a(edge, unique=True)) == 6

        # first orbital on first two atoms
        edge = H2.edges(orbitals=[0, 2], exclude=[0, 1, 2, 3])
        assert len(edge) == 8
        assert len(H2.geometry.o2a(edge, unique=True)) == 4


def test_wavefunction1():
    N = 50
    o1 = SphericalOrbital(0, (np.linspace(0, 2, N), np.exp(-np.linspace(0, 100, N))))
    G = Geometry([[1] * 3, [2] * 3], Atom(6, o1), sc=[4, 4, 4])
    H = Hamiltonian(G)
    R, param = [0.1, 1.5], [1., 0.1]
    H.construct([R, param])
    ES = H.eigenstate(dtype=np.float64)
    # Plot in the full thing
    grid = Grid(0.1, geometry=H.geometry)
    grid.fill(0.)
    ES.sub(0).wavefunction(grid)


def test_wavefunction2():
    N = 50
    o1 = SphericalOrbital(0, (np.linspace(0, 2, N), np.exp(-np.linspace(0, 100, N))))
    G = Geometry([[1] * 3, [2] * 3], Atom(6, o1), sc=[4, 4, 4])
    H = Hamiltonian(G)
    R, param = [0.1, 1.5], [1., 0.1]
    H.construct([R, param])
    ES = H.eigenstate(dtype=np.float64)
    # This is effectively plotting outside where no atoms exists
    # (there could however still be psi weight).
    grid = Grid(0.1, sc=SuperCell([2, 2, 2], origo=[2] * 3))
    grid.fill(0.)
    ES.sub(0).wavefunction(grid)


def test_wavefunction3():
    N = 50
    o1 = SphericalOrbital(0, (np.linspace(0, 2, N), np.exp(-np.linspace(0, 100, N))))
    G = Geometry([[1] * 3, [2] * 3], Atom(6, o1), sc=[4, 4, 4])
    H = Hamiltonian(G, spin=Spin('nc'))
    R, param = [0.1, 1.5], [[0., 0., 0.1, -0.1],
                            [1., 1., 0.1, -0.1]]
    H.construct([R, param])
    ES = H.eigenstate()
    # Plot in the full thing
    grid = Grid(0.1, dtype=np.complex128, sc=SuperCell([2, 2, 2], origo=[-1] * 3))
    grid.fill(0.)
    ES.sub(0).wavefunction(grid)


def test_wavefunction_eta():
    N = 50
    o1 = SphericalOrbital(0, (np.linspace(0, 2, N), np.exp(-np.linspace(0, 100, N))))
    G = Geometry([[1] * 3, [2] * 3], Atom(6, o1), sc=[4, 4, 4])
    H = Hamiltonian(G, spin=Spin('nc'))
    R, param = [0.1, 1.5], [[0., 0., 0.1, -0.1],
                            [1., 1., 0.1, -0.1]]
    H.construct([R, param])
    ES = H.eigenstate()
    # Plot in the full thing
    grid = Grid(0.1, dtype=np.complex128, sc=SuperCell([2, 2, 2], origo=[-1] * 3))
    grid.fill(0.)
    ES.sub(0).wavefunction(grid, eta=True)
