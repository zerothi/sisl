# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import warnings
from functools import partial

import numpy as np

if np.lib.NumpyVersion(np.__version__) >= "2.0.0b1":
    from numpy.exceptions import ComplexWarning
else:
    from numpy import ComplexWarning
import pytest
from scipy.linalg import block_diag
from scipy.sparse import SparseEfficiencyWarning, issparse

from sisl import (
    Atom,
    BandStructure,
    BrillouinZone,
    Geometry,
    Grid,
    Hamiltonian,
    Lattice,
    MonkhorstPack,
    SislError,
    SphericalOrbital,
    Spin,
    get_distribution,
    oplist,
)
from sisl.physics.electron import ahc, berry_phase, shc, spin_contamination

pytestmark = [
    pytest.mark.physics,
    pytest.mark.hamiltonian,
    pytest.mark.filterwarnings("ignore", category=SparseEfficiencyWarning),
]


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

            C = Atom(Z=6, R=[bond * 1.01])
            self.g = Geometry(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
                atoms=C,
                lattice=self.lattice,
            )
            self.H = Hamiltonian(self.g)
            self.HS = Hamiltonian(self.g, orthogonal=False)

            C = Atom(Z=6, R=[bond * 1.01] * 2)
            self.g2 = Geometry(
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float64) * bond,
                atoms=C,
                lattice=self.lattice,
            )
            self.H2 = Hamiltonian(self.g2)
            self.HS2 = Hamiltonian(self.g2, orthogonal=False)

    return t()


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

    def test_set1(self, setup):
        setup.H.H[0, 0] = 1.0
        assert setup.H[0, 0] == 1.0
        assert setup.H[1, 0] == 0.0
        setup.H.empty()

        setup.HS.H[0, 0] = 1.0
        assert setup.HS.H[0, 0] == 1.0
        assert setup.HS.H[1, 0] == 0.0
        assert setup.HS.S[0, 0] == 0.0
        assert setup.HS.S[1, 0] == 0.0
        setup.HS.S[0, 0] = 1.0
        assert setup.HS.H[0, 0] == 1.0
        assert setup.HS.H[1, 0] == 0.0
        assert setup.HS.S[0, 0] == 1.0
        assert setup.HS.S[1, 0] == 0.0

        # delete before creating the same content
        setup.HS.empty()
        # THIS IS A CHECK FOR BACK_WARD COMPATIBILITY!
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setup.HS[0, 0] = 1.0, 1.0
        assert setup.HS.H[0, 0] == 1.0
        assert setup.HS.S[0, 0] == 1.0
        setup.HS.empty()

    def test_set2(self, setup):
        setup.H.construct([(0.1, 1.5), (1.0, 0.1)])
        assert setup.H[0, 0] == 1.0
        assert setup.H[1, 0] == 0.1
        assert setup.H[0, 1] == 0.1
        setup.H.empty()

    def test_set3(self, setup):
        setup.HS.construct([(0.1, 1.5), ((1.0, 2.0), (0.1, 0.2))])
        assert setup.HS.H[0, 0] == 1.0
        assert setup.HS.S[0, 0] == 2.0
        assert setup.HS.H[1, 1] == 1.0
        assert setup.HS.S[1, 1] == 2.0
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
            setup.H[ia, idx[0]] = 1.0
            setup.H[ia, idx[1]] = 0.1
        assert setup.H.H[0, 0] == 1.0
        assert setup.H.H[1, 1] == 1.0
        assert setup.H.H[1, 0] == 0.1
        assert setup.H.H[0, 1] == 0.1
        assert setup.H.nnz == len(setup.H) * 4
        setup.H.empty()

    @pytest.mark.slow
    def test_set5(self, setup):
        # Test of HUGE construct
        g = setup.g.tile(10, 0).tile(10, 1).tile(10, 2)
        H = Hamiltonian(g)
        H.construct([(0.1, 1.5), (1.0, 0.1)])
        assert H.H[0, 0] == 1.0
        assert H.H[1, 1] == 1.0
        assert H.H[1, 0] == 0.1
        assert H.H[0, 1] == 0.1
        # This is graphene
        # on-site == len(H)
        # nn == 3 * len(H)
        assert H.nnz == len(H) * 4
        del H

    def test_iter1(self, setup):
        setup.HS.construct([(0.1, 1.5), ((1.0, 2.0), (0.1, 0.2))])
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
        setup.HS.H[0, 0] = 1.0
        nnz = 0
        for io, jo in setup.HS:
            nnz = nnz + 1
        assert nnz == setup.HS.nnz
        assert nnz == 1
        setup.HS.empty()

    @pytest.mark.filterwarnings("ignore", category=ComplexWarning)
    def test_Hk1(self, setup):
        H = setup.HS.copy()
        H.construct([(0.1, 1.5), ((1.0, 2.0), (0.1, 0.2))])
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
        H.construct([(0.1, 1.5), ((1.0, 2.0), (0.1, 0.2))])
        h = H.copy()
        Hk = h.Hk(k=[0.15, 0.15, 0.15])
        assert Hk.dtype == np.complex128
        h = H.copy()
        Hk = h.Hk(k=[0.15, 0.15, 0.15], dtype=np.complex64)
        assert Hk.dtype == np.complex64

    def test_Hk5(self, setup, sisl_complex):
        dtype, *_ = sisl_complex
        H = setup.H.copy()
        H.construct([(0.1, 1.5), (1.0, 0.1)])
        Hk = H.Hk(k=[0.15, 0.15, 0.15], dtype=dtype)
        assert Hk.dtype == dtype
        Sk = H.Sk(k=[0.15, 0.15, 0.15])
        # orthogonal basis sets always returns a diagonal in float64
        assert Sk.dtype == np.float64
        Sk = H.Sk(k=[0.15, 0.15, 0.15], dtype=dtype)
        # orthogonal basis sets always returns a diagonal in float64
        assert Sk.dtype == dtype

    def test_eigenstate_gauge(self, setup):
        H = setup.HS.copy()
        H.construct([(0.1, 1.5), ((1.0, 2.0), (0.1, 0.2))])

        # Try with different gauges
        for gauge in ("cell", "R", "lattice"):
            assert H.eigenstate(gauge=gauge).info["gauge"] == "lattice"

        for gauge in ("atom", "atoms", "atom", "orbitals", "r", "atomic"):
            assert H.eigenstate(gauge=gauge).info["gauge"] == "atomic"

    @pytest.mark.parametrize(
        "spin", ["unpolarized", "polarized", "non-collinear", "spin-orbit", "nambu"]
    )
    @pytest.mark.parametrize(
        "dtype", [np.float32, np.float64, np.complex64, np.complex128]
    )
    @pytest.mark.parametrize("k", [[0, 0, 0], [0.15, 0.25, 0.35]])
    def test_Hk_format(self, setup, k, spin, dtype):
        H = Hamiltonian(setup.g, spin=spin, dtype=dtype)
        t0 = np.random.rand(H.shape[-1])
        t1 = np.random.rand(H.shape[-1])
        H.construct([(0.1, 1.5), (t0, t1)])
        csr = H.Hk(k, format="csr").toarray()
        mat = H.Hk(k, format="matrix")
        arr = H.Hk(k, format="array")
        coo = H.Hk(k, format="coo").toarray()
        assert np.allclose(csr, mat)
        assert np.allclose(csr, arr)
        assert np.allclose(csr, coo)

    @pytest.mark.parametrize("orthogonal", [True, False])
    @pytest.mark.parametrize("gauge", ["cell", "atom"])
    @pytest.mark.parametrize(
        "spin", ["unpolarized", "polarized", "non-collinear", "spin-orbit", "nambu"]
    )
    def test_format_sc(self, orthogonal, gauge, spin, sisl_complex):
        g = Geometry(
            [[i, 0, 0] for i in range(10)],
            Atom(6, R=1.01),
            lattice=Lattice([10, 1, 5.0], nsc=[3, 3, 1]),
        )
        dtype, atol, rtol = sisl_complex
        allclose = partial(np.allclose, atol=atol, rtol=rtol)

        H = Hamiltonian(g, dtype=np.float64, orthogonal=orthogonal, spin=Spin(spin))
        nd = H._csr._D.shape[-1]
        # this will correctly account for the double size for NC/SOC/Nambu
        no = len(H)
        no_s = H.geometry.no_s
        for ia in g:
            idx = g.close(ia, R=(0.1, 1.01))[1]
            H[ia, ia] = 1.0
            H[ia, idx] = np.random.rand(nd)

        H = (H + H.transpose(conjugate=True, spin=True)) / 2
        n_s = H.geometry.lattice.n_s

        for k in [[0, 0, 0], [0.15, 0.1, 0.05]]:
            for attr, kwargs in [("Hk", {"gauge": gauge}), ("Sk", {})]:
                Mk = getattr(H, attr)
                csr = Mk(k, format="csr", **kwargs, dtype=dtype)
                sc_csr1 = Mk(k, format="sc:csr", **kwargs, dtype=dtype)
                sc_csr2 = Mk(k, format="sc", **kwargs, dtype=dtype)
                sc_mat = Mk(k, format="sc:array", **kwargs, dtype=dtype)
                mat = sc_mat.reshape(no, n_s, no).sum(1)

                assert sc_mat.shape == sc_csr1.shape
                assert allclose(csr.toarray(), mat)
                assert allclose(sc_csr1.toarray(), sc_csr2.toarray())
                for isc in range(n_s):
                    csr -= sc_csr1[:, isc * no : (isc + 1) * no]
                assert allclose(csr.toarray(), 0.0)

    def test_construct_raise_default(self, setup):
        # Test that construct fails with more than one
        # orbital
        with pytest.raises(ValueError):
            setup.H2.construct([(0.1, 1.5), (1.0, 0.1)])

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
        setup.H.construct([(0.1, 1.5), (1.0, 0.1)])
        csr = setup.H.tocsr(0)
        H = Hamiltonian.fromsp(setup.H.geometry, csr)
        assert H.spsame(setup.H)
        setup.H.empty()

    def test_fromsp2(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), (1.0, 0.1)])
        csr = H.tocsr(0)
        with pytest.raises(ValueError):
            Hamiltonian.fromsp(setup.H.geometry.tile(2, 0), csr)

    def test_fromsp3(self, setup):
        H = setup.HS.copy()
        H.construct([(0.1, 1.5), ([1.0, 1.0], [0.1, 0])])
        h = Hamiltonian.fromsp(H.geometry.copy(), H.tocsr(0), H.tocsr(1))
        assert H.spsame(h)

    def test_op1(self, setup):
        g = Geometry([[i, 0, 0] for i in range(100)], Atom(6, R=1.01), lattice=[100])
        H = Hamiltonian(g, dtype=np.int32)
        for i in range(10):
            j = range(i * 4, i * 4 + 3)
            H[0, j] = i

            # i+
            H += 1
            for jj in j:
                assert H[0, jj] == i + 1
                assert H[1, jj] == 0

            # i-
            H -= 1
            for jj in j:
                assert H[0, jj] == i
                assert H[1, jj] == 0

            # i*
            H *= 2
            for jj in j:
                assert H[0, jj] == i * 2
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
        g = Geometry([[i, 0, 0] for i in range(100)], Atom(6, R=1.01), lattice=[100])
        H = Hamiltonian(g, dtype=np.int32)
        for i in range(10):
            j = range(i * 4, i * 4 + 3)
            H[0, j] = i

            # +
            s = H + 1
            for jj in j:
                assert s[0, jj] == i + 1
                assert H[0, jj] == i
                assert s[1, jj] == 0

            # -
            s = H - 1
            for jj in j:
                assert s[0, jj] == i - 1
                assert H[0, jj] == i
                assert s[1, jj] == 0

            # -
            s = 1 - H
            for jj in j:
                assert s[0, jj] == 1 - i
                assert H[0, jj] == i
                assert s[1, jj] == 0

            # *
            s = H * 2
            for jj in j:
                assert s[0, jj] == i * 2
                assert H[0, jj] == i
                assert s[1, jj] == 0

            # //
            s = s // 2
            for jj in j:
                assert s[0, jj] == i
                assert H[0, jj] == i
                assert s[1, jj] == 0

            # **
            s = H**2
            for jj in j:
                assert s[0, jj] == i**2
                assert H[0, jj] == i
                assert s[1, jj] == 0

            # ** (r)
            s = 2**H
            for jj in j:
                assert s[0, jj], 2 ** H[0 == jj]
                assert H[0, jj] == i
                assert s[1, jj] == 0

    def test_op3(self, setup):
        g = Geometry([[i, 0, 0] for i in range(100)], Atom(6, R=1.01), lattice=[100])
        H = Hamiltonian(g, dtype=np.int32)
        Hc = H.copy()
        del Hc

        # Create initial stuff
        for i in range(10):
            j = range(i * 4, i * 4 + 3)
            H[0, j] = i

        for op in ["add", "sub", "mul", "pow"]:
            func = getattr(H, f"__{op}__")
            h = func(1)
            assert h.dtype == np.int32
            h = func(1.0)
            assert h.dtype == np.float64
            if op != "pow":
                h = func(1.0j)
                assert h.dtype == np.complex128

        H = H.copy(dtype=np.float64)
        for op in ["add", "sub", "mul", "pow"]:
            func = getattr(H, f"__{op}__")
            h = func(1)
            assert h.dtype == np.float64
            h = func(1.0)
            assert h.dtype == np.float64
            if op != "pow":
                h = func(1.0j)
                assert h.dtype == np.complex128

        H = H.copy(dtype=np.complex128)
        for op in ["add", "sub", "mul", "pow"]:
            func = getattr(H, f"__{op}__")
            h = func(1)
            assert h.dtype == np.complex128
            h = func(1.0)
            assert h.dtype == np.complex128
            if op != "pow":
                h = func(1.0j)
                assert h.dtype == np.complex128

    def test_op4(self, setup):
        g = Geometry([[i, 0, 0] for i in range(100)], Atom(6, R=1.01), lattice=[100])
        H = Hamiltonian(g, dtype=np.int32)
        # Create initial stuff
        for i in range(10):
            j = range(i * 4, i * 4 + 3)
            H[0, j] = i

        h = 1 + H
        assert h.dtype == np.int32
        h = 1.0 + H
        assert h.dtype == np.float64
        h = 1.0j + H
        assert h.dtype == np.complex128

        h = 1 - H
        assert h.dtype == np.int32
        h = 1.0 - H
        assert h.dtype == np.float64
        h = 1.0j - H
        assert h.dtype == np.complex128

        h = 1 * H
        assert h.dtype == np.int32
        h = 1.0 * H
        assert h.dtype == np.float64
        h = 1.0j * H
        assert h.dtype == np.complex128

        h = 1**H
        assert h.dtype == np.int32
        h = 1.0**H
        assert h.dtype == np.float64
        h = 1.0j**H
        assert h.dtype == np.complex128

    def test_untile1(self, setup):
        # Test of eigenvalues using a cut
        # Hamiltonian
        R, param = [0.1, 1.5], [1.0, 0.1]

        # Create reference
        Hg = Hamiltonian(setup.g)
        Hg.construct([R, param])
        g = setup.g.tile(2, 0).tile(2, 1)
        H = Hamiltonian(g)
        H.construct([R, param])
        # Create cut Hamiltonian
        Hc = H.untile(2, 1).untile(2, 0)
        eigc = Hc.eigh()
        eigg = Hg.eigh()
        assert np.allclose(eigc, eigg)
        assert np.allclose(Hg.eigh(), Hc.eigh())
        del Hc, H

    def test_untile2(self, setup):
        # Test of eigenvalues using a cut
        # Hamiltonian
        R, param = [0.1, 1.5], [(1.0, 1.0), (0.1, 0.1)]

        # Create reference
        Hg = Hamiltonian(setup.g, orthogonal=False)
        Hg.construct([R, param])

        g = setup.g.tile(2, 0).tile(2, 1)
        H = Hamiltonian(g, orthogonal=False)
        H.construct([R, param])
        # Create cut Hamiltonian
        Hc = H.untile(2, 1).untile(2, 0)
        eigc = Hc.eigh()
        eigg = Hg.eigh()
        assert np.allclose(Hg.eigh(), Hc.eigh())
        del Hc, H

    def test_eigh_vs_eig(self, setup, sisl_tolerance):
        atol, rtol = sisl_tolerance[np.complex64]
        allclose = partial(np.allclose, atol=atol, rtol=rtol)

        # Test of eigenvalues
        R, param = [0.1, 1.5], [1.0, 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param), eta=True)

        eig1 = H.eigh(dtype=np.complex64)
        eig2 = np.sort(H.eig(dtype=np.complex64).real)
        eig3 = np.sort(H.eig(eigvals_only=False, dtype=np.complex64)[0].real)
        assert allclose(eig1, eig2)
        assert allclose(eig1, eig3)

        eig1 = H.eigh([0.01] * 3, dtype=np.complex64)
        eig2 = np.sort(H.eig([0.01] * 3, dtype=np.complex64).real)
        eig3 = np.sort(
            H.eig([0.01] * 3, eigvals_only=False, dtype=np.complex64)[0].real
        )
        assert allclose(eig1, eig2)
        assert allclose(eig1, eig3)

    def test_eig1(self, setup):
        # Test of eigenvalues
        R, param = [0.1, 1.5], [1.0, 0.1]
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
        HS.construct([(0.1, 1.5), ((1.0, 1.0), (0.1, 0.1))])
        eig1 = HS.eigh(dtype=np.complex64)
        assert np.allclose(eig1, HS.eigh(dtype=np.complex128))
        setup.HS.empty()

    def test_eig3(self, setup):
        setup.HS.construct([(0.1, 1.5), ((1.0, 1.0), (0.1, 0.1))])
        BS = BandStructure(setup.HS, [[0, 0, 0], [0.5, 0.5, 0]], 10)
        eigs = BS.apply.array.eigh()
        assert len(BS) == eigs.shape[0]
        assert len(setup.HS) == eigs.shape[1]
        eig2 = np.array([eig for eig in BS.apply.iter.eigh()])
        assert np.allclose(eigs, eig2)
        setup.HS.empty()

    @pytest.mark.filterwarnings("ignore", message="*uses an overlap matrix that")
    def test_eig4(self, setup):
        # Test of eigenvalues vs eigenstate class
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((1.0, 1.0), (0.1, 0.1))])

        for k in ([0] * 3, [0.2] * 3):
            e, v = HS.eigh(k, eigvals_only=False)
            es = HS.eigenstate(k)
            assert np.allclose(e, es.eig)
            assert np.allclose(v, es.state.T)
            assert np.allclose(es.norm2(), 1)
            # inner is not norm2
            assert not np.allclose(es.inner(diag=False), np.eye(len(es)))

            with pytest.raises(ValueError):
                es.inner(es.sub(0))

            assert es.inner(es.sub(0), diag=False).shape == (len(es), 1)

            eig1 = HS.eigh(k)
            eig2 = np.sort(HS.eig(k).real)
            eig3 = np.sort(HS.eig(k, eigvals_only=False)[0].real)
            eig4 = es.inner(matrix=HS.Hk(k))
            eig5 = es.inner(ket=es, matrix=HS.Hk(k))
            assert np.allclose(eig1, eig2, atol=1e-5)
            assert np.allclose(eig1, eig3, atol=1e-5)
            assert np.allclose(eig1, eig4, atol=1e-5)
            assert np.allclose(eig1, eig5, atol=1e-5)

            assert es.inner(
                matrix=HS.Hk([0.1] * 3), ket=HS.eigenstate([0.3] * 3), diag=False
            ).shape == (len(es), len(es))

    @pytest.mark.filterwarnings("ignore", message="*uses an overlap matrix that")
    def test_inner(self, setup):
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((2.0, 1.0), (3.0, 0.0))])
        HS = HS.tile(2, 0).tile(2, 1)

        es1 = HS.eigenstate([0.1] * 3)
        es2 = HS.eigenstate([0.2] * 3)
        m1 = es1.inner(es2, diag=False)
        assert m1.shape == (len(es1), len(es2))
        m2 = es2.inner(es1, diag=False)
        assert np.allclose(m1, m2.conj().T)
        m3 = es2.inner(es1.state, diag=False)
        assert np.allclose(m1, m3.conj().T)

        r = range(3)
        m1 = es1.sub(r).inner(es2, diag=False)
        m2 = es2.inner(es1.sub(r), diag=False)
        assert np.allclose(m1, m2.conj().T)
        assert es1.sub(r).inner(es2, diag=False).shape == (len(r), len(es2))
        assert es1.inner(es2.sub(r), diag=False).shape == (len(es1), len(r))

    def test_gauge_eig(self, setup):
        # Test of eigenvalues
        R, param = [0.1, 1.5], [1.0, 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))

        k = [0.1] * 3
        es1 = H.eigenstate(k, gauge="cell")
        es2 = H.eigenstate(k, gauge="atom")
        assert np.allclose(es1.eig, es2.eig)
        assert not np.allclose(es1.state, es2.state)

        es1 = H.eigenstate(k, gauge="cell", dtype=np.complex64)
        es2 = H.eigenstate(k, gauge="atom", dtype=np.complex64)
        assert np.allclose(es1.eig, es2.eig)
        assert not np.allclose(es1.state, es2.state)

    def test_eigenstate_ipr(self, setup):
        # Test of eigenvalues
        R, param = [0.1, 1.5], [1.0, 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))

        k = [0.1] * 3
        es = H.eigenstate(k)
        ipr = es.ipr()
        assert ipr.shape == (len(es),)

    def test_eigenstate_tile(self, setup):
        # Test of eigenvalues
        R, param = [0.1, 1.5], [0.0, 2.7]
        H1 = setup.H.copy()
        H1.construct((R, param))
        H2 = H1.tile(2, 1)

        k = [0] * 3
        # we must select a k that does not fold on
        # itself which then creates degenerate states
        for k1 in [0.5, 1 / 3]:
            k[1] = k1
            es1 = H1.eigenstate(k)
            es1_2 = es1.tile(2, 1, normalize=True)
            es2 = H2.eigenstate(es1_2.info["k"])

            # we need to check that these are somewhat the same
            out = es1_2.inner(es2, diag=False)
            abs_out = np.absolute(out)
            assert np.isclose(abs_out, 1).sum() == len(es1)

    def test_eigenstate_tile_offset(self, setup):
        # Test of eigenvalues
        R, param = [0.1, 1.5], [0.0, 2.7]
        H1 = setup.H.copy()
        H1.construct((R, param))
        H2 = H1.tile(2, 1)

        k = [0] * 3
        # we must select a k that does not fold on
        # itself which then creates degenerate states
        for k1 in [0.5, 1 / 3]:
            k[1] = k1
            es1 = H1.eigenstate(k)
            es1_2 = es1.tile(2, 1, normalize=True, offset=1)
            es2 = H2.eigenstate(es1_2.info["k"]).translate([0, 1, 0])

            # we need to check that these are somewhat the same
            out = es1_2.inner(es2, diag=False)
            abs_out = np.absolute(out)
            assert np.isclose(abs_out, 1).sum() == len(es1)

    def test_eigenstate_translate(self, setup):
        # Test of eigenvalues
        R, param = [0.1, 1.5], [0.0, 2.7]
        H = setup.H.copy()
        H.construct((R, param))

        k = [0] * 3
        # we must select a k that does not fold on
        # itself which then creates degenerate states
        for k1 in [0.5, 1 / 3]:
            k[1] = k1
            es1 = H.eigenstate(k)
            es1_2 = es1.tile(2, 1)
            es2 = es1.translate([0, 1, 0])
            assert np.allclose(es1_2.state[:, : len(es1)], es1.state)
            assert np.allclose(es1_2.state[:, len(es1) :], es2.state)

    def test_gauge_velocity(self, setup):
        R, param = [0.1, 1.5], [1.0, 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))

        k = [0.1] * 3
        # This is a degenerate eigenstate:
        #  2, 2, 4, 4, 2, 2
        # It would be nice to check decoupling.
        # Currently this is disabled, but should be used.

        # Some comments from the older code which enabled automatic
        # decoupling:
        # This test is the reason why a default degenerate=1e-5 is used
        # since the gauge='cell' yields the correct *decoupled* states
        # where as gauge='orbital' mixes them in a bad way.
        es1 = H.eigenstate(k, gauge="cell")
        es2 = H.eigenstate(k, gauge="atom")
        assert not np.allclose(es1.velocity(), es2.velocity())

        es2.change_gauge("cell")
        assert not np.allclose(es1.velocity(), es2.velocity())

        es2.change_gauge("atom")
        es1.change_gauge("atom")
        v1 = es1.velocity()
        v2 = es2.velocity()
        assert not np.allclose(v1, v2)

        # Projected velocity
        vv1 = es1.velocity(matrix=True)
        vv2 = es2.velocity(matrix=True)
        assert not np.allclose(np.diagonal(vv1, axis1=1, axis2=2), v2)
        assert not np.allclose(np.diagonal(vv2, axis1=1, axis2=2), v1)

    def test_derivative_orthogonal(self, setup):
        R, param = [0.1, 1.5], [1.0, 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))

        k = [0.1] * 3
        es = H.eigenstate()
        v1, vv1 = es.derivative(2)
        v = es.derivative(1)
        assert np.allclose(v1, v)

    def test_derivative_orthogonal_axis(self, setup):
        R, param = [0.1, 1.5], [1.0, 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))

        k = [0.1] * 3
        es = H.eigenstate()

        v1, vv1 = es.derivative(2, axes="x")
        assert len(v1) == 1
        assert len(vv1) == 1

        v1, vv1 = es.derivative(2, axes="xy")
        assert len(v1) == 2
        assert len(vv1) == 3

    def test_derivative_non_orthogonal(self, setup):
        R, param = [0.1, 1.5], [(1.0, 1.0), (0.1, 0.1)]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g, orthogonal=False)
        H.construct((R, param))

        k = [0.1] * 3
        es = H.eigenstate()
        v1, vv1 = es.derivative(2)
        v = es.derivative(1)
        assert np.allclose(v1, v)

    def test_berry_phase(self, setup):
        R, param = [0.1, 1.5], [1.0, 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))
        bz = BandStructure.param_circle(H, 20, 0.01, [0, 0, 1], [1 / 3] * 3)
        berry_phase(bz)
        berry_phase(bz, sub=0)
        berry_phase(bz, eigvals=True, sub=0, method="berry:svd")

    def test_berry_phase_fail_sc(self, setup):
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        bz = BandStructure.param_circle(
            H.geometry.lattice, 20, 0.01, [0, 0, 1], [1 / 3] * 3
        )
        with pytest.raises(SislError):
            berry_phase(bz)

    def test_berry_phase_loop(self, setup):
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        bz1 = BandStructure.param_circle(H, 20, 0.01, [0, 0, 1], [1 / 3] * 3)
        bz2 = BandStructure.param_circle(H, 20, 0.01, [0, 0, 1], [1 / 3] * 3, loop=True)
        assert np.allclose(berry_phase(bz1), berry_phase(bz2))

    def test_berry_phase_non_orthogonal(self, setup):
        R, param = [0.1, 1.5], [(1.0, 1.0), (0.1, 0.1)]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g, orthogonal=False)
        H.construct((R, param))

        bz = BandStructure.param_circle(H, 20, 0.01, [0, 0, 1], [1 / 3] * 3)
        berry_phase(bz)

    def test_berry_phase_orthogonal_spin_down(self, setup):
        R, param = [0.1, 1.5], [(1.0, 1.0), (0.1, 0.2)]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g, spin=Spin.POLARIZED)
        H.construct((R, param))

        bz = BandStructure.param_circle(H, 20, 0.01, [0, 0, 1], [1 / 3] * 3)
        bp1 = berry_phase(bz)
        bp2 = berry_phase(bz, eigenstate_kwargs={"spin": 1})
        assert bp1 != bp2

    def test_berry_phase_zak_x_topological(self):
        # SSH model, topological cell
        # |t2| < |t1|
        g = Geometry([[0, 0, 0], [1.2, 0, 0]], Atom(1, 1.001), lattice=[2, 10, 10])
        g.set_nsc([3, 1, 1])
        H = Hamiltonian(g)
        H.construct([(0.1, 1.0, 1.5), (0, 1.0, 0.5)])

        # Contour
        def func(parent, N, i):
            return [i / N, 0, 0]

        bz = BrillouinZone.parametrize(H, func, 101)
        assert np.allclose(np.abs(berry_phase(bz, sub=0, method="zak")), np.pi)
        # Just to do the other branch
        berry_phase(bz, method="zak")

    def test_berry_phase_zak_x_topological_non_orthogonal(self):
        # SSH model, topological cell
        # |t2| < |t1|
        g = Geometry([[0, 0, 0], [1.2, 0, 0]], Atom(1, 1.001), lattice=[2, 10, 10])
        g.set_nsc([3, 1, 1])
        H = Hamiltonian(g, orthogonal=False)
        H.construct([(0.1, 1.0, 1.5), ((0, 1), (1.0, 0.25), (0.5, 0.1))])

        # Contour
        def func(parent, N, i):
            return [i / N, 0, 0]

        bz = BrillouinZone.parametrize(H, func, 101)
        assert np.allclose(np.abs(berry_phase(bz, sub=0, method="zak")), np.pi)
        # Just to do the other branch
        berry_phase(bz, method="zak")

    def test_berry_phase_zak_x_trivial(self):
        # SSH model, trivial cell
        # |t2| > |t1|
        g = Geometry([[0, 0, 0], [1.2, 0, 0]], Atom(1, 1.001), lattice=[2, 10, 10])
        g.set_nsc([3, 1, 1])
        H = Hamiltonian(g)
        H.construct([(0.1, 1.0, 1.5), (0, 0.5, 1.0)])

        # Contour
        def func(parent, N, i):
            return [i / N, 0, 0]

        bz = BrillouinZone.parametrize(H, func, 101)
        assert np.allclose(np.abs(berry_phase(bz, sub=0, method="zak")), 0.0)
        # Just to do the other branch
        berry_phase(bz, method="zak")

    def test_berry_phase_zak_y(self):
        # SSH model, topological cell
        g = Geometry([[0, -0.6, 0], [0, 0.6, 0]], Atom(1, 1.001), lattice=[10, 2, 10])
        g.set_nsc([1, 3, 1])
        H = Hamiltonian(g)
        H.construct([(0.1, 1.0, 1.5), (0, 1.0, 0.5)])

        # Contour
        def func(parent, N, i):
            return [0, i / N, 0]

        bz = BrillouinZone.parametrize(H, func, 101)
        assert np.allclose(np.abs(berry_phase(bz, sub=0, method="zak")), np.pi)
        # Just to do the other branch
        berry_phase(bz, method="zak")

    def test_berry_phase_zak_offset(self):
        # SSH model, topological cell
        g = Geometry([[0.0, 0, 0], [1.2, 0, 0]], Atom(1, 1.001), lattice=[2, 10, 10])
        g.set_nsc([3, 1, 1])
        H = Hamiltonian(g)
        H.construct([(0.1, 1.0, 1.5), (0, 1.0, 0.5)])

        # Contour
        def func(parent, N, i):
            return [i / N, 0, 0]

        bz = BrillouinZone.parametrize(H, func, 101)
        zak = berry_phase(bz, sub=0, method="zak")
        assert np.allclose(np.abs(zak), np.pi)

    def test_berry_phase_method_fail(self):
        # wrong method keyword
        g = Geometry([[-0.6, 0, 0], [0.6, 0, 0]], Atom(1, 1.001), lattice=[2, 10, 10])
        g.set_nsc([3, 1, 1])
        H = Hamiltonian(g)

        def func(parent, N, i):
            return [0, i / N, 0]

        bz = BrillouinZone.parametrize(H, func, 101)
        with pytest.raises(ValueError):
            berry_phase(bz, method="unknown")

    def test_berry_curvature(self, setup):
        R, param = [0.1, 1.5], [[1.0, 0.1, 0, 0], [0.4, 0.2, 0.3, 0.2]]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g, spin=Spin.NONCOLINEAR)
        H.construct((R, param))

        k = [0.1] * 3
        ie1 = H.eigenstate(k, gauge="cell").berry_curvature()
        ie2 = H.eigenstate(k, gauge="atom").berry_curvature()
        assert not np.allclose(ie1, ie2)

    def test_spin_berry_curvature(self, setup):
        R, param = [0.1, 1.5], [[1.0, 0.1, 0, 0], [0.4, 0.2, 0.3, 0.2]]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g, spin=Spin.NONCOLINEAR)
        H.construct((R, param))

        k = [0.1] * 3
        ie1 = H.eigenstate(k, gauge="cell").spin_berry_curvature()
        ie2 = H.eigenstate(k, gauge="atom").spin_berry_curvature()
        assert not np.allclose(ie1, ie2)

    @pytest.mark.filterwarnings("ignore", category=ComplexWarning)
    def test_ahc(self, setup):
        R, param = [0.1, 1.5], [1.0, 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))

        mp = MonkhorstPack(H, [5, 5, 1])
        ahc(mp)

    @pytest.mark.filterwarnings("ignore", category=ComplexWarning)
    def test_ahc_spin(self, setup):
        R, param = [0.1, 1.5], [[1.0, 2.0], [0.1, 0.2]]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g, spin=Spin.POLARIZED)
        H.construct((R, param))

        mp = MonkhorstPack(H, [3, 3, 1])
        cond = ahc(mp)
        cond2 = ahc(mp, sum=False)
        assert np.allclose(cond, cond2.sum(-1))

    @pytest.mark.filterwarnings("ignore", category=ComplexWarning)
    def test_shc(self, setup):
        R, param = [0.1, 1.5], [[1.0, 0.1, 0, 0], [0.4, 0.2, 0.3, 0.2]]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g, spin=Spin.NONCOLINEAR)
        H.construct((R, param))

        mp = MonkhorstPack(H, [3, 3, 1])
        cond = shc(mp)
        cond2 = shc(mp, sum=False)
        assert np.allclose(cond, cond2.sum(-1))

        cond2 = shc(mp, sigma=Spin.Z)
        assert np.allclose(cond, cond2)

    @pytest.mark.filterwarnings("ignore", category=ComplexWarning)
    def test_shc_and_ahc(self, setup):
        R, param = [0.1, 1.5], [
            [1.0, 0.1, 0, 0, 0, 0, 0, 0],
            [0.4, 0.2, 0.3, 0.3, 0.5, 0.4, 0.2, 0.3],
        ]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g, spin=Spin.SPINORBIT)
        H.construct((R, param))

        mp = MonkhorstPack(H, [3, 3, 1])
        c_ahc = ahc(mp)
        # Ensure that shc calculates AHC in other segments
        c_shc = shc(mp, J_axes="y")
        assert np.allclose(c_ahc[0], c_shc[0])
        assert not np.allclose(c_ahc[1], c_shc[1])
        assert np.allclose(c_ahc[2], c_shc[2])

    @pytest.mark.xfail(reason="Gauges make different decouplings")
    def test_gauge_eff(self, setup):
        # it is not fully clear to me why they are different
        R, param = [0.1, 1.5], [1.0, 0.1]
        g = setup.g.tile(2, 0).tile(2, 1).tile(2, 2)
        H = Hamiltonian(g)
        H.construct((R, param))

        k = [0.1] * 3
        ie1 = H.eigenstate(k, gauge="cell").effective_mass()
        ie2 = H.eigenstate(k, gauge="atom").effective_mass()
        assert np.allclose(abs(ie1), abs(ie2))

    def test_eigenstate_polarized_orthogonal_sk(self, setup):
        R, param = [0.1, 1.5], [1.0, [0.1, 0.1]]
        H = Hamiltonian(setup.g, spin="P")
        H.construct((R, param))

        k = [0.1] * 3
        ie1 = H.eigenstate(k, spin=0, format="array").Sk()
        ie2 = H.eigenstate(k, spin=1, format="array").Sk()
        assert np.allclose(ie1.dot(1.0), 1.0)
        assert np.allclose(ie2.dot(1.0), 1.0)

    def test_eigenstate_polarized_non_ortogonal_sk(self, setup):
        R, param = [0.1, 1.5], [[1.0, 1.0, 1.0], [0.1, 0.1, 0.05]]
        H = Hamiltonian(setup.g, spin="P", orthogonal=False)
        H.construct((R, param))

        k = [0.1] * 3
        ie1 = H.eigenstate(k, spin=0, format="array").Sk()
        ie2 = H.eigenstate(k, spin=1, format="array").Sk()
        assert np.allclose(ie1, ie2)

    def test_change_gauge(self, setup):
        # Test of eigenvalues vs eigenstate class
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((1.0, 1.0), (0.1, 0.1))])
        es = HS.eigenstate()
        es2 = es.copy()
        es2.change_gauge("atom")
        assert np.allclose(es2.state, es.state)

        es = HS.eigenstate(k=(0.2, 0.2, 0.2))
        es2 = es.copy()
        es2.change_gauge("atom")
        assert not np.allclose(es2.state, es.state)
        es2.change_gauge("cell")
        assert np.allclose(es2.state, es.state)

    def test_expectation_value(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), ((1.0, 1.0))])
        D = np.ones(len(H))
        I = np.identity(len(H))
        for k in ([0] * 3, [0.2] * 3):
            es = H.eigenstate(k)

            d = es.inner(matrix=D)
            assert np.allclose(d, D)
            d = es.inner(matrix=D, diag=False)
            assert np.allclose(d, I)

            d = es.inner(matrix=I)
            assert np.allclose(d, D)
            d = es.inner(matrix=I, diag=False)
            assert np.allclose(d, I)

    def test_velocity_orthogonal(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), ((1.0, 1.0))])
        E = np.linspace(-4, 4, 21)
        for k in ([0] * 3, [0.2] * 3):
            es = H.eigenstate(k)
            v = es.velocity()
            vsub = es.sub([0]).velocity()[:, 0]
            assert np.allclose(v[:, 0], vsub)

    @pytest.mark.filterwarnings("ignore", category=ComplexWarning)
    def test_velocity_nonorthogonal(self, setup):
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((1.0, 1.0), (0.1, 0.1))])
        E = np.linspace(-4, 4, 21)
        for k in ([0] * 3, [0.2] * 3):
            es = HS.eigenstate(k)
            v = es.velocity()
            vsub = es.sub([0]).velocity()
            assert np.allclose(v[:, 0], vsub)

    def test_velocity_matrix_orthogonal(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), ((1.0, 1.0))])
        E = np.linspace(-4, 4, 21)
        for k in ([0] * 3, [0.2] * 3):
            es = H.eigenstate(k)
            v = es.velocity(matrix=True)
            vsub = es.sub([0, 1]).velocity(matrix=True)
            assert np.allclose(v[:, :2, :2], vsub)

    @pytest.mark.filterwarnings("ignore", category=ComplexWarning)
    def test_velocity_matrix_nonorthogonal(self, setup):
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((1.0, 1.0), (0.1, 0.1))])
        E = np.linspace(-4, 4, 21)
        for k in ([0] * 3, [0.2] * 3):
            es = HS.eigenstate(k)
            v = es.velocity(matrix=True)
            vsub = es.sub([0, 1]).velocity(matrix=True)
            assert np.allclose(v[:, :2, :2], vsub)

    def test_dos1(self, setup):
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((1.0, 1.0), (0.1, 0.1))])
        E = np.linspace(-4, 4, 21)
        for k in ([0] * 3, [0.2] * 3):
            es = HS.eigenstate(k)
            DOS = es.DOS(E)
            assert DOS.dtype.kind == "f"
            assert np.allclose(es.norm2(), 1)
            str(es)

    def test_pdos1(self, setup):
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((0.0, 1.0), (1.0, 0.1))])
        E = np.linspace(-4, 4, 21)
        for k in ([0] * 3, [0.2] * 3):
            es = HS.eigenstate(k)
            DOS = es.DOS(E, "lorentzian")
            PDOS = es.PDOS(E, "lorentzian")
            assert PDOS.dtype.kind == "f"
            assert PDOS.shape[0] == 1
            assert PDOS.shape[1] == len(HS)
            assert PDOS.shape[2] == len(E)
            assert np.allclose(PDOS.sum(1), DOS)

    def test_pdos2(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), (0.0, 0.1)])
        E = np.linspace(-4, 4, 21)
        for k in ([0] * 3, [0.2] * 3):
            es = H.eigenstate(k)
            DOS = es.DOS(E)
            PDOS = es.PDOS(E)
            assert PDOS.dtype.kind == "f"
            assert np.allclose(PDOS.sum(1), DOS)

    def test_pdos3(self, setup):
        # check whether the default S(Gamma) works
        # In this case we will assume an orthogonal
        # basis, however, the basis is not orthogonal.
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((0.0, 1.0), (1.0, 0.1))])
        E = np.linspace(-4, 4, 21)
        es = HS.eigenstate()
        es.parent = None
        DOS = es.DOS(E)
        PDOS = es.PDOS(E)
        assert not np.allclose(PDOS.sum(1), DOS)

    def test_pdos4(self, setup):
        # check whether the default S(Gamma) works
        # In this case we will assume an orthogonal
        # basis. If the basis *is* orthogonal, then
        # regardless of k, the PDOS will be correct.
        H = setup.H.copy()
        H.construct([(0.1, 1.5), (0.0, 0.1)])
        E = np.linspace(-4, 4, 21)
        es = H.eigenstate()
        es.parent = None
        DOS = es.DOS(E)
        PDOS = es.PDOS(E)
        assert PDOS.dtype.kind == "f"
        assert np.allclose(PDOS.sum(1), DOS)
        es = H.eigenstate([0.25] * 3)
        DOS = es.DOS(E)
        es.parent = None
        PDOS = es.PDOS(E)
        assert PDOS.dtype.kind == "f"
        assert np.allclose(PDOS.sum(1), DOS)

    def test_pdos_nc(self):
        geom = Geometry([0] * 3)
        H = Hamiltonian(geom, spin="nc")
        spin = H.spin
        # this should be Hermitian
        H[0, 0] = np.array([1, 2, 3, 4])
        E = [0]

        def dist(E, *args):
            return np.ones(len(E))

        # just get a fictional PDOS
        es = H.eigenstate()
        PDOS = es.PDOS(E, dist)[..., 0]
        SM = es.spin_moment()
        SMp = es.spin_moment(projection=True)

        # now check with spin stuff
        pdos = es.inner().real
        assert np.allclose(PDOS[0, 0], pdos.sum())

        pdos = es.inner(matrix=spin.X).real
        assert np.allclose(PDOS[1, 0], pdos.sum())
        assert np.allclose(SM[0], pdos)
        assert np.allclose(SMp[0].sum(-1), pdos)

        pdos = es.inner(matrix=spin.Y).real
        assert np.allclose(PDOS[2, 0], pdos.sum())
        assert np.allclose(SM[1], pdos)
        assert np.allclose(SMp[1].sum(-1), pdos)

        pdos = es.inner(matrix=spin.Z).real
        assert np.allclose(PDOS[3, 0], pdos.sum())
        assert np.allclose(SM[2], pdos)
        assert np.allclose(SMp[2].sum(-1), pdos)

    def test_pdos_so(self):
        geom = Geometry([0] * 3)
        H = Hamiltonian(geom, spin="soc")
        spin = H.spin
        # this should be Hermitian
        H[0, 0] = np.array([1, 2, 3, 4, 0, 0, 3, -4])
        E = [0]

        def dist(E, *args):
            return np.ones(len(E))

        # just get a fictional PDOS
        es = H.eigenstate()
        PDOS = es.PDOS(E, dist)[..., 0]
        SM = es.spin_moment()
        SMp = es.spin_moment(projection=True)

        # now check with spin stuff
        pdos = es.inner().real
        assert np.allclose(PDOS[0, 0], pdos.sum())

        pdos = es.inner(matrix=spin.X).real
        assert np.allclose(PDOS[1, 0], pdos.sum())
        assert np.allclose(SM[0], pdos)
        assert np.allclose(SMp[0].sum(-1), pdos)

        pdos = es.inner(matrix=spin.Y).real
        assert np.allclose(PDOS[2, 0], pdos.sum())
        assert np.allclose(SM[1], pdos)
        assert np.allclose(SMp[1].sum(-1), pdos)

        pdos = es.inner(matrix=spin.Z).real
        assert np.allclose(PDOS[3, 0], pdos.sum())
        assert np.allclose(SM[2], pdos)
        assert np.allclose(SMp[2].sum(-1), pdos)

    def test_coop_against_pdos_nonortho(self, setup):
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((0.0, 1.0), (1.0, 0.1))])
        E = np.linspace(-4, 4, 21)
        for k in ([0] * 3, [0.2] * 3):
            es = HS.eigenstate(k)
            COOP = es.COOP(E, "lorentzian")

            DOS = es.DOS(E, "lorentzian")
            COOP2DOS = np.array([C.sum() for C in COOP])
            assert DOS.shape == COOP2DOS.shape
            assert np.allclose(DOS, COOP2DOS)

            # This one returns sparse matrices, so we have to
            # deal with that.
            DOS = es.PDOS(E, "lorentzian")[0]
            COOP2DOS = np.array([C.sum(1).A1 for C in COOP]).T
            assert DOS.shape == COOP2DOS.shape
            assert np.allclose(DOS, COOP2DOS)

    def test_coop_against_pdos_ortho(self, setup):
        H = setup.H.copy()
        H.construct([(0.1, 1.5), (0.0, 1.0)])
        E = np.linspace(-4, 4, 21)
        for k in ([0] * 3, [0.2] * 3):
            es = H.eigenstate(k)
            COOP = es.COOP(E, "lorentzian")

            DOS = es.DOS(E, "lorentzian")
            COOP2DOS = np.array([C.sum() for C in COOP])
            assert DOS.shape == COOP2DOS.shape
            assert np.allclose(DOS, COOP2DOS)

            DOS = es.PDOS(E, "lorentzian")
            COOP2DOS = np.array([C.sum(1).A1 for C in COOP]).T
            assert DOS.shape[1:] == COOP2DOS.shape
            assert np.allclose(DOS, COOP2DOS)

    def test_coop_sp_vs_np(self, setup):
        HS = setup.HS.copy()
        HS.construct([(0.1, 1.5), ((0.0, 1.0), (1.0, 0.1))])
        E = np.linspace(-4, 4, 21)
        for k in ([0] * 3, [0.2] * 3):
            es = HS.eigenstate(k)
            COOP_sp = es.COOP(E, "lorentzian")
            assert issparse(COOP_sp[0])

            es = HS.eigenstate(k, format="array")
            COOP_np = es.COOP(E, "lorentzian")
            assert isinstance(COOP_np[0], np.ndarray)

            for c_sp, c_np in zip(COOP_sp, COOP_np):
                assert np.allclose(c_sp.toarray(), c_np)

    def test_spin1(self, setup):
        g = Geometry(
            [[i, 0, 0] for i in range(10)],
            Atom(6, R=1.01),
            lattice=Lattice(100, nsc=[3, 3, 1]),
        )
        H = Hamiltonian(g, dtype=np.int32, spin=Spin.POLARIZED)
        for i in range(10):
            j = range(i * 2, i * 2 + 3)
            H[0, j] = (i, i * 2)

        H2 = Hamiltonian(g, 2, dtype=np.int32)
        for i in range(10):
            j = range(i * 2, i * 2 + 3)
            H2[0, j] = (i, i * 2)
        assert H.spsame(H2)

    def test_spin2(self, setup):
        g = Geometry(
            [[i, 0, 0] for i in range(10)],
            Atom(6, R=1.01),
            lattice=Lattice(100, nsc=[3, 3, 1]),
        )
        H = Hamiltonian(g, dtype=np.int32, spin=Spin.POLARIZED)
        for i in range(10):
            j = range(i * 2, i * 2 + 3)
            H[0, j] = (i, i * 2)

        H2 = Hamiltonian(g, 2, dtype=np.int32)
        for i in range(10):
            j = range(i * 2, i * 2 + 3)
            H2[0, j] = (i, i * 2)
        assert H.spsame(H2)

        H2 = Hamiltonian(g, Spin(Spin.POLARIZED), dtype=np.int32)
        for i in range(10):
            j = range(i * 2, i * 2 + 3)
            H2[0, j] = (i, i * 2)
        assert H.spsame(H2)

        H2 = Hamiltonian(g, Spin("polarized"), dtype=np.int32)
        for i in range(10):
            j = range(i * 2, i * 2 + 3)
            H2[0, j] = (i, i * 2)
        assert H.spsame(H2)

    def test_transform_up(self):
        g = Geometry(
            [[i, 0, 0] for i in range(10)],
            Atom(6, R=1.01),
            lattice=Lattice(100, nsc=[3, 3, 1]),
        )
        H = Hamiltonian(g, dtype=np.float64, spin=Spin.UNPOLARIZED)
        for i in range(10):
            H[0, i] = i + 0.1
        Hcsr = [H.tocsr(i) for i in range(H.shape[2])]

        for spin in (Spin.POLARIZED, Spin.NONCOLINEAR, Spin.SPINORBIT, Spin.NAMBU):
            Ht = H.transform(spin=spin)
            assert np.abs(Hcsr[0] - Ht.tocsr(0)).sum() == 0
            assert np.abs(Hcsr[0] - Ht.tocsr(1)).sum() == 0

    def test_transform_up_nonortho(self):
        g = Geometry(
            [[i, 0, 0] for i in range(10)],
            Atom(6, R=1.01),
            lattice=Lattice(100, nsc=[3, 3, 1]),
        )
        H = Hamiltonian(g, dtype=np.float64, spin=Spin.UNPOLARIZED, orthogonal=False)
        for i in range(10):
            H[0, i] = (i + 0.1, 1.0)
        Hcsr = [H.tocsr(i) for i in range(H.shape[2])]

        Ht = H.transform(spin=Spin.POLARIZED)
        assert np.abs(Hcsr[0] - Ht.tocsr(0)).sum() == 0
        assert np.abs(Hcsr[0] - Ht.tocsr(1)).sum() == 0
        assert np.abs(Hcsr[-1] - Ht.tocsr(-1)).sum() == 0

        Ht2 = H.transform([[1], [1]], spin=Spin.POLARIZED) - Ht
        assert np.abs(Ht2.tocsr(0)).sum() == 0
        assert np.abs(Ht2.tocsr(1)).sum() == 0
        assert np.abs(Ht2.tocsr(-1)).sum() == 0

        Ht = H.transform(spin=Spin.NONCOLINEAR)
        assert np.abs(Hcsr[0] - Ht.tocsr(0)).sum() == 0
        assert np.abs(Hcsr[0] - Ht.tocsr(1)).sum() == 0
        assert np.abs(Hcsr[-1] - Ht.tocsr(-1)).sum() == 0

        Ht = H.transform(spin=Spin.SPINORBIT)
        assert np.abs(Hcsr[0] - Ht.tocsr(0)).sum() == 0
        assert np.abs(Hcsr[0] - Ht.tocsr(1)).sum() == 0
        assert np.abs(Hcsr[-1] - Ht.tocsr(-1)).sum() == 0

    def test_transform_down(self):
        g = Geometry(
            [[i, 0, 0] for i in range(10)],
            Atom(6, R=1.01),
            lattice=Lattice(100, nsc=[3, 3, 1]),
        )
        H = Hamiltonian(g, dtype=np.float64, spin=Spin.NAMBU)
        for i in range(10):
            for j in range(16):
                H[0, i, j] = i + 0.1 + j
        Hcsr = [H.tocsr(i) for i in range(H.shape[2])]

        Ht = H.transform(spin=Spin.UNPOLARIZED)
        assert np.abs(0.5 * Hcsr[0] + 0.5 * Hcsr[1] - Ht.tocsr(0)).sum() == 0

        Ht = H.transform(spin=Spin.POLARIZED)
        assert np.abs(Hcsr[0] - Ht.tocsr(0)).sum() == 0
        assert np.abs(Hcsr[1] - Ht.tocsr(1)).sum() == 0

        Ht = H.transform(spin=Spin.NONCOLINEAR)
        for i in range(4):
            assert np.abs(Hcsr[i] - Ht.tocsr(i)).sum() == 0

        Ht = H.transform(spin=Spin.SPINORBIT)
        for i in range(8):
            assert np.abs(Hcsr[i] - Ht.tocsr(i)).sum() == 0

    def test_transform_down_nonortho(self):
        g = Geometry(
            [[i, 0, 0] for i in range(10)],
            Atom(6, R=1.01),
            lattice=Lattice(100, nsc=[3, 3, 1]),
        )
        H = Hamiltonian(g, dtype=np.float64, spin=Spin.NAMBU, orthogonal=False)
        for i in range(10):
            for j in range(16):
                H[0, i, j] = i + 0.1 + j
            H[0, i, -1] = 1.0
        Hcsr = [H.tocsr(i) for i in range(H.shape[2])]

        Ht = H.transform(spin=Spin.UNPOLARIZED)
        assert np.abs(0.5 * Hcsr[0] + 0.5 * Hcsr[1] - Ht.tocsr(0)).sum() == 0
        assert np.abs(Hcsr[-1] - Ht.tocsr(-1)).sum() == 0

        Ht = H.transform(spin=Spin.POLARIZED)
        assert np.abs(Hcsr[0] - Ht.tocsr(0)).sum() == 0
        assert np.abs(Hcsr[1] - Ht.tocsr(1)).sum() == 0
        assert np.abs(Hcsr[-1] - Ht.tocsr(-1)).sum() == 0

        Ht = H.transform(spin=Spin.NONCOLINEAR)
        for i in range(4):
            assert np.abs(Hcsr[i] - Ht.tocsr(i)).sum() == 0
        assert np.abs(Hcsr[-1] - Ht.tocsr(-1)).sum() == 0

        Ht = H.transform(spin=Spin.SPINORBIT)
        for i in range(8):
            assert np.abs(Hcsr[i] - Ht.tocsr(i)).sum() == 0
        assert np.abs(Hcsr[-1] - Ht.tocsr(-1)).sum() == 0

    @pytest.mark.parametrize("k", [[0, 0, 0], [0.1, 0, 0]])
    def test_spin_contamination(self, setup, k):
        g = Geometry(
            [[i, 0, 0] for i in range(10)],
            Atom(6, R=1.01),
            lattice=Lattice(1, nsc=[3, 1, 1]),
        )
        H = Hamiltonian(g, spin=Spin.POLARIZED)
        H.construct(([0.1, 1.1], [[0, 0.1], [1, 1.1]]))
        H[0, 0] = (0.1, 0.0)
        H[0, 1] = (0.5, 0.4)
        es_alpha = H.eigenstate(k, spin=0)
        es_beta = H.eigenstate(k, spin=1)

        sup, sdn = spin_contamination(es_alpha.state, es_beta.state, sum=False)
        assert sup.sum() == pytest.approx(sdn.sum())
        assert len(sup) == es_alpha.shape[0]
        assert len(sdn) == es_beta.shape[0]
        s = spin_contamination(es_alpha.state, es_beta.state)
        assert sup.sum() == pytest.approx(s)

        sup, sdn = spin_contamination(
            es_alpha.sub(range(2)).state, es_beta.state, sum=False
        )
        assert sup.sum() == pytest.approx(sdn.sum())
        assert len(sup) == 2
        assert len(sdn) == es_beta.shape[0]

        sup, sdn = spin_contamination(
            es_alpha.sub(range(3)).state, es_beta.sub(range(2)).state, sum=False
        )
        assert sup.sum() == pytest.approx(sdn.sum())
        assert len(sup) == 3
        assert len(sdn) == 2

        sup, sdn = spin_contamination(
            es_alpha.sub(0).state.ravel(), es_beta.sub(range(2)).state, sum=False
        )
        assert sup.sum() == pytest.approx(sdn.sum())
        assert sup.ndim == 1
        assert len(sup) == 1
        assert len(sdn) == 2

        sup, sdn = spin_contamination(
            es_alpha.sub(0).state.ravel(), es_beta.sub(0).state.ravel(), sum=False
        )
        assert sup.sum() == pytest.approx(sdn.sum())
        assert sup.ndim == 0
        assert sdn.ndim == 0

        sup, sdn = spin_contamination(
            es_alpha.sub(range(2)).state, es_beta.sub(0).state.ravel(), sum=False
        )
        assert sup.sum() == pytest.approx(sdn.sum())
        assert len(sup) == 2
        assert len(sdn) == 1

    def test_non_colinear_orthogonal(self, setup, sisl_tolerance):
        atol, rtol = sisl_tolerance[np.complex64]
        allclose = partial(np.allclose, atol=atol, rtol=rtol)

        g = Geometry(
            [[i, 0, 0] for i in range(10)],
            Atom(6, R=1.01),
            lattice=Lattice(100, nsc=[3, 3, 1]),
        )
        H = Hamiltonian(g, dtype=np.float64, spin=Spin.NONCOLINEAR)
        for i in range(10):
            j = range(i * 2, i * 2 + 3)
            H[i, i, 0] = 0.05
            H[i, i, 1] = 0.1
            H[i, i, 2] = 0.1
            H[i, i, 3] = 0.1
            if i > 0:
                H[i, i - 1, 0] = 1.0
                H[i, i - 1, 1] = 1.0
            if i < 9:
                H[i, i + 1, 0] = 1.0
                H[i, i + 1, 1] = 1.0

        eig1 = H.eigh(dtype=np.complex64)
        assert allclose(H.eigh(dtype=np.complex128), eig1)
        assert allclose(H.eigh(gauge="atom", dtype=np.complex128), eig1)
        assert len(eig1) == len(H)

        H1 = Hamiltonian(g, dtype=np.float64, spin=Spin("non-collinear"))
        for i in range(10):
            j = range(i * 2, i * 2 + 3)
            H1[i, i, 0] = 0.05
            H1[i, i, 1] = 0.1
            H1[i, i, 2] = 0.1
            H1[i, i, 3] = 0.1
            if i > 0:
                H1[i, i - 1, 0] = 1.0
                H1[i, i - 1, 1] = 1.0
            if i < 9:
                H1[i, i + 1, 0] = 1.0
                H1[i, i + 1, 1] = 1.0
        assert H1.spsame(H)

        eig1 = H1.eigh(dtype=np.complex64)
        assert allclose(H1.eigh(dtype=np.complex128), eig1)
        assert np.allclose(H.eigh(), H1.eigh())

        # Create the block matrix for expectation
        SZ = block_diag(*([H1.spin.Z] * H1.no))

        for dtype in (np.complex64, np.complex128):
            es = H1.eigenstate(dtype=dtype)
            assert allclose(es.eig, eig1)
            assert np.allclose(es.inner(), 1)

            # Perform spin-moment calculation
            sm = es.spin_moment()
            sm2 = es.inner(matrix=SZ).real
            sm3 = np.diag(np.dot(np.conj(es.state), SZ).dot(es.state.T)).real
            assert np.allclose(sm[2], sm2)
            assert np.allclose(sm[2], sm3)

            om = es.spin_moment(projection=True)
            assert np.allclose(sm, om.sum(-1))

            PDOS = es.PDOS(np.linspace(-1, 1, 21))
            DOS = es.DOS(np.linspace(-1, 1, 21))
            assert np.allclose(PDOS.sum(1)[0, :], DOS)
            es.velocity(matrix=True)

        # Check the velocities
        # But only compare for np.float64, we need the precision
        v = es.velocity()
        vv = es.velocity(matrix=True)
        assert np.allclose(np.diagonal(vv).T, v)

        # Ensure we can change gauge for NC stuff
        es.change_gauge("cell")
        es.change_gauge("atom")

    def test_non_colinear_non_orthogonal(self, sisl_tolerance):
        atol, rtol = sisl_tolerance[np.complex64]
        allclose = partial(np.allclose, atol=atol * 10, rtol=rtol * 100)

        g = Geometry(
            [[i, 0, 0] for i in range(10)],
            Atom(6, R=1.01),
            lattice=Lattice(100, nsc=[3, 3, 1]),
        )
        H = Hamiltonian(g, dtype=np.float64, orthogonal=False, spin=Spin.NONCOLINEAR)
        for i in range(10):
            j = range(i * 2, i * 2 + 3)
            H[i, i, 0] = 0.1
            H[i, i, 1] = 0.05
            H[i, i, 2] = 0.1
            H[i, i, 3] = 0.1
            if i > 0:
                H[i, i - 1, 0] = 1.0
                H[i, i - 1, 1] = 1.0
            if i < 9:
                H[i, i + 1, 0] = 1.0
                H[i, i + 1, 1] = 1.0
            H.S[i, i] = 1.0

        eig1 = H.eigh(dtype=np.complex64)
        assert allclose(H.eigh(dtype=np.complex128), eig1)
        assert len(eig1) == len(H)

        H1 = Hamiltonian(
            g, dtype=np.float64, orthogonal=False, spin=Spin("non-collinear")
        )
        for i in range(10):
            j = range(i * 2, i * 2 + 3)
            H1[i, i, 0] = 0.1
            H1[i, i, 1] = 0.05
            H1[i, i, 2] = 0.1
            H1[i, i, 3] = 0.1
            if i > 0:
                H1[i, i - 1, 0] = 1.0
                H1[i, i - 1, 1] = 1.0
            if i < 9:
                H1[i, i + 1, 0] = 1.0
                H1[i, i + 1, 1] = 1.0
            H1.S[i, i] = 1.0
        assert H1.spsame(H)

        eig1 = H1.eigh(dtype=np.complex64)
        assert allclose(H1.eigh(dtype=np.complex128), eig1)
        assert allclose(H.eigh(dtype=np.complex64), H1.eigh(dtype=np.complex128))

        for dtype in (np.complex64, np.complex128):
            es = H1.eigenstate(dtype=dtype)
            assert allclose(es.eig, eig1)

            sm = es.spin_moment()

            om = es.spin_moment(projection=True)
            assert np.allclose(sm, om.sum(-1))

            PDOS = es.PDOS(np.linspace(-1, 1, 21))
            DOS = es.DOS(np.linspace(-1, 1, 21))
            assert np.allclose(PDOS.sum(1)[0, :], DOS)
            es.velocity(matrix=True)

        # Check the velocities
        # But only compare for np.float64, we need the precision
        v = es.velocity()
        vv = es.velocity(matrix=True)
        assert np.allclose(np.diagonal(vv).T, v)

        # Ensure we can change gauge for NC stuff
        es.change_gauge("cell")
        es.change_gauge("atom")

    def test_spin_orbit_orthogonal(self, sisl_tolerance):
        atol, rtol = sisl_tolerance[np.complex64]
        allclose = partial(np.allclose, atol=atol * 10, rtol=rtol * 100)

        g = Geometry(
            [[i, 0, 0] for i in range(10)],
            Atom(6, R=1.01),
            lattice=Lattice(100, nsc=[3, 3, 1]),
        )
        H = Hamiltonian(g, dtype=np.float64, spin=Spin.SPINORBIT)
        for i in range(10):
            j = range(i * 2, i * 2 + 3)
            H[i, i, 0] = 0.1
            H[i, i, 1] = 0.05
            H[i, i, 2] = 0.1
            H[i, i, 3] = 0.1
            H[i, i, 4] = 0.1
            H[i, i, 5] = 0.1
            H[i, i, 6] = 0.1
            H[i, i, 7] = 0.1
            if i > 0:
                H[i, i - 1, 0] = 1.0
                H[i, i - 1, 1] = 1.0
            if i < 9:
                H[i, i + 1, 0] = 1.0
                H[i, i + 1, 1] = 1.0

        eig1 = H.eigh(dtype=np.complex64)
        assert allclose(H.eigh(dtype=np.complex128), eig1)
        assert len(H.eigh()) == len(H)

        H1 = Hamiltonian(g, dtype=np.float64, spin=Spin("spin-orbit"))
        for i in range(10):
            j = range(i * 2, i * 2 + 3)
            H1[i, i, 0] = 0.1
            H1[i, i, 1] = 0.05
            H1[i, i, 2] = 0.1
            H1[i, i, 3] = 0.1
            H1[i, i, 4] = 0.1
            H1[i, i, 5] = 0.1
            H1[i, i, 6] = 0.1
            H1[i, i, 7] = 0.1
            if i > 0:
                H1[i, i - 1, 0] = 1.0
                H1[i, i - 1, 1] = 1.0
            if i < 9:
                H1[i, i + 1, 0] = 1.0
                H1[i, i + 1, 1] = 1.0
        assert H1.spsame(H)

        eig1 = H1.eigh(dtype=np.complex64)
        assert allclose(H1.eigh(dtype=np.complex128), eig1)
        assert allclose(H.eigh(dtype=np.complex64), H1.eigh(dtype=np.complex128))

        # Create the block matrix for expectation
        SZ = block_diag(*([H1.spin.Z] * H1.no))

        for dtype in (np.complex64, np.complex128):
            es = H.eigenstate(dtype=dtype)
            assert allclose(es.eig, eig1)

            sm = es.spin_moment()
            sm2 = es.inner(matrix=SZ).real
            sm3 = np.diag(np.dot(np.conj(es.state), SZ).dot(es.state.T)).real
            assert np.allclose(sm[2], sm2)
            assert np.allclose(sm[2], sm3)

            om = es.spin_moment(projection=True)
            assert np.allclose(sm, om.sum(-1))

            PDOS = es.PDOS(np.linspace(-1, 1, 21))
            DOS = es.DOS(np.linspace(-1, 1, 21))
            assert np.allclose(PDOS.sum(1)[0, :], DOS)
            es.velocity(matrix=True)

        # Check the velocities
        # But only compare for np.float64, we need the precision
        v = es.velocity()
        vv = es.velocity(matrix=True)
        assert np.allclose(np.diagonal(vv).T, v)

        # Ensure we can change gauge for SO stuff
        es.change_gauge("cell")
        es.change_gauge("atom")

    def test_finalized(self, setup):
        assert not setup.H.finalized
        setup.H.H[0, 0] = 1.0
        setup.H.finalize()
        assert setup.H.finalized
        assert setup.H.nnz == 1
        setup.H.empty()
        assert not setup.HS.finalized
        setup.HS.H[0, 0] = 1.0
        setup.HS.S[0, 0] = 1.0
        setup.HS.finalize()
        assert setup.HS.finalized
        assert setup.HS.nnz == 1
        setup.HS.empty()

    @pytest.mark.slow
    @pytest.mark.parametrize("nx", [1, 4])
    @pytest.mark.parametrize("ny", [1, 5])
    @pytest.mark.parametrize("nz", [1, 6])
    def test_tile_same(self, setup, nx, ny, nz):
        R, param = [0.1, 1.5], [1.0, 0.1]

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
        assert np.allclose(
            Hg.Hk([0.1, 0.2, 0.3], format="array"),
            H.Hk([0.1, 0.2, 0.3], format="array"),
        )

    @pytest.mark.slow
    def test_tile3(self, setup):
        R, param = [0.1, 1.1, 2.1, 3.1], [1.0, 2.0, 3.0, 4.0]

        # Create reference
        g = Geometry([[0] * 3], Atom("H", R=[4.0]), lattice=[1.0] * 3)
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
            self[io, odx] = -1.0
            self[io + 1, odx + 1] = 1.0

            # Set connecting
            odx = self.geometry.a2o(idx[1])
            self[io, odx] = 0.2
            self[io, odx + 1] = 0.01
            self[io + 1, odx] = 0.01
            self[io + 1, odx + 1] = 0.3

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
        R, param = [0.1, 1.5], [1.0, 0.1]

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
        R, param = [0.1, 1.5], [1.0, 0.1]

        # Create reference
        Hg = Hamiltonian(setup.g.repeat(2, 0).repeat(2, 1).repeat(2, 2))
        Hg.construct([R, param])
        Hg.finalize()
        H = Hamiltonian(setup.g)
        H.construct([R, param])
        H = H.repeat(2, 0).repeat(2, 1).repeat(2, 2)
        assert Hg.spsame(H)
        H.finalize()
        Hg.finalize()
        assert np.allclose(H._csr._D, Hg._csr._D)

    @pytest.mark.slow
    def test_repeat3(self, setup):
        R, param = [0.1, 1.1, 2.1, 3.1], [1.0, 2.0, 3.0, 4.0]

        # Create reference
        g = Geometry([[0] * 3], Atom("H", R=[4.0]), lattice=[1.0] * 3)
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
            self[io, odx] = -1.0
            self[io + 1, odx + 1] = 1.0

            # Set connecting
            odx = self.geometry.a2o(idx[1])
            self[io, odx] = 0.2
            self[io, odx + 1] = 0.01
            self[io + 1, odx] = 0.01
            self[io + 1, odx + 1] = 0.3

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
        R, param = [0.1, 1.5], [1.0, 0.1]

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
        R, param = [0.1, 1.5], [1.0, 0.1]

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
        R, param = [0.1, 1.5], [1.0, 0.1]
        H = Hamiltonian(setup.g.copy())
        H.construct([R, param])
        eig0 = H.eigh()[0]
        H.shift(0.2)
        assert H.eigh()[0] == pytest.approx(eig0 + 0.2)

    def test_shift2(self, setup):
        R, param = [0.1, 1.5], [(1.0, 1.0), (0.1, 0.1)]
        H = Hamiltonian(setup.g.copy(), orthogonal=False)
        H.construct([R, param])
        eig0 = H.eigh()[0]
        H.shift(0.2)
        assert H.eigh()[0] == pytest.approx(eig0 + 0.2)

    def test_shift3(self, setup):
        R, param = [0.1, 1.5], [(1.0, -1.0, 1.0), (0.1, 0.1, 0.1)]
        H = Hamiltonian(setup.g.copy(), spin=Spin("P"), orthogonal=False)
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
        R, param = [0.1, 1.5], [(1.0, 1.0), (2.1, 0.1)]
        H = Hamiltonian(setup.g.copy(), orthogonal=False)
        H.construct([R, param])
        bz = MonkhorstPack(H, [10, 10, 1])
        q = 0.9
        Ef = H.fermi_level(bz, q=q)
        H.shift(-Ef)
        assert H.fermi_level(bz, q=q) == pytest.approx(0.0, abs=1e-6)

    def test_fermi_level_spin(self, setup):
        R, param = [0.1, 1.5], [(1.0, 1.0), (2.1, 0.1)]
        H = Hamiltonian(setup.g.copy(), spin=Spin("P"))
        H.construct([R, param])
        bz = MonkhorstPack(H, [10, 10, 1])
        q = 1.1
        Ef = H.fermi_level(bz, q=q)
        assert np.asarray(Ef).ndim == 0
        H.shift(-Ef)
        assert H.fermi_level(bz, q=q) == pytest.approx(0.0, abs=1e-6)

    def test_fermi_level_spin_separate(self, setup):
        R, param = [0.1, 1.5], [(1.0, 1.0), (2.1, 0.1)]
        H = Hamiltonian(setup.g.copy(), spin=Spin("P"))
        H.construct([R, param])
        bz = MonkhorstPack(H, [10, 10, 1])
        q = [0.5, 0.3]
        Ef = H.fermi_level(bz, q=q)
        assert len(Ef) == 2
        H.shift(-Ef)
        assert np.allclose(H.fermi_level(bz, q=q), 0.0)

    def test_wrap_oplist(self, setup):
        R, param = [0.1, 1.5], [1, 2.1]
        H = Hamiltonian(setup.g.copy())
        H.construct([R, param])
        bz = MonkhorstPack(H, [10, 10, 1])
        E = np.linspace(-4, 4, 21)
        dist = get_distribution("gaussian", smearing=0.05)

        def wrap(es, parent, k, weight):
            DOS = es.DOS(E, distribution=dist)
            PDOS = es.PDOS(E, distribution=dist)
            vel = es.velocity() * es.occupation()
            return oplist([DOS, PDOS, vel])

        bz_avg = bz.apply.average
        results = bz_avg.eigenstate(wrap=wrap)
        assert np.allclose(
            bz_avg.eigenstate(wrap=lambda es: es.DOS(E, distribution=dist)), results[0]
        )
        assert np.allclose(
            bz_avg.eigenstate(wrap=lambda es: es.PDOS(E, distribution=dist)), results[1]
        )

    def test_edges1(self, setup):
        R, param = [0.1, 1.5], [1.0, 0.1]
        H = Hamiltonian(setup.g)
        H.construct([R, param])
        assert len(H.edges(0)) == 4

    def test_edges2(self, setup):
        R, param = [0.1, 1.5], [1.0, 0.1]
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
            self[io, odx] = -1.0
            self[io + 1, odx + 1] = 1.0

            # Set connecting
            odx = self.geometry.a2o(idx[1])
            self[io, odx] = 0.2
            self[io, odx + 1] = 0.01
            self[io + 1, odx] = 0.01
            self[io + 1, odx + 1] = 0.3

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
    G = Geometry([[1] * 3, [2] * 3], Atom(6, o1), lattice=[4, 4, 4])
    H = Hamiltonian(G)
    R, param = [0.1, 1.5], [1.0, 0.1]
    H.construct([R, param])
    ES = H.eigenstate(dtype=np.float64)
    # Plot in the full thing
    grid = Grid(0.1, geometry=H.geometry)
    grid.fill(0.0)
    ES.sub(0).wavefunction(grid)


def test_wavefunction2():
    N = 50
    o1 = SphericalOrbital(0, (np.linspace(0, 2, N), np.exp(-np.linspace(0, 100, N))))
    G = Geometry([[1] * 3, [2] * 3], Atom(6, o1), lattice=[4, 4, 4])
    H = Hamiltonian(G)
    R, param = [0.1, 1.5], [1.0, 0.1]
    H.construct([R, param])
    ES = H.eigenstate(dtype=np.float64)
    # This is effectively plotting outside where no atoms exists
    # (there could however still be psi weight).
    grid = Grid(0.1, lattice=Lattice([2, 2, 2], origin=[2] * 3))
    grid.fill(0.0)
    ES.sub(0).wavefunction(grid)


def test_wavefunction3():
    N = 50
    o1 = SphericalOrbital(0, (np.linspace(0, 2, N), np.exp(-np.linspace(0, 100, N))))
    G = Geometry([[1] * 3, [2] * 3], Atom(6, o1), lattice=[4, 4, 4])
    H = Hamiltonian(G, spin=Spin("nc"))
    R, param = [0.1, 1.5], [[0.0, 0.0, 0.1, -0.1], [1.0, 1.0, 0.1, -0.1]]
    H.construct([R, param])
    ES = H.eigenstate()
    # Plot in the full thing
    grid = Grid(0.1, dtype=np.complex128, lattice=Lattice([2, 2, 2], origin=[-1] * 3))
    grid.fill(0.0)
    ES.sub(0).wavefunction(grid)


def test_wavefunction_eta():
    N = 50
    o1 = SphericalOrbital(0, (np.linspace(0, 2, N), np.exp(-np.linspace(0, 100, N))))
    G = Geometry([[1] * 3, [2] * 3], Atom(6, o1), lattice=[4, 4, 4])
    H = Hamiltonian(G, spin=Spin("nc"))
    R, param = [0.1, 1.5], [[0.0, 0.0, 0.1, -0.1], [1.0, 1.0, 0.1, -0.1]]
    H.construct([R, param])
    ES = H.eigenstate()
    # Plot in the full thing
    grid = Grid(0.1, dtype=np.complex128, lattice=Lattice([2, 2, 2], origin=[-1] * 3))
    grid.fill(0.0)
    ES.sub(0).wavefunction(grid, eta=True)


def test_hamiltonian_fromsp_overlap():
    G = Geometry([[1] * 3, [2] * 3], Atom(6), lattice=[4, 4, 4])
    H = Hamiltonian(G, spin=Spin("nc"), orthogonal=False)
    R, param = [0.1, 1.5], [[0.0, 0.0, 0.1, -0.1, 1], [1.0, 1.0, 0.1, -0.1, 0.1]]
    H.construct([R, param])

    # original shape
    shape = list(H.shape)

    # Try and merge into something new
    H1 = H.fromsp(H.geometry, [H, H])
    assert not H.orthogonal
    assert H1.orthogonal
    shape[-1] = H.shape[-1] * 2 - 2
    assert H1.shape == tuple(shape)

    H1 = H.fromsp(H.geometry, [H, H], S=H)
    assert not H.orthogonal
    assert not H1.orthogonal
    shape[-1] = H.shape[-1] * 2 - 1
    assert H1.shape == tuple(shape)
