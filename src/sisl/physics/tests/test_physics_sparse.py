# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import platform
import sys

import numpy as np
import pytest
import scipy.sparse as sps

from sisl import Atom, SislWarning, SparseCSR, Spin, geom
from sisl.physics.sparse import SparseOrbitalBZ, SparseOrbitalBZSpin

pytestmark = [pytest.mark.physics, pytest.mark.sparse]


def _get():
    gr = geom.graphene()
    return gr


def test_str():
    gr = _get()
    # The most simple setup.
    sp = SparseOrbitalBZ(gr)
    str(sp)


def run_Pk_hermitian_tests(M):
    for k in ([0, 0, 0], [0.1, 0.2, 0.3]):
        Mk = M.Pk(k=k, format="array")
        assert np.allclose(Mk, Mk.T.conj())
        Mk = M.Pk(k=k).toarray()
        assert np.allclose(Mk, Mk.T.conj())


def test_S():
    gr = _get()
    # The most simple setup.
    sp = SparseOrbitalBZ(gr, orthogonal=False)
    sp[0, 0] = 0.5
    sp[1, 1] = 0.5
    sp.S[0, 0] = 1.0
    sp.S[1, 1] = 1.0
    assert sp[1, 1, 0] == pytest.approx(0.5)
    assert sp.S[1, 1] == pytest.approx(1.0)


def test_eigh_orthogonal():
    gr = _get()
    # The most simple setup.
    sp = SparseOrbitalBZ(gr)
    sp[0, 0] = 0.5
    sp[1, 1] = 0.5
    assert np.allclose(sp.eigh(), [0.5, 0.5])
    assert np.allclose(sp.eigh([0.5] * 3), [0.5, 0.5])


def test_eigh_non_orthogonal():
    gr = _get()
    # The most simple setup.
    sp = SparseOrbitalBZ(gr, orthogonal=False)
    sp[0, 0] = 0.5
    sp[1, 1] = 0.5
    sp.S[0, 0] = 1.0
    sp.S[1, 1] = 1.0
    assert np.allclose(sp.eigh(), [0.5, 0.5])


def test_eigsh_orthogonal():
    gr = _get()
    # The most simple setup.
    sp = SparseOrbitalBZ(gr)
    sp[0, 0] = 0.5
    sp[1, 1] = 0.5
    # Fails due to too many requested eigenvalues
    with pytest.raises(TypeError):
        sp.eigsh(n=3)


def test_eigsh_non_orthogonal():
    sp = SparseOrbitalBZ(_get(), orthogonal=False)
    sp.construct([(0.1, 1.44), ([0, 1.0], [-2.7, 0])])
    sp.eigsh(n=1)


def test_pickle_non_orthogonal():
    import pickle as p

    gr = _get()
    sp = SparseOrbitalBZ(gr, orthogonal=False)
    sp[0, 0] = 0.5
    sp[1, 1] = 0.5
    sp.S[0, 0] = 1.0
    sp.S[1, 1] = 1.0
    s = p.dumps(sp)
    SP = p.loads(s)
    assert sp.spsame(SP)
    assert np.allclose(sp.eigh(), SP.eigh())


def test_pickle_non_orthogonal_spin():
    import pickle as p

    gr = _get()
    sp = SparseOrbitalBZSpin(gr, spin=Spin("p"), orthogonal=False)
    sp[0, 0, :] = 0.5
    sp[1, 1, :] = 0.5
    sp.S[0, 0] = 1.0
    sp.S[1, 1] = 1.0
    s = p.dumps(sp)
    SP = p.loads(s)
    assert sp.spsame(SP)
    assert np.allclose(sp.eigh(), SP.eigh())


@pytest.mark.parametrize("n0", [1, 2])
@pytest.mark.parametrize("n1", [1, 3])
@pytest.mark.parametrize("n2", [1, 4])
def test_sparse_orbital_bz_hermitian(sisl_allclose, n0, n1, n2):
    g = geom.fcc(1.0, Atom(1, R=1.5)) * 2
    s = SparseOrbitalBZ(g)
    s.construct([[0.1, 1.51], [1, 2]])
    s = s.tile(n0, 0).tile(n1, 1).tile(n2, 2)
    no = s.geometry.no

    nnz = 0
    for io in range(no):
        # orbitals connecting to io
        edges = s.edges(io)
        # Figure out the transposed supercell indices of the edges
        isc = -s.geometry.o2isc(edges)
        # Convert to supercell
        IO = s.geometry.lattice.sc_index(isc) * no + io
        # Figure out if 'io' is also in the back-edges
        for jo, edge in zip(IO, edges % no):
            assert jo in s.edges(edge)
            nnz += 1

    # Check that we have counted all nnz
    assert s.nnz == nnz

    # Since we are also dealing with f32 data-types we cannot go beyond 1e-7
    for k0 in [0, 0.1]:
        for k1 in [0, -0.15]:
            for k2 in [0, 0.33333]:
                k = (k0, k1, k2)

                if np.allclose(k, 0.0):
                    dtypes = [None, np.float32, np.float64]
                else:
                    dtypes = [None, np.complex64, np.complex128]

                # Also assert Pk == Pk.H for all data-types
                for dtype in dtypes:
                    allclose = sisl_allclose[dtype]

                    Pk = s.Pk(k=k, format="csr", dtype=dtype)
                    assert allclose(Pk.toarray(), Pk.getH().toarray())

                    Pk = s.Pk(k=k, format="array", dtype=dtype)
                    assert allclose(Pk, Pk.T.conj())


def test_sparse_orbital_bz_non_colinear(sisl_allclose):
    M = SparseOrbitalBZSpin(geom.graphene(), spin=Spin("NC"))
    M.construct(([0.1, 1.44], [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]))
    M.finalize()

    MT = M.transpose(spin=True)
    MH = M.transpose(conjugate=True, spin=True)

    allclose = sisl_allclose[M.dtype]

    assert not allclose((M - MT)._csr._D, 0)
    # For a non-collinear with construct we don't take
    # into account the imaginary parts... :(
    # Transposing and Hermitian transpose are the same for NC
    # There are only 1 imaginary part which will change sign regardless
    assert not allclose((MT - MH)._csr._D, 0)
    assert allclose((M - MH)._csr._D, 0)
    run_Pk_hermitian_tests(M)
    run_Pk_hermitian_tests(MH)


def _so_real2cmplx(p):
    return [p[0] + 1j * p[4], p[1] + 1j * p[5], p[2] + 1j * p[3], p[6] + 1j * p[7]]


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_sparse_orbital_bz_spin_orbit_warns_hermitian(dtype):
    M = SparseOrbitalBZSpin(geom.graphene(), spin=Spin("SO"), dtype=dtype)

    p0 = np.arange(1, 9) / 10
    p1 = np.arange(2, 10) / 10

    if dtype == np.complex128:
        p0 = _so_real2cmplx(p0)
        p1 = _so_real2cmplx(p1)

    with pytest.warns(SislWarning, match="Hermitian"):
        M.construct(([0.1, 1.44], [p0, p1]))


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_sparse_orbital_bz_spin_orbit(dtype):
    M = SparseOrbitalBZSpin(geom.graphene(), spin=Spin("SO"), dtype=dtype)

    p0 = [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.3, -0.4]
    p1 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if dtype == np.complex128:
        p0 = _so_real2cmplx(p0)
        p1 = _so_real2cmplx(p1)

    M.construct(
        (
            [0.1, 1.44],
            [p0, p1],
        )
    )
    M.finalize()

    MT = M.transpose(spin=True)
    MH = M.transpose(conjugate=True, spin=True)

    assert np.abs((M - MT)._csr._D).sum() != 0
    assert np.abs((M - MH)._csr._D).sum() == 0
    assert np.abs((MT - MH)._csr._D).sum() != 0
    run_Pk_hermitian_tests(M)
    run_Pk_hermitian_tests(MH)


def test_sparse_orbital_bz_spin_orbit_astype():
    Mr = SparseOrbitalBZSpin(geom.graphene(), spin=Spin("SO"), dtype=np.float64)
    Mc = SparseOrbitalBZSpin(geom.graphene(), spin=Spin("SO"), dtype=np.complex128)

    p0 = [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.3, -0.4]
    p1 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    Mr.construct(
        (
            [0.1, 1.44],
            [p0, p1],
        )
    )
    p0 = _so_real2cmplx(p0)
    p1 = _so_real2cmplx(p1)
    Mc.construct(
        (
            [0.1, 1.44],
            [p0, p1],
        )
    )

    assert np.allclose(Mr.astype(np.complex128)._csr._D, Mc._csr._D)
    assert np.allclose(Mc.astype(np.float64)._csr._D, Mr._csr._D)
    assert np.allclose(Mr.astype(np.complex128).astype(np.float64)._csr._D, Mr._csr._D)
    assert np.allclose(Mc.astype(np.float64).astype(np.complex128)._csr._D, Mc._csr._D)


def _nambu_cmplx2real(p):
    return [
        p[0].real,
        p[1].real,
        p[2].real,
        p[2].imag,
        p[0].imag,
        p[1].imag,
        p[3].real,
        p[3].imag,
        p[4].real,
        p[4].imag,
        p[5].real,
        p[5].imag,
        p[6].real,
        p[6].imag,
        p[7].real,
        p[7].imag,
    ]


@pytest.mark.filterwarnings("ignore", message="*non-Hermitian on-site")
def test_sparse_orbital_bz_nambu_astype():
    Mr = SparseOrbitalBZSpin(geom.graphene(), spin=Spin("nambu"), dtype=np.float64)
    Mc = SparseOrbitalBZSpin(geom.graphene(), spin=Spin("nambu"), dtype=np.complex128)

    p0 = [
        0.1 + 1j * 0.0,
        0.2 + 1j * 0.0,
        0.3 + 1j * 0.4,
        0.3 - 1j * 0.4,
        # onsite S must have zero real
        # onsite triplet states must have 0 imaginary
        1j * 0.6,
        0.3,
        0.4,
        0.3,
    ]

    p1 = [
        0.2 + 1j * 0.6,
        0.3 + 1j * 0.7,
        0.4 + 1j * 0.5,
        0.3 + 1j * 0.9,
        0.3 + 1j * 0.7,
        0.4 + 1j * 0.8,
        0.5 + 1j * 0.6,
        0.4 + 1j * 1.0,
    ]

    Mc.construct(
        (
            [0.1, 1.44],
            [p0, p1],
        )
    )
    p0 = _nambu_cmplx2real(p0)
    p1 = _nambu_cmplx2real(p1)
    Mr.construct(
        (
            [0.1, 1.44],
            [p0, p1],
        )
    )

    assert np.allclose(Mr.astype(np.complex128)._csr._D, Mc._csr._D)
    assert np.allclose(Mc.astype(np.float64)._csr._D, Mr._csr._D)
    assert np.allclose(Mr.astype(np.complex128).astype(np.float64)._csr._D, Mr._csr._D)
    assert np.allclose(Mc.astype(np.float64).astype(np.complex128)._csr._D, Mc._csr._D)


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_sparse_orbital_bz_spin_orbit_hermitian_not(dtype):
    M = SparseOrbitalBZSpin(geom.graphene(), spin="SO", dtype=dtype)

    p0 = [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.3, -0.4]
    p1 = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    if dtype == np.complex128:
        p0 = _so_real2cmplx(p0)
        p1 = _so_real2cmplx(p1)

    M.construct(
        (
            [0.1, 1.44],
            [p0, p1],
        )
    )
    M.finalize()
    new = (M + M.transpose(conjugate=True, spin=True)) / 2
    assert np.abs((M - new)._csr._D).sum() == 0


@pytest.mark.filterwarnings("ignore", message="*non-Hermitian on-site")
@pytest.mark.parametrize(
    "spin", ["unpolarized", "polarized", "non-colinear", "spin-orbit", "nambu"]
)
@pytest.mark.parametrize("finalize", [True, False])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_sparse_orbital_spin_make_hermitian(spin, finalize, dtype, sisl_allclose):
    M = SparseOrbitalBZSpin(
        geom.graphene(), spin=spin, dtype=np.complex128, orthogonal=False
    )
    ns = M.shape[-1]

    p0 = np.random.rand(ns) + 1j * np.random.rand(ns)
    p1 = np.random.rand(ns) + 1j * np.random.rand(ns)

    M.construct(
        (
            [0.1, 1.44],
            [p0, p1],
        )
    )
    if finalize:
        M.finalize()
    # We have to do it after to ensure we correctly populate NC+Nambu
    M = M.astype(dtype)

    allclose = sisl_allclose[M.dtype]

    MH = (M + M.transpose(conjugate=True, spin=True)) / 2
    assert allclose((MH - MH.transpose(conjugate=True, spin=True))._csr._D, 0)

    for format, proc in (("array", lambda x: x), ("csr", lambda x: x.toarray())):
        for k in ([0, 0, 0], [0.1, 0.2, 0.3]):
            Mk = proc(MH.Pk(k=k, format=format))
            assert allclose(Mk, Mk.T.conj())
            Mk = proc(MH.Sk(k=k, format=format))
            assert allclose(Mk, Mk.T.conj())


@pytest.mark.filterwarnings("ignore", message="*non-Hermitian on-site")
@pytest.mark.parametrize(
    "spin", ["unpolarized", "polarized", "non-colinear", "spin-orbit", "nambu"]
)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_sparse_orbital_spin_make_trs(spin, dtype, sisl_allclose):
    M = SparseOrbitalBZSpin(
        geom.graphene(), spin=spin, dtype=np.complex128, orthogonal=False
    )
    ns = M.shape[-1]

    p0 = np.random.rand(ns) + 1j * np.random.rand(ns)
    p1 = np.random.rand(ns) + 1j * np.random.rand(ns)
    # overlap matrices needs to have certain structure, otherwise
    # diagonalization will fail!
    p0[-1] = 1 + 1j * p0[-1].imag
    p1[-1] = 0.2 + 1j * p1[-1].imag

    M.construct(
        (
            [0.1, 1.44],
            [p0, p1],
        )
    )
    M.finalize()
    M = M.astype(dtype)

    allclose = sisl_allclose[M.dtype]

    MTRS = M.trs()
    if spin == "unpolarized":
        if np.dtype(dtype).kind == "c":
            assert not allclose((M - MTRS)._csr._D, 0)
    else:
        assert not allclose((M - MTRS)._csr._D, 0)
    MTRS = (M + MTRS) / 2
    assert allclose((MTRS - MTRS.trs())._csr._D, 0)

    k = np.array([0.1, 0.2, 0.3])
    eig1 = MTRS.eigh(k=k)
    eig2 = MTRS.eigh(k=-k)
    assert allclose(eig1, eig2)


@pytest.mark.filterwarnings("ignore", message="*non-Hermitian on-site")
@pytest.mark.parametrize(
    "spin", ["unpolarized", "polarized", "non-colinear", "spin-orbit", "nambu"]
)
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("orthogonal", [True, False])
def test_sparse_orbital_spin_transpose(spin, dtype, orthogonal, sisl_allclose):
    M = SparseOrbitalBZSpin(
        geom.graphene(), spin=spin, orthogonal=orthogonal, dtype=np.complex128
    )
    ns = M.shape[-1]

    p0 = np.random.rand(ns) + 1j * np.random.rand(ns)
    p1 = np.random.rand(ns) + 1j * np.random.rand(ns)

    M.construct(
        (
            [0.1, 1.44],
            [p0, p1],
        )
    )
    M.finalize()

    # We have to do it after to ensure we correctly populate NC+Nambu
    M = M.astype(dtype)

    allclose = sisl_allclose[M.dtype]

    cFsF = {"conjugate": False, "spin": False}
    cTsF = {"conjugate": True, "spin": False}
    cFsT = {"conjugate": False, "spin": True}
    cTsT = {"conjugate": True, "spin": True}

    MT = M.transpose(**cFsF)
    MH = M.transpose(**cTsT)
    MS = M.transpose(**cFsT)
    MC = M.transpose(**cTsF)

    # just do the same op, twice, should back-transform to the same matrix
    # just do the same op, twice, should back-transform to the same matrix
    assert allclose((M - MT.transpose(**cFsF))._csr._D, 0)
    assert allclose((M - MS.transpose(**cFsT))._csr._D, 0)
    assert allclose((M - MH.transpose(**cTsT))._csr._D, 0)
    assert allclose((M - MC.transpose(**cTsF))._csr._D, 0)

    # And different conversions
    # MT ^H = M (spin-transpose and conjugate)
    assert allclose((MT.transpose(**cTsT) - MS.transpose(**cTsF))._csr._D, 0)
    assert allclose((MT.transpose(**cTsT) - MH.transpose(**cFsF))._csr._D, 0)
    assert allclose((MT.transpose(**cTsT) - MC.transpose(**cFsT))._csr._D, 0)
    # MS ^* = M (spin-transpose and conjugate)
    assert allclose((MS.transpose(**cTsF) - MT.transpose(**cTsT))._csr._D, 0)
    assert allclose((MS.transpose(**cTsF) - MH.transpose(**cFsF))._csr._D, 0)
    assert allclose((MS.transpose(**cTsF) - MC.transpose(**cFsT))._csr._D, 0)


def test_sparse_orbital_transform_ortho_unpolarized():
    M = SparseOrbitalBZSpin(geom.graphene(), spin="unpolarized")
    a = np.arange(M.spin.size(M.dtype)) + 0.3
    M.construct(([0.1, 1.44], [a, a + 0.1]))
    M.finalize()
    Mcsr = [M.tocsr(i) for i in range(M.shape[2])]

    Mt = M.transform(spin="unpolarized")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0

    Mt = M.transform(spin="polarized")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[0] - Mt.tocsr(1)).sum() == 0

    Mt = M.transform(spin="non-colinear")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[0] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mt.tocsr(2)).sum() == 0
    assert np.abs(Mt.tocsr(-1)).sum() == 0

    for s in ("so", "nambu"):
        Mt = M.transform(spin=s)
        assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
        assert np.abs(Mcsr[0] - Mt.tocsr(1)).sum() == 0
        assert np.abs(Mt.tocsr(2)).sum() == 0
        assert np.abs(Mt.tocsr(-1)).sum() == 0


def test_sparse_orbital_transform_nonortho_unpolarized():
    M = SparseOrbitalBZSpin(geom.graphene(), spin="unpolarized", orthogonal=False)
    a = np.arange(M.spin.size(M.dtype) + 1) + 0.3
    M.construct(([0.1, 1.44], [a, a + 0.1]))
    M.finalize()
    Mcsr = [M.tocsr(i) for i in range(M.shape[2])]

    Mt = M.transform(spin="unpolarized")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0

    Mt = M.transform(spin="polarized")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[0] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0

    Mt = M.transform(spin="non-colinear")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[0] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mt.tocsr(2)).sum() == 0
    assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0

    for s in ("so", "nambu"):
        Mt = M.transform(spin=s)
        assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
        assert np.abs(Mcsr[0] - Mt.tocsr(1)).sum() == 0
        assert np.abs(Mt.tocsr(2)).sum() == 0
        assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0


def test_sparse_orbital_transform_ortho_polarized():
    M = SparseOrbitalBZSpin(geom.graphene(), spin="polarized")
    a = np.arange(M.spin.size(M.dtype)) + 0.3
    M.construct(([0.1, 1.44], [a, a + 0.1]))
    M.finalize()
    Mcsr = [M.tocsr(i) for i in range(M.shape[2])]

    Mt = M.transform(spin="unpolarized")
    assert np.abs(0.5 * Mcsr[0] + 0.5 * Mcsr[1] - Mt.tocsr(0)).sum() == 0

    Mt = M.transform(spin="polarized")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0

    Mt = M.transform(spin="non-colinear")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mt.tocsr(2)).sum() == 0
    assert np.abs(Mt.tocsr(-1)).sum() == 0

    for s in ("so", "nambu"):
        Mt = M.transform(spin=s)
        assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
        assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
        assert np.abs(Mt.tocsr(2)).sum() == 0
        assert np.abs(Mt.tocsr(-1)).sum() == 0


def test_sparse_orbital_transform_ortho_nc():
    M = SparseOrbitalBZSpin(geom.graphene(), spin="non-colinear")
    a = np.arange(M.spin.size(M.dtype)) + 0.3
    M.construct(([0.1, 1.44], [a, a + 0.1]))
    M.finalize()
    Mcsr = [M.tocsr(i) for i in range(M.shape[2])]

    Mt = M.transform(spin="unpolarized")
    assert np.abs(0.5 * Mcsr[0] + 0.5 * Mcsr[1] - Mt.tocsr(0)).sum() == 0

    Mt = M.transform(spin="polarized")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0

    Mt = M.transform(spin="non-colinear")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mcsr[2] - Mt.tocsr(2)).sum() == 0
    assert np.abs(Mcsr[3] - Mt.tocsr(3)).sum() == 0

    for s in ("so", "nambu"):
        Mt = M.transform(spin=s)
        assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
        assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
        assert np.abs(Mcsr[2] - Mt.tocsr(2)).sum() == 0
        assert np.abs(Mcsr[3] - Mt.tocsr(3)).sum() == 0


@pytest.mark.filterwarnings("ignore", message="*non-Hermitian on-site")
def test_sparse_orbital_transform_ortho_so():
    M = SparseOrbitalBZSpin(geom.graphene(), spin="so")
    a = np.arange(M.spin.size(M.dtype)) + 0.3
    M.construct(([0.1, 1.44], [a, a + 0.1]))
    M.finalize()
    Mcsr = [M.tocsr(i) for i in range(M.shape[2])]

    Mt = M.transform(spin="unpolarized")
    assert np.abs(0.5 * Mcsr[0] + 0.5 * Mcsr[1] - Mt.tocsr(0)).sum() == 0

    Mt = M.transform(spin="polarized")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0

    Mt = M.transform(spin="non-colinear")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mcsr[2] - Mt.tocsr(2)).sum() == 0
    assert np.abs(Mcsr[3] - Mt.tocsr(3)).sum() == 0

    for s in ("so", "nambu"):
        Mt = M.transform(spin=s)
        assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
        assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
        assert np.abs(Mcsr[2] - Mt.tocsr(2)).sum() == 0
        assert np.abs(Mcsr[3] - Mt.tocsr(3)).sum() == 0


@pytest.mark.filterwarnings("ignore", message="*non-Hermitian on-site")
def test_sparse_orbital_transform_nonortho_so():
    M = SparseOrbitalBZSpin(geom.graphene(), spin="so", orthogonal=False)
    a = np.arange(M.spin.size(M.dtype) + 1) + 0.3
    M.construct(([0.1, 1.44], [a, a + 0.1]))
    M.finalize()
    Mcsr = [M.tocsr(i) for i in range(M.shape[2])]

    Mt = M.transform(spin="unpolarized")
    assert np.abs(0.5 * Mcsr[0] + 0.5 * Mcsr[1] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0

    Mt = M.transform(spin="polarized")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0

    Mt = M.transform(spin="non-colinear")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mcsr[2] - Mt.tocsr(2)).sum() == 0
    assert np.abs(Mcsr[3] - Mt.tocsr(3)).sum() == 0
    assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0

    for s in ("so", "nambu"):
        Mt = M.transform(spin=s)
        assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
        assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
        assert np.abs(Mcsr[2] - Mt.tocsr(2)).sum() == 0
        assert np.abs(Mcsr[3] - Mt.tocsr(3)).sum() == 0
        assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0


def test_sparse_orbital_transform_basis():
    M = SparseOrbitalBZSpin(geom.graphene(), spin="polarized", orthogonal=False)
    M.construct(([0.1, 1.44], [(3.0, 2.0, 1.0), (0.3, 0.2, 0.0)]))
    M.finalize()
    Mcsr = [M.tocsr(i) for i in range(M.shape[2])]

    Mt = M.transform(orthogonal=True).transform(orthogonal=False)
    assert M.dim == Mt.dim
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0


@pytest.mark.skipif(
    sys.platform.startswith("win") or "arm" in platform.machine().lower(),
    reason="Data type cannot be float128",
)
def test_sparse_orbital_transform_combinations():
    M = SparseOrbitalBZSpin(
        geom.graphene(), spin="polarized", orthogonal=False, dtype=np.int32
    )
    M.construct(([0.1, 1.44], [(3, 2, 1), (2, 1, 0)]))
    M.finalize()
    Mcsr = [M.tocsr(i) for i in range(M.shape[2])]

    Mt = (
        M.transform(spin="non-colinear", orthogonal=True)
        .astype(np.float64)
        .transform(spin="polarized", orthogonal=False)
    )
    assert M.dim == Mt.dim
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0

    Mt = (
        M.transform(orthogonal=True)
        .astype(np.float128)
        .transform(spin="so", orthogonal=False)
        .astype(np.float64)
    )
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mt.tocsr(2)).sum() == 0
    assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0

    Mt = (
        M.transform(spin="polarized", orthogonal=True)
        .transform(spin="so", orthogonal=False)
        .astype(np.float64)
    )
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mt.tocsr(2)).sum() == 0
    assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0

    Mt = (
        M.transform(spin="unpolarized", orthogonal=True)
        .astype(np.float32)
        .transform(orthogonal=False)
        .astype(np.complex128)
    )
    assert np.abs(0.5 * Mcsr[0] + 0.5 * Mcsr[1] - Mt.tocsr(0)).sum() == 0


def test_sparse_orbital_transform_matrix():
    M = SparseOrbitalBZSpin(
        geom.graphene(), spin="polarized", orthogonal=False, dtype=np.int32
    )
    M.construct(([0.1, 1.44], [(1, 2, 3), (4, 5, 6)]))
    M.finalize()
    Mcsr = [M.tocsr(i) for i in range(M.shape[2])]

    Mt = M.transform(spin="unpolarized", matrix=np.ones((1, 3)), orthogonal=True)
    assert Mt.dim == 1
    assert np.abs(Mcsr[0] + Mcsr[1] + Mcsr[2] - Mt.tocsr(0)).sum() == 0

    Mt = M.transform(spin="polarized", matrix=np.ones((2, 3)), orthogonal=True)
    assert Mt.dim == 2
    assert np.abs(Mcsr[0] + Mcsr[1] + Mcsr[2] - Mt.tocsr(1)).sum() == 0

    Mt = M.transform(matrix=np.ones((3, 3)))
    assert Mt.dim == 3
    assert np.abs(Mcsr[0] + Mcsr[1] + Mcsr[2] - Mt.tocsr(2)).sum() == 0

    Mt = M.transform(spin="non-colinear", matrix=np.ones((4, 3)), orthogonal=True)
    assert Mt.dim == 4
    assert np.abs(Mcsr[0] + Mcsr[1] + Mcsr[2] - Mt.tocsr(3)).sum() == 0

    Mt = M.transform(spin="non-colinear", matrix=np.ones((5, 3)))
    assert Mt.dim == 5
    assert np.abs(Mcsr[0] + Mcsr[1] + Mcsr[2] - Mt.tocsr(4)).sum() == 0

    Mt = M.transform(spin="so", matrix=np.ones((8, 3)), orthogonal=True)
    assert Mt.dim == 8
    assert np.abs(Mcsr[0] + Mcsr[1] + Mcsr[2] - Mt.tocsr(7)).sum() == 0

    Mt = M.transform(spin="so", matrix=np.ones((9, 3)))
    assert Mt.dim == 9
    assert np.abs(Mcsr[0] + Mcsr[1] + Mcsr[2] - Mt.tocsr(8)).sum() == 0


def test_sparse_orbital_transform_fail():
    M = SparseOrbitalBZSpin(
        geom.graphene(), spin="polarized", orthogonal=False, dtype=np.int32
    )
    M.construct(([0.1, 1.44], [(1, 2, 3), (4, 5, 6)]))
    M.finalize()

    with pytest.raises(ValueError):
        M.transform(np.zeros([2, 2]), spin="unpolarized")


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize(
    "spin", ["unpolarized", "polarized", "non-colinear", "spin-orbit"]
)
def test_sparseorbital_spin_dtypes(dtype, spin):
    gr = geom.graphene()

    M = SparseOrbitalBZSpin(gr, spin=Spin(spin), dtype=dtype)
    assert M.dtype == dtype


def test_sparseorbital_fromsp_csr_matrix():
    gr = geom.graphene()
    no, no_s = gr.no, gr.no_s
    s1 = sps.csr_matrix((no, no_s))

    M = SparseOrbitalBZ.fromsp(gr, s1)
    assert M.shape == (no, no_s, 1)
    assert M.nnz == 0

    M = SparseOrbitalBZ.fromsp(gr, [s1, s1], orthogonal=False)
    assert M.shape == (no, no_s, 2)

    M = SparseOrbitalBZ.fromsp(gr, [s1, s1], S=s1, orthogonal=False)
    assert M.shape == (no, no_s, 3)


def test_sparseorbital_fromsp_sparsecsr():
    gr = geom.graphene()
    no, no_s = gr.no, gr.no_s
    s1 = SparseCSR((no, no_s, 2))

    M = SparseOrbitalBZ.fromsp(gr, s1)
    assert M.shape == (no, no_s, 2)
    assert M.nnz == 0

    M = SparseOrbitalBZ.fromsp(gr, [s1, s1])
    assert M.shape == (no, no_s, 4)

    M = SparseOrbitalBZ.fromsp(gr, [s1, s1], orthogonal=False)
    assert M.shape == (no, no_s, 4)


def test_sparseorbital_fromsp_sparsecsr_overlap_error():
    gr = geom.graphene()
    no, no_s = gr.no, gr.no_s
    s1 = SparseCSR((no, no_s, 2))
    with pytest.raises(ValueError):
        SparseOrbitalBZ.fromsp(gr, s1, S=s1)


def test_sparseorbital_fromsp_combined():
    gr = geom.graphene()
    no, no_s = gr.no, gr.no_s
    s1 = SparseCSR((no, no_s, 2))
    s2 = sps.csr_matrix((no, no_s))

    M = SparseOrbitalBZ.fromsp(gr, [s1, s2])
    assert M.shape == (no, no_s, 3)
    assert M.nnz == 0

    M = SparseOrbitalBZ.fromsp(gr, s1, S=s2)
    assert M.shape == (no, no_s, 3)

    M = SparseOrbitalBZ.fromsp(gr, [s1, s2], S=s2)
    assert M.shape == (no, no_s, 4)


def test_sparseorbital_fromsp_orthogonal():
    gr = geom.graphene()
    no, no_s = gr.no, gr.no_s
    s1 = SparseCSR((no, no_s, 2))
    s2 = sps.csr_matrix((no, no_s))

    M = SparseOrbitalBZ.fromsp(gr, [s1, s2], orthogonal=False)
    assert M.shape == (no, no_s, 3)
    assert M.nnz == 0
    assert not M.orthogonal

    M1 = SparseOrbitalBZ.fromsp(gr, M)
    assert M1.shape == (no, no_s, 2)
    assert M1.orthogonal

    # we only extract the matrix elements (not overlap)
    # and then add the overlap part explicitly
    M1 = SparseOrbitalBZ.fromsp(gr, M, S=M)
    assert M1.shape == (no, no_s, 3)
    assert not M1.orthogonal
