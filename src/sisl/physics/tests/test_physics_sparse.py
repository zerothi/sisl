# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import platform
import sys

import numpy as np
import pytest

from sisl import Atom, SislWarning, Spin, geom
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
def test_sparse_orbital_bz_hermitian(n0, n1, n2):
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
    approx_zero = pytest.approx(0.0, abs=1e-5)
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
                    Pk = s.Pk(k=k, format="csr", dtype=dtype)
                    assert abs(Pk - Pk.getH()).toarray().max() == approx_zero

                    Pk = s.Pk(k=k, format="array", dtype=dtype)
                    assert np.abs(Pk - np.conj(Pk.T)).max() == approx_zero


def test_sparse_orbital_bz_non_colinear():
    M = SparseOrbitalBZSpin(geom.graphene(), spin=Spin("NC"))
    M.construct(([0.1, 1.44], [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]))
    M.finalize()

    MT = M.transpose()
    MH = M.transpose(hermitian=True)

    assert np.abs((M - MT)._csr._D).sum() != 0
    # For a non-collinear with construct we don't take
    # into account the imaginary parts... :(
    # Transposing and Hermitian transpose are the same for NC
    # There are only 1 imaginary part which will change sign regardless
    assert np.abs((MT - MH)._csr._D).sum() != 0
    assert np.abs((M - MH)._csr._D).sum() == 0


def test_sparse_orbital_bz_non_colinear_trs_kramers_theorem():
    M = SparseOrbitalBZSpin(geom.graphene(), spin=Spin("NC"))

    M.construct(([0.1, 1.44], [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]))
    M.finalize()

    M = (M + M.transpose(hermitian=True)) * 0.5
    MTRS = (M + M.trs()) * 0.5

    # This will in principle also work for M since the above parameters preserve
    # TRS
    k = np.array([0.1, 0.1, 0])
    eig1 = MTRS.eigh(k=k)
    eig2 = MTRS.eigh(k=-k)
    assert np.allclose(eig1, eig2)


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

    MT = M.transpose()
    MH = M.transpose(hermitian=True)

    assert np.abs((M - MT)._csr._D).sum() != 0
    assert np.abs((M - MH)._csr._D).sum() == 0
    assert np.abs((MT - MH)._csr._D).sum() != 0


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


@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_sparse_orbital_bz_nambu(dtype):
    M = SparseOrbitalBZSpin(geom.graphene(), spin=Spin("nambu"), dtype=dtype)

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

    if dtype == np.float64:
        p0 = _nambu_cmplx2real(p0)
        p1 = _nambu_cmplx2real(p1)

    M.construct(
        (
            [0.1, 1.44],
            [p0, p1],
        )
    )
    M.finalize()

    MT = M.transpose()
    MH = M.transpose(hermitian=True)

    assert np.abs((M - MT)._csr._D).sum() != 0
    assert np.abs((M - MH)._csr._D).sum() == 0
    assert np.abs((MT - MH)._csr._D).sum() != 0


@pytest.mark.filterwarnings("ignore", message="*is NOT Hermitian for on-site")
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_sparse_orbital_bz_spin_orbit_trs_kramers_theorem(dtype):
    M = SparseOrbitalBZSpin(geom.graphene(), spin="SO", dtype=dtype)

    p0 = np.arange(1, 9) / 10
    p1 = np.arange(2, 10) / 10

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

    M = (M + M.transpose(hermitian=True)) / 2
    MTRS = (M + M.trs()) * 0.5

    # This will in principle also work for M since the above parameters preserve
    # TRS
    k = np.array([0.1, 0.1, 0])
    eig1 = MTRS.eigh(k=k)
    eig2 = MTRS.eigh(k=-k)
    assert np.allclose(eig1, eig2)


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
    new = (M + M.transpose(hermitian=True)) / 2
    assert np.abs((M - new)._csr._D).sum() == 0


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

    Mt = M.transform(spin="so")
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

    Mt = M.transform(spin="so")
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

    Mt = M.transform(spin="so")
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

    Mt = M.transform(spin="so")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mcsr[2] - Mt.tocsr(2)).sum() == 0
    assert np.abs(Mcsr[3] - Mt.tocsr(3)).sum() == 0


@pytest.mark.filterwarnings("ignore", message="*is NOT Hermitian for on-site")
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

    Mt = M.transform(spin="so")
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mcsr[2] - Mt.tocsr(2)).sum() == 0
    assert np.abs(Mcsr[3] - Mt.tocsr(3)).sum() == 0


@pytest.mark.filterwarnings("ignore", message="*is NOT Hermitian for on-site")
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

    Mt = M.transform(spin="so")
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

    Mt = M.transform(spin="non-colinear", dtype=np.float64, orthogonal=True).transform(
        spin="polarized", orthogonal=False
    )
    assert M.dim == Mt.dim
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0

    Mt = M.transform(dtype=np.float128, orthogonal=True).transform(
        spin="so", dtype=np.float64, orthogonal=False
    )
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mt.tocsr(2)).sum() == 0
    assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0

    Mt = M.transform(spin="polarized", orthogonal=True).transform(
        spin="so", dtype=np.float64, orthogonal=False
    )
    assert np.abs(Mcsr[0] - Mt.tocsr(0)).sum() == 0
    assert np.abs(Mcsr[1] - Mt.tocsr(1)).sum() == 0
    assert np.abs(Mt.tocsr(2)).sum() == 0
    assert np.abs(Mcsr[-1] - Mt.tocsr(-1)).sum() == 0

    Mt = M.transform(spin="unpolarized", dtype=np.float32, orthogonal=True).transform(
        dtype=np.complex128, orthogonal=False
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

    Mt = M.transform(matrix=np.ones((3, 3)), dtype=np.float64)
    assert Mt.dim == 3
    assert np.abs(Mcsr[0] + Mcsr[1] + Mcsr[2] - Mt.tocsr(2)).sum() == 0

    Mt = M.transform(
        spin="non-colinear", matrix=np.ones((4, 3)), orthogonal=True, dtype=np.float64
    )
    assert Mt.dim == 4
    assert np.abs(Mcsr[0] + Mcsr[1] + Mcsr[2] - Mt.tocsr(3)).sum() == 0

    Mt = M.transform(spin="non-colinear", matrix=np.ones((5, 3)), dtype=np.float64)
    assert Mt.dim == 5
    assert np.abs(Mcsr[0] + Mcsr[1] + Mcsr[2] - Mt.tocsr(4)).sum() == 0

    Mt = M.transform(
        spin="so", matrix=np.ones((8, 3)), orthogonal=True, dtype=np.float64
    )
    assert Mt.dim == 8
    assert np.abs(Mcsr[0] + Mcsr[1] + Mcsr[2] - Mt.tocsr(7)).sum() == 0

    Mt = M.transform(spin="so", matrix=np.ones((9, 3)), dtype=np.float64)
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
