from __future__ import print_function, division

import pytest

import math as m
import numpy as np

from sisl import geom, Atom, Geometry, Spin
from sisl.physics.sparse import SparseOrbitalBZ, SparseOrbitalBZSpin

pytestmark = pytest.mark.sparse


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
    sp.S[0, 0] = 1.
    sp.S[1, 1] = 1.
    assert sp[1, 1, 0] == pytest.approx(0.5)
    assert sp.S[1, 1] == pytest.approx(1.)


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
    sp.S[0, 0] = 1.
    sp.S[1, 1] = 1.
    assert np.allclose(sp.eigh(), [0.5, 0.5])


@pytest.mark.xfail
def test_eigsh_orthogonal():
    gr = _get()
    # The most simple setup.
    sp = SparseOrbitalBZ(gr)
    sp[0, 0] = 0.5
    sp[1, 1] = 0.5
    # Fails due to too many requested eigenvalues
    sp.eigsh()


@pytest.mark.xfail
def test_eigsh_non_orthogonal():
    gr = _get()
    # The most simple setup.
    sp = SparseOrbitalBZ(gr, orthogonal=False)
    sp.eigsh()


def test_pickle_non_orthogonal():
    import pickle as p
    gr = _get()
    sp = SparseOrbitalBZ(gr, orthogonal=False)
    sp[0, 0] = 0.5
    sp[1, 1] = 0.5
    sp.S[0, 0] = 1.
    sp.S[1, 1] = 1.
    s = p.dumps(sp)
    SP = p.loads(s)
    assert sp.spsame(SP)
    assert np.allclose(sp.eigh(), SP.eigh())


def test_pickle_non_orthogonal_spin():
    import pickle as p
    gr = _get()
    sp = SparseOrbitalBZSpin(gr, spin=Spin('p'), orthogonal=False)
    sp[0, 0, :] = 0.5
    sp[1, 1, :] = 0.5
    sp.S[0, 0] = 1.
    sp.S[1, 1] = 1.
    s = p.dumps(sp)
    SP = p.loads(s)
    assert sp.spsame(SP)
    assert np.allclose(sp.eigh(), SP.eigh())


@pytest.mark.parametrize("n0", [1, 2, 4])
@pytest.mark.parametrize("n1", [1, 2, 4])
@pytest.mark.parametrize("n2", [1, 2, 4])
def test_sparse_orbital_bz_hermitian(n0, n1, n2):
    g = geom.fcc(1., Atom(1, R=1.5)) * 2
    s = SparseOrbitalBZ(g)
    s.construct([[0.1, 1.51], [1, 2]])
    s = s.tile(n0, 0).tile(n1, 1).tile(n2, 2)
    no = s.geometry.no

    nnz = 0
    for io in range(no):
        # orbitals connecting to io
        edges = s.edges(io)
        # Figure out the transposed supercell indices of the edges
        isc = - s.geometry.o2isc(edges)
        # Convert to supercell
        IO = s.geometry.sc.sc_index(isc) * no + io
        # Figure out if 'io' is also in the back-edges
        for jo, edge in zip(IO, edges % no):
            assert jo in s.edges(edge)
            nnz += 1

    # Check that we have counted all nnz
    assert s.nnz == nnz

    # Since we are also dealing with f32 data-types we cannot go beyond 1e-7
    approx_zero = pytest.approx(0., abs=1e-5)
    for k0 in [0, 0.1]:
        for k1 in [0, -0.15]:
            for k2 in [0, 0.33333]:
                k = (k0, k1, k2)

                if np.allclose(k, 0.):
                    dtypes = [None, np.float32, np.float64]
                else:
                    dtypes = [None, np.complex64, np.complex128]

                # Also assert Pk == Pk.H for all data-types
                for dtype in dtypes:
                    Pk = s.Pk(k=k, format='csr', dtype=dtype)
                    assert abs(Pk - Pk.getH()).toarray().max() == approx_zero

                    Pk = s.Pk(k=k, format='array', dtype=dtype)
                    assert np.abs(Pk - np.conj(Pk.T)).max() == approx_zero


def test_sparse_orbital_bz_non_colinear():
    M = SparseOrbitalBZSpin(geom.graphene(), spin=Spin('NC'))
    M.construct(([0.1, 1.44],
                 [[0.1, 0.2, 0.3, 0.4],
                  [0.2, 0.3, 0.4, 0.5]]))

    MT = M.transpose()
    MH = M.transpose(True)

    assert np.abs((M - MT)._csr._D).sum() != 0
    assert np.abs((M - MH)._csr._D).sum() != 0
    assert np.abs((MT - MH)._csr._D).sum() != 0


def test_sparse_orbital_bz_non_colinear_trs_kramers_theorem():
    M = SparseOrbitalBZSpin(geom.graphene(), spin=Spin('NC'))

    M.construct(([0.1, 1.44],
                 [[0.1, 0.2, 0.3, 0.4],
                  [0.2, 0.3, 0.4, 0.5]]))

    M = (M + M.transpose(True)) * 0.5
    MTRS = (M + M.trs()) * 0.5

    # This will in principle also work for M since the above parameters preserve
    # TRS
    k = np.array([0.1, 0.1, 0])
    eig1 = MTRS.eigh(k=k)
    eig2 = MTRS.eigh(k=-k)
    assert np.allclose(eig1, eig2)


def test_sparse_orbital_bz_spin_orbit():
    M = SparseOrbitalBZSpin(geom.graphene(), spin=Spin('SO'))

    M.construct(([0.1, 1.44],
                 [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                  [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]))

    MT = M.transpose()
    MH = M.transpose(True)

    assert np.abs((M - MT)._csr._D).sum() != 0
    assert np.abs((M - MH)._csr._D).sum() != 0
    assert np.abs((MT - MH)._csr._D).sum() != 0


def test_sparse_orbital_bz_spin_orbit_trs_kramers_theorem():
    M = SparseOrbitalBZSpin(geom.graphene(), spin=Spin('SO'))

    M.construct(([0.1, 1.44],
                 [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                  [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]))

    M = (M + M.transpose(True)) / 2
    MTRS = (M + M.trs()) * 0.5

    # This will in principle also work for M since the above parameters preserve
    # TRS
    k = np.array([0.1, 0.1, 0])
    eig1 = MTRS.eigh(k=k)
    eig2 = MTRS.eigh(k=-k)
    assert np.allclose(eig1, eig2)


@pytest.mark.xfail(reason="Sparse.construct does not obey Hermitivity for complex values")
def test_sparse_orbital_bz_spin_orbit():
    M = SparseOrbitalBZSpin(geom.graphene(), spin=Spin('SO'))

    M.construct(([0.1, 1.44],
                 [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                  [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]))
    new = (M + M.transpose(True)) / 2
    assert np.abs((M - new)._csr._D).sum() == 0
