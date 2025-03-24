# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import SparseEfficiencyWarning

import sisl
from sisl import (
    Atom,
    Bloch,
    BrillouinZone,
    Geometry,
    Hamiltonian,
    Lattice,
    RealSpaceSE,
    RealSpaceSI,
    RecursiveSI,
    SemiInfinite,
    WideBandSE,
)

pytestmark = [
    pytest.mark.physics,
    pytest.mark.self_energy,
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
            func = self.H.create_construct([0.1, bond + 0.1], [0.0, -2.7])
            self.H.construct(func)
            self.HS = Hamiltonian(self.g, orthogonal=False)
            func = self.HS.create_construct(
                [0.1, bond + 0.1], [(0.0, 1.0), (-2.7, 0.0)]
            )
            self.HS.construct(func)

    return t()


def test_objects(setup):
    for D, si, sid in [("+A", 0, 1), ("-A", 0, -1), ("+B", 1, 1), ("-B", 1, -1)]:
        SE = SemiInfinite(setup.H.copy(), D)
        assert SE.semi_inf == si
        assert SE.semi_inf_dir == sid
        assert D in str(SE)


def test_sancho_orthogonal(setup):
    SE = RecursiveSI(setup.H.copy(), "+A")
    assert not np.allclose(SE.self_energy(0.1), SE.self_energy(0.1, bulk=True))


def test_sancho_bloch_zero_off_diag(setup):
    # check whether the bloch-expansion with 0 transverse
    # connections returns a block-diagonal matrix
    H = setup.H.copy()
    # disconnect transverse directions
    H.set_nsc(b=1)
    no = len(H)
    SE = RecursiveSI(H, "+A")

    for nb in [2, 4, 5]:
        bloch = Bloch(1, nb, 1)
        for E in [-0.1, 0.2, 0.4, -0.5]:
            se = bloch(SE.self_energy, [0.2, 0.2, 0.2], E=E)
            for b in range(1, nb):
                off = b * no
                assert np.allclose(se[:no, :no], se[off : off + no, off : off + no])
                se[off : off + no, off : off + no] = 0.0
            se[:no, :no] = 0.0
            assert np.allclose(se, 0.0)


def test_sancho_orthogonal_dtype(setup):
    SE = RecursiveSI(setup.H, "+A")
    s64 = SE.self_energy(0.1, dtype=np.complex64)
    s128 = SE.self_energy(0.1)
    assert s64.dtype == np.complex64
    assert s128.dtype == np.complex128
    assert np.allclose(s64, s128)


def test_sancho_warning():
    lattice = Lattice([1, 1, 10], nsc=[5, 5, 1])
    C = Atom(Z=6, R=[2 * 1.01])
    g = Geometry([[0.0, 0.0, 0.0]], atoms=C, lattice=lattice)
    H = Hamiltonian(g)
    func = H.create_construct([0.1, 1.01, 2.01], [0.0, -2.0, -1.0])
    H.construct(func)

    with pytest.warns(sisl.SislWarning, match=r"first neighboring cell.*\[1.\]"):
        RecursiveSI(H, "+A")


def test_sancho_non_orthogonal(setup):
    SE = RecursiveSI(setup.HS, "-A")
    assert not np.allclose(SE.self_energy(0.1), SE.self_energy(0.1, bulk=True))


def test_sancho_broadening_matrix(setup):
    SE = RecursiveSI(setup.HS, "-A")
    assert np.allclose(SE.broadening_matrix(0.1), SE.se2broadening(SE.self_energy(0.1)))


def test_sancho_non_orthogonal_dtype(setup):
    SE = RecursiveSI(setup.HS, "-A")
    s64 = SE.self_energy(0.1, dtype=np.complex64)
    s128 = SE.self_energy(0.1)
    assert s64.dtype == np.complex64
    assert s128.dtype == np.complex128
    assert np.allclose(s64, s128)


def test_sancho_lr(setup):
    SL = RecursiveSI(setup.HS, "-A")
    SR = RecursiveSI(setup.HS, "+A")

    E = 0.1
    k = [0, 0.13, 0]

    # Check that left/right are different

    L_SE = SL.self_energy(E, k)
    R_SE = SR.self_energy(E, k)
    assert not np.allclose(L_SE, R_SE)

    LB_SEL, LB_SER = SL.self_energy_lr(E, k)
    RB_SEL, RB_SER = SR.self_energy_lr(E, k)

    assert np.allclose(LB_SEL, L_SE)
    assert np.allclose(LB_SER, R_SE)
    assert np.allclose(RB_SEL, L_SE)
    assert np.allclose(RB_SER, R_SE)

    LB_SEL, LB_SER = SL.self_energy_lr(E, k, bulk=True)
    L_SE = SL.self_energy(E, k, bulk=True)
    R_SE = SR.self_energy(E, k, bulk=True)
    assert not np.allclose(L_SE, R_SE)
    RB_SEL, RB_SER = SR.self_energy_lr(E, k, bulk=True)

    assert np.allclose(LB_SEL, L_SE)
    assert np.allclose(LB_SER, R_SE)
    assert np.allclose(RB_SEL, L_SE)
    assert np.allclose(RB_SER, R_SE)


def test_sancho_green(setup):
    SL = RecursiveSI(setup.HS, "-A")
    SR = RecursiveSI(setup.HS, "+A")

    E = 0.1
    k = [0, 0.13, 0]

    # Check that left/right are different

    L_SE = SL.self_energy(E, k, bulk=True)
    R_SE = SR.self_energy(E, k)
    g = np.linalg.inv(L_SE - R_SE)
    G = SL.green(E, k)
    assert np.allclose(g, G)
    assert np.allclose(SL.green(E, k), SR.green(E, k))


def test_wideband_1(setup):
    SE = WideBandSE(10, 1e-2)
    se = SE.self_energy()
    assert se.shape == (10, 10)
    assert np.allclose(np.diag(se), -1j * np.pi * 1e-2)
    assert np.allclose(np.diag(SE.self_energy(1j)), -1j * np.pi * 1.0)
    assert np.allclose(np.diag(SE.broadening_matrix(1j)), np.pi * 2.0)
    # ensure our custom function works!
    assert np.allclose(SE.broadening_matrix(1j), SE.se2broadening(SE.self_energy(1j)))


@pytest.mark.parametrize("k_axes", [0, 1])
@pytest.mark.parametrize("semi_axis", [0, 1])
@pytest.mark.parametrize("trs", [True, False])
@pytest.mark.parametrize("bz", [None, BrillouinZone([1])])
@pytest.mark.parametrize("unfold", [1, 2])
def test_real_space_HS(setup, k_axes, semi_axis, trs, bz, unfold):
    if k_axes == semi_axis:
        return
    RSE = RealSpaceSE(setup.HS, semi_axis, k_axes, (unfold, unfold, unfold))
    RSE.set_options(dk=100, trs=trs, bz=bz)
    RSE.setup()
    RSE.green(0.1)


@pytest.mark.parametrize("k_axes", [0, 1])
@pytest.mark.parametrize("semi_axis", [0, 1])
@pytest.mark.parametrize("trs", [True, False])
@pytest.mark.parametrize("bz", [None, BrillouinZone([1])])
@pytest.mark.parametrize("unfold", [1, 2])
def test_real_space_H(setup, k_axes, semi_axis, trs, bz, unfold):
    if k_axes == semi_axis:
        return
    RSE = RealSpaceSE(
        setup.H, semi_axis, k_axes, (unfold, unfold, 1), trs=trs, dk=100, bz=bz
    )
    RSE.green(0.1)
    RSE.self_energy(0.1)


def test_real_space_H_3d():
    lattice = Lattice(1.0, nsc=[3] * 3)
    H = Atom(Z=1, R=[1.001])
    geom = Geometry([0] * 3, atoms=H, lattice=lattice)
    H = Hamiltonian(geom)
    H.construct(([0.001, 1.01], (0, -1)))
    RSE = RealSpaceSE(H, 0, [1, 2], (3, 4, 2))
    RSE.set_options(dk=100, trs=True)
    RSE.setup()
    nk = np.ones(3, np.int32)
    nk[[1, 2]] = 23
    bz = BrillouinZone(H, nk)
    RSE.set_options(bz=bz)

    RSE.green(0.1)
    # Since there is only 2 repetitions along one direction we will have the full matrix
    # coupled!
    assert np.allclose(RSE.self_energy(0.1), RSE.self_energy(0.1, coupling=True))
    assert np.allclose(
        RSE.self_energy(0.1, bulk=True), RSE.self_energy(0.1, bulk=True, coupling=True)
    )


@pytest.mark.xfail(
    int(np.__version__.split(".")[0]) >= 2,
    reason="some numpy2 bug means it will not honor dtype=complex64",
)
def test_real_space_H_dtype(setup):
    RSE = RealSpaceSE(setup.H, 0, 1, (2, 2, 1), dk=100)
    g64 = RSE.green(0.1, dtype=np.complex64)
    g128 = RSE.green(0.1, dtype=np.complex128)
    assert g64.dtype == np.complex64
    assert g128.dtype == np.complex128
    assert np.allclose(g64, g128, atol=1.0e-4)

    s64 = RSE.self_energy(0.1, dtype=np.complex64)
    s128 = RSE.self_energy(0.1, dtype=np.complex128)
    assert s64.dtype == np.complex64
    assert s128.dtype == np.complex128
    assert np.allclose(s64, s128, atol=1e-2)
    RSE.real_space_coupling()
    RSE.clear()


def test_real_space_H_SE_unfold(setup):
    # check that calculating the real-space Green function is equivalent for two equivalent systems
    RSE = RealSpaceSE(setup.H, 0, 1, (2, 2, 1), dk=100)
    RSE_big = RealSpaceSE(setup.H.tile(2, 0).tile(2, 1), semi_axis=0, k_axes=1, dk=100)

    for E in [0.1, 1.5]:
        G = RSE.green(E)
        G_big = RSE_big.green(E)
        assert np.allclose(G, G_big)

        SE = RSE.self_energy(E)
        SE_big = RSE_big.self_energy(E)
        assert np.allclose(SE, SE_big)


def test_real_space_HS_SE_unfold(setup):
    # check that calculating the real-space Green function is equivalent for two equivalent systems
    RSE = RealSpaceSE(setup.HS, 0, 1, (2, 2, 1), dk=100)
    RSE_big = RealSpaceSE(setup.HS.tile(2, 0).tile(2, 1), semi_axis=0, k_axes=1, dk=100)

    for E in [0.1, 1.5]:
        G = RSE.green(E)
        G_big = RSE_big.green(E)
        assert np.allclose(G, G_big)

        SE = RSE.self_energy(E)
        SE_big = RSE_big.self_energy(E)
        assert np.allclose(SE, SE_big)


def test_real_space_HS_SE_unfold_with_k():
    # check that calculating the real-space Green function is equivalent for two equivalent systems
    sq = Geometry([0] * 3, Atom(1, 1.01), [1])
    sq.set_nsc([3] * 3)
    H = Hamiltonian(sq)
    H.construct([(0.1, 1.1), (4, -1)])

    RSE = RealSpaceSE(H, 0, 1, (3, 4, 1), dk=100, trs=False)

    k1 = [0, 0, 0.2]
    k2 = [0, 0, 0.3]
    for E in [0.1, 1.5]:
        G1 = RSE.green(E, k1)
        G2 = RSE.green(E, k2)
        assert not np.allclose(G1, G2)

        SE1 = RSE.self_energy(E, k1)
        SE2 = RSE.self_energy(E, k2)
        assert not np.allclose(SE1, SE2)


def test_real_space_SE_fail_k_trs():
    sq = Geometry([0] * 3, Atom(1, 1.01), [1])
    sq.set_nsc([3] * 3)
    H = Hamiltonian(sq)
    H.construct([(0.1, 1.1), (4, -1)])

    RSE = RealSpaceSE(H, 0, 1, (3, 4, 1))
    with pytest.raises(ValueError):
        RSE.green(0.1, [0, 0, 0.2])


def test_real_space_SE_fail_k_semi_same():
    sq = Geometry([0] * 3, Atom(1, 1.01), [1])
    sq.set_nsc([3] * 3)
    H = Hamiltonian(sq)
    H.construct([(0.1, 1.1), (4, -1)])

    with pytest.raises(ValueError):
        RSE = RealSpaceSE(H, 0, 0, (3, 4, 1))


def test_real_space_SE_fail_nsc_semi():
    sq = Geometry([0] * 3, Atom(1, 1.01), [1])
    sq.set_nsc([5, 3, 1])
    H = Hamiltonian(sq)
    H.construct([(0.1, 1.1), (4, -1)])

    with pytest.raises(ValueError):
        RSE = RealSpaceSE(H, 0, 1, (3, 4, 1), dk=100)


def test_real_space_SE_fail_nsc_k():
    sq = Geometry([0] * 3, Atom(1, 1.01), [1])
    sq.set_nsc([3, 1, 1])
    H = Hamiltonian(sq)
    H.construct([(0.1, 1.1), (4, -1)])

    with pytest.raises(ValueError):
        RSE = RealSpaceSE(H, 0, 1, (3, 4, 1), dk=100)


def test_real_space_SE_fail_nsc_semi_fully_periodic():
    sq = Geometry([0] * 3, Atom(1, 1.01), [1])
    sq.set_nsc([3, 5, 3])
    H = Hamiltonian(sq)
    H.construct([(0.1, 1.1), (4, -1)])

    with pytest.raises(ValueError):
        RSE = RealSpaceSE(H, 1, 0, (3, 4, 1), dk=100)


def test_real_space_SE_spin_orbit():
    sq = Geometry([0] * 3, Atom(1, 1.01), [1])
    sq.set_nsc([3, 3, 3])
    H = Hamiltonian(sq, spin="SOC")
    on = [4, 0, 0, 0, 0, 0, 0, 0]
    off = [-1, 0, 0, 0, 0, 0, 0, 0]
    H.construct([(0.1, 1.1), (on, off)])
    RS = RealSpaceSE(H, 0, (1, 2), (2, 2, 2), dk=1.4)
    RS.self_energy(0.1)
    RS.self_energy(0.1, coupling=True)


@pytest.mark.parametrize("k_axes", [0])
@pytest.mark.parametrize("trs", [True, False])
@pytest.mark.parametrize("bz", [None, BrillouinZone([1])])
@pytest.mark.parametrize("unfold", [1, 3])
@pytest.mark.parametrize("bulk", [True, False])
@pytest.mark.parametrize("coupling", [True, False])
def test_real_space_SI_HS(setup, k_axes, trs, bz, unfold, bulk, coupling):
    semi = RecursiveSI(setup.HS, "-B")
    surf = setup.HS.tile(4, 1)
    surf.set_nsc(b=1)
    RSI = RealSpaceSI(semi, surf, k_axes, (unfold, 1, unfold))
    RSI.setup(dk=100, trs=trs, bz=bz)
    RSI.self_energy(0.1, bulk=bulk, coupling=coupling)


@pytest.mark.parametrize("semi_dir", ["-B", "+B"])
@pytest.mark.parametrize("k_axes", [0])
@pytest.mark.parametrize("trs", [True, False])
@pytest.mark.parametrize("bz", [None, BrillouinZone([1])])
@pytest.mark.parametrize("unfold", [1, 3])
@pytest.mark.parametrize("bulk", [True, False])
@pytest.mark.parametrize("semi_bulk", [True, False])
@pytest.mark.parametrize("coupling", [True, False])
def test_real_space_SI_H(
    setup, semi_dir, k_axes, trs, bz, unfold, bulk, semi_bulk, coupling
):
    semi = RecursiveSI(setup.H, semi_dir)
    surf = setup.H.tile(4, 1)
    surf.set_nsc(b=1)
    RSI = RealSpaceSI(semi, surf, k_axes, (unfold, 1, unfold))
    RSI.set_options(dk=100, trs=trs, bz=bz, semi_bulk=semi_bulk)
    RSI.setup()
    RSI.self_energy(0.1, bulk=bulk, coupling=coupling)


def test_real_space_SI_H_test(setup):
    semi = RecursiveSI(setup.H, "-B")
    surf = setup.H.tile(4, 1)
    surf.set_nsc(b=1)
    RSI = RealSpaceSI(semi, surf, 0, (3, 1, 3))
    RSI.set_options(dk=100, trs=False, bz=None)
    RSI.setup()
    RSI.green(0.1, [0, 0, 0.1])
    RSI.self_energy(0.1, [0, 0, 0.1])
    RSI.clear()


def test_real_space_SI_H_k_trs(setup):
    semi = RecursiveSI(setup.H, "-B")
    surf = setup.H.tile(4, 1)
    surf.set_nsc(b=1)
    RSI = RealSpaceSI(semi, surf, 0, (3, 1, 3))
    RSI.setup(dk=100, trs=True, bz=None)
    with pytest.raises(ValueError):
        RSI.green(0.1, [0, 0, 0.1])


def test_real_space_SI_fail_semi_in_k(setup):
    semi = RecursiveSI(setup.H, "-B")
    surf = setup.H.tile(4, 1)
    surf.set_nsc(b=1)
    with pytest.raises(ValueError):
        RSI = RealSpaceSI(semi, surf, [0, 1], (2, 1, 1))


def test_real_space_SI_fail_surf_nsc(setup):
    semi = RecursiveSI(setup.H, "-B")
    surf = setup.H.tile(4, 1)
    with pytest.raises(ValueError):
        RSI = RealSpaceSI(semi, surf, 0, (2, 1, 1))


def test_real_space_SI_fail_k_no_nsc(setup):
    semi = RecursiveSI(setup.H, "-B")
    surf = setup.H.tile(4, 1)
    surf.set_nsc([1] * 3)
    with pytest.raises(ValueError):
        RSI = RealSpaceSI(semi, surf, 0, (2, 1, 1))


def test_real_space_SI_fail_unfold_in_semi(setup):
    semi = RecursiveSI(setup.H, "-B")
    surf = setup.H.tile(4, 1)
    surf.set_nsc(b=1)
    with pytest.raises(ValueError):
        RSI = RealSpaceSI(semi, surf, 0, (2, 2, 1))


def test_real_space_SI_spin_orbit():
    sq = Geometry([0] * 3, Atom(1, 1.01), [1])
    sq.set_nsc([3, 3, 1])
    H = Hamiltonian(sq, spin="SOC")
    on = [4, 0, 0, 0, 0, 0, 0, 0]
    off = [-1, 0, 0, 0, 0, 0, 0, 0]
    H.construct([(0.1, 1.1), (on, off)])
    semi = RecursiveSI(H, "-B")
    surf = H.tile(4, 1)
    surf.set_nsc(b=1)
    RS = RealSpaceSI(semi, surf, 0, (2, 1, 1), dk=1.5)
    RS.self_energy(0.1)
    RS.self_energy(0.1, coupling=True)
