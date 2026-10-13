# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as ssp

import sisl as si
from sisl.conftest import sisl_importskip
from sisl_toolbox.btd import (
    BlochElectrode,
    DeviceGreen,
    DownfoldElectrode,
    Electrode,
    PermuteBTD,
    RealSpaceElectrode,
)

pytestmark = [pytest.mark.btd]

# number of cells along the transport direction (c)
# 0: left electrode, 1: left downfolding, 2-5: device,
# 6: right downfolding, 7: right electrode
NCELL = 8
ETA = 0.05
K = (0.15, 0, 0)


def _electrode(hop, spin="unpolarized", orthogonal=True):
    """Electrode with 2 atoms (2 orbitals each) along c, periodic along a

    For non-orthogonal electrodes, only unpolarized spin is implemented.
    """
    atom = si.Atom(1, R=[1.1, 1.1])
    geom = si.Geometry([[0, 0, 0], [0, 0, 1]], atom, lattice=[1, 10, 2])
    geom.set_nsc([3, 1, 3])
    H = si.Hamiltonian(geom, spin=spin, orthogonal=orthogonal)
    onsite = np.array([[0.1, 0.3], [0.3, 0.5]])
    # overlap matrix elements (positive definite)
    S_onsite = np.array([[1.0, 0.05], [0.05, 1.0]])
    S_hop = -0.08 * hop
    for ia in geom:
        for ja in geom.close(ia, R=1.1):
            ib = geom.a2o(ia)
            jb = geom.a2o(ja)
            block = onsite if ia == ja else hop
            S_block = S_onsite if ia == ja else S_hop
            for io in range(2):
                for jo in range(2):
                    if not orthogonal:
                        H[ib + io, jb + jo] = (block[io, jo], S_block[io, jo])
                    elif spin == "unpolarized":
                        H[ib + io, jb + jo] = block[io, jo]
                    elif spin == "polarized":
                        H[ib + io, jb + jo] = (block[io, jo], 0.7 * block[io, jo])
                    else:
                        # non-colinear, with a small off-diagonal spin part
                        H[ib + io, jb + jo] = (
                            block[io, jo],
                            0.9 * block[io, jo],
                            0.1 * block[io, jo],
                            0.05 * block[io, jo],
                        )
    return H


def _system(hop, spin="unpolarized", orthogonal=True):
    Hel = _electrode(hop, spin, orthogonal)
    # Bloch expand along a
    H = Hel.tile(2, 0).tile(NCELL, 2)
    H.set_nsc(c=1)
    # perturb the device region
    rng = np.random.default_rng(1234)
    no_cell = 8
    for io in range(2 * no_cell, 6 * no_cell):
        if not orthogonal:
            H[io, io, 0] = H[io, io, 0] + rng.random()
        elif spin == "unpolarized":
            H[io, io] = H[io, io] + rng.random()
        elif spin == "polarized":
            H[io, io] = H[io, io] + rng.random(2)
        else:
            H[io, io] = H[io, io] + np.append(rng.random(2), [0, 0])
    return Hel, H


def _cell_orbs(H, cell):
    no_cell = H.no // NCELL
    return np.arange(cell * no_cell, (cell + 1) * no_cell)


def _expand(H, orbs):
    """Spin expanded orbital indices"""
    n = len(H) // H.no
    return (np.repeat(orbs * n, n).reshape(-1, n) + np.arange(n)).ravel()


def _electrodes(Hel, H, bulk, rng, last=None):
    """Create left and right downfolded electrodes

    Within each block the orbitals are shuffled.
    """
    ret = []
    for name, semi_inf, cells in (
        ("Left", "-C", (0, 1, 2)),
        ("Right", "+C", (7, 6, 5)),
    ):
        se = si.RecursiveSI(Hel, semi_inf, eta=ETA)
        elec = BlochElectrode(se, bloch=(2, 1, 1), name=name)
        blocks = [_cell_orbs(H, c) for c in cells]
        if last is not None:
            blocks[-1] = blocks[-1][last]
        blocks = [rng.permutation(b) for b in blocks]
        orbs_down = np.concatenate(blocks)
        blocks_down = [len(b) for b in blocks]
        down = DownfoldElectrode(
            elec,
            H,
            {"orbitals": _cell_orbs(H, cells[0])},
            {"orbitals": PermuteBTD(orbs_down, blocks_down)},
            eta_device=ETA,
            bulk=bulk,
        )
        ret.append((down, {"orbitals": blocks[-1]}))
    return ret


def _reference(Hel, H, E, k, **kwargs):
    """Inverse Green function of the full system (with bulk electrode self-energies)"""
    M = (E + 1j * ETA) * H.Sk(k, format="array") - H.Hk(k, format="array", **kwargs)
    for semi_inf, cell in (("-C", 0), ("+C", NCELL - 1)):
        se = si.RecursiveSI(Hel.tile(2, 0), semi_inf, eta=ETA)
        o = _expand(H, _cell_orbs(H, cell))
        M[np.ix_(o, o)] -= se.self_energy(E, k=k, **kwargs)
    return M


@pytest.mark.parametrize("spin", ["unpolarized", "nc"])
def test_bloch_self_energy(spin):
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel = _electrode(hop, spin)
    elec = BlochElectrode(
        si.RecursiveSI(Hel, "-C", eta=ETA), bloch=(2, 1, 1), name="Left"
    )
    ref = si.RecursiveSI(Hel.tile(2, 0), "-C", eta=ETA)
    assert len(elec) == len(ref)
    assert np.allclose(elec.self_energy(0.3, k=K), ref.self_energy(0.3, k=K))
    # bulk matrix includes the electrode eta
    Href = Hel.tile(2, 0)
    # principal layer, without couplings along the semi-infinite direction
    Href.set_nsc(c=1)
    Mref = (0.3 + 1j * ETA) * Href.Sk(K, format="array") - Href.Hk(K, format="array")
    assert elec.has_bulk
    assert np.allclose(elec.bulk_matrix(0.3, k=K), Mref)


def test_electrode_abstract():
    with pytest.raises(TypeError):
        Electrode("Left")


def test_real_space_unknown_se():
    with pytest.raises(ValueError):
        RealSpaceElectrode(None, name="Left")


def _write_gf(path, H, se, E, k=(0, 0, 0), **kwargs):
    """Write a GF file of the bulk self-energies of `se` (Gamma-point only)"""
    with si.get_sile(path, mode="w") as gf:
        gf.write_header(si.BrillouinZone(H, [k]), E)
        for _, new_k, k, e in gf:
            if new_k:
                gf.write_hamiltonian(
                    H.Hk(k, format="array", dtype=np.complex128),
                    H.Sk(k, format="array", dtype=np.complex128),
                )
            gf.write_self_energy(se.self_energy(e, k=k, bulk=True, **kwargs))
    return si.get_sile(path)


def _real_space_se():
    g = si.geom.graphene(orthogonal=True)
    H = si.Hamiltonian(g)
    H.construct([[0.1, 1.43], [0, -2.7]])
    # coarse k-sampling for speed
    return si.RealSpaceSE(H, 0, 1, (3, 3, 1), eta=ETA, dk=20)


def test_real_space_self_energy():
    se = _real_space_se()
    elec = RealSpaceElectrode(se, name="Left")
    Hc = se.real_space_coupling()
    assert len(elec) == len(Hc)
    assert elec.has_bulk
    assert np.isclose(elec.eta, ETA)
    assert np.allclose(elec.self_energy(0.3), se.self_energy(0.3, coupling=True))
    # E S - H - Σ is the bulk real-space self-energy
    assert np.allclose(
        elec.bulk_matrix(0.3) - elec.self_energy(0.3),
        se.self_energy(0.3, bulk=True, coupling=True),
    )


def test_real_space_gf(tmp_path):
    se = _real_space_se()
    Hc = se.real_space_coupling()
    E = np.linspace(-1, 1, 3)
    gf = _write_gf(tmp_path / "rs.TBTGF", Hc, se, E + 1j * ETA, coupling=True)
    ref = RealSpaceElectrode(se, name="Left")
    elec = RealSpaceElectrode(gf, name="Left")
    assert len(elec) == len(ref)
    assert elec.has_bulk
    assert np.isclose(elec.eta, ETA)
    # also read energies in reverse order
    for e in np.append(E, E[::-1]):
        assert np.allclose(elec.self_energy(e), ref.self_energy(e))
        assert np.allclose(elec.bulk_matrix(e), ref.bulk_matrix(e))
    assert ssp.issparse(elec.bulk_matrix(E[0], format="csr"))


def test_bloch_gf(tmp_path):
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel = _electrode(hop, orthogonal=False)
    se = si.RecursiveSI(Hel, "-C", eta=ETA)
    E = np.linspace(-1, 1, 3)
    gf = _write_gf(tmp_path / "elec.TBTGF", se.spgeom0, se, E + 1j * ETA)
    ref = BlochElectrode(se, name="Left")
    elec = BlochElectrode(gf, name="Left")
    with pytest.raises(ValueError):
        BlochElectrode(gf, bloch=(2, 1, 1), name="Left")
    assert len(elec) == len(ref)
    assert elec.has_bulk
    for e in E:
        assert np.allclose(elec.self_energy(e), ref.self_energy(e))
        assert np.allclose(elec.bulk_matrix(e), ref.bulk_matrix(e))


@pytest.mark.parametrize("bulk", [True, False])
def test_downfold_atoms_auto(bulk):
    # atoms and automatic downfolding regions equals the explicit orbitals
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel, H = _system(hop)
    se = si.RecursiveSI(Hel, "-C", eta=ETA)
    elec = BlochElectrode(se, bloch=(2, 1, 1), name="Left")
    cells = [_cell_orbs(H, c) for c in range(3)]
    atoms = [np.unique(H.o2a(c)) for c in cells]
    kwargs = dict(eta_device=ETA, bulk=bulk)
    ref = DownfoldElectrode(
        elec, H, {"orbitals": cells[0]}, {"orbitals": PermuteBTD(cells)}, **kwargs
    )
    SE_ref = ref.self_energy(0.3, k=K)

    # atoms (default) are expanded to their orbitals
    for down in (
        DownfoldElectrode(elec, H, atoms[0], PermuteBTD(atoms), **kwargs),
        DownfoldElectrode(
            elec, H, {"atoms": atoms[0]}, {"atoms": PermuteBTD(atoms)}, **kwargs
        ),
    ):
        assert np.array_equal(down.orbs, ref.orbs)
        assert np.allclose(down.self_energy(0.3, k=K), SE_ref)

    # automatic order and blocks of the region, the last block are the
    # nodes coupling to the region (a subset of the next cell)
    for down in (
        DownfoldElectrode(elec, H, atoms[0], np.concatenate(atoms[:2]), **kwargs),
        DownfoldElectrode(
            elec,
            H,
            {"orbitals": cells[0]},
            {"orbitals": np.concatenate(cells[:2])},
            **kwargs,
        ),
    ):
        assert np.isin(down.orbs, cells[2]).all()
        idx = np.searchsorted(ref.orbs, down.orbs)
        SE = down.self_energy(0.3, k=K)
        assert np.allclose(SE, SE_ref[np.ix_(idx, idx)])
        # the remaining orbitals have no self-energy
        rest = np.setdiff1d(np.arange(len(ref.orbs)), idx)
        assert np.allclose(SE_ref[rest], 0)

    with pytest.raises(ValueError, match="contain the electrode"):
        DownfoldElectrode(elec, H, atoms[0], atoms[1], **kwargs)
    with pytest.raises(ValueError, match="not connected"):
        DownfoldElectrode(
            elec, H, atoms[0], np.concatenate([atoms[0], atoms[2]]), **kwargs
        )
    # orbitals passed as atoms (out of range)
    with pytest.raises(ValueError, match="orbitals"):
        DownfoldElectrode(
            elec, H, _cell_orbs(H, NCELL - 1), PermuteBTD(cells), **kwargs
        )


def test_downfold_nested():
    # downfolding a downfolded electrode equals downfolding everything at once
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel, H = _system(hop)
    se = si.RecursiveSI(Hel, "-C", eta=ETA)
    elec = BlochElectrode(se, bloch=(2, 1, 1), name="Left")
    cells = [_cell_orbs(H, c) for c in range(4)]
    kwargs = dict(eta_device=ETA, bulk=True)
    ref = DownfoldElectrode(
        elec, H, {"orbitals": cells[0]}, {"orbitals": PermuteBTD(cells)}, **kwargs
    )
    down1 = DownfoldElectrode(
        elec, H, {"orbitals": cells[0]}, {"orbitals": PermuteBTD(cells[:3])}, **kwargs
    )
    with pytest.raises(ValueError, match="bulk"):
        DownfoldElectrode(
            down1,
            H,
            {"orbitals": cells[2]},
            {"orbitals": PermuteBTD(cells[2:])},
            **kwargs,
        )
    down2 = DownfoldElectrode(
        down1,
        H,
        {"orbitals": cells[2]},
        {"orbitals": PermuteBTD(cells[2:])},
        eta_device=ETA,
        bulk=False,
    )
    assert down2.name == "Left"
    assert np.allclose(down2.self_energy(0.3, k=K), ref.self_energy(0.3, k=K))


@pytest.mark.parametrize("bulk", [True, False])
@pytest.mark.parametrize("spin", ["unpolarized", "nc"])
def test_downfold_self_energy(bulk, spin):
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel, H = _system(hop, spin)
    rng = np.random.default_rng(42)
    E = 0.3
    M = _reference(Hel, H, E, K)
    for down, _ in _electrodes(Hel, H, bulk, rng):
        # downfold everything but the device orbitals
        if down.name == "Left":
            orbs_x = np.concatenate([_cell_orbs(H, 0), _cell_orbs(H, 1)])
        else:
            orbs_x = np.concatenate([_cell_orbs(H, 7), _cell_orbs(H, 6)])
        x = _expand(H, orbs_x)
        d = _expand(H, down.orbs)
        ref = M[np.ix_(d, x)] @ np.linalg.solve(M[np.ix_(x, x)], M[np.ix_(x, d)])
        assert len(down) == len(d)
        assert np.allclose(down.self_energy(E, k=K), ref)


def test_downfold_partial_atoms():
    # only the first orbital couples between atoms, so the
    # downfolded self-energy only lives on the first orbitals
    hop = np.array([[-1.0, 0.0], [0.0, 0.0]])
    Hel, H = _system(hop)
    rng = np.random.default_rng(42)
    E = 0.3
    M = _reference(Hel, H, E, K)
    for down, _ in _electrodes(Hel, H, True, rng, last=slice(0, None, 2)):
        if down.name == "Left":
            orbs_x = np.concatenate([_cell_orbs(H, 0), _cell_orbs(H, 1)])
        else:
            orbs_x = np.concatenate([_cell_orbs(H, 7), _cell_orbs(H, 6)])
        x = orbs_x
        d = down.orbs
        ref = M[np.ix_(d, x)] @ np.linalg.solve(M[np.ix_(x, x)], M[np.ix_(x, d)])
        assert np.allclose(down.self_energy(E, k=K), ref)


@pytest.mark.parametrize("bulk", [True, False])
@pytest.mark.parametrize("spin", ["unpolarized", "nc"])
@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("in_device", ["inner", "outer"])
def test_downfold_partial_electrode(bulk, spin, explicit, in_device):
    # Only one layer of the left electrode is downfolded, the other layer is
    # in the device region (the last block).
    # The self-energy of the -C electrode lives on the outer layer (z=0) only,
    # so for in_device == "outer" the electrode self-energy in the device is non-zero.
    Hel, H = _system(np.array([[-1.0, -0.2], [-0.2, -0.7]]), spin)
    E = 0.3
    M = _reference(Hel, H, E, K)
    elec = BlochElectrode(
        si.RecursiveSI(Hel, "-C", eta=ETA), bloch=(2, 1, 1), name="Left"
    )
    outer = H.a2o([0, 2], all=True)
    inner = H.a2o([1, 3], all=True)
    if in_device == "inner":
        orbs_x, orbs_dev = outer, inner
        # the last block: only the inner layer couples to the outer layer
        orbs_d = inner
    else:
        orbs_x, orbs_dev = inner, outer
        # the last block: the inner layer couples to both outer layers
        orbs_d = np.concatenate([outer, H.a2o([4, 6], all=True)])
    if explicit:
        down = {"orbitals": PermuteBTD([orbs_x, orbs_d[::-1]])}
    else:
        # the last block is found from the coupling to the region
        down = {"orbitals": orbs_x}
    down = DownfoldElectrode(
        elec, H, {"orbitals": _cell_orbs(H, 0)}, down, eta_device=ETA, bulk=bulk
    )
    assert np.array_equal(np.sort(down.orbs), np.sort(orbs_d))
    assert np.array_equal(down.orbs_down, orbs_x)

    # Schur complement of the full system
    x = _expand(H, orbs_x)
    d = _expand(H, down.orbs)
    Hd = (E + 1j * ETA) * H.Sk(K, format="array") - H.Hk(K, format="array")
    downfolded = M[np.ix_(d, x)] @ np.linalg.solve(M[np.ix_(x, x)], M[np.ix_(x, d)])
    SE = down.self_energy(E, k=K)
    assert np.allclose(SE, Hd[np.ix_(d, d)] - M[np.ix_(d, d)] + downfolded)

    # explicitly: the real self-energy (on the electrode orbitals in the
    # device region) + the downfolded self-energy
    se = elec.self_energy(E, k=K)
    # rows of the electrode self-energy are _cell_orbs(H, 0) (spin expanded)
    is_el = d < len(se)
    se_d = np.zeros_like(SE)
    se_d[np.ix_(is_el, is_el)] = se[np.ix_(d[is_el], d[is_el])]
    assert np.allclose(se_d, 0) == (in_device == "inner")
    assert np.allclose(SE, se_d + downfolded)

    # and it reproduces the Green function of the full system
    right = _electrodes(Hel, H, bulk, np.random.default_rng(42))[1]
    device = np.concatenate([orbs_dev] + [_cell_orbs(H, c) for c in range(1, 6)])
    G = DeviceGreen(H, [down, right], {"orbitals": device}, eta=ETA)
    # the device permutation is spin expanded
    Gref = np.linalg.inv(M)
    assert np.allclose(G.green(E, k=K), Gref[np.ix_(G.permute, G.permute)])


def test_downfold_partial_electrode_errors():
    Hel, H = _system(np.array([[-1.0, -0.2], [-0.2, -0.7]]))
    elec = BlochElectrode(
        si.RecursiveSI(Hel, "-C", eta=ETA), bloch=(2, 1, 1), name="Left"
    )
    orbs = _cell_orbs(H, 0)
    orbs_x = H.a2o([0, 2], all=True)
    orbs_d = H.a2o([1, 3], all=True)
    # the electrode is fully in the last block
    with pytest.raises(ValueError, match="nothing to downfold"):
        DownfoldElectrode(
            elec,
            H,
            {"orbitals": orbs},
            {"orbitals": PermuteBTD([_cell_orbs(H, 1), orbs])},
        )
    # the downfolded electrode orbitals must neighbor the last block
    with pytest.raises(ValueError, match="before the last block"):
        DownfoldElectrode(
            elec,
            H,
            {"orbitals": orbs},
            {"orbitals": PermuteBTD([orbs_x, _cell_orbs(H, 1), orbs_d])},
        )


def test_downfold_spin_cache():
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel, H = _system(hop, "polarized")
    rng = np.random.default_rng(42)
    (down, _), _ = _electrodes(Hel, H, True, rng)
    up = down.self_energy(0.3, k=K, spin=0)
    dn = down.self_energy(0.3, k=K, spin=1)
    assert not np.allclose(up, dn)
    assert np.allclose(down.self_energy(0.3, k=K, spin=0), up)


@pytest.mark.parametrize("bulk", [True, False])
@pytest.mark.parametrize("spin", ["unpolarized", "nc"])
def test_device_green(bulk, spin):
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel, H = _system(hop, spin)
    rng = np.random.default_rng(42)
    elecs = _electrodes(Hel, H, bulk, rng)

    # device region, shuffled within each block
    blocks = [rng.permutation(_cell_orbs(H, c)) for c in range(2, 6)]
    permbtd = PermuteBTD(np.concatenate(blocks), [len(b) for b in blocks])
    G = DeviceGreen(H, elecs, {"orbitals": permbtd}, eta=ETA)

    E = 0.3
    Gref = np.linalg.inv(_reference(Hel, H, E, K))
    permute = _expand(H, permbtd.permute)
    assert np.allclose(G.green(E, k=K), Gref[np.ix_(permute, permute)])

    # Caroli formula in the device region, with the downfolded broadening
    # matrices (equivalent to TBtrans)
    M = _reference(Hel, H, E, K)
    gam = []
    for (down, _), cells in zip(elecs, ((0, 1), (7, 6))):
        x = _expand(H, np.concatenate([_cell_orbs(H, c) for c in cells]))
        d = _expand(H, down.orbs)
        se = M[np.ix_(d, x)] @ np.linalg.solve(M[np.ix_(x, x)], M[np.ix_(x, d)])
        gam.append((d, down.se2broadening(se)))

    (dL, gL), (dR, gR) = gam
    GRL = Gref[np.ix_(dR, dL)]
    GLR = Gref[np.ix_(dL, dR)]
    T_LR = np.trace(gR @ GRL @ gL @ GRL.conj().T).real
    T_RL = np.trace(gL @ GLR @ gR @ GLR.conj().T).real
    assert np.isclose(G.transmission(E, "Left", "Right", k=K), T_LR)
    assert np.isclose(G.transmission(E, "Right", "Left", k=K), T_RL)


def test_device_green_outside():
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel, H = _system(hop)
    rng = np.random.default_rng(42)
    elecs = _electrodes(Hel, H, True, rng)
    blocks = [_cell_orbs(H, c) for c in range(3, 6)]
    permbtd = PermuteBTD(np.concatenate(blocks), [len(b) for b in blocks])
    with pytest.raises(ValueError, match="outside"):
        DeviceGreen(H, elecs, {"orbitals": permbtd})


def test_device_green_downfold_overlap():
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel, H = _system(hop)
    se = si.RecursiveSI(Hel, "-C", eta=ETA)
    elec = BlochElectrode(se, bloch=(2, 1, 1), name="Left")
    cells = [_cell_orbs(H, c) for c in range(4)]
    down = DownfoldElectrode(
        elec, H, {"orbitals": cells[0]}, {"orbitals": PermuteBTD(cells[:3])}
    )
    device = {"orbitals": np.concatenate(cells[2:])}

    # the downfolding region (cell 1) is in the device region
    with pytest.raises(ValueError, match="overlapping the device"):
        DeviceGreen(H, [down], {"orbitals": np.concatenate(cells[1:])})

    # two electrodes in the same downfolding region
    other = DownfoldElectrode(
        elec,
        H,
        {"orbitals": cells[0]},
        {"orbitals": PermuteBTD(cells[:3])},
        name="Other",
    )
    with pytest.raises(ValueError, match="overlapping downfolding regions"):
        DeviceGreen(H, [down, other], device)

    # nested downfolding includes the inner region
    inner = DownfoldElectrode(
        elec, H, {"orbitals": cells[0]}, {"orbitals": PermuteBTD(cells[:2])}
    )
    outer = DownfoldElectrode(
        inner,
        H,
        {"orbitals": cells[1]},
        {"orbitals": PermuteBTD(cells[1:3])},
        bulk=False,
    )
    assert np.array_equal(outer.orbs_down, np.concatenate(cells[:2]))
    with pytest.raises(ValueError, match="overlapping downfolding regions"):
        DeviceGreen(H, [outer, other], device)


@pytest.mark.parametrize("prefix", ["TBT", "TS"])
def test_electrode_from_fdf(tmp_path, prefix, sisl_importskip):
    sisl_importskip("netCDF4")
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel = _electrode(hop)
    Hel.write(tmp_path / "elec.nc")
    fdf = tmp_path / "RUN.fdf"
    fdf.write_text(f"""
TS.Elecs.Eta 0.02 eV
TBT.Elecs.Bulk false
%block TS.Elec.Left
  HS {tmp_path / "elec.nc"}
  semi-inf-direction -a3
  bloch 2 1 1
  delta-Ef 0.1 eV
  tbt.eta 0.03 eV
%endblock
""")
    elec = Electrode.from_fdf(fdf, "Left", prefix=prefix, mu=0.25, downfold=False)
    tbt = prefix == "TBT"
    assert isinstance(elec, BlochElectrode)
    eta = 0.03 if tbt else 0.02
    assert elec.name == "Left"
    assert np.allclose(elec.eta, eta)
    assert np.array_equal(elec.bloch.bloch, [2, 1, 1])
    assert elec.info.bulk == (not tbt)
    assert np.isclose(elec.info.dEf, 0.1)
    assert np.isclose(elec.info.mu, 0.25)

    Hel.shift(0.35)
    ref = si.RecursiveSI(Hel.tile(2, 0), "-C", eta=eta)
    assert np.allclose(elec.self_energy(0.3, k=K), ref.self_energy(0.3, k=K))


def test_device_green_unknown_electrode():
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel, H = _system(hop)
    rng = np.random.default_rng(42)
    elecs = _electrodes(Hel, H, True, rng)
    blocks = [_cell_orbs(H, c) for c in range(2, 6)]
    permbtd = PermuteBTD(np.concatenate(blocks), [len(b) for b in blocks])
    G = DeviceGreen(H, elecs, {"orbitals": permbtd}, eta=ETA)
    assert [e.name for e in G.elecs] == ["Left", "Right"]
    assert G._elec("Right") == 1
    with pytest.raises(KeyError, match="Middle"):
        G.transmission(0.3, "Middle", "Right")


@pytest.mark.parametrize(
    "semi_inf, error",
    [
        # real-space electrodes requires a GF file
        ("ab", ValueError),
        ("-a1+a2", ValueError),
        ("abc", ValueError),
        ("a3", ValueError),
        ("-a3-c", ValueError),
        ("-x", ValueError),
    ],
)
def test_electrode_from_fdf_semi_inf(tmp_path, semi_inf, error, sisl_importskip):
    sisl_importskip("netCDF4")
    Hel = _electrode(np.array([[-1.0, -0.2], [-0.2, -0.7]]))
    Hel.write(tmp_path / "elec.nc")
    fdf = tmp_path / "RUN.fdf"
    fdf.write_text(f"""
%block TS.Elec.Left
  HS elec.nc
  semi-inf-direction {semi_inf}
%endblock
""")
    with pytest.raises(error):
        Electrode.from_fdf(fdf, "Left", mu=0.0, downfold=False)


@pytest.mark.parametrize("use_gf", [True, False])
def test_electrode_from_fdf_gf(tmp_path, use_gf, sisl_importskip):
    sisl_importskip("netCDF4")
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel = _electrode(hop)
    Hel.write(tmp_path / "elec.nc")
    se = si.RecursiveSI(Hel, "-C", eta=ETA)
    E = np.linspace(-1, 1, 3)
    _write_gf(tmp_path / "elec.TBTGF", se.spgeom0, se, E + 1j * ETA)

    rs = _real_space_se()
    Hc = rs.real_space_coupling()
    Hc.write(tmp_path / "rs.nc")
    _write_gf(tmp_path / "rs.TBTGF", Hc, rs, E + 1j * ETA, coupling=True)

    fdf = tmp_path / "RUN.fdf"
    fdf.write_text(f"""
TS.Elecs.Eta {ETA} eV
%block TBT.Elecs
  Left
  surface
  Right
%endblock
%block TBT.Elec.Left
  HS elec.nc
  semi-inf-direction -a3
  GF elec.TBTGF
%endblock
%block TBT.Elec.surface
  HS rs.nc
  semi-inf-direction ab
  GF rs.TBTGF
%endblock
%block TBT.Elec.Right
  HS elec.nc
  semi-inf-direction +a3
%endblock
""")
    # no chemical potentials and no bias, all are at 0
    elec = Electrode.from_fdf(fdf, "Left", use_gf=use_gf, downfold=False)
    assert isinstance(elec, BlochElectrode)
    assert isinstance(elec.se, si.RecursiveSI) != use_gf
    assert elec.info.gf == "elec.TBTGF"
    assert elec.info.mu == 0.0
    assert np.allclose(elec.self_energy(E[1]), se.self_energy(E[1]))

    # no GF file, always in memory
    elec = Electrode.from_fdf(fdf, "Right", use_gf=use_gf, downfold=False)
    assert isinstance(elec.se, si.RecursiveSI)
    assert elec.info.gf is None

    if use_gf:
        elec = Electrode.from_fdf(fdf, "surface", use_gf=use_gf, downfold=False)
        assert isinstance(elec, RealSpaceElectrode)
        assert elec.info.semi_inf == ["a", "b"]
        assert elec.info.na == Hc.na
        ref = rs.self_energy(E[1], coupling=True)
        assert np.allclose(elec.self_energy(E[1]), ref)
    else:
        with pytest.raises(ValueError, match="use_gf"):
            Electrode.from_fdf(fdf, "surface", use_gf=use_gf, downfold=False)


class _ScaledSE(si.RecursiveSI):
    """Self-energy with a custom broadening conversion"""

    def se2broadening(self, SE):
        return 2 * si.RecursiveSI.se2broadening(SE)


def test_se2broadening_delegation():
    # electrodes must use the self-energy object to convert to broadening matrices
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel, H = _system(hop)
    elec = BlochElectrode(_ScaledSE(Hel, "-C", eta=ETA), bloch=(2, 1, 1), name="Left")
    cells = [_cell_orbs(H, c) for c in range(3)]
    down = DownfoldElectrode(
        elec, H, {"orbitals": cells[0]}, {"orbitals": PermuteBTD(cells)}, eta_device=ETA
    )
    for el in (elec, down):
        SE = el.self_energy(0.3, k=K)
        assert np.allclose(el.se2broadening(SE), 2j * (SE - SE.conj().T))


@pytest.mark.parametrize("format", ["array", "csr", "coo"])
@pytest.mark.parametrize(
    "spin, orthogonal",
    [("unpolarized", True), ("polarized", True), ("nc", True), ("unpolarized", False)],
)
def test_bulk_matrix_format(format, spin, orthogonal):
    # the bulk matrix equals the Bloch unfolded E S - H of the principal layer
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel = _electrode(hop, spin, orthogonal)
    se = si.RecursiveSI(Hel, "-C", eta=ETA)
    elec = BlochElectrode(se, bloch=(2, 1, 1), name="Left")
    kwargs = {"spin": 1} if spin == "polarized" else {}

    E = 0.3 + 1j * ETA
    H0 = se.spgeom0

    def func(k, **kwargs):
        return H0.Sk(k, format="array") * E - H0.Hk(k, format="array", **kwargs)

    ref = elec.bloch(func, K, **kwargs)
    M = elec.bulk_matrix(0.3, k=K, format=format, **kwargs)
    if format == "array":
        assert isinstance(M, np.ndarray)
    else:
        assert ssp.issparse(M)
        assert M.format == format
        M = M.toarray()
    assert M.shape == (len(elec), len(elec))
    assert np.allclose(M, ref, atol=1e-14, rtol=0)
