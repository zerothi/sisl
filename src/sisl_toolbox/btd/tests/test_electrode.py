# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl as si
from sisl_toolbox.btd import (
    BlochSelfEnergy,
    DeviceGreen,
    DownfoldSelfEnergy,
    PermuteBTD,
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
        elec = BlochSelfEnergy(name, se, bloch=(2, 1, 1))
        blocks = [_cell_orbs(H, c) for c in cells]
        if last is not None:
            blocks[-1] = blocks[-1][last]
        blocks = [rng.permutation(b) for b in blocks]
        orbs_down = np.concatenate(blocks)
        blocks_down = [len(b) for b in blocks]
        down = DownfoldSelfEnergy(
            elec,
            H,
            orbs_down,
            _cell_orbs(H, cells[0]),
            blocks_down,
            eta_device=ETA,
            bulk=bulk,
        )
        ret.append((down, blocks[-1]))
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
    elec = BlochSelfEnergy("Left", si.RecursiveSI(Hel, "-C", eta=ETA), bloch=(2, 1, 1))
    ref = si.RecursiveSI(Hel.tile(2, 0), "-C", eta=ETA)
    assert len(elec) == len(ref)
    assert np.allclose(elec.self_energy(0.3, k=K), ref.self_energy(0.3, k=K))
    assert np.allclose(
        elec.broadening_matrix(0.3, k=K), ref.broadening_matrix(0.3, k=K)
    )


@pytest.mark.parametrize("bulk", [True, False])
@pytest.mark.parametrize("spin", ["unpolarized", "nc"])
def test_downfold_self_energy(bulk, spin):
    hop = np.array([[-1.0, -0.2], [-0.2, -0.7]])
    Hel, H = _system(hop, spin)
    rng = np.random.default_rng(42)
    E = 0.3
    M = _reference(Hel, H, E, K)
    for down, orbs_dev in _electrodes(Hel, H, bulk, rng):
        # downfold everything but the device orbitals
        if down.name == "Left":
            orbs_x = np.concatenate([_cell_orbs(H, 0), _cell_orbs(H, 1)])
        else:
            orbs_x = np.concatenate([_cell_orbs(H, 7), _cell_orbs(H, 6)])
        x = _expand(H, orbs_x)
        d = _expand(H, orbs_dev)
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
    for down, orbs_dev in _electrodes(Hel, H, True, rng, last=slice(0, None, 2)):
        if down.name == "Left":
            orbs_x = np.concatenate([_cell_orbs(H, 0), _cell_orbs(H, 1)])
        else:
            orbs_x = np.concatenate([_cell_orbs(H, 7), _cell_orbs(H, 6)])
        x = orbs_x
        d = orbs_dev
        ref = M[np.ix_(d, x)] @ np.linalg.solve(M[np.ix_(x, x)], M[np.ix_(x, d)])
        assert np.allclose(down.self_energy(E, k=K), ref)


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
    G = DeviceGreen(H, elecs, permbtd, eta=ETA)

    E = 0.3
    Gref = np.linalg.inv(_reference(Hel, H, E, K))
    permute = _expand(H, permbtd.permute)
    assert np.allclose(G.green(E, k=K), Gref[np.ix_(permute, permute)])

    # Caroli formula in the device region, with the downfolded broadening
    # matrices (equivalent to TBtrans)
    M = _reference(Hel, H, E, K)
    gam = []
    for (down, orbs), cells in zip(elecs, ((0, 1), (7, 6))):
        x = _expand(H, np.concatenate([_cell_orbs(H, c) for c in cells]))
        d = _expand(H, orbs)
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
        DeviceGreen(H, elecs, permbtd)


@pytest.mark.parametrize("prefix", ["TBT", "TS"])
def test_bloch_from_fdf(tmp_path, prefix):
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
    elec = BlochSelfEnergy.from_fdf(fdf, "Left", prefix=prefix, mu=0.25)
    tbt = prefix == "TBT"
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
    G = DeviceGreen(H, elecs, permbtd, eta=ETA)
    assert [e.name for e in G.elecs] == ["Left", "Right"]
    assert G._elec("Right") == 1
    with pytest.raises(KeyError, match="Middle"):
        G.transmission(0.3, "Middle", "Right")
