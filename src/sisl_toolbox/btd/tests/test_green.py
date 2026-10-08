# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest

import sisl as si
from sisl_toolbox.btd import BlochSelfEnergy, DeviceGreen, PermuteBTD

from .test_electrode import (
    ETA,
    NCELL,
    K,
    _cell_orbs,
    _electrode,
    _electrodes,
    _expand,
    _reference,
    _system,
)

pytestmark = [pytest.mark.btd]

HOP = np.array([[-1.0, -0.2], [-0.2, -0.7]])


def _caroli(Gref, gam_from, gam_to):
    """Transmission from the dense Green function and (index, Gamma) pairs"""
    (d_from, g_from), (d_to, g_to) = gam_from, gam_to
    G = Gref[np.ix_(d_to, d_from)]
    return np.trace(g_to @ G @ g_from @ G.conj().T).real


def _downfolded_gammas(H, M, elecs):
    """Downfolded broadening matrices (index in H, Gamma) from the dense matrix"""
    gam = []
    for (down, orbs), cells in zip(elecs, ((0, 1), (7, 6))):
        x = _expand(H, np.concatenate([_cell_orbs(H, c) for c in cells]))
        d = _expand(H, orbs)
        se = M[np.ix_(d, x)] @ np.linalg.solve(M[np.ix_(x, x)], M[np.ix_(x, d)])
        gam.append((d, down.se2broadening(se)))
    return gam


def test_permute_array():
    Hel, H = _system(HOP)
    rng = np.random.default_rng(42)
    elecs = _electrodes(Hel, H, True, rng)

    # only the permutation table, the blocks are calculated
    permute = np.concatenate([_cell_orbs(H, c) for c in range(2, 6)])
    G = DeviceGreen(H, elecs, permute, eta=ETA)
    assert G.blocks.sum() == len(permute)
    assert len(G.blocks) > 1

    E = 0.3
    Gref = np.linalg.inv(_reference(Hel, H, E, K))
    assert np.allclose(G.green(E, k=K), Gref[np.ix_(permute, permute)])


@pytest.mark.parametrize("method", ["gps", "electrode"])
def test_permute_str_bulk_electrodes(method):
    # The full H is the device, the electrodes are placed directly
    # on the electrode orbitals (no downfolding)
    Hel, H = _system(HOP)
    elecs = []
    for name, semi_inf, cell in (("Left", "-C", 0), ("Right", "+C", NCELL - 1)):
        se = si.RecursiveSI(Hel, semi_inf, eta=ETA)
        elecs.append((BlochSelfEnergy(name, se, bloch=(2, 1, 1)), _cell_orbs(H, cell)))
    G = DeviceGreen(H, elecs, method, eta=ETA)
    # each electrode must be within 2 neighboring blocks
    blocks_end = np.cumsum(G.blocks)
    for permute in (el.permute for el in G._elecs):
        b = np.searchsorted(blocks_end, permute[[0, -1]], side="right")
        assert b[1] - b[0] <= 1

    E = 0.3
    Gref = np.linalg.inv(_reference(Hel, H, E, K))
    assert np.allclose(G.green(E, k=K), Gref[np.ix_(G.permute, G.permute)])

    def gamma(semi_inf, cell):
        se = si.RecursiveSI(Hel.tile(2, 0), semi_inf, eta=ETA)
        return _cell_orbs(H, cell), se.broadening_matrix(E, k=K)

    T = _caroli(Gref, gamma("-C", 0), gamma("+C", NCELL - 1))
    assert np.isclose(G.transmission(E, "Left", "Right", k=K), T)


def test_spin_cache():
    Hel, H = _system(HOP, "polarized")
    rng = np.random.default_rng(42)
    elecs = _electrodes(Hel, H, True, rng)
    blocks = [rng.permutation(_cell_orbs(H, c)) for c in range(2, 6)]
    permbtd = PermuteBTD(np.concatenate(blocks), [len(b) for b in blocks])
    G = DeviceGreen(H, elecs, permbtd, eta=ETA)

    E = 0.3
    T = []
    Gref = []
    for spin in (0, 1):
        M = _reference(Hel, H, E, K, spin=spin)
        Gref.append(np.linalg.inv(M))
        T.append(_caroli(Gref[-1], *_downfolded_gammas(H, M, elecs)))
    assert not np.isclose(T[0], T[1])

    permute = permbtd.permute
    # alternate the spin at the same E and k, the cache must not mix them
    for spin in (0, 1, 0, 1):
        assert np.allclose(
            G.green(E, k=K, spin=spin), Gref[spin][np.ix_(permute, permute)]
        )
        assert np.isclose(G.transmission(E, "Left", "Right", k=K, spin=spin), T[spin])

    # scattering states remember the spin they were calculated with
    for spin in (1, 0):
        state = G.scattering_state(E, "Left", k=K, cutoff=1e-10, spin=spin)
        G.green(E, k=K, spin=1 - spin)
        eig = G.eigenchannel(state, "Right")
        assert np.isclose(eig.c.sum(), T[spin])

        state = G.scattering_state(
            E, "Left", k=K, cutoff=1e-10, method="eig", spin=spin
        )
        G.green(E, k=K, spin=1 - spin)
        eig = G.eigenchannel(state, "Right")
        assert np.isclose(eig.c.sum(), T[spin])

        S = G.scattering_matrix(E, "Left", "Right", k=K, cutoff=1e-12, spin=spin)
        assert np.isclose(np.sum(np.abs(S.state) ** 2), T[spin])


def test_dtype_cache():
    Hel, H = _system(HOP)
    rng = np.random.default_rng(42)
    elecs = _electrodes(Hel, H, True, rng)
    blocks = [_cell_orbs(H, c) for c in range(2, 6)]
    permbtd = PermuteBTD(np.concatenate(blocks), [len(b) for b in blocks])
    G = DeviceGreen(H, elecs, permbtd, eta=ETA)
    assert G.green(0.3, k=K, dtype=np.complex64).dtype == np.complex64
    assert G.green(0.3, k=K, dtype=np.complex128).dtype == np.complex128


def test_permute_electrode_span():
    # a user supplied partition where the electrode spans 3 blocks
    Hel, H = _system(HOP)
    rng = np.random.default_rng(42)
    elecs = _electrodes(Hel, H, True, rng)
    permute = np.concatenate([_cell_orbs(H, c) for c in range(2, 6)])
    permbtd = PermuteBTD(permute, [2, 2, 4] + [8] * 3)
    with pytest.raises(ValueError, match="spans"):
        DeviceGreen(H, elecs, permbtd)


def test_btd_partition_dense():
    from sisl_toolbox.btd import btd_partition

    _, H = _system(HOP)
    dense = [_cell_orbs(H, 0), _cell_orbs(H, NCELL - 1)]
    for method in ("gps", "rcm"):
        p = btd_partition(H, method=method, dense=dense)
        assert np.array_equal(np.sort(p.permute), np.arange(H.no))
        blocks_end = np.cumsum(p.blocks)
        inv = np.empty_like(p.permute)
        inv[p.permute] = np.arange(len(p.permute))
        for orbs in dense:
            pos = np.sort(inv[orbs])
            b = np.searchsorted(blocks_end, pos[[0, -1]], side="right")
            assert b[1] - b[0] <= 1


MU = {"Left": 0.1, "Right": -0.1}


def _reference_mu(Hel, H, E, k, device):
    """Device Green function with electrodes shifted by MU

    The electrode regions are bulk, i.e. also shifted by MU.
    """
    M = (E + 1j * ETA) * np.eye(len(H)) - H.Hk(k, format="array")
    for name, semi_inf, cell in (("Left", "-C", 0), ("Right", "+C", NCELL - 1)):
        He = Hel.tile(2, 0)
        He.shift(MU[name])
        se = si.RecursiveSI(He, semi_inf, eta=ETA)
        o = _cell_orbs(H, cell)
        M[o, o] -= MU[name]
        M[np.ix_(o, o)] -= se.self_energy(E, k=k)
    return np.linalg.inv(M)[np.ix_(device, device)]


class _FakeTBT:
    """Minimal stand-in for a tbtncSileTBtrans"""

    def __init__(self, H, rng):
        self.elecs = ["Left", "Right"]
        self._H = H
        self._down = {}
        for name, cells in (("Left", (0, 1, 2)), ("Right", (7, 6, 5))):
            self._down[name] = [rng.permutation(_cell_orbs(H, c)) for c in cells]
        self._dev = [rng.permutation(_cell_orbs(H, c)) for c in range(2, 6)]

    def mu(self, name):
        return MU[name]

    def eta(self, name=None):
        return ETA

    def bloch(self, name):
        return np.array([2, 1, 1])

    def a_elec(self, name):
        cell = {"Left": 0, "Right": NCELL - 1}[name]
        return np.unique(self._H.o2a(_cell_orbs(self._H, cell)))

    def pivot_down(self, name):
        return np.concatenate(self._down[name])

    def btd(self, name=None):
        if name is None:
            return np.array([len(b) for b in self._dev])
        return np.array([len(b) for b in self._down[name]])

    def pivot(self, name=None):
        if name is None:
            return np.concatenate(self._dev)
        return self._down[name][-1]


@pytest.mark.parametrize("as_bloch", [True, False])
def test_from_tbt(as_bloch):
    Hel, H = _system(HOP)
    rng = np.random.default_rng(42)
    tbt = _FakeTBT(H, rng)

    if as_bloch:
        elecs = {}
        for name, semi_inf in (("Left", "-C"), ("Right", "+C")):
            He = Hel.copy()
            He.shift(MU[name])
            se = si.RecursiveSI(He, semi_inf, eta=ETA)
            elecs[name] = BlochSelfEnergy(name, se, bloch=(2, 1, 1))
    else:
        elecs = {"Left": (Hel, "-C"), "Right": (Hel, "+C")}
    G = DeviceGreen.from_tbt(tbt, H, elecs)

    E = 0.3
    permute = tbt.pivot()
    assert np.allclose(G.green(E, k=K), _reference_mu(Hel, H, E, K, permute))

    with pytest.raises(ValueError, match="missing"):
        DeviceGreen.from_tbt(tbt, H, {"Left": elecs["Left"]})


def _write_fdf(tmp_path, Hel, H, extra="", right="end -1"):
    Hel.write(tmp_path / "ELEC.nc")
    H.write(tmp_path / "DEVICE.nc")
    fdf = tmp_path / "RUN.fdf"
    fdf.write_text(f"""
SystemLabel device
TBT.HS DEVICE.nc
TS.Voltage 0.2 eV
TS.Elecs.Eta {ETA} eV
TBT.Contours.Eta {ETA} eV
%block TS.ChemPots
  Left
  Right
%endblock
%block TS.ChemPot.Left
  mu V/2
%endblock
%block TS.ChemPot.Right
  mu -V/2
%endblock
%block TS.Elecs
  Left
  Right
%endblock
%block TS.Elec.Left
  HS ELEC.nc
  semi-inf-direction -a3
  bloch 2 1 1
  electrode-position 1
  chemical-potential Left
%endblock
%block TS.Elec.Right
  HS ELEC.nc
  semi-inf-direction +a3
  bloch 2 1 1
  electrode-position {right}
  chemical-potential Right
%endblock
{extra}
""")
    return fdf


@pytest.mark.parametrize(
    "extra, cells",
    [
        ("", range(1, NCELL - 1)),
        ("%block TBT.Atoms.Device\n  atom [ 9 -- 24 ]\n%endblock", range(2, 6)),
        ("TBT.BTD.Pivot.Device orb+rev-CM", range(1, NCELL - 1)),
    ],
)
def test_from_fdf(tmp_path, extra, cells):
    Hel, H = _system(HOP)
    fdf = _write_fdf(tmp_path, Hel, H, extra)
    G = DeviceGreen.from_fdf(fdf)
    assert [e.name for e in G.elecs] == ["Left", "Right"]
    assert np.isclose(G.eta, ETA)

    device = np.concatenate([_cell_orbs(H, c) for c in cells])
    assert np.array_equal(np.sort(G.permute), device)

    E = 0.3
    Gref = _reference_mu(Hel, H, E, K, G.permute)
    assert np.allclose(G.green(E, k=K), Gref)


# the right electrode (4 atoms) starts at atom 4 * (NCELL - 1) + 1 == -4
@pytest.mark.parametrize("right", ["-4", f"{4 * (NCELL - 1) + 1}", "end -1"])
def test_from_fdf_position(tmp_path, right):
    Hel, H = _system(HOP)
    fdf = _write_fdf(tmp_path, Hel, H, right=right)
    G = DeviceGreen.from_fdf(fdf)
    device = np.concatenate([_cell_orbs(H, c) for c in range(1, NCELL - 1)])
    assert np.array_equal(np.sort(G.permute), device)
    assert np.allclose(G.green(0.3, k=K), _reference_mu(Hel, H, 0.3, K, G.permute))


def test_from_fdf_position_outside(tmp_path):
    # -1 is the first atom of the electrode, which then extends beyond the atoms
    Hel, H = _system(HOP)
    fdf = _write_fdf(tmp_path, Hel, H, right="-1")
    with pytest.raises(ValueError, match="outside"):
        DeviceGreen.from_fdf(fdf)


def test_from_fdf_overlap(tmp_path):
    Hel, H = _system(HOP)
    fdf = _write_fdf(tmp_path, Hel, H, "TS.Atoms.Buffer [ 1 -- 2 ]")
    with pytest.raises(ValueError, match="overlapping"):
        DeviceGreen.from_fdf(fdf)


@pytest.mark.parametrize("method", ["svd:gamma", "svd:A", "eig"])
@pytest.mark.parametrize("orthogonal", [True, False])
def test_scattering_state(method, orthogonal):
    Hel, H = _system(HOP, orthogonal=orthogonal)
    rng = np.random.default_rng(42)
    elecs = _electrodes(Hel, H, True, rng)
    blocks = [rng.permutation(_cell_orbs(H, c)) for c in range(2, 6)]
    permbtd = PermuteBTD(np.concatenate(blocks), [len(b) for b in blocks])
    G = DeviceGreen(H, elecs, permbtd, eta=ETA)

    E = 0.3
    # the device Green function (with overlap)
    Gref = np.linalg.inv(_reference(Hel, H, E, K))
    assert np.allclose(G.green(E, k=K), Gref[np.ix_(permbtd.permute, permbtd.permute)])

    for elec in ("Left", "Right"):
        A = G.spectral(E, elec, k=K)
        state = G.scattering_state(E, elec, k=K, cutoff=1e-12, method=method)
        psi = state.state.T
        # the states are returned as-is (orthonormal)
        assert np.allclose(psi.conj().T @ psi, np.eye(psi.shape[1]))
        # and they reproduce the spectral function
        assert np.allclose(2 * np.pi * (psi * state.c) @ psi.conj().T, A)
        # largest first
        assert np.all(np.diff(state.c) <= 0)

        other = "Right" if elec == "Left" else "Left"
        T = G.transmission(E, elec, other, k=K)
        eig = G.eigenchannel(state, other)
        assert np.isclose(eig.c.sum(), T)
