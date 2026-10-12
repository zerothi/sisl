# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import warnings

import numpy as np
import pytest

import sisl as si
from sisl.messages import SislInfo, SislWarning
from sisl_toolbox.btd import (
    BlochElectrode,
    DeviceGreen,
    DownfoldElectrode,
    Electrode,
    PermuteBTD,
    downfold_partition,
)

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
    for (down, _), cells in zip(elecs, ((0, 1), (7, 6))):
        x = _expand(H, np.concatenate([_cell_orbs(H, c) for c in cells]))
        d = _expand(H, down.orbs)
        se = M[np.ix_(d, x)] @ np.linalg.solve(M[np.ix_(x, x)], M[np.ix_(x, d)])
        gam.append((d, down.se2broadening(se)))
    return gam


@pytest.mark.parametrize("space", ["atoms", "orbitals", None])
def test_permute_array(space):
    Hel, H = _system(HOP)
    rng = np.random.default_rng(42)
    elecs = _electrodes(Hel, H, True, rng)

    # only the permutation table, the blocks are calculated
    permute = np.concatenate([_cell_orbs(H, c) for c in range(2, 6)])
    if space == "orbitals":
        arg = {"orbitals": permute}
    else:
        # atoms are expanded to their orbitals (in order)
        atoms = H.o2a(permute[:: H.no // H.na])
        assert np.array_equal(H.a2o(atoms, all=True), permute)
        arg = atoms if space is None else {"atoms": atoms}
    G = DeviceGreen(H, elecs, arg, eta=ETA)
    assert np.array_equal(G.permute, permute)
    assert G.blocks.sum() == len(permute)
    assert len(G.blocks) > 1

    E = 0.3
    Gref = np.linalg.inv(_reference(Hel, H, E, K))
    assert np.allclose(G.green(E, k=K), Gref[np.ix_(permute, permute)])


@pytest.mark.parametrize(
    "method", ["gps", "electrode", "atom+gps", "orb+gps", "orb+electrode"]
)
def test_permute_str_bulk_electrodes(method):
    # The full H is the device, the electrodes are placed directly
    # on the electrode orbitals (no downfolding)
    Hel, H = _system(HOP)
    elecs = []
    for name, semi_inf, cell in (("Left", "-C", 0), ("Right", "+C", NCELL - 1)):
        se = si.RecursiveSI(Hel, semi_inf, eta=ETA)
        # atom indices (default)
        atoms = np.unique(H.o2a(_cell_orbs(H, cell)))
        elecs.append((BlochElectrode(name, se, bloch=(2, 1, 1)), atoms))
    G = DeviceGreen(H, elecs, method, eta=ETA)
    if not method.startswith("orb+"):
        # atoms are partitioned, an atom is never split between blocks
        iblock = np.repeat(np.arange(len(G.blocks)), G.blocks)
        atoms = H.o2a(G.permute)
        for ia in np.unique(atoms):
            assert len(np.unique(iblock[atoms == ia])) == 1
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
    G = DeviceGreen(H, elecs, {"orbitals": permbtd}, eta=ETA)

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
    G = DeviceGreen(H, elecs, {"orbitals": permbtd}, eta=ETA)
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
        DeviceGreen(H, elecs, {"orbitals": permbtd})


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


def _reference_mu(Hel, H, E, k, device, bulk=True):
    """Device Green function with electrodes shifted by MU

    For `bulk` the electrode regions are also shifted by MU.
    """
    M = (E + 1j * ETA) * np.eye(len(H)) - H.Hk(k, format="array")
    for name, semi_inf, cell in (("Left", "-C", 0), ("Right", "+C", NCELL - 1)):
        He = Hel.tile(2, 0)
        He.shift(MU[name])
        se = si.RecursiveSI(He, semi_inf, eta=ETA)
        o = _cell_orbs(H, cell)
        if bulk:
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
            elecs[name] = BlochElectrode(name, se, bloch=(2, 1, 1))
    else:
        elecs = {"Left": (Hel, "-C"), "Right": (Hel, "+C")}
    G = DeviceGreen.from_tbt(tbt, H, elecs)

    E = 0.3
    permute = tbt.pivot()
    assert np.allclose(G.green(E, k=K), _reference_mu(Hel, H, E, K, permute))

    with pytest.raises(ValueError, match="missing"):
        DeviceGreen.from_tbt(tbt, H, {"Left": elecs["Left"]})


def _write_fdf(tmp_path, Hel, H, extra="", right="end -1", voltage=0.2):
    pytest.importorskip("netCDF4")
    Hel.write(tmp_path / "ELEC.nc")
    H.write(tmp_path / "DEVICE.nc")
    fdf = tmp_path / "RUN.fdf"
    fdf.write_text(f"""
SystemLabel device
TBT.HS DEVICE.nc
TS.Voltage {voltage} eV
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


@pytest.mark.parametrize("bulk", [True, False])
def test_from_fdf_device_elecs(tmp_path, bulk):
    # electrodes in the device region are not downfolded, i.e. the full system is the device.
    # bulk uses the (shifted) electrode Hamiltonian in the electrode region
    Hel, H = _system(HOP)
    fdf = _write_fdf(
        tmp_path, Hel, H, f"TBT.Elecs.Bulk {bulk}\nTBT.Atoms.Device [ 1 -- {H.na} ]"
    )
    G = DeviceGreen.from_fdf(fdf)
    assert all(isinstance(e, BlochElectrode) for e in G.elecs)
    assert [el.bulk for el in G._elecs] == [bulk, bulk]
    orbs = np.arange(H.no)
    assert np.array_equal(np.sort(G.permute), orbs)

    E = 0.3
    Gref = _reference_mu(Hel, H, E, K, G.permute, bulk=bulk)
    assert np.allclose(G.green(E, k=K), Gref)


# The electrode cells have the atoms (z=0, z=1, z=0, z=1), only the outer
# layers are downfolded: z=0 for Left, z=1 for Right.
PARTIAL_DEVICE = f"TBT.Atoms.Device [ 2 4 5 -- {4 * NCELL - 3} {4 * NCELL - 1} ]"


@pytest.mark.parametrize("bulk", [True, False])
def test_from_fdf_partial(tmp_path, bulk):
    # electrodes partially in the device region are partially downfolded
    Hel, H = _system(HOP)
    fdf = _write_fdf(tmp_path, Hel, H, f"TBT.Elecs.Bulk {bulk}\n{PARTIAL_DEVICE}")
    G = DeviceGreen.from_fdf(fdf)
    assert all(isinstance(e, DownfoldElectrode) for e in G.elecs)
    inner = {
        "Left": H.a2o([1, 3], all=True),
        "Right": H.a2o([H.na - 4, H.na - 2], all=True),
    }
    device = np.concatenate(
        [inner["Left"], _orbs(H, *range(1, NCELL - 1)), inner["Right"]]
    )
    assert np.array_equal(np.sort(G.permute), np.sort(device))
    for elec in G.elecs:
        assert np.array_equal(np.sort(elec.orbs), inner[elec.name])

    # bulk only shifts the downfolded electrode atoms (outer layers)
    E = 0.3
    M = (E + 1j * ETA) * np.eye(len(H)) - H.Hk(K, format="array")
    for name, semi_inf, cell in (("Left", "-C", 0), ("Right", "+C", NCELL - 1)):
        He = Hel.tile(2, 0)
        He.shift(MU[name])
        se = si.RecursiveSI(He, semi_inf, eta=ETA)
        o = _cell_orbs(H, cell)
        if bulk:
            outer = np.setdiff1d(o, inner[name])
            M[outer, outer] -= MU[name]
        M[np.ix_(o, o)] -= se.self_energy(E, k=K)
    Gref = np.linalg.inv(M)[np.ix_(G.permute, G.permute)]
    assert np.allclose(G.green(E, k=K), Gref)


def test_in_device_transmission():
    # electrodes placed directly in the device (bulk), Caroli formula
    # with the broadening matrices of the electrodes
    Hel, H = _system(HOP)
    elecs = []
    for name, semi_inf, cell in (("Left", "-C", 0), ("Right", "+C", NCELL - 1)):
        He = Hel.copy()
        He.shift(MU[name])
        se = si.RecursiveSI(He, semi_inf, eta=ETA)
        elec = BlochElectrode(name, se, bloch=(2, 1, 1))
        elecs.append((elec, {"atoms": np.unique(H.o2a(_cell_orbs(H, cell)))}, True))
    G = DeviceGreen(H, elecs, "gps", eta=ETA)

    E = 0.3
    Gref = _reference_mu(Hel, H, E, K, np.arange(H.no))
    assert np.allclose(G.green(E, k=K), Gref[np.ix_(G.permute, G.permute)])

    def gamma(name, semi_inf, cell):
        He = Hel.tile(2, 0)
        He.shift(MU[name])
        se = si.RecursiveSI(He, semi_inf, eta=ETA)
        return _cell_orbs(H, cell), se.broadening_matrix(E, k=K)

    T = _caroli(Gref, gamma("Left", "-C", 0), gamma("Right", "+C", NCELL - 1))
    assert np.isclose(G.transmission(E, "Left", "Right", k=K), T)


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


def _orbs(H, *cells):
    return np.concatenate([_cell_orbs(H, c) for c in cells])


def test_downfold_partition():
    _, H = _system(HOP)
    left, right = _orbs(H, 0), _orbs(H, NCELL - 1)
    # device defaults to all orbitals not in the electrodes
    parts = downfold_partition(H, [left, right])
    for part, (elec, dev) in zip(parts, ((0, 1), (NCELL - 1, NCELL - 2))):
        assert isinstance(part, PermuteBTD)
        n = part.blocks[-1]
        assert np.array_equal(np.sort(part.permute[:-n]), _orbs(H, elec))
        # the last block are the device orbitals coupling to the electrode
        assert np.isin(part.permute[-n:], _orbs(H, dev)).all()

    # buffer orbitals are not part of the device
    parts = downfold_partition(H, [left, right], buffer=_orbs(H, 3, 4))
    assert parts[0].permute[-parts[0].blocks[-1] :].max() < _orbs(H, 3).min()

    # electrodes in the device are not downfolded
    parts = downfold_partition(H, [left, right], device=_orbs(H, *range(2, NCELL)))
    assert parts[1] is None
    n = parts[0].blocks[-1]
    assert np.array_equal(np.sort(parts[0].permute[:-n]), _orbs(H, 0, 1))
    assert np.isin(parts[0].permute[-n:], _orbs(H, 2)).all()

    # electrodes partially in the device are partially downfolded
    outer, inner = H.a2o([0, 2], all=True), H.a2o([1, 3], all=True)
    device = np.concatenate([inner, _orbs(H, *range(1, NCELL))])
    parts = downfold_partition(H, [left, right], device=device)
    assert parts[1] is None
    n = parts[0].blocks[-1]
    assert np.array_equal(np.sort(parts[0].permute[:-n]), outer)
    assert np.array_equal(np.sort(parts[0].permute[-n:]), inner)
    # the atomic graph gives the same partition
    parts = downfold_partition(
        H.toSparseAtom(), [[0, 1, 2, 3]], device=np.arange(1, H.na)
    )
    assert np.array_equal(np.sort(parts[0].permute), [0, 1, 2, 3])
    assert parts[0].blocks.tolist() == [1, 3]

    # the downfolding region of the left electrode couples to the buffer
    with pytest.raises(ValueError, match="excluded"):
        downfold_partition(H, [left, right], device=_orbs(H, 3, 4), buffer=_orbs(H, 2))


@pytest.mark.parametrize(
    "extra, in_device",
    [
        ("", False),
        ("%block TBT.Atoms.Device\n  atom [ 9 -- 24 ]\n%endblock", False),
        ("TBT.Elecs.Bulk false", False),
        ("TBT.BTD.Pivot.Elecs orb+", False),
        ("TBT.BTD.Pivot.Elec.Left atom+\nTBT.BTD.Pivot.Elec.Right orb+", False),
        (PARTIAL_DEVICE, False),
        (f"{PARTIAL_DEVICE}\nTBT.BTD.Pivot.Elecs orb+", False),
        (f"TBT.Atoms.Device [ 1 -- {4 * NCELL} ]", True),
    ],
)
def test_electrode_from_fdf_downfold(tmp_path, extra, in_device):
    # Electrode.from_fdf downfolds the electrode as DeviceGreen.from_fdf
    Hel, H = _system(HOP)
    fdf = _write_fdf(tmp_path, Hel, H, extra)
    G = DeviceGreen.from_fdf(fdf)
    elecs = []
    for name, cell in (("Left", 0), ("Right", NCELL - 1)):
        el = Electrode.from_fdf(fdf, name)
        ref = G.elecs[G._elec(name)]
        assert type(el) is type(ref)
        if in_device:
            assert isinstance(el, BlochElectrode)
            assert np.array_equal(el.orbs, _orbs(H, cell))
        else:
            assert isinstance(el, DownfoldElectrode)
            assert np.array_equal(el.orbs, ref.orbs)
            assert el.info.bulk == ref._data.bulk
        assert np.allclose(el.self_energy(0.3, k=K), ref.self_energy(0.3, k=K))
        elecs.append(el)

    # the electrodes carry their orbitals
    G2 = DeviceGreen(
        G.H, elecs, {"orbitals": PermuteBTD(G.permute, G.blocks)}, eta=G.eta
    )
    assert [el.bulk for el in G2._elecs] == [el.bulk for el in G._elecs]
    assert np.allclose(G2.green(0.3, k=K), G.green(0.3, k=K))

    with pytest.raises(ValueError, match="downfold=False"):
        Electrode.from_fdf(fdf, "Middle")


def test_device_green_no_orbs():
    Hel, H = _system(HOP)
    elec = BlochElectrode("Left", si.RecursiveSI(Hel, "-C", eta=ETA), bloch=(2, 1, 1))
    with pytest.raises(ValueError, match="no orbitals"):
        DeviceGreen(H, [elec])
    with pytest.raises(ValueError, match="'atoms' or 'orbitals'"):
        DeviceGreen(H, [(elec, {"orbs": _cell_orbs(H, 0)})])
    with pytest.raises(ValueError, match="'atoms' or 'orbitals'"):
        DeviceGreen(H, [(elec, {"atoms": [0], "orbitals": [0]})])
    # orbital indices passed as atoms
    with pytest.raises(ValueError, match="orbitals"):
        DeviceGreen(H, [(elec, _cell_orbs(H, NCELL - 1))])
    with pytest.raises(ValueError, match="orbitals"):
        DeviceGreen(H, [(elec, [0, 1])], np.arange(H.no))


@pytest.mark.parametrize("method", ["svd:gamma", "svd:A", "eig"])
@pytest.mark.parametrize("orthogonal", [True, False])
def test_scattering_state(method, orthogonal):
    Hel, H = _system(HOP, orthogonal=orthogonal)
    rng = np.random.default_rng(42)
    elecs = _electrodes(Hel, H, True, rng)
    blocks = [rng.permutation(_cell_orbs(H, c)) for c in range(2, 6)]
    permbtd = PermuteBTD(np.concatenate(blocks), [len(b) for b in blocks])
    G = DeviceGreen(H, elecs, {"orbitals": permbtd}, eta=ETA)

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


def _bulk_elecs(Hel, H, semi_infs=("-C", "+C"), cells=(0, NCELL - 1), eta=ETA):
    """Left/Right Bloch electrodes placed directly on cells of H (not downfolded)"""
    elecs = []
    for name, semi_inf, cell in zip(("Left", "Right"), semi_infs, cells):
        se = si.RecursiveSI(Hel, semi_inf, eta=eta)
        elec = BlochElectrode(name, se, bloch=(2, 1, 1))
        elecs.append((elec, {"orbitals": _cell_orbs(H, cell)}))
    return elecs


def test_eta_default():
    # eta defaults to the smallest electrode eta / 10
    Hel, H = _system(HOP)
    elecs = _bulk_elecs(Hel, H)
    elecs[1] = (
        BlochElectrode(
            "Right", si.RecursiveSI(Hel, "+C", eta=ETA / 2), bloch=(2, 1, 1)
        ),
        elecs[1][1],
    )
    G = DeviceGreen(H, elecs)
    assert np.isclose(G.eta, ETA / 20)
    # and is usable
    assert np.isfinite(G.transmission(0.3, "Left", "Right", k=K))
    str(G)

    # also for downfolded electrodes
    permute = {"orbitals": np.concatenate([_cell_orbs(H, c) for c in range(2, 6)])}
    G = DeviceGreen(H, _electrodes(Hel, H, True, np.random.default_rng(42)), permute)
    assert np.isclose(G.eta, ETA / 10)


def test_eta_default_unknown():
    class NoEta(Electrode):
        def __len__(self):
            return 1

        def self_energy(self, E, k=(0, 0, 0), **kwargs):
            return np.zeros([1, 1], dtype=np.complex128)

        def se2broadening(self, SE):
            return SE

    g = si.Geometry(
        [[i, 0, 0] for i in range(4)], si.Atom(1, R=1.1), lattice=[4, 10, 10]
    )
    H = si.Hamiltonian(g)
    H.construct([(0.1, 1.1), (0, -1)])
    with pytest.raises(ValueError, match="eta"):
        DeviceGreen(H, [(NoEta("E"), [0])])
    assert DeviceGreen(H, [(NoEta("E"), [0])], eta=0.1).eta == 0.1


@pytest.mark.parametrize("downfold", [True, False])
def test_restrict_nsc(downfold):
    # H with couplings along the semi-infinite direction must give the
    # same result as H without them
    Hel, _ = _system(HOP)
    Hp = Hel.tile(2, 0).tile(NCELL, 2)
    assert Hp.nsc[2] == 3
    H = Hp.copy()
    H.set_nsc(c=1)

    def green(H):
        if downfold:
            elecs = _electrodes(Hel, H, True, np.random.default_rng(42))
            permute = {
                "orbitals": np.concatenate([_cell_orbs(H, c) for c in range(2, 6)])
            }
        else:
            elecs = _bulk_elecs(Hel, H)
            permute = "gps"
        G = DeviceGreen(H, elecs, permute, eta=ETA)
        return G, G.green(0.3, k=K)

    Gp, gp = green(Hp)
    G, g = green(H)
    # the input is not changed
    assert Hp.nsc[2] == 3
    assert Gp.H.nsc[2] == 1
    assert np.allclose(gp, g)


def test_placement_correct_no_warning():
    Hel, H = _system(HOP)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SislWarning)
        DeviceGreen(H, _bulk_elecs(Hel, H), eta=ETA)
        permute = {"orbitals": np.concatenate([_cell_orbs(H, c) for c in range(2, 6)])}
        elecs = _electrodes(Hel, H, True, np.random.default_rng(42))
        DeviceGreen(H, elecs, permute, eta=ETA)


@pytest.mark.parametrize(
    "semi_infs, cells, match",
    [
        # wrong semi-infinite directions
        (("+C", "+C"), (0, NCELL - 1), "semi-infinite side"),
        # electrode not at the edge of the device
        (("-C", "+C"), (3, NCELL - 1), "semi-infinite side"),
    ],
)
def test_placement_warning_side(semi_infs, cells, match):
    Hel, H = _system(HOP)
    with pytest.warns(SislWarning, match=match):
        DeviceGreen(H, _bulk_elecs(Hel, H, semi_infs, cells), eta=ETA)


def test_placement_warning_order():
    Hel, H = _system(HOP)
    elecs = _bulk_elecs(Hel, H)
    elec, orbs = elecs[0]
    elecs[0] = (elec, {"orbitals": orbs["orbitals"][::-1]})
    with pytest.warns(SislWarning, match="do not match the electrode orbitals"):
        DeviceGreen(H, elecs, eta=ETA)


def test_placement_downfold_warning():
    Hel, H = _system(HOP)
    elec = BlochElectrode("Left", si.RecursiveSI(Hel, "+C", eta=ETA), bloch=(2, 1, 1))
    with pytest.warns(SislWarning, match="semi-infinite side"):
        DownfoldElectrode(
            elec,
            H,
            {"orbitals": _cell_orbs(H, 0)},
            {"orbitals": np.concatenate([_cell_orbs(H, c) for c in (0, 1)])},
            eta_device=ETA,
        )


def test_placement_skewed_lattice():
    # graphene, transport along a, where b is not perpendicular to a
    Hel = si.Hamiltonian(si.geom.graphene())
    Hel.construct([(0.1, 1.44), (0, -2.7)])
    H = Hel.tile(6, 0)
    no = Hel.no

    def elecs(left):
        return [
            (BlochElectrode("Left", si.RecursiveSI(Hel, left)), np.arange(no)),
            (
                BlochElectrode("Right", si.RecursiveSI(Hel, "+A")),
                np.arange(no) + 5 * no,
            ),
        ]

    with warnings.catch_warnings():
        warnings.simplefilter("error", SislWarning)
        # the removal of the couplings along a is only informative
        warnings.simplefilter("ignore", SislInfo)
        G = DeviceGreen(H, elecs("-A"), eta=ETA)

    # compare with the dense Green function (transverse k along b)
    E, k = 0.5, (0, 0.15, 0)
    Hd = H.copy()
    Hd.set_nsc(a=1)
    M = (E + 1j * ETA) * np.eye(Hd.no) - Hd.Hk(k, format="array")
    gam = []
    for semi_inf, o in (("-A", np.arange(no)), ("+A", np.arange(no) + 5 * no)):
        se = si.RecursiveSI(Hel, semi_inf).self_energy(E, k=k)
        M[np.ix_(o, o)] -= se
        gam.append((o, 1j * (se - se.conj().T)))
    Gref = np.linalg.inv(M)
    T = _caroli(Gref, *gam)
    assert np.isclose(G.transmission(E, "Left", "Right", k=k), T)
    with pytest.warns(SislWarning, match="semi-infinite side"):
        DeviceGreen(H, elecs("+A"))
