# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Block-tri-diagonal electrode handling

The electrode classes calculate the self-energies of the electrodes. They have
no knowledge of the device region, nor its permutation. The rows (and columns) of the returned
self-energies are in the orbital order the electrode was created with,
`DeviceGreen` is responsible for placing them in the device region.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Literal, Optional, Self

import numpy as np
import numpy.typing as npt
import scipy.sparse as ssp

import sisl as si
import sisl.io.siesta
import sisl.io.siesta.binaries
from sisl import _array as _a
from sisl._internal import set_module
from sisl.io.siesta.fdf import _parse_list
from sisl.linalg import solve
from sisl.messages import warn
from sisl.typing import KPoint
from sisl.typing._core import SileLike
from sisl.utils.misc import PropertyDict

from ._btd import (
    PermuteBTD,
    _downfold_permute,
    _graph,
    _permute_a2o,
    _split_pivot,
    downfold_partition,
)
from ._help import (
    _check_atoms,
    _split_space,
    _to_orbs,
    expand_blocks,
    expand_orbs,
    get_expand,
)

__all__ = ["Electrode", "BlochElectrode", "DownfoldElectrode", "RealSpaceElectrode"]


# In the following code sections the comments
# will be referring to systems of varying details.
#
# Imagine a system of atoms:
#
#        Device
#        ⌜   ⌝
# [1 2 3 4 5 6 7 8]
#  ⌞ ⌟         ⌞ ⌟
#   |           |
#   E1          E2
#
# We will use these terms:
#
# - full system: atoms 1-8
# - device: atoms 4-6
# - downfolding of E1: 1-3
# - downfolding of E1 and device: 1-4
# - pure electrode E1 in full: 1-2
# - electrode E1 in device: 4
# - downfolding in device: 4 (assuming only NN interactions)


# [+-] [a [*]] V [/ b]
_MU_V = re.compile(
    r"^(?P<sign>[+-])?\s*(?:(?P<a>\d*\.?\d+(?:[eE][+-]?\d+)?)\s*\*?\s*)?V\s*(?:/\s*(?P<b>\d*\.?\d+(?:[eE][+-]?\d+)?))?$"
)


def _fdf_mu(value: str, V: float) -> float:
    """Convert a chemical potential expression of the bias ``[+-][a[*]]V[/b]``

    Parameters
    ----------
    value :
        the expression, e.g. ``V/2`` or ``-V/2``
    V :
        the applied bias
    """
    m = _MU_V.match(value.strip())
    if m is None:
        raise ValueError(
            f"could not parse chemical potential {value!r}, only "
            "'<value> <unit>' and '[+-][a*]V[/b]' are allowed"
        )
    mu = V
    if m["a"] is not None:
        mu *= float(m["a"])
    if m["b"] is not None:
        mu /= float(m["b"])
    if m["sign"] == "-":
        mu = -mu
    return mu


# [+-] (a | b | c | a1 | a2 | a3), repeated
_SEMI_INF = re.compile(r"([+-]?)(a[123]|[abc])")


def _fdf_semi_inf(value: str) -> list[str]:
    """Convert a semi-infinite direction to a list of sisl directions

    E.g. ``-a3`` is ``["-c"]`` and ``+a1-a2`` is ``["+a", "-b"]``.
    A direction without a sign has an empty sign, e.g. ``ab`` is ``["a", "b"]``.
    """
    value = value.lower().replace(" ", "")
    if not re.fullmatch(f"(?:{_SEMI_INF.pattern})+", value):
        raise ValueError(f"could not parse semi-infinite direction {value!r}")
    dirs = []
    for sign, d in _SEMI_INF.findall(value):
        dirs.append(sign + {"a1": "a", "a2": "b", "a3": "c"}.get(d, d))
    letters = [d[-1] for d in dirs]
    if len(set(letters)) != len(letters):
        raise ValueError(f"semi-infinite direction {value!r} has duplicate directions")
    return dirs


def _bulk_matrix(
    H: si.Hamiltonian, E: complex, eta, k: KPoint, dtype, format, **kwargs
):
    r"""Return :math:`E\mathbf S - \mathbf H`, for real energies `eta` is added to `E`"""
    if np.isrealobj(E) and eta is not None:
        E = E + 1j * eta
    M = H.Sk(k, format=format, dtype=dtype) * E - H.Hk(
        k, format=format, dtype=dtype, **kwargs
    )
    if ssp.issparse(M):
        # sparse arithmetic may change the format
        M = M.asformat(format)
    return M


def _is_gf(se) -> bool:
    """Whether `se` is a GF file (TranSiesta/TBtrans/sisl created)"""
    return isinstance(se, sisl.io.siesta.binaries._gfSileSiesta)


class _GFSelfEnergy:
    r"""Self-energies read from a GF file

    The GF file stores, per k-point, :math:`\mathbf H` and :math:`\mathbf S`, and per
    energy point the *bulk* self-energy
    :math:`\mathbf S E - \mathbf H - \boldsymbol\Sigma(E)`.
    The self-energies are taken at the closest energy point in the file
    (with the imaginary part of the file).

    Parameters
    ----------
    gf :
        the GF file
    """

    def __init__(self, gf):
        self.gf = gf
        gf.read_header()
        # caches of the last read k-point: (key, H, S)
        # and energy-point: (key, Σ)
        self._cache_HS = None
        self._cache_SE = None

    def __len__(self) -> int:
        """Size of the self-energy matrix (including spin)"""
        return self.gf._no_u

    def __str__(self) -> str:
        return f"{self.__class__.__name__}{{file: {self.gf.file}, no: {len(self)}, eta: {self.eta:.3e}}}"

    @property
    def eta(self) -> float:
        """Imaginary part of the energies in the GF file"""
        return float(np.mean(self.gf._E.imag))

    def _index(self, E: complex, k: KPoint) -> tuple[int, int]:
        """Return the k-point and energy-point indices in the GF file"""
        gf = self.gf
        Egf = gf._E
        # find the energy-point by the real part (Eindex warns on mismatches)
        iE = np.abs(Egf.real - np.real(E)).argmin()
        iE = gf.Eindex(np.real(E) + 1j * Egf[iE].imag)
        ik = gf.kindex(np.asarray(k, dtype=np.float64))
        return ik, iE

    def _HkSk(self, ik: int, spin: int) -> tuple[np.ndarray, np.ndarray]:
        r"""Return :math:`\mathbf H` and :math:`\mathbf S` (cached, do not modify)"""
        key = (spin, ik)
        if self._cache_HS is None or self._cache_HS[0] != key:
            self._cache_HS = (key, *self.gf.HkSk(ik, spin=spin))
        return self._cache_HS[1:]

    def _SeH(self, ik: int, iE: int, spin: int) -> np.ndarray:
        r"""Return :math:`E\mathbf S - \mathbf H` at the energy in the GF file"""
        H, S = self._HkSk(ik, spin)
        return S * self.gf._E[iE] - H

    def self_energy(
        self, E: complex, k: KPoint = (0, 0, 0), dtype=None, spin: int = 0
    ) -> np.ndarray:
        """Return the self-energy (the cached array, do not modify it)"""
        ik, iE = self._index(E, k)
        key = (spin, ik, iE)
        if self._cache_SE is None or self._cache_SE[0] != key:
            # the GF file stores E S - H - Σ
            SE = self._SeH(ik, iE, spin) - self.gf.self_energy(iE, ik, spin=spin)
            self._cache_SE = (key, SE)
        SE = self._cache_SE[1]
        if dtype is None:
            return SE
        return SE.astype(dtype, copy=False)

    def bulk_matrix(
        self,
        E: complex,
        k: KPoint = (0, 0, 0),
        *,
        dtype=np.complex128,
        format: str = "array",
        spin: int = 0,
    ):
        r"""Return :math:`E\mathbf S - \mathbf H` of the GF file (at the energy in the file)"""
        M = self._SeH(*self._index(E, k), spin).astype(dtype, copy=False)
        if format in ("array", "dense"):
            return M
        return ssp.csr_array(M).asformat(format)

    @staticmethod
    def se2broadening(SE: np.ndarray) -> np.ndarray:
        """Convert the self-energy `SE` to the broadening matrix"""
        return si.physics.SelfEnergy.se2broadening(SE)


class _TransportFdf:
    """TranSiesta/TBtrans options of an fdf file

    For the ``TBT`` prefix, ``TBT`` prefixed options take precedence over
    ``TS`` prefixed options. For the ``TS`` prefix, only ``TS`` options are used.
    """

    def __init__(self, fdf: SileLike, prefix: Literal["TBT", "TS"] = "TBT"):
        if not isinstance(fdf, si.BaseSile):
            fdf = si.io.siesta.fdfSileSiesta(fdf)
        self.fdf = fdf
        self.is_tbtrans = prefix.upper() == "TBT"

    def read_device_hamiltonian(self):
        """Returns the device Hamiltonian (without stripping device atoms)"""
        fdf = self.fdf
        slabel = self.fdf.get("SystemLabel", "siesta")
        for hs_ext in ("TS.HSX", "TSHS", "HSX", "nc"):
            if fdf.dir_file(f"{slabel}.{hs_ext}").is_file():
                # choose a sane default (if it exists!)
                hs_default = f"{slabel}.{hs_ext}"
                break
        else:
            hs_default = f"{slabel}.TS.HSX"
        if self.is_tbtrans:
            hs_default = fdf.get("TBT.HS", hs_default)
        return si.get_sile(fdf.dir_file(hs_default)).read_hamiltonian()

    def get(self, key: str, default=None, **kwargs):
        """Get TS.<key>, which is overwritten by TBT.<key>"""
        ret = self.fdf.get(f"TS.{key}", default, **kwargs)
        if self.is_tbtrans:
            ret = self.fdf.get(f"TBT.{key}", ret, **kwargs)
        return ret

    def block(self, key: str) -> str:
        """The label of the block TS.<key> or TBT.<key> (if present)"""
        if self.is_tbtrans and self.fdf.type(f"TBT.{key}") == "B":
            return f"TBT.{key}"
        return f"TS.{key}"

    def block_get(self, block: str, key: str, default=None, **kwargs):
        """Get <key> in `block`, which is overwritten by tbt.<key>"""
        ret = self.fdf.block_get(block, key, default, **kwargs)
        if self.is_tbtrans:
            ret = self.fdf.block_get(block, f"tbt.{key}", ret, **kwargs)
        return ret

    def names(self, key: str) -> list[str]:
        """The (non-comment) lines of the block TS.<key> (or TBT.<key>)"""
        names = []
        for line in self.get(key) or []:
            line = line.split("#", 1)[0].strip()
            if line:
                names.append(line)
        return names


def _fdf_atoms(value, na: int) -> np.ndarray:
    """Convert a TranSiesta/TBtrans atom list to (sorted) 0-based atomic indices

    The value is from `fdfSileSiesta.get`, either a list (``[1 -- 4]``),
    or a block where each line may start with ``atom`` followed by a list or
    integers.
    Indices are 1-based, negative values counts from the end (``-1`` is the last atom).
    """
    if value is None:
        return _a.arrayi([])
    if isinstance(value, list):
        lst = []
        for line in value:
            line = line.split("#", 1)[0].strip()
            word, *rest = line.split(maxsplit=1)
            if word.lower() in ("atom", "atoms"):
                line = rest[0] if rest else ""
            line = line.strip()
            if not line:
                continue
            if not line.startswith("["):
                line = f"[{line}]"
            lst.append(_parse_list(line))
        value = np.concatenate(lst) if lst else _a.arrayi([])
    value = np.atleast_1d(np.asarray(value, dtype=np.int64))
    value = np.where(value < 0, value + na + 1, value)
    if ((value < 1) | (value > na)).any():
        raise ValueError(f"atom indices are out of range [1, {na}]")
    return np.unique(value - 1)


def _fdf_electrode_atoms(opts: PropertyDict, na: int) -> np.ndarray:
    """Atoms (0-based) of the electrode from its ``electrode-position``

    ``<int>`` is the first atom, and ``end <int>`` is the last atom.
    Negative values counts from the end (``-1`` is the last atom).
    """
    position = opts.info.position
    if position is None:
        raise ValueError(f"electrode {opts.name} has no electrode-position")
    n = opts.info.na
    tokens = position.split()
    atom = int(tokens[-1])
    is_end = tokens[0].lower() == "end"
    if atom < 0:
        atom += na + 1
    if is_end:
        atom -= n - 1
    if atom < 1 or atom + n - 1 > na:
        raise ValueError(
            f"electrode {opts.name} with electrode-position {position!r} "
            f"and {n} atoms is outside the {na} atoms"
        )
    return np.arange(atom - 1, atom - 1 + n)


def _restrict_nsc(H: si.Hamiltonian, axes) -> si.Hamiltonian:
    """Remove couplings of `H` along the lattice vectors `axes`

    A copy is returned if `H` needs to be changed.
    """
    nsc = {"abc"[ax]: 1 for ax in axes if H.nsc[ax] > 1}
    if nsc:
        H = H.copy()
        H.set_nsc(**nsc)
    return H


def _semi_inf(elec: Electrode) -> list[tuple[int, int]]:
    """Semi-infinite lattice vectors of `elec` as ``(axis, direction)``

    The direction is ``1`` (``-1``) when the bulk electrode extends along
    the positive (negative) lattice vector, and ``0`` for directions without
    a side, e.g. the periodic directions of a real-space electrode.
    A downfolded electrode has the directions of the electrode it downfolds.
    """
    while isinstance(elec, DownfoldElectrode):
        elec = elec.electrode
    se = getattr(elec, "se", None)
    if isinstance(se, si.physics.SemiInfinite):
        return [(se.semi_inf, se.semi_inf_dir)]
    if isinstance(se, si.physics.RealSpaceSI):
        return [(se.semi.semi_inf, se.semi.semi_inf_dir)] + [
            (ax, 0) for ax in se._k_axes
        ]
    if isinstance(se, si.physics.RealSpaceSE):
        return [(se._semi_axis, 0)] + [(ax, 0) for ax in se._k_axes]
    # e.g. GF files, use the directions from `Electrode.from_fdf`
    return [
        ("abc".index(d[-1]), {"+": 1, "-": -1}.get(d[0], 0))
        for d in elec.info.get("semi_inf", [])
    ]


def _restrict_nsc_elecs(H: si.Hamiltonian, elecs) -> si.Hamiltonian:
    """Remove couplings of `H` along the semi-infinite directions of `elecs`

    A copy is returned if `H` needs to be changed.
    """
    return _restrict_nsc(H, {ax for elec in elecs for ax, _ in _semi_inf(elec)})


# tolerance [Ang] used when checking the electrode placement, see `_check_placement`
_PLACEMENT_TOL = 0.05


def _check_placement(
    H: si.Hamiltonian, elec: Electrode, orbs: np.ndarray, region: np.ndarray, what: str
) -> None:
    """Warn when `elec` does not seem to be placed correctly on the orbitals `orbs` of `H`

    This catches common setup mistakes, which otherwise silently gives wrong results:

    1. the orbitals `orbs` must have the same order, and the same relative positions
       (modulo the electrode lattice), as the orbitals of the electrode
       (only when the electrode geometry is known, see `Electrode._geometry`).
    2. for a single signed semi-infinite direction, `H` must not couple `orbs` to
       orbitals in `region` that lie on the semi-infinite side of the electrode.
       This catches wrong semi-infinite directions and electrodes not
       placed at the edge of `region`.

    Parameters
    ----------
    orbs :
        orbitals of `H` (without spin) of the rows of the self-energy
    region :
        orbitals of `H` (without spin) that are coupled to the electrode
        in the calculation, e.g. the device region
    """
    geom = H.geometry
    orbs = np.asarray(orbs).ravel()
    atoms = geom.o2a(orbs)

    egeom = elec._geometry()
    if egeom is not None and egeom.no == len(orbs):
        eorbs = np.arange(egeom.no)
        eatoms = egeom.o2a(eorbs)
        # same orbital (in each atom) and same relative positions
        same = np.array_equal(orbs - geom.firsto[atoms], eorbs - egeom.firsto[eatoms])
        if same:
            d = (geom.xyz[atoms] - geom.xyz[atoms[0]]) - (
                egeom.xyz[eatoms] - egeom.xyz[eatoms[0]]
            )
            frac = d @ egeom.lattice.icell.T
            d = (frac - np.rint(frac)) @ egeom.lattice.cell
            same = bool(np.all(np.linalg.norm(d, axis=1) <= _PLACEMENT_TOL))
        if not same:
            warn(
                f"{what} electrode {elec.name}: the orbitals it is placed on do not "
                "match the electrode orbitals (orbital order, or relative atomic "
                f"positions differ by more than {_PLACEMENT_TOL} Ang). "
                "The indices must be in the same order as the electrode orbitals "
                "(including Bloch expansions, as Hamiltonian.tile along a, b, c)."
            )

    sides = _semi_inf(elec)
    if len(sides) != 1 or sides[0][1] == 0:
        return
    ax, side = sides[0]
    # the surface normal (pointing into the bulk electrode), the transverse
    # lattice vectors are perpendicular to it
    n = geom.icell[ax] * side
    n /= np.linalg.norm(n)

    # all couplings of the electrode orbitals (any spin component, or overlap)
    M = abs(H.tocsr(0)[orbs])
    for dim in range(1, H.dim):
        M = M + abs(H.tocsr(dim)[orbs])
    M = M.tocsr()
    M.eliminate_zeros()
    jo = np.unique(M.indices)
    jo_uc = jo % geom.no
    is_region = np.zeros(geom.no, dtype=bool)
    is_region[np.asarray(region).ravel()] = True
    is_region[orbs] = False
    jo = jo[is_region[jo_uc]]
    if len(jo) == 0:
        return
    ja = np.unique(geom.o2a(jo))
    outer = (geom.xyz[np.unique(atoms)] @ n).max()
    bad = geom.axyz(ja) @ n > outer + _PLACEMENT_TOL
    if bad.any():
        direction = "+-"[side < 0] + "abc"[ax]
        warn(
            f"{what} electrode {elec.name} is coupled to atoms "
            f"{np.unique(ja[bad] % geom.na).tolist()} lying on its semi-infinite "
            f"side ({direction}). Likely the semi-infinite direction is wrong, or "
            "the electrode is not placed at the edge of the region."
        )


def _fdf_electrode_options(
    tfdf: _TransportFdf, name: str, clsname: str
) -> PropertyDict:
    """Parse the electrode block ``TS.Elec.<name>``, no self-energies are created

    The chemical potential is not resolved, see `_fdf_electrode_mu`.
    """
    fdf = tfdf.fdf
    block = tfdf.block(f"Elec.{name}")
    if fdf.type(block) != "B":
        raise ValueError(f"{clsname} could not find block: {block}")

    def block_get(key, default=None, **kwargs):
        return tfdf.block_get(block, key, default, **kwargs)

    opts = PropertyDict(name=name, block=block)

    # Get semi-infinite direction(s), this determines the electrode class
    semi_inf = None
    for suf in ("-direction", "-dir", ""):
        semi_inf = block_get(f"semi-inf{suf}", semi_inf)
    if semi_inf is None:
        raise ValueError(
            f"{clsname} could not find electrode semi-inf-direction "
            f"in block: {block}"
        )
    try:
        semi_inf = _fdf_semi_inf(semi_inf)
    except ValueError as e:
        raise ValueError(f"{clsname}: {e}") from None
    if len(semi_inf) == 1 and semi_inf[0][0] not in "+-":
        raise ValueError(
            f"{clsname} requires a signed semi-inf-direction, got {semi_inf[0]}"
        )

    gf = tfdf.get(f"Elec.{name}.GF")
    for key in ("gf", "gf-file"):
        gf = block_get(key, gf)

    Helec = tfdf.get(f"Elec.{name}.HS")
    for key in ("hs", "hs-file", "tshs", "tshs-file"):
        Helec = block_get(key, Helec)
    if not Helec:
        raise ValueError(f"{clsname} could not find electrode HS in block: {block}")
    Helec = si.get_sile(fdf.dir_file(Helec)).read_hamiltonian()

    used_atoms = block_get("used-atoms")
    if used_atoms is not None and int(used_atoms) != Helec.na:
        raise NotImplementedError(
            f"{clsname} does not implement used-atoms different "
            "from the number of atoms in the electrode."
        )

    bloch = [tfdf.get(f"Elec.{name}.Bloch.A{i+1}", 1) for i in range(3)]
    for i, sufs in enumerate([("a", "a1"), ("b", "a2"), ("c", "a3")]):
        for suf in sufs:
            bloch[i] = block_get(f"bloch-{suf}", bloch[i])
    bloch = block_get("bloch", f"{bloch[0]} {bloch[1]} {bloch[2]}")
    bloch = [int(b) for b in bloch.split()]

    opts.H = Helec
    opts.bloch = bloch
    opts.eta = block_get("eta", tfdf.get("Elecs.Eta", 1e-3, unit="eV"), unit="eV")

    info = PropertyDict()
    info.bulk = bool(block_get("bulk", tfdf.get("Elecs.Bulk", True)))
    info.position = None
    for key in ("electrode-position", "elec-pos"):
        info.position = block_get(key, info.position)
    if info.position is not None:
        info.position = str(info.position)
    info.chemical_potential = None
    for key in ("chemical-potential", "chem-pot", "mu"):
        info.chemical_potential = block_get(key, info.chemical_potential)
    # manual shift of the fermi-level
    info.dEf = block_get("delta-Ef", 0.0, unit="eV")
    # add a fraction of the bias in the coupling elements of the
    # E-C region, only meaningful for
    info.V_fraction = float(block_get("V-fraction", 0.0))
    info.semi_inf = semi_inf
    info.na = Helec.na * int(np.prod(bloch))
    info.gf = gf
    opts.info = info
    return opts


def _fdf_electrode_mu(tfdf: _TransportFdf, opts: PropertyDict, clsname: str) -> float:
    """Chemical potential of the electrode, see `Electrode.from_fdf`"""
    fdf = tfdf.fdf
    chem_pots = tfdf.names("ChemPots")

    # Same defaults as TranSiesta/TBtrans: the only chemical potential,
    # or the one with the same name as the electrode
    cp = opts.info.chemical_potential
    if cp is None:
        cp = chem_pots[0] if len(chem_pots) == 1 else opts.name

    V = tfdf.get("Voltage", 0.0, unit="eV")
    cp_block = tfdf.block(f"ChemPot.{cp}")
    if not chem_pots and tfdf.is_tbtrans and abs(V) <= 1e-5:
        # TBtrans uses a single chemical potential when there
        # are no chemical potentials and no bias
        mu = 0.0
    elif fdf.type(cp_block) == "B":
        mu = tfdf.block_get(cp_block, "mu", 0.0, unit="eV")
    elif not chem_pots and cp.lower() in ("left", "right"):
        # TranSiesta fakes the Left/Right chemical potentials
        # when none are specified
        mu = {"left": "V/2", "right": "-V/2"}[cp.lower()]
    else:
        raise ValueError(
            f"{clsname} could not find the chemical-potential block: {cp_block}"
        )
    if isinstance(mu, str):
        mu = _fdf_mu(mu, V)
    return mu


def _fdf_electrode(
    tfdf: _TransportFdf,
    opts: PropertyDict,
    mu: Optional[float],
    use_gf: bool,
    clsname: str,
) -> Electrode:
    """Create the (not downfolded) electrode from the parsed electrode block `opts`"""
    name = opts.name
    info = PropertyDict(opts.info)
    semi_inf = info.semi_inf
    is_real_space = len(semi_inf) > 1

    # electrodes without a GF file are always calculated in memory
    use_gf = use_gf and bool(info.gf)
    if not use_gf and is_real_space:
        raise ValueError(
            f"{clsname} found the semi-inf-direction {''.join(semi_inf)}, "
            "real-space self-energies can not be calculated from the fdf "
            "information alone. Use use_gf=True to read them from the GF file "
            "(GF option in the block), or create a RealSpaceElectrode with a "
            "sisl.physics.RealSpaceSE."
        )

    if mu is None:
        mu = _fdf_electrode_mu(tfdf, opts, clsname)
    info.mu = mu
    if info.V_fraction > 0.0:
        warn(
            f"{clsname} found a non-zero V-fraction value. "
            "This is currently not implemented."
        )

    if use_gf:
        gf = si.get_sile(tfdf.fdf.dir_file(info.gf))
        if is_real_space:
            elec = RealSpaceElectrode(gf, name=name)
        else:
            # GF files are already Bloch expanded
            elec = BlochElectrode(gf, name=name)
    else:
        Helec = opts.H.copy()
        Helec.shift(info.dEf)
        # shift according to potential
        Helec.shift(mu)

        if is_real_space:
            raise ValueError(
                f"{clsname} currently does not implement "
                "real-space electrodes directly reading the fdf"
            )
        else:
            elec = BlochElectrode(
                si.RecursiveSI(Helec, semi_inf[0], eta=opts.eta),
                bloch=opts.bloch,
                name=name,
            )

    if len(elec) != len(opts.H) * int(np.prod(opts.bloch)):
        raise ValueError(
            f"{clsname} found an inconsistent size of the GF file "
            f"({len(elec)}) and the electrode HS ({len(opts.H)}) "
            f"with Bloch expansion {opts.bloch}."
        )
    elec.info.update(info)
    return elec


def _fdf_device(tfdf: _TransportFdf, clsname: str) -> PropertyDict:
    """Read the device Hamiltonian, the electrode blocks and the regions

    See `DeviceGreen.from_fdf` for the fdf flags used.

    Returns
    -------
    PropertyDict
        with ``H`` (restricted along the semi-infinite directions), ``names``,
        ``opts`` (`_fdf_electrode_options` for each electrode), the atoms
        ``elec_atoms``, ``device_atoms``, ``buffer_atoms``, the equivalent orbitals
        ``elec_orbs``, ``device_orbs``, ``buffer_orbs`` and ``eta`` (of the device region).
    """
    fdf = tfdf.fdf

    # Read the device H
    H = tfdf.read_device_hamiltonian()

    # Read the electrodes
    # same default as TranSiesta/TBtrans
    names = tfdf.names("Elecs") or ["Left", "Right"]
    opts = [_fdf_electrode_options(tfdf, name, clsname) for name in names]
    # Append H_device
    for opt in opts:
        opt.H_device = H

    # default to the smallest electrode eta
    # Correct by a factor 1/10 to minimize smearing for device states.
    # We want the electrode to smear.
    eta = min((o.eta for o in opts), default=0.0) / 10
    if tfdf.is_tbtrans:
        eta = fdf.get("TBT.Contours.Eta", eta, unit="eV")
    else:
        eta = fdf.get("TS.Contours.nEq.Eta", eta, unit="eV")

    # Limit connections of the device along the semi-inf directions
    axes = {"abc".index(d[-1]) for o in opts for d in o.info.semi_inf}
    H = _restrict_nsc(H, axes)
    na = H.na

    # Regions
    elec_atoms = [_fdf_electrode_atoms(o, na) for o in opts]
    buffer_atoms = _fdf_atoms(tfdf.get("Atoms.Buffer"), na)
    regions = [("buffer", buffer_atoms)]
    regions.extend(zip(names, elec_atoms))

    count = np.zeros(na, dtype=np.int32)
    for _, atoms in regions:
        count[atoms] += 1
    if (count > 1).any():
        raise ValueError(
            f"{clsname} found overlapping regions for atoms "
            f"{(np.flatnonzero(count > 1) + 1).tolist()} (1-based)"
        )

    device = fdf.get("TBT.Atoms.Device") if tfdf.is_tbtrans else None
    if device is None:
        # all atoms not in the buffer, nor in the electrodes
        device_atoms = np.flatnonzero(count == 0)
    else:
        # the device region may contain (parts of) the electrodes,
        # which are then (partially) not downfolded
        device_atoms = _fdf_atoms(device, na)
        overlap = np.intersect1d(device_atoms, buffer_atoms)
        if len(overlap) > 0:
            raise ValueError(
                f"{clsname} found overlapping device and buffer atoms "
                f"{(overlap + 1).tolist()} (1-based)"
            )

    def a2o(atoms):
        return H.a2o(atoms, all=True)

    return PropertyDict(
        H=H,
        names=names,
        opts=opts,
        elec_atoms=elec_atoms,
        device_atoms=device_atoms,
        buffer_atoms=buffer_atoms,
        elec_orbs=[a2o(atoms) for atoms in elec_atoms],
        device_orbs=a2o(device_atoms),
        buffer_orbs=a2o(buffer_atoms),
        eta=eta,
    )


def _fdf_downfold(
    tfdf: _TransportFdf, setup: PropertyDict
) -> list[Optional[PermuteBTD]]:
    """Downfolding regions (in orbitals) of the electrodes in `setup` (from `_fdf_device`)

    The regions are found in the atomic or orbital space, according to the
    ``atom+``/``orb+`` prefix of ``TBT.BTD.Pivot.Elec.<name>`` (or ``TBT.BTD.Pivot.Elecs``),
    which defaults to atoms (as TBtrans).
    """
    spaces = []
    for name in setup.names:
        pivot = "atom+"
        if tfdf.is_tbtrans:
            pivot = tfdf.fdf.get("TBT.BTD.Pivot.Elecs", pivot)
            pivot = tfdf.fdf.get(f"TBT.BTD.Pivot.Elec.{name}", pivot)
        spaces.append(_split_pivot(pivot)[0])

    H = setup.H
    ret = [None] * len(spaces)
    for space in set(spaces):
        if space == "atoms":
            parts = downfold_partition(
                H.toSparseAtom(),
                setup.elec_atoms,
                setup.device_atoms,
                setup.buffer_atoms,
            )
            parts = [p if p is None else _permute_a2o(H.geometry, p) for p in parts]
        else:
            parts = downfold_partition(
                H, setup.elec_orbs, setup.device_orbs, setup.buffer_orbs
            )
        for i, s in enumerate(spaces):
            if s == space:
                ret[i] = parts[i]
    return ret


@set_module("sisl_toolbox.btd")
class Electrode(ABC):
    """Base class for electrodes used in `DeviceGreen`

    This class is abstract, use `Electrode.from_fdf` to create the electrode
    class matching an fdf electrode block, or one of the sub-classes
    directly.

    Sub-classes must implement `__len__`, `self_energy` and `se2broadening`.

    Parameters
    ----------
    name :
        name of the electrode, used to reference it in `DeviceGreen`.
        Without a name the electrode can only be referenced by its index
        (or the electrode object itself).
    """

    def __init__(self, *, name: Optional[str] = None):
        self.name = name
        #: orbitals of the device Hamiltonian where the self-energy is placed
        #: (without spin), None if unknown.
        #: When set, `DeviceGreen` accepts the electrode without its orbitals.
        self.orbs = None
        # additional information, e.g. from `Electrode.from_fdf`
        self.info = PropertyDict()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}{{name: {self.name}, no: {len(self)}}}"

    @abstractmethod
    def __len__(self) -> int:
        """Size of the self-energy matrix (including spin)"""

    @abstractmethod
    def self_energy(self, E: complex, k: KPoint = (0, 0, 0), **kwargs) -> np.ndarray:
        """Return the self-energy matrix"""

    @abstractmethod
    def se2broadening(self, SE: np.ndarray) -> np.ndarray:
        """Convert the self-energy `SE` to the broadening matrix

        This is delegated to the underlying self-energy object.
        """

    @property
    def eta(self) -> float | None:
        """Imaginary part used for the self-energy calculation (if known)"""
        return None

    @property
    def has_bulk(self) -> bool:
        """Whether `bulk_matrix` is implemented for this electrode"""
        return False

    def _geometry(self) -> si.Geometry | None:
        """Geometry of the rows of the self-energy (without spin), None if unknown"""
        return None

    def bulk_matrix(
        self,
        E: complex,
        k: KPoint = (0, 0, 0),
        *,
        dtype=np.complex128,
        format: str = "array",
        **kwargs,
    ):
        r"""Return :math:`E\mathbf S - \mathbf H` of the bulk electrode

        The rows (and columns) are the same as for `self_energy`.
        For real energies, `eta` is added to `E`.
        Only available when `has_bulk` is true.

        Parameters
        ----------
        E :
            energy
        k :
            k-point
        dtype :
            data-type of the returned matrix
        format :
            format of the returned matrix, ``"array"`` for a dense matrix,
            or a sparse format (e.g. ``"csr"``), see `sisl.physics.Hamiltonian.Hk`
        **kwargs :
            passed to `sisl.physics.Hamiltonian.Hk` (e.g. ``spin``)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}(name={self.name}) has no bulk matrix."
        )

    @classmethod
    def from_fdf(
        cls,
        fdf: SileLike,
        name: str,
        prefix: Literal["TBT", "TS"] = "TBT",
        mu: Optional[float] = None,
        use_gf: bool = False,
        downfold: bool = True,
    ) -> Electrode:
        """Create the electrode from the ``%block TS.Elec.<name>`` in an fdf file

        The returned class depends on the content of the block:

        - a single ``semi-inf-direction`` (e.g. ``-a3``) returns a `BlochElectrode`
          using a `sisl.physics.RecursiveSI` self-energy, or the ``GF`` file
          (when `use_gf` is true).
        - multiple ``semi-inf-direction`` (e.g. ``ab`` or ``abc``) returns a
          `RealSpaceElectrode` using the ``GF`` file. This requires `use_gf`,
          since the real-space self-energy cannot be created from the fdf file
          (create a `RealSpaceElectrode` with a `sisl.physics.RealSpaceSE` instead).

        When `downfold` is true, and the electrode is not (fully) in the device
        region, the above electrode is (partially) downfolded into the device
        region, and a `DownfoldElectrode` is returned. This reads the device Hamiltonian
        (``TBT.HS``) and the other electrodes (``TS.Elecs``) to determine the
        downfolding region, see `downfold_partition` and `DeviceGreen.from_fdf`
        for the fdf flags used. The downfolding region is found for atoms
        or orbitals according to the ``atom+``/``orb+`` prefix of
        ``TBT.BTD.Pivot.Elec.<name>`` (or ``TBT.BTD.Pivot.Elecs``), defaulting to atoms.

        The electrode Hamiltonian is shifted by ``delta-Ef`` and the chemical potential.
        A GF file should already contain these shifts.
        File names are relative to the directory of the fdf file.

        Other options from the fdf file are stored in the ``info`` attribute:

        - ``bulk``: whether the electrode Hamiltonian is used in the electrode region
          (true), or the device Hamiltonian (false), in the downfolding region, or
          the device region (for electrodes fully in ``TBT.Atoms.Device``)
        - ``dEf``: the ``delta-Ef`` shift of the electrode Hamiltonian
        - ``mu``: the chemical potential shift
        - ``chemical_potential``: name of the chemical potential (or None)
        - ``position``: the ``electrode-position`` value (or None)
        - ``V_fraction``: the fraction of the bias in the electrode couplings
        - ``semi_inf``: the semi-infinite directions (sisl notation, e.g. ``["-c"]``)
        - ``na``: number of atoms of the electrode in the device (including
          Bloch expansions)
        - ``gf``: the ``GF`` file name (or None)

        When `downfold` is true, the `orbs` attribute holds the orbitals of the
        device Hamiltonian where the self-energy is placed (the electrode orbitals
        for electrodes fully in the device region).

        Parameters
        ----------
        fdf :
           fdf file to read the parameters from
        name :
           name of the electrode, i.e. the block ``TS.Elec.<name>``
        prefix :
           which prefix to use, if TBT it will prefer TBT prefix, but fall back
           to TS prefixes.
           If TS, only these prefixes will be used.
        mu :
           chemical potential of the electrode [eV], if None it is read from the
           ``%block TS.ChemPot.<chemical-potential>`` (and the bias).
           As in TranSiesta, ``chemical-potential`` defaults to the only chemical
           potential in ``TS.ChemPots``, or the one named as the electrode.
           Without ``TS.ChemPots``, ``Left`` and ``Right`` are ``V/2`` and ``-V/2``.
           Without ``TBT.ChemPots`` (and ``TS.ChemPots``) and no bias, TBtrans
           uses a single chemical potential at 0 for all electrodes.
        use_gf :
           whether the self-energies are read from the ``GF`` file of the block
           (``GF`` or ``TBT.Elec.<name>.GF``), otherwise they are calculated
           in memory (the ``GF`` file is ignored).
           Electrodes without a ``GF`` file are always calculated in memory.
        downfold :
           whether the electrode is downfolded into the device region (unless
           it is fully in the device region). Use false to get the electrode itself,
           e.g. for `DeviceGreen.from_tbt`.
        """
        tfdf = _TransportFdf(fdf, prefix)
        clsname = f"{cls.__name__}.from_fdf(electrode={name})"
        if not downfold:
            opts = _fdf_electrode_options(tfdf, name, clsname)
            return _fdf_electrode(tfdf, opts, mu, use_gf, clsname)

        setup = _fdf_device(tfdf, clsname)
        if name not in setup.names:
            raise ValueError(
                f"{clsname} could not find the electrode in the electrodes "
                f"{setup.names}, use downfold=False for electrodes not in the device."
            )
        i = setup.names.index(name)
        elec = _fdf_electrode(tfdf, setup.opts[i], mu, use_gf, clsname)
        orbs = setup.elec_orbs[i]

        down = _fdf_downfold(tfdf, setup)[i]
        if down is None:
            elec.orbs = orbs
            return elec
        ret = DownfoldElectrode(
            elec,
            setup.H,
            {"orbitals": orbs},
            {"orbitals": down},
            eta_device=setup.eta,
            bulk=elec.info.bulk,
        )
        ret.info.update(elec.info)
        return ret


@set_module("sisl_toolbox.btd")
class BlochElectrode(Electrode):
    """Electrode with a (possibly Bloch expanded) self-energy

    The rows of the self-energy are the orbitals of the Bloch expanded electrode,
    i.e. matching ``se.spgeom0.tile(...)`` for the `bloch` expansions.

    Parameters
    ----------
    se :
        the self-energy calculator, a `sisl.physics.SelfEnergy` or a
        Siesta GF file (`~sisl.io.siesta.gfSileSiesta`).
        A GF file is already Bloch expanded.
    bloch :
        Bloch expansion of the self-energy
    name :
        name of the electrode, see `Electrode`
    """

    def __init__(
        self,
        se,
        bloch: si.Bloch | tuple[int, int, int] | None = None,
        *,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        if bloch is None:
            bloch = (1, 1, 1)
        if not isinstance(bloch, si.Bloch):
            bloch = si.Bloch(bloch)
        self.bloch = bloch

        if _is_gf(se):
            if len(bloch) > 1:
                raise ValueError(
                    f"{self.__class__.__name__} got a GF file and a Bloch expansion, "
                    "GF files are already Bloch expanded."
                )
            se = _GFSelfEnergy(se)
        elif not isinstance(se, si.physics.SelfEnergy):
            raise ValueError(f"{self.__class__.__name__} got an unknown 'se' argument.")
        self.se = se

    @property
    def eta(self) -> float | None:
        """Imaginary part used for the self-energy calculation (if known)"""
        return getattr(self.se, "eta", None)

    @property
    def has_bulk(self) -> bool:
        """Whether `bulk_matrix` is implemented, requires a `sisl.physics.RecursiveSI` self-energy or a GF file"""
        return isinstance(self.se, (si.physics.RecursiveSI, _GFSelfEnergy))

    def __len__(self) -> int:
        """Size of the self-energy matrix (including spin)"""
        if isinstance(self.se, _GFSelfEnergy):
            # Includes spin (non-colinear) and Bloch expansions
            return len(self.se)
        return len(self.se) * len(self.bloch)

    def __str__(self) -> str:
        se = str(self.se).replace("\n", "\n ")
        return f"{self.__class__.__name__}{{name: {self.name}, no: {len(self)}, bloch: {self.bloch.bloch},\n {se}\n}}"

    def _call_bloch(self, func, k: KPoint, **kwargs) -> np.ndarray:
        """Call `func` with the Bloch expansion"""
        if len(self.bloch) == 1:
            return func(k=k, **kwargs)
        return self.bloch(func, k, **kwargs)

    def self_energy(self, E: complex, k: KPoint = (0, 0, 0), **kwargs) -> np.ndarray:
        """Return the (Bloch expanded) self-energy"""
        return self._call_bloch(self.se.self_energy, k, E=E, **kwargs)

    def se2broadening(self, SE: np.ndarray) -> np.ndarray:
        """Convert the self-energy `SE` to the broadening matrix, see `sisl.physics.SelfEnergy.se2broadening`"""
        return self.se.se2broadening(SE)

    def _geometry(self) -> si.Geometry | None:
        """Geometry of the (Bloch expanded) electrode, None if unknown"""
        spgeom0 = getattr(self.se, "spgeom0", None)
        if spgeom0 is None:
            return None
        geom = spgeom0.geometry
        for ax, reps in enumerate(self.bloch.bloch):
            if reps > 1:
                geom = geom.tile(reps, ax)
        return geom

    def _bulk_hamiltonian(self) -> si.Hamiltonian:
        """The (Bloch expanded) electrode Hamiltonian of the principal layer

        The Bloch expansion is equivalent to tiling along the lattice vectors
        in the order a, b, c (same orbital order and matrix elements).
        The tiled Hamiltonian is cached.
        """
        bloch = tuple(self.bloch.bloch)
        cache = getattr(self, "_bulk_cache", None)
        if cache is None or cache[0] != bloch or cache[1] is not self.se:
            H = self.se.spgeom0
            for ax, reps in enumerate(bloch):
                if reps > 1:
                    H = H.tile(reps, ax)
            self._bulk_cache = cache = (bloch, self.se, H)
        return cache[2]

    def bulk_matrix(
        self,
        E: complex,
        k: KPoint = (0, 0, 0),
        *,
        dtype=np.complex128,
        format: str = "array",
        **kwargs,
    ):
        r"""Return :math:`E\mathbf S - \mathbf H` of the (Bloch expanded) electrode

        For real energies, `eta` is added to `E`.
        For GF files, the energy (and :math:`\mathbf H`, :math:`\mathbf S`) of the file is used.
        See `Electrode.bulk_matrix` for the arguments.
        """
        if not self.has_bulk:
            return super().bulk_matrix(E, k, dtype=dtype, format=format, **kwargs)
        if isinstance(self.se, _GFSelfEnergy):
            return self.se.bulk_matrix(E, k, dtype=dtype, format=format, **kwargs)
        return _bulk_matrix(
            self._bulk_hamiltonian(), E, self.eta, k, dtype, format, **kwargs
        )


@set_module("sisl_toolbox.btd")
class RealSpaceElectrode(Electrode):
    """Electrode with a real-space self-energy

    Used for electrodes with more than one semi-infinite direction.
    The rows of the self-energy are the orbitals of the real-space coupling
    region, i.e. matching ``se.real_space_coupling()``.

    Parameters
    ----------
    se :
        the real-space self-energy calculator (`sisl.physics.RealSpaceSE` or
        `sisl.physics.RealSpaceSI`), or a Siesta/TBtrans GF file
        (`~sisl.io.tbtrans.tbtgfSileTBtrans`) containing the real-space
        self-energies (see the `bulk` and `coupling` arguments of
        `sisl.physics.RealSpaceSE.self_energy`).
    name :
        name of the electrode, see `Electrode`
    """

    def __init__(
        self,
        se: si.physics.RealSpaceSE | si.physics.RealSpaceSI,
        *,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if _is_gf(se):
            se = _GFSelfEnergy(se)
            self._H = None
        elif isinstance(se, (si.physics.RealSpaceSE, si.physics.RealSpaceSI)):
            # The self-energies are on the coupling region only
            self._H = se.real_space_coupling()
        else:
            raise ValueError(f"{self.__class__.__name__} got an unknown 'se' argument.")
        self.se = se

    @property
    def eta(self) -> float | None:
        """Imaginary part used for the self-energy calculation"""
        return self.se.eta

    @property
    def has_bulk(self) -> bool:
        """Whether `bulk_matrix` is implemented (always true)"""
        return True

    def __len__(self) -> int:
        """Size of the self-energy matrix (including spin)"""
        if self._H is None:
            return len(self.se)
        return len(self._H)

    def __str__(self) -> str:
        se = str(self.se).replace("\n", "\n ")
        return (
            f"{self.__class__.__name__}{{name: {self.name}, no: {len(self)},\n {se}\n}}"
        )

    def _geometry(self) -> si.Geometry | None:
        """Geometry of the real-space coupling region, None if unknown"""
        if self._H is None:
            return None
        return self._H.geometry

    def self_energy(self, E: complex, k: KPoint = (0, 0, 0), **kwargs) -> np.ndarray:
        """Return the real-space self-energy of the coupling region"""
        if self._H is None:
            return self.se.self_energy(E, k=k, **kwargs)
        return self.se.self_energy(E, k=k, coupling=True, **kwargs)

    def se2broadening(self, SE: np.ndarray) -> np.ndarray:
        """Convert the self-energy `SE` to the broadening matrix, see `sisl.physics.SelfEnergy.se2broadening`"""
        return self.se.se2broadening(SE)

    def bulk_matrix(
        self,
        E: complex,
        k: KPoint = (0, 0, 0),
        *,
        dtype=np.complex128,
        format: str = "array",
        **kwargs,
    ):
        r"""Return :math:`E\mathbf S - \mathbf H` of the real-space coupling region

        For real energies, `eta` is added to `E`.
        For GF files, the energy (and :math:`\mathbf H`, :math:`\mathbf S`) of the file is used.
        See `Electrode.bulk_matrix` for the arguments.
        """
        if self._H is None:
            return self.se.bulk_matrix(E, k, dtype=dtype, format=format, **kwargs)
        return _bulk_matrix(self._H, E, self.eta, k, dtype, format, **kwargs)


def _downfold_region(
    H: si.Hamiltonian, orbs_elec: np.ndarray, down, what: str
) -> PermuteBTD:
    """The downfolding region `down` (see `DownfoldElectrode`) in orbitals"""
    space, down = _split_space(down, what)
    if isinstance(down, PermuteBTD):
        if space == "atoms":
            _check_atoms(H, down.permute, what)
            return _permute_a2o(H.geometry, down)
        return down

    # find the order and blocks of the region from the electrode
    down = np.unique(np.asarray(down).ravel())
    if space == "atoms":
        _check_atoms(H, down, what)
        csr = H.toSparseAtom()
        elec = np.unique(H.o2a(orbs_elec))
        n = H.na
    else:
        csr = H
        elec = orbs_elec
        n = H.no
    is_down = np.zeros(n, dtype=bool)
    is_down[down] = True
    if not is_down[elec].any():
        raise ValueError(f"{what} must contain the electrode (or a part of it)")
    region, blocks = _downfold_permute(
        _graph(csr), elec, np.flatnonzero(~is_down), np.array([], dtype=np.int64)
    )
    if not np.array_equal(np.sort(region[: -blocks[-1]]), down):
        raise ValueError(f"{what} contains indices not connected to the electrode")
    down = PermuteBTD(region, blocks)
    if space == "atoms":
        return _permute_a2o(H.geometry, down)
    return down


@set_module("sisl_toolbox.btd")
class DownfoldElectrode(Electrode):
    r"""Electrode downfolded through a region of `H`

    The self-energy of `electrode` (any `Electrode`, also a `DownfoldElectrode`)
    is placed on `elec` of `H`, and is then downfolded through the region
    `down` to the last block of the region.

    The electrode may be partially downfolded, i.e. part of `elec` may be in
    the last block (the device region). Then the electrode self-energy
    couples the downfolded electrode indices directly to the last block, and
    they must all be in the block before the last block. The self-energy
    of the electrode indices in the last block is added to the downfolded
    self-energy.

    The rows of the returned self-energy are the orbitals of the last block,
    in the order they appear in the downfolding region (see `orbs`).

    Indices are atoms, ``{"atoms": idx}`` (equivalent) or ``{"orbitals": idx}``
    (not accounting for spin). Atoms are expanded to all their orbitals (in order).

    Parameters
    ----------
    electrode :
        the electrode being downfolded
    H :
        Hamiltonian containing (at least) the downfolding region.
        Couplings along the semi-infinite directions of `electrode` are
        removed (on a copy), the electrode self-energy replaces them.
    elec :
        indices of `H` corresponding to the rows of the electrode self-energy
        (in that order)
    down :
        the downfolding region of `H`, either

        - a `PermuteBTD`: the region (including `elec` and the last block) in
          block-tri-diagonal order, the last block are the indices that the
          self-energy is downfolded to.
        - an array: the indices of the region (including the downfolded part of
          `elec`, excluding the last block). The order and blocks are found from
          `elec` (see `downfold_partition`), and the last block are the indices
          of `H` coupling to the region (and the indices of `elec` not in `down`).
    eta_device :
        imaginary part used in the downfolding region (for real energies)
    bulk :
        whether the electrode region of `H` (the downfolded part of `elec`) is
        replaced by the bulk electrode matrix (`Electrode.bulk_matrix`).
        Couplings to the rest of `H` (including the electrode indices
        in the last block) are taken from `H`. Requires ``electrode.has_bulk``.
    name :
        name of the electrode, defaults to the `electrode` name (see `Electrode`)
    """

    def __init__(
        self,
        electrode: Electrode,
        H: si.Hamiltonian,
        elec: npt.ArrayLike | Mapping[str, npt.ArrayLike],
        down: PermuteBTD | npt.ArrayLike | Mapping[str, PermuteBTD | npt.ArrayLike],
        eta_device: float = 0.0,
        bulk: bool = True,
        *,
        name: Optional[str] = None,
    ):
        if name is None:
            name = electrode.name
        super().__init__(name=name)
        self.electrode = electrode
        self._eta_device = eta_device
        cls = self.__class__.__name__
        H = _restrict_nsc_elecs(H, [electrode])

        # everything is stored in orbitals
        orbs_elec = _to_orbs(H, elec, f"{cls} elec")
        down = _downfold_region(H, orbs_elec, down, f"{cls} down")
        orbs_down = down.permute
        blocks_down = down.blocks
        #: orbitals of `H` of the rows of the self-energy (the last block)
        self.orbs = orbs_down[-blocks_down[-1] :]
        #: orbitals of `H` that are downfolded (the region without the last block),
        #: including the downfolded orbitals of a downfolded `electrode`.
        #: `DeviceGreen` checks these are not in the device, nor other downfolding regions.
        region = [orbs_down[: -blocks_down[-1]]]
        if isinstance(electrode, DownfoldElectrode):
            region.append(electrode.orbs_down)
        self.orbs_down = np.unique(np.concatenate(region))
        if len(np.unique(orbs_down)) != len(orbs_down):
            raise ValueError(f"{cls} down has duplicate indices")
        if len(blocks_down) < 2:
            raise ValueError(f"{cls} down must have at least 2 blocks")
        # electrode orbitals that are downfolded (r), and in the last block (d)
        is_elec_r = np.isin(orbs_elec, orbs_down[: -blocks_down[-1]])
        is_elec_d = np.isin(orbs_elec, self.orbs)
        if not (is_elec_r | is_elec_d).all():
            raise ValueError(f"{cls} elec must be a subset of down")
        if not is_elec_r.any():
            raise ValueError(
                f"{cls} elec is fully in the last block of down, nothing to downfold"
            )
        if is_elec_d.any():
            # the self-energy couples the downfolded electrode orbitals
            # directly to the last block
            prev = orbs_down[-blocks_down[-1] - blocks_down[-2] : -blocks_down[-1]]
            if not np.isin(orbs_elec[is_elec_r], prev).all():
                raise ValueError(
                    f"{cls} elec is partially in the last block of down, "
                    "the downfolded electrode indices must be in the block "
                    "before the last block"
                )
        if not isinstance(electrode, DownfoldElectrode):
            _check_placement(H, electrode, orbs_elec, orbs_down, cls)
        if bulk and not electrode.has_bulk:
            raise ValueError(
                f"{cls}(bulk=True) requires an electrode with a bulk matrix, "
                f"{electrode.__class__.__name__}(name={electrode.name}) has none."
            )

        # storage data
        self._data = PropertyDict()
        data = self._data
        data.bulk = bulk

        # The Hamiltonian of the atoms in the downfolding region.
        # Its orbitals are the (sorted) orbitals of all atoms in the downfolding
        # region, they need not all be part of `orbs_down`.
        atoms_down = H.o2a(orbs_down, unique=True)
        data.H = H.sub(atoms_down)
        orbs_sub = H.a2o(atoms_down, all=True)

        def positions(orbs):
            """Positions of `orbs` in `orbs_sub` (sorted)"""
            idx = np.searchsorted(orbs_sub, orbs)
            assert np.array_equal(orbs_sub[idx], orbs)
            return idx

        # Spin expansion (non-colinear) of the matrices
        expand = get_expand(len(data.H), data.H.no)
        if len(electrode) != len(orbs_elec) * expand:
            raise ValueError(
                f"{cls} electrode {electrode.name} has size {len(electrode)}, "
                f"but orbs_elec gives {len(orbs_elec) * expand}"
            )

        # Indices in data.H (expanded)
        #  elec: the electrode orbitals, in the order of the electrode self-energy
        #  down: the downfolding orbitals, in the BTD order
        data.elec = expand_orbs(positions(orbs_elec), expand).reshape(-1, 1)
        data.down = expand_orbs(positions(orbs_down), expand).reshape(-1, 1)
        # Indices in the electrode self-energy (expanded)
        #  elec_r: the downfolded electrode orbitals
        #  elec_d: the electrode orbitals in the last block
        data.elec_r = expand_orbs(np.flatnonzero(is_elec_r), expand).reshape(-1, 1)
        data.elec_d = expand_orbs(np.flatnonzero(is_elec_d), expand).reshape(-1, 1)
        # Indices of elec_d in the last block (expanded)
        last = np.argsort(self.orbs)
        last = last[np.searchsorted(self.orbs, orbs_elec[is_elec_d], sorter=last)]
        data.last_d = expand_orbs(last, expand).reshape(-1, 1)
        self.blocks = expand_blocks(blocks_down, expand)
        data.blocks_cum0 = np.append(0, np.cumsum(self.blocks))

    def __len__(self) -> int:
        """Size of the self-energy matrix (including spin)"""
        return self.blocks[-1]

    @property
    def eta(self) -> float | None:
        """Imaginary part used for the self-energy calculation of `electrode` (if known)"""
        return self.electrode.eta

    def __str__(self) -> str:
        se = str(self.electrode).replace("\n", "\n ")
        return f"{self.__class__.__name__}{{name: {self.name}, no: {len(self)}, blocks: {len(self.blocks)}, eta_device: {self._eta_device}, bulk: {self._data.bulk},\n {se}\n}}"

    def se2broadening(self, SE: np.ndarray) -> np.ndarray:
        """Convert the self-energy `SE` to the broadening matrix, see `Electrode.se2broadening` of `electrode`"""
        return self.electrode.se2broadening(SE)

    def _check_Ek(self, E: complex, k: KPoint, dtype, **kwargs) -> bool:
        """Check whether we have already runned this exact energy point.

        If not, then setup the data set for the next iteration.
        """
        data = self._data
        if hasattr(data, "E"):
            if (
                np.allclose(data.E, E)
                and np.allclose(data.k, k)
                and data.dtype == dtype
                and data.kwargs == kwargs
            ):
                # we have already prepared the calculation
                return True

        data.E = E
        data.Ed = E
        if np.isrealobj(E):
            data.Ed = E + 1j * self._eta_device
        data.k = np.asarray(k, dtype=np.float64)
        data.dtype = dtype
        data.kwargs = kwargs

        return False

    def _prepare(
        self, E: complex, k: KPoint = (0, 0, 0), dtype=np.complex128, **kwargs
    ) -> None:
        if self._check_Ek(E, k, dtype, **kwargs):
            return

        # Prepare the matrices
        data = self._data
        H = data.H

        data.SeH = H.Sk(k, dtype=dtype) * data.Ed - H.Pk(k, dtype=dtype, **kwargs)
        if data.bulk:
            # lil, since the bulk matrix may change the sparsity structure
            SeH = data.SeH.tolil()
            elec_r = data.elec[data.elec_r.ravel()]
            SeH[elec_r, elec_r.T] = self.electrode.bulk_matrix(
                E, k, dtype=dtype, **kwargs
            )[data.elec_r, data.elec_r.T]
            data.SeH = SeH.tocsr()

    def self_energy(
        self, E: complex, k: KPoint = (0, 0, 0), *, dtype=np.complex128, **kwargs
    ) -> np.ndarray:
        """Return the downfolded self-energy"""
        self._prepare(E, k, dtype, **kwargs)
        data = self._data
        se = self.electrode.self_energy(E, k=k, dtype=dtype, **kwargs)

        # now put it in the matrix, the (dense) self-energy may change
        # the sparsity structure, so it is subtracted as a sparse matrix
        elec = data.elec.ravel()
        n = len(elec)
        M = data.SeH - ssp.csr_array(
            (np.ravel(se), (np.repeat(elec, n), np.tile(elec, n))),
            shape=data.SeH.shape,
        )
        # Reduce to the downfolding region, in the BTD order
        M = M[data.down, data.down.T]

        def gM(M, idx1, idx2):
            return M[idx1, :][:, idx2].toarray()

        Mr = 0
        cblocks = data.blocks_cum0
        sli = slice(cblocks[0], cblocks[1])
        for b in range(1, len(self.blocks)):
            sli1 = slice(cblocks[b], cblocks[b + 1])

            # Do the downfolding of the self-energies
            Mr = gM(M, sli1, sli) @ solve(
                gM(M, sli, sli) - Mr,
                gM(M, sli, sli1),
                overwrite_a=True,
                overwrite_b=True,
            )
            sli = sli1

        if len(data.elec_d) > 0:
            # electrode orbitals in the last block (partially downfolded)
            Mr[data.last_d, data.last_d.T] += se[data.elec_d, data.elec_d.T]
        return Mr
