# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Block-tri-diagonal electrode handling

The electrode classes only calculate self-energies. They have no knowledge of
the device region, nor its permutation. The rows (and columns) of the returned
self-energies are in the orbital order the electrode was created with,
`DeviceGreen` is responsible for placing them in the device region.
"""

from __future__ import annotations

import re
from typing import Literal, Optional, Self

import numpy as np
import numpy.typing as npt

import sisl as si
import sisl.io.siesta
import sisl.io.siesta.binaries
from sisl._internal import set_module
from sisl.linalg import solve
from sisl.messages import warn
from sisl.typing import KPoint
from sisl.typing._core import SileLike
from sisl.utils.misc import PropertyDict

from ._help import expand_blocks, expand_orbs, get_expand

__all__ = ["ElectrodeSelfEnergy", "BlochSelfEnergy", "DownfoldSelfEnergy"]


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


@set_module("sisl_toolbox.btd")
class ElectrodeSelfEnergy(si.physics.SelfEnergy):
    """Base class for electrode self-energies used in `DeviceGreen`

    Sub-classes must implement `__len__` and `self_energy`.

    Parameters
    ----------
    name :
        name of the electrode
    """

    def __init__(self, name: str):
        self.name = name
        # additional information, e.g. from `BlochSelfEnergy.from_fdf`
        self.info = PropertyDict()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}{{name: {self.name}, no: {len(self)}}}"

    def broadening_matrix(self, *args, **kwargs) -> np.ndarray:
        """Return the broadening matrix, see `self_energy` for arguments"""
        return self.se2broadening(self.self_energy(*args, **kwargs))


@set_module("sisl_toolbox.btd")
class BlochSelfEnergy(ElectrodeSelfEnergy):
    """Self-energy of a (possibly Bloch expanded) electrode

    The rows of the self-energy are the orbitals of the Bloch expanded electrode,
    i.e. matching ``se.spgeom0.tile(...)`` for the `bloch` expansions.

    Parameters
    ----------
    name :
        name of the electrode
    se :
        the self-energy calculator, a `sisl.physics.SelfEnergy` or a
        Siesta GF file (`~sisl.io.siesta.gfSileSiesta`).
        A GF file is already Bloch expanded.
    bloch :
        Bloch expansion of the self-energy
    """

    def __init__(
        self,
        name: str,
        se,
        bloch: si.Bloch | tuple[int, int, int] | None = None,
    ):
        super().__init__(name)
        self.se = se

        if bloch is None:
            bloch = (1, 1, 1)
        if not isinstance(bloch, si.Bloch):
            bloch = si.Bloch(bloch)
        self.bloch = bloch

        if isinstance(se, sisl.io.siesta.binaries._gfSileSiesta):
            if len(bloch) > 1:
                raise ValueError(
                    f"{self.__class__.__name__} got a GF file and a Bloch expansion, "
                    "GF files are already Bloch expanded."
                )
        elif not isinstance(se, si.physics.SelfEnergy):
            raise ValueError(f"{self.__class__.__name__} got an unknown 'se' argument.")

    @classmethod
    def from_fdf(
        cls,
        fdf: SileLike,
        name: str,
        prefix: Literal["TBT", "TS"] = "TBT",
        mu: Optional[float] = None,
    ) -> Self:
        """Create the electrode from the ``%block TS.Elec.<name>`` in an fdf file

        The returned electrode uses a `sisl.physics.RecursiveSI` self-energy.
        The electrode Hamiltonian is shifted by ``delta-Ef`` and the chemical potential.
        File names are relative to the directory of the fdf file.

        Other options from the fdf file are stored in the ``info`` attribute:

        - ``bulk``: whether the electrode is bulk in the downfolding region
        - ``dEf``: the ``delta-Ef`` shift of the electrode Hamiltonian
        - ``mu``: the chemical potential shift
        - ``chemical_potential``: name of the chemical potential (or None)
        - ``position``: the ``electrode-position`` value (or None)
        - ``V_fraction``: the fraction of the bias in the electrode couplings

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
        """
        if not isinstance(fdf, si.BaseSile):
            fdf = si.io.siesta.fdfSileSiesta(fdf)
        is_tbtrans = prefix.upper() == "TBT"
        clsname = f"{cls.__name__}.from_fdf(electrode={name})"

        def fdf_get(key, default=None, **kwargs):
            """Get TS.<key>, which is overwritten by TBT.<key>"""
            ret = fdf.get(f"TS.{key}", default, **kwargs)
            if is_tbtrans:
                ret = fdf.get(f"TBT.{key}", ret, **kwargs)
            return ret

        def fdf_block(key) -> str:
            """The label of the block TS.<key> or TBT.<key> (if present)"""
            if is_tbtrans and fdf.type(f"TBT.{key}") == "B":
                return f"TBT.{key}"
            return f"TS.{key}"

        block = fdf_block(f"Elec.{name}")
        if fdf.type(block) != "B":
            raise ValueError(f"{clsname} could not find block: {block}")

        def block_get(key, default=None, **kwargs):
            ret = fdf.block_get(block, key, default, **kwargs)
            if is_tbtrans:
                # TBT prefixed keys take precedence
                ret = fdf.block_get(block, f"tbt.{key}", ret, **kwargs)
            return ret

        Helec = fdf_get(f"Elec.{name}.HS")
        for key in ("hs", "hs-file", "tshs", "tshs-file"):
            Helec = block_get(key, Helec)
        if not Helec:
            raise ValueError(f"{clsname} could not find electrode HS in block: {block}")
        Helec = si.get_sile(fdf.dir_file(Helec)).read_hamiltonian()

        # Get semi-infinite direction
        semi_inf = None
        for suf in ("-direction", "-dir", ""):
            semi_inf = block_get(f"semi-inf{suf}", semi_inf)
        if semi_inf is None:
            raise ValueError(
                f"{clsname} could not find electrode semi-inf-direction "
                f"in block: {block}"
            )
        # convert to sisl infinite
        semi_inf = semi_inf.lower()
        semi_inf = semi_inf[0] + {"a1": "a", "a2": "b", "a3": "c"}.get(
            semi_inf[1:], semi_inf[1:]
        )
        # Check that semi_inf is a recursive one!
        if semi_inf not in ("-a", "+a", "-b", "+b", "-c", "+c"):
            raise NotImplementedError(
                f"{clsname} does not implement other "
                "self energies than the recursive one."
            )

        used_atoms = block_get("used-atoms")
        if used_atoms is not None and int(used_atoms) != Helec.na:
            raise NotImplementedError(
                f"{clsname} does not implement used-atoms different "
                "from the number of atoms in the electrode."
            )

        info = PropertyDict()
        info.bulk = bool(block_get("bulk", fdf_get("Elecs.Bulk", True)))

        bloch = [fdf_get(f"Elec.{name}.Bloch.A{i+1}", 1) for i in range(3)]
        for i, sufs in enumerate([("a", "a1"), ("b", "a2"), ("c", "a3")]):
            for suf in sufs:
                bloch[i] = block_get(f"bloch-{suf}", bloch[i])
        bloch = block_get("bloch", f"{bloch[0]} {bloch[1]} {bloch[2]}")
        bloch = [int(b) for b in bloch.split()]

        eta = block_get("eta", fdf_get("Elecs.Eta", 1e-3, unit="eV"), unit="eV")

        info.position = None
        for key in ("electrode-position", "elec-pos"):
            info.position = block_get(key, info.position)
        if info.position is not None:
            info.position = str(info.position)

        info.chemical_potential = None
        for key in ("chemical-potential", "chem-pot", "mu"):
            info.chemical_potential = block_get(key, info.chemical_potential)
        if mu is None:
            chem_pots = []
            for line in fdf_get("ChemPots") or []:
                line = line.split("#", 1)[0].strip()
                if line:
                    chem_pots.append(line)

            # Same defaults as TranSiesta/TBtrans: the only chemical potential,
            # or the one with the same name as the electrode
            cp = info.chemical_potential
            if cp is None:
                cp = chem_pots[0] if len(chem_pots) == 1 else name

            cp_block = fdf_block(f"ChemPot.{cp}")
            if fdf.type(cp_block) == "B":
                mu = fdf.block_get(cp_block, "mu", 0.0, unit="eV")
                if is_tbtrans:
                    mu = fdf.block_get(cp_block, "tbt.mu", mu, unit="eV")
            elif not chem_pots and cp.lower() in ("left", "right"):
                # TranSiesta fakes the Left/Right chemical potentials
                # when none are specified
                mu = {"left": "V/2", "right": "-V/2"}[cp.lower()]
            else:
                raise ValueError(
                    f"{clsname} could not find the chemical-potential block: {cp_block}"
                )
            if isinstance(mu, str):
                mu = _fdf_mu(mu, fdf_get("Voltage", 0.0, unit="eV"))

        # manual shift of the fermi-level
        info.dEf = block_get("delta-Ef", 0.0, unit="eV")
        Helec.shift(info.dEf)
        # shift according to potential
        info.mu = mu
        Helec.shift(mu)
        # add a fraction of the bias in the coupling elements of the
        # E-C region, only meaningful for
        info.V_fraction = float(block_get("V-fraction", 0.0))
        if info.V_fraction > 0.0:
            warn(
                f"{clsname} found a non-zero V-fraction value. "
                "This is currently not implemented."
            )

        elec = cls(name, si.RecursiveSI(Helec, semi_inf, eta=eta), bloch=bloch)
        elec.info.update(info)
        return elec

    @property
    def eta(self) -> float | None:
        """Imaginary part used for the self-energy calculation (if known)"""
        return getattr(self.se, "eta", None)

    def __len__(self) -> int:
        """Size of the self-energy matrix (including spin)"""
        if isinstance(self.se, sisl.io.siesta.binaries._gfSileSiesta):
            # Includes spin (non-colinear) and Bloch expansions
            return self.se._no_u
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
        if isinstance(self.se, sisl.io.siesta.binaries._gfSileSiesta):
            # GF files have no dtype argument
            dtype = kwargs.pop("dtype", None)
            SE = self.se.self_energy(E, k=k, **kwargs)
            if dtype is None:
                return SE
            return SE.astype(dtype, copy=False)
        return self._call_bloch(self.se.self_energy, k, E=E, **kwargs)


@set_module("sisl_toolbox.btd")
class DownfoldSelfEnergy(ElectrodeSelfEnergy):
    r"""Self-energy of an electrode downfolded through a region of `H`

    The downfolding region is described by the orbitals `orbs_down` of `H`, in
    the block-tri-diagonal order given by `blocks_down`.
    The electrode self-energy is placed on the orbitals `orbs_elec` of `H`
    (which must be part of `orbs_down`), and is then downfolded to the
    orbitals of the last block.

    The rows of the returned self-energy are the orbitals of the last block,
    in the order they appear in `orbs_down`.

    Parameters
    ----------
    electrode :
        the electrode self-energy
    H :
        Hamiltonian containing (at least) the downfolding region
    orbs_down :
        orbitals of `H` in the downfolding region (including the electrode
        and the returned orbitals), in the order of the BTD blocks
    orbs_elec :
        orbitals of `H` corresponding to the rows of the electrode self-energy
    blocks_down :
        BTD block sizes of `orbs_down`, the last block are the orbitals that the
        self-energy is downfolded to
    eta_device :
        imaginary part used in the downfolding region (for real energies)
    bulk :
        whether the electrode region of `H` is replaced by the
        (Bloch expanded) electrode Hamiltonian. Requires the `electrode` to
        be a `BlochSelfEnergy` with a `sisl.physics.RecursiveSI` self-energy.
    name :
        name of the electrode, defaults to the `electrode` name
    """

    def __init__(
        self,
        electrode: ElectrodeSelfEnergy,
        H: si.Hamiltonian,
        orbs_down: npt.ArrayLike,
        orbs_elec: npt.ArrayLike,
        blocks_down: npt.ArrayLike,
        eta_device: float = 0.0,
        bulk: bool = True,
        name: Optional[str] = None,
    ):
        if name is None:
            name = electrode.name
        super().__init__(name)
        self.electrode = electrode
        self._eta_device = eta_device

        orbs_down = np.asarray(orbs_down).ravel()
        orbs_elec = np.asarray(orbs_elec).ravel()
        blocks_down = np.asarray(blocks_down).ravel()
        cls = self.__class__.__name__
        if blocks_down.sum() != len(orbs_down):
            raise ValueError(
                f"{cls} blocks_down must sum to the length of orbs_down "
                f"({blocks_down.sum()} != {len(orbs_down)})"
            )
        if len(np.unique(orbs_down)) != len(orbs_down):
            raise ValueError(f"{cls} orbs_down has duplicate orbitals")
        if len(blocks_down) < 2:
            raise ValueError(f"{cls} blocks_down must have at least 2 blocks")
        if not np.isin(orbs_elec, orbs_down[: -blocks_down[-1]]).all():
            raise ValueError(
                f"{cls} orbs_elec must be a subset of orbs_down, excluding the last block"
            )

        # storage data
        self._data = PropertyDict()
        data = self._data
        data.bulk = bulk

        if bulk:
            if not (
                isinstance(electrode, BlochSelfEnergy)
                and isinstance(electrode.se, si.physics.RecursiveSI)
            ):
                raise ValueError(
                    f"{cls}(bulk=True) requires a BlochSelfEnergy with a "
                    "RecursiveSI self-energy."
                )

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
        self.blocks = expand_blocks(blocks_down, expand)
        data.blocks_cum0 = np.append(0, np.cumsum(self.blocks))

    def __len__(self) -> int:
        """Size of the self-energy matrix (including spin)"""
        return self.blocks[-1]

    def __str__(self) -> str:
        se = str(self.electrode).replace("\n", "\n ")
        return f"{self.__class__.__name__}{{name: {self.name}, no: {len(self)}, blocks: {len(self.blocks)}, eta_device: {self._eta_device}, bulk: {self._data.bulk},\n {se}\n}}"

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
        data.Eb = E
        if np.isrealobj(E):
            data.Ed = E + 1j * self._eta_device
            eta = getattr(self.electrode, "eta", None)
            if eta is not None:
                data.Eb = E + 1j * eta
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
            Eb = data.Eb
            Helec = self.electrode.se.spgeom0

            def hsk(k, **kwargs):
                # constructor for the H and S part
                return Helec.Sk(k, format="array", dtype=dtype) * Eb - Helec.Hk(
                    k, format="array", dtype=dtype, **kwargs
                )

            data.SeH[data.elec, data.elec.T] = self.electrode._call_bloch(
                hsk, k, **kwargs
            )

    def self_energy(
        self, E: complex, k: KPoint = (0, 0, 0), *, dtype=np.complex128, **kwargs
    ) -> np.ndarray:
        """Return the downfolded self-energy"""
        self._prepare(E, k, dtype, **kwargs)
        data = self._data
        se = self.electrode.self_energy(E, k=k, dtype=dtype, **kwargs)

        # now put it in the matrix
        M = data.SeH.copy()
        M[data.elec, data.elec.T] -= se
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
        return Mr
