# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Block-tri-diagonal electrode handling

The electrode classes only calculate self-energies. They have no knowledge of
the device region, nor its pivoting. The rows (and columns) of the returned
self-energies are in the orbital order the electrode was created with,
`DeviceGreen` is responsible for placing them in the device region.
"""

from __future__ import annotations

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

from ._help import expand_btd, expand_orbs, get_expand

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
        mu: float = 0.0,
    ) -> Self:
        """Create the electrode from the ``%block TS.Elec.<name>`` in an fdf file

        The returned electrode uses a `sisl.physics.RecursiveSI` self-energy.
        The electrode Hamiltonian is shifted by ``delta-Ef`` and `mu`.

        Other options from the fdf file are stored in the ``info`` attribute:

        - ``bulk``: whether the electrode is bulk in the downfolding region
        - ``dEf``: the ``delta-Ef`` shift of the electrode Hamiltonian
        - ``mu``: the chemical potential shift (`mu`)
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
           chemical potential of the electrode, the electrode Hamiltonian is shifted
           by this value [eV]
        """
        from sisl.unit.siesta import unit_convert

        if not isinstance(fdf, si.BaseSile):
            fdf = si.io.siesta.fdfSileSiesta(fdf)
        is_tbtrans = prefix.upper() == "TBT"
        clsname = f"{cls.__name__}.from_fdf(electrode={name})"

        def get_line(line):
            """Parse lines in the %block constructs of fdf's"""
            key, val = line.split(None, 1)
            return key.lower().strip(), val.split("#", 1)[0].strip()

        def block_get(dic, key, default=None, unit=None):
            # keys are stored in lower-case, see get_line
            key = key.lower()
            ret = dic.get(key, default)
            if is_tbtrans:
                # TBT prefixed keys take precedence
                ret = dic.get(f"tbt.{key}", ret)
            if unit is None or not isinstance(ret, str):
                return ret
            ret, un = ret.split()
            return float(ret) * unit_convert(un, unit)

        tbt_prefix = f"TBT.Elec.{name}"
        ts_prefix = f"TS.Elec.{name}"

        block = fdf.get(f"{ts_prefix}")
        Helec = fdf.get(f"{ts_prefix}.HS")
        bulk = fdf.get("TS.Elecs.Bulk", True)
        eta = fdf.get("TS.Elecs.Eta", 1e-3, unit="eV")
        bloch = [1, 1, 1]
        for i in range(3):
            bloch[i] = fdf.get(f"{ts_prefix}.Bloch.A{i+1}", 1)
        if is_tbtrans:
            block = fdf.get(f"{tbt_prefix}", block)
            Helec = fdf.get(f"{tbt_prefix}.HS", Helec)
            bulk = fdf.get("TBT.Elecs.Bulk", bulk)
            eta = fdf.get("TBT.Elecs.Eta", eta, unit="eV")
            for i in range(3):
                bloch[i] = fdf.get(f"{tbt_prefix}.Bloch.A{i+1}", bloch[i])
        if block is None:
            raise ValueError(f"{clsname} could not find block: {ts_prefix}")

        # Convert to key value based function
        dic = {key: val for key, val in map(get_line, block)}

        # Retrieve data
        for key in ("hs", "hs-file", "tshs", "tshs-file"):
            Helec = block_get(dic, key, Helec)
        if Helec:
            Helec = si.get_sile(Helec).read_hamiltonian()
        else:
            raise ValueError(
                f"{clsname} could not find electrode HS in block: {ts_prefix}"
            )

        # Get semi-infinite direction
        semi_inf = None
        for suf in ("-direction", "-dir", ""):
            semi_inf = block_get(dic, f"semi-inf{suf}", semi_inf)
        if semi_inf is None:
            raise ValueError(
                f"{clsname} could not find electrode semi-inf-direction "
                f"in block: {ts_prefix}"
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

        info = PropertyDict()
        info.bulk = bool(block_get(dic, "bulk", bulk))
        # loop for 0
        for i, sufs in enumerate([("a", "a1"), ("b", "a2"), ("c", "a3")]):
            for suf in sufs:
                bloch[i] = block_get(dic, f"bloch-{suf}", bloch[i])

        bloch = [
            int(b)
            for b in block_get(
                dic, "bloch", f"{bloch[0]} {bloch[1]} {bloch[2]}"
            ).split()
        ]

        eta = block_get(dic, "eta", eta, unit="eV")
        # manual shift of the fermi-level
        info.dEf = block_get(dic, "delta-Ef", 0.0, unit="eV")
        Helec.shift(info.dEf)
        # shift according to potential
        info.mu = mu
        Helec.shift(mu)
        # add a fraction of the bias in the coupling elements of the
        # E-C region, only meaningful for
        info.V_fraction = block_get(dic, "V-fraction", 0.0)
        if info.V_fraction > 0.0:
            warn(
                f"{clsname} found a non-zero V-fraction value. "
                "This is currently not implemented."
            )

        elec = cls(name, si.RecursiveSI(Helec, semi_inf, eta=eta), bloch=bloch)
        elec.info = info
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
    the block-tri-diagonal order given by `btd_down`.
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
    btd_down :
        BTD block sizes of `orbs_down`, the last block are the orbitals that the
        self-energy is downfolded to
    eta_device :
        imaginary part used in the downfolding region (for real energies)
    bulk :
        whether the electrode region of `H` is replaced by the
        (Bloch expanded) electrode Hamiltonian. Requires the `electrode` to
        be a `BlochSelfEnergy` with a `sisl.physics.SemiInfinite` self-energy.
    name :
        name of the electrode, defaults to the `electrode` name
    """

    def __init__(
        self,
        electrode: ElectrodeSelfEnergy,
        H: si.Hamiltonian,
        orbs_down: npt.ArrayLike,
        orbs_elec: npt.ArrayLike,
        btd_down: npt.ArrayLike,
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
        btd_down = np.asarray(btd_down).ravel()
        cls = self.__class__.__name__
        if btd_down.sum() != len(orbs_down):
            raise ValueError(
                f"{cls} btd_down must sum to the length of orbs_down "
                f"({btd_down.sum()} != {len(orbs_down)})"
            )
        if len(np.unique(orbs_down)) != len(orbs_down):
            raise ValueError(f"{cls} orbs_down has duplicate orbitals")
        if len(btd_down) < 2:
            raise ValueError(f"{cls} btd_down must have at least 2 blocks")
        if not np.isin(orbs_elec, orbs_down[: -btd_down[-1]]).all():
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
                and hasattr(electrode.se, "spgeom0")
            ):
                raise ValueError(
                    f"{cls}(bulk=True) requires a BlochSelfEnergy with a "
                    "SemiInfinite self-energy."
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
        self.btd = expand_btd(btd_down, expand)
        data.btd_cum0 = np.append(0, np.cumsum(self.btd))

    def __len__(self) -> int:
        """Size of the self-energy matrix (including spin)"""
        return self.btd[-1]

    def __str__(self) -> str:
        se = str(self.electrode).replace("\n", "\n ")
        return f"{self.__class__.__name__}{{name: {self.name}, no: {len(self)}, blocks: {len(self.btd)}, eta_device: {self._eta_device}, bulk: {self._data.bulk},\n {se}\n}}"

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
        cbtd = data.btd_cum0
        sli = slice(cbtd[0], cbtd[1])
        for b in range(1, len(self.btd)):
            sli1 = slice(cbtd[b], cbtd[b + 1])

            # Do the downfolding of the self-energies
            Mr = gM(M, sli1, sli) @ solve(
                gM(M, sli, sli) - Mr,
                gM(M, sli, sli1),
                overwrite_a=True,
                overwrite_b=True,
            )
            sli = sli1
        return Mr
