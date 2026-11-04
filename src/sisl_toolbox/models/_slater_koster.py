# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import re
from typing import Optional

import numpy as np

import sisl as si
from sisl.geom import NeighborFinder
from sisl_toolbox.models._base import BaseModel, ReferenceDispatch

__all__ = ["SlaterKosterHamiltonian", "slater_koster"]


class SlaterKosterHamiltonian(BaseModel):
    """Hamiltonians from published Slater-Koster parameterizations"""

    ref = BaseModel.ref.copy()


def _hamiltonian(
    sk: si.SlaterKoster, geometry: si.Geometry, atoms: list[si.Atom]
) -> si.Hamiltonian:
    """Hamiltonian of `geometry` with its atoms replaced by `atoms` (by atomic number)"""
    by_Z = {atom.Z: atom for atom in atoms}
    missing = set(geometry.atoms.Z) - set(by_Z)
    if missing:
        raise ValueError(
            f"the model has no parameters for the atomic numbers {sorted(missing)}"
        )
    geometry = si.Geometry(
        geometry.xyz, [by_Z[Z] for Z in geometry.atoms.Z], lattice=geometry.lattice
    )
    # room for all elements of a row, sparse rows grow slowly
    n = NeighborFinder(geometry.translate2uc(), R=sk.R).find_neighbors().n_neighbors
    nnzpr = int(n.max() + 1) * int(geometry.atoms.orbitals.max())
    H = si.Hamiltonian(geometry, nnzpr=nnzpr, orthogonal=sk.orthogonal)
    H.construct(sk)
    return H


def _period(Z: int) -> int:
    return int(np.searchsorted([2, 10, 18, 36, 54, 86], Z)) + 1


# Vogl, Hjalmarson, Dow, Table 1 (eV):
# E(s,a) E(p,a) E(s,c) E(p,c) E(s*,a) E(s*,c)
# V(s,s) V(x,x) V(x,y) V(sa,pc) V(sc,pa) V(s*a,pc) V(pa,s*c)
# and Table 2: the bond length d (Ang)
# fmt: off
_VOGL1983 = {
    "C": ((-4.5450, 3.8400, -4.5450, 3.8400, 11.3700, 11.3700),
          (-22.7250, 3.8400, 11.6700, 15.2206, 15.2206, 8.2109, 8.2109), 1.54),
    "Si": ((-4.2000, 1.7150, -4.2000, 1.7150, 6.6850, 6.6850),
           (-8.3000, 1.7150, 4.5750, 5.7292, 5.7292, 5.3749, 5.3749), 2.35),
    "Ge": ((-5.8800, 1.6100, -5.8800, 1.6100, 6.3900, 6.3900),
           (-6.7800, 1.6100, 4.9000, 5.4649, 5.4649, 5.2191, 5.2191), 2.45),
    "Sn": ((-5.6700, 1.3300, -5.6700, 1.3300, 5.9000, 5.9000),
           (-5.6700, 1.3300, 4.0800, 4.5116, 4.5116, 5.8939, 5.8939), 2.81),
    "SiC": ((-8.4537, 2.1234, -4.8463, 4.3466, 9.6534, 9.3166),
            (-12.4197, 3.0380, 5.9216, 9.4900, 9.2007, 8.7138, 4.4051), 1.88),
    "AlP": ((-7.8466, 1.3169, -1.2534, 4.2831, 8.7069, 7.4231),
            (-7.4535, 2.3749, 4.8378, 5.2451, 5.2775, 5.2508, 4.6388), 2.36),
    "AlAs": ((-7.5273, 0.9833, -1.1627, 3.5867, 7.4833, 6.7267),
             (-6.6642, 1.8780, 4.2919, 5.1106, 5.4965, 4.5216, 4.9950), 2.45),
    "AlSb": ((-6.1714, 0.9807, -2.0716, 3.0163, 6.7607, 6.1543),
             (-5.6448, 1.7199, 3.6648, 4.9121, 4.2137, 4.3662, 3.0739), 2.66),
    "GaP": ((-8.1124, 1.1250, -2.1976, 4.1150, 8.5150, 7.1850),
            (-7.4709, 2.1516, 5.1369, 4.2771, 6.3190, 4.6541, 5.0950), 2.36),
    "GaAs": ((-8.3431, 1.0414, -2.6569, 3.6686, 8.5914, 6.7386),
             (-6.4513, 1.9546, 5.0779, 4.4800, 5.7839, 4.8422, 4.8077), 2.45),
    "GaSb": ((-7.3207, 0.8554, -3.8993, 2.9146, 6.6354, 5.9846),
             (-6.1567, 1.5789, 4.1285, 4.9601, 4.6675, 4.9895, 4.2180), 2.64),
    "InP": ((-8.5274, 0.8735, -1.4826, 4.0465, 8.2635, 7.0665),
            (-5.3614, 1.8801, 4.2324, 2.2265, 5.5825, 3.4623, 4.4814), 2.54),
    "InAs": ((-9.5381, 0.9099, -2.7219, 3.7201, 7.4099, 6.7401),
             (-5.6052, 1.8398, 4.4693, 3.0354, 5.4389, 3.3744, 3.9097), 2.62),
    "InSb": ((-8.0157, 0.6738, -3.4643, 2.9162, 6.4530, 5.9362),
             (-5.5193, 1.4018, 3.8761, 3.7880, 4.5900, 3.5666, 3.4048), 2.81),
    "ZnSe": ((-11.8383, 1.5072, 0.0183, 5.9928, 7.5872, 8.9928),
             (-6.2163, 3.0054, 5.9942, 3.4980, 6.3191, 2.5891, 3.9533), 2.45),
    "ZnTe": ((-9.8150, 1.4834, -0.9350, 5.2666, 7.0834, 8.2666),
             (-6.5765, 2.7951, 5.4670, 5.9827, 5.8199, 1.3196, 0.0000), 2.64),
}
# fmt: on


class Vogl1983Dispatch(ReferenceDispatch):
    r"""The :math:`sp^3s^*` model of Vogl, Hjalmarson and Dow, :doi:`10.1016/0022-3697(83)90064-1`"""

    doi = "10.1016/0022-3697(83)90064-1"

    def dispatch(
        self, material: str = "Si", geometry: Optional[si.Geometry] = None
    ) -> si.Hamiltonian:
        r"""Nearest neighbor :math:`sp^3s^*` Hamiltonian of a diamond or zincblende semiconductor

        The orbitals are :math:`ns`, :math:`np` and :math:`(n+1)s` (:math:`s^*`),
        with :math:`n` the period of the element.

        Parameters
        ----------
        material :
            one of C, Si, Ge, Sn, SiC, AlP, AlAs, AlSb, GaP, GaAs, GaSb, InP, InAs,
            InSb, ZnSe, ZnTe. A compound is written cation first.
        geometry :
            the geometry; its atoms get the orbitals by atomic number.
            Default is the crystal with the bond length of the paper.
        """
        E, V, d = _VOGL1983[material]
        cation, anion = (re.findall("[A-Z][a-z]?", material) * 2)[:2]
        Es_a, Ep_a, Es_c, Ep_c, Ess_a, Ess_c = E
        Vss, Vxx, Vxy, Vsa_pc, Vsc_pa, Vssa_pc, Vpa_ssc = V
        # 4 nearest neighbors along (1, 1, 1) / sqrt(3)
        sp = 3**0.5 / 4
        R = d * (1 + (8 / 3) ** 0.5) / 2

        def atom(symbol):
            Z = si.Atom(symbol).Z
            n = _period(Z)
            orbs = [si.AtomicOrbital(n=n, l=0, R=R)]
            orbs += [si.AtomicOrbital(n=n, l=1, m=m, R=R) for m in (-1, 0, 1)]
            orbs.append(si.AtomicOrbital(n=n + 1, l=0, R=R))
            return si.Atom(Z, orbs, tag=symbol), f"{n}s", f"{n}p", f"{n + 1}s"

        A, sa, pa, ssa = atom(anion)
        C, sc, pc, ssc = atom(cation)
        a, c = A.tag, C.tag
        hopping = {
            (a, c): {
                (sa, sc): {"sigma": Vss / 4},
                (pa, pc): {"sigma": (Vxx + 2 * Vxy) / 4, "pi": (Vxx - Vxy) / 4},
                (sa, pc): {"sigma": sp * Vsa_pc},
                (ssa, pc): {"sigma": sp * Vssa_pc},
            },
        }
        if a != c:
            # an element has V(sc,pa) == V(sa,pc): the reverse key gives it
            hopping[(c, a)] = {
                (sc, pa): {"sigma": sp * Vsc_pa},
                (ssc, pa): {"sigma": sp * Vpa_ssc},
            }
        onsite = {
            a: {sa: Es_a, pa: Ep_a, ssa: Ess_a},
            c: {sc: Es_c, pc: Ep_c, ssc: Ess_c},
        }
        sk = si.SlaterKoster(hopping, onsite=onsite, R=R)
        if geometry is None:
            geometry = si.geom.diamond(4 * d / 3**0.5, [A, C])
        return _hamiltonian(sk, geometry, [A, C])


SlaterKosterHamiltonian.ref.register("Vogl1983", Vogl1983Dispatch)
SlaterKosterHamiltonian.ref.register(Vogl1983Dispatch.doi, Vogl1983Dispatch)


def _xu1992_scaling(r):
    """Distance scaling of the hopping of Xu 1992 (1 at r0, 0 at rm)"""
    r0, n, nc, rc, r1 = 1.536329, 2.0, 6.5, 2.18, 2.45
    c = (6.7392620074314e-3, -8.1885359517898e-2, 0.1932365259144, 0.354287433238)
    x = r - r1
    tail = c[0] + x * (c[1] + x * (c[2] + x * c[3]))
    with np.errstate(over="ignore", divide="ignore"):
        gsp = (r0 / r) ** n * np.exp(n * ((r0 / rc) ** nc - (r / rc) ** nc))
    return np.where(r < r1, gsp, tail)


class Xu1992Dispatch(ReferenceDispatch):
    r"""The orthogonal :math:`sp^3` carbon model of Xu, Wang, Chan and Ho, :doi:`10.1088/0953-8984/4/28/006`"""

    doi = "10.1088/0953-8984/4/28/006"

    def dispatch(self, geometry: Optional[si.Geometry] = None) -> si.Hamiltonian:
        r"""Orthogonal :math:`sp^3` Hamiltonian of carbon, with hopping up to 2.6 Ang

        Only the Hamiltonian; the repulsive energy of the paper is not included.

        Parameters
        ----------
        geometry :
            the geometry; its carbon atoms get the :math:`2s` and :math:`2p` orbitals.
            Default is diamond with the bond length 1.536329 Ang.
        """
        R = 2.6
        orbs = [si.AtomicOrbital(n=2, l=0, R=R)]
        orbs += [si.AtomicOrbital(n=2, l=1, m=m, R=R) for m in (-1, 0, 1)]
        C = si.Atom(6, orbs, tag="C")

        def V(V0):
            return lambda r: V0 * _xu1992_scaling(r)

        hopping = {
            ("C", "C"): {
                ("2s", "2s"): {"sigma": V(-5.0)},
                ("2s", "2p"): {"sigma": V(4.7)},
                ("2p", "2p"): {"sigma": V(5.5), "pi": V(-1.55)},
            }
        }
        sk = si.SlaterKoster(hopping, onsite={"C": {"2s": -2.99, "2p": 3.71}}, R=R)
        if geometry is None:
            geometry = si.geom.diamond(4 * 1.536329 / 3**0.5, C)
        return _hamiltonian(sk, geometry, [C])


SlaterKosterHamiltonian.ref.register("Xu1992", Xu1992Dispatch)
SlaterKosterHamiltonian.ref.register(Xu1992Dispatch.doi, Xu1992Dispatch)


slater_koster = si.utils.PropertyDict()
slater_koster.hamiltonian = SlaterKosterHamiltonian()
slater_koster.H = slater_koster.hamiltonian
