# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from sisl._internal import set_module

from ..sile import add_sile
from .tbt import ElecType, Ry2eV, Ry2K, tbtavncSileTBtrans, tbtncSileTBtrans

__all__ = ["phtncSilePHtrans", "phtavncSilePHtrans"]


@set_module("sisl.io.phtrans")
class phtncSilePHtrans(tbtncSileTBtrans):
    """PHtrans file object"""

    _trans_type = "PHT"
    _E2eV = Ry2eV**2

    def phonon_temperature(self, elec: ElecType) -> float:
        """Phonon bath temperature [Kelvin]"""
        return self._value("kT", self._elec(elec))[0] * Ry2K

    def kT(self, elec: ElecType) -> float:
        """Phonon bath temperature [eV]"""
        return self._value("kT", self._elec(elec))[0] * Ry2eV


@set_module("sisl.io.phtrans")
class phtavncSilePHtrans(tbtavncSileTBtrans):
    """PHtrans file object"""

    _trans_type = "PHT"
    _E2eV = Ry2eV**2

    def phonon_temperature(self, elec: ElecType) -> float:
        """Phonon bath temperature [Kelvin]"""
        return self._value("kT", self._elec(elec))[0] * Ry2K

    def kT(self, elec: ElecType) -> float:
        """Phonon bath temperature [eV]"""
        return self._value("kT", self._elec(elec))[0] * Ry2eV


for _name in (
    "chemical_potential",
    "electron_temperature",
    "kT",
    "orbital_current",
    "bond_current",
    "vector_current",
    "current",
    "current_parameter",
    "shot_noise",
    "noise_power",
):
    # TODO change this such that the intrinsic details
    # are separated.
    setattr(phtncSilePHtrans, _name, None)
    setattr(phtavncSilePHtrans, _name, None)


add_sile("PHT.nc", phtncSilePHtrans)
add_sile("PHT.AV.nc", phtavncSilePHtrans)
