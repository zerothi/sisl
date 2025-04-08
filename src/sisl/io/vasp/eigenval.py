# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np

from sisl._internal import set_module
from sisl.messages import deprecate_argument
from sisl.typing import UnitsVar
from sisl.unit import serialize_units_arg, unit_convert

from ..sile import add_sile, sile_fh_open

# Import sile objects
from .sile import SileVASP

__all__ = ["eigenvalSileVASP"]


@set_module("sisl.io.vasp")
class eigenvalSileVASP(SileVASP):
    """Kohn-Sham eigenvalues"""

    @sile_fh_open()
    @deprecate_argument(
        "k",
        "ret_k",
        "use ret_k instead of k",
        "0.15",
        "0.17",
    )
    def read_data(self, ret_k: bool = False, units: UnitsVar = "eV"):
        r"""Read eigenvalues as calculated by VASP

        Parameters
        ----------
        ret_k :
           also return k points and weights
        units :
           selects units in the returned data

        Returns
        -------
        eigenvalues : numpy.ndarray
            all eigenvalues, shape ``(ns, nk, nb)``
            where ``ns`` number of spin-components, ``nk`` number of k-points and
            ``nb`` number of bands
        k_points : numpy.ndarray
            k-points (if `ret_k` is true), shape ``(nk, 3)``
        weights: numpy.ndarray
            weights for k-points (if `ret_k` is true), shape ``(nk)``
        """
        units = serialize_units_arg(units)
        eV2unit = unit_convert("eV", units["energy"])

        # read first line
        line = self.readline()  # NIONS, NIONS, NBLOCK * KBLOCK, NSPIN
        ns = int(line.split()[-1])
        self.readline()  # AOMEGA, LATT_CUR%ANORM(1:3) *1e-10, POTIM * 1e-15
        self.readline()  # TEMP
        self.readline()  # ' CAR '
        self.readline()  # name
        line = list(map(int, self.readline().split()))  # electrons, k-points, bands
        nk = line[1]
        nb = line[2]
        eigs = np.empty([ns, nk, nb], np.float64)
        k = np.empty([nk, 3], np.float64)
        w = np.empty([nk], np.float64)
        for ik in range(nk):
            self.readline()  # empty line
            line = self.readline().split()  # k-point, weight
            k[ik, :] = list(map(float, line[:3]))
            w[ik] = float(line[3])
            for ib in range(nb):
                # band, eig_UP, eig_DOWN, pop_UP, pop_DOWN
                # We currently neglect the populations
                E = map(float, self.readline().split()[1 : ns + 1])
                eigs[:, ik, ib] = list(E)

        eigs *= eV2unit

        if ret_k:
            return eigs, k, w
        return eigs


add_sile("EIGENVAL", eigenvalSileVASP, gzip=True)
